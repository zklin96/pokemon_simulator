"""Benchmark suite for VGC AI.

This module provides automated benchmarking against
a suite of baseline opponents.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from loguru import logger

from .evaluator import VGCEvaluator, EvaluationResult


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    
    model_id: str
    timestamp: str
    results: Dict[str, EvaluationResult]
    aggregate: Dict[str, float] = field(default_factory=dict)
    
    @property
    def overall_win_rate(self) -> float:
        """Aggregate win rate across all opponents."""
        if not self.results:
            return 0.0
        rates = [r.win_rate for r in self.results.values()]
        return np.mean(rates)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "timestamp": self.timestamp,
            "overall_win_rate": self.overall_win_rate,
            "aggregate": self.aggregate,
            "results": {k: v.to_dict() for k, v in self.results.items()},
        }
    
    def summary(self) -> str:
        """Get summary string."""
        lines = [
            f"Benchmark: {self.model_id}",
            f"Timestamp: {self.timestamp}",
            f"Overall Win Rate: {self.overall_win_rate:.1%}",
            "",
            "Per-Opponent Results:",
        ]
        
        for opp_id, result in sorted(self.results.items(), key=lambda x: -x[1].win_rate):
            lines.append(f"  {opp_id}: {result.win_rate:.1%}")
        
        return "\n".join(lines)


class BenchmarkSuite:
    """Automated benchmark suite.
    
    Evaluates models against a standard set of opponents
    for consistent comparison.
    
    Example:
        suite = BenchmarkSuite()
        
        result = suite.run(model, model_id="my_model")
        print(result.summary())
        
        # Compare multiple models
        comparison = suite.compare([model1, model2], ["v1", "v2"])
    """
    
    def __init__(
        self,
        opponents: Optional[Dict[str, Callable]] = None,
        games_per_opponent: int = 50,
        save_dir: Optional[Path] = None,
    ):
        """Initialize benchmark suite.
        
        Args:
            opponents: Dict of opponent_id -> factory function
            games_per_opponent: Games to play per opponent
            save_dir: Directory to save results
        """
        self.opponents = opponents or self._default_opponents()
        self.games_per_opponent = games_per_opponent
        self.save_dir = save_dir
        
        self.evaluator = VGCEvaluator(save_dir=save_dir / "evals" if save_dir else None)
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
    
    def _default_opponents(self) -> Dict[str, Callable]:
        """Get default opponents."""
        def random_agent():
            from ..ml.battle_ai.agents import RandomAgent
            return RandomAgent(action_dim=144)
        
        def heuristic_agent():
            from ..ml.battle_ai.agents import HeuristicAgent
            return HeuristicAgent()
        
        return {
            "random": random_agent,
            "heuristic": heuristic_agent,
        }
    
    def run(
        self,
        model: Any,
        model_id: str = "model",
    ) -> BenchmarkResult:
        """Run full benchmark suite.
        
        Args:
            model: Model to benchmark
            model_id: Model identifier
            
        Returns:
            BenchmarkResult with all evaluations
        """
        logger.info(f"Running benchmark for {model_id}")
        
        results = {}
        
        for opp_id, opp_factory in self.opponents.items():
            logger.info(f"  vs {opp_id}...")
            opponent = opp_factory()
            
            result = self.evaluator.evaluate(
                model=model,
                opponent=opponent,
                num_games=self.games_per_opponent,
                model_id=model_id,
                opponent_id=opp_id,
            )
            
            results[opp_id] = result
        
        # Compute aggregates
        aggregate = self._compute_aggregates(results)
        
        benchmark = BenchmarkResult(
            model_id=model_id,
            timestamp=datetime.utcnow().isoformat(),
            results=results,
            aggregate=aggregate,
        )
        
        logger.info(f"Benchmark complete: {benchmark.overall_win_rate:.1%} overall")
        
        # Save
        if self.save_dir:
            self._save_result(benchmark)
        
        return benchmark
    
    def _compute_aggregates(
        self,
        results: Dict[str, EvaluationResult],
    ) -> Dict[str, float]:
        """Compute aggregate statistics."""
        if not results:
            return {}
        
        all_win_rates = [r.win_rate for r in results.values()]
        
        aggregates = {
            "mean_win_rate": np.mean(all_win_rates),
            "min_win_rate": np.min(all_win_rates),
            "max_win_rate": np.max(all_win_rates),
            "std_win_rate": np.std(all_win_rates),
        }
        
        # Compute weighted score (harder opponents weighted more)
        weights = {
            "random": 1.0,
            "heuristic": 2.0,
            "imitation": 3.0,
            "self": 4.0,
        }
        
        weighted_sum = 0.0
        weight_total = 0.0
        
        for opp_id, result in results.items():
            w = weights.get(opp_id, 1.0)
            weighted_sum += result.win_rate * w
            weight_total += w
        
        aggregates["weighted_score"] = weighted_sum / weight_total if weight_total > 0 else 0
        
        return aggregates
    
    def compare(
        self,
        models: List[Any],
        model_ids: List[str],
    ) -> Dict[str, BenchmarkResult]:
        """Compare multiple models.
        
        Args:
            models: List of models
            model_ids: List of model identifiers
            
        Returns:
            Dict of model_id -> BenchmarkResult
        """
        results = {}
        
        for model, model_id in zip(models, model_ids):
            results[model_id] = self.run(model, model_id)
        
        # Log comparison
        logger.info("\nModel Comparison:")
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].overall_win_rate,
            reverse=True,
        )
        
        for rank, (model_id, result) in enumerate(sorted_results, 1):
            logger.info(
                f"  {rank}. {model_id}: "
                f"{result.overall_win_rate:.1%} "
                f"(weighted: {result.aggregate.get('weighted_score', 0):.2f})"
            )
        
        return results
    
    def _save_result(self, result: BenchmarkResult):
        """Save benchmark result."""
        filename = f"benchmark_{result.model_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.save_dir / filename
        
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.info(f"Saved benchmark to {path}")
    
    def load_result(self, path: Path) -> BenchmarkResult:
        """Load a benchmark result from file."""
        with open(path) as f:
            data = json.load(f)
        
        # Would need to reconstruct EvaluationResult objects
        return BenchmarkResult(
            model_id=data["model_id"],
            timestamp=data["timestamp"],
            results={},  # Simplified
            aggregate=data["aggregate"],
        )


class ContinuousBenchmark:
    """Continuous benchmarking for training monitoring.
    
    Periodically evaluates model during training and
    tracks improvement over time.
    """
    
    def __init__(
        self,
        suite: BenchmarkSuite,
        interval_steps: int = 10000,
        save_dir: Optional[Path] = None,
    ):
        """Initialize continuous benchmark.
        
        Args:
            suite: Benchmark suite to use
            interval_steps: Steps between benchmarks
            save_dir: Directory to save history
        """
        self.suite = suite
        self.interval_steps = interval_steps
        self.save_dir = save_dir
        
        self.history: List[Dict[str, Any]] = []
        self.last_benchmark_step = 0
    
    def should_benchmark(self, step: int) -> bool:
        """Check if should run benchmark at this step."""
        return step - self.last_benchmark_step >= self.interval_steps
    
    def benchmark(
        self,
        model: Any,
        step: int,
        model_id: str = "model",
    ) -> BenchmarkResult:
        """Run benchmark and record result.
        
        Args:
            model: Model to benchmark
            step: Current training step
            model_id: Model identifier
            
        Returns:
            BenchmarkResult
        """
        result = self.suite.run(model, f"{model_id}_step{step}")
        
        self.history.append({
            "step": step,
            "timestamp": result.timestamp,
            "overall_win_rate": result.overall_win_rate,
            "aggregate": result.aggregate,
        })
        
        self.last_benchmark_step = step
        
        # Save history
        if self.save_dir:
            self._save_history()
        
        return result
    
    def get_improvement(self) -> float:
        """Get improvement since start."""
        if len(self.history) < 2:
            return 0.0
        
        first = self.history[0]["overall_win_rate"]
        last = self.history[-1]["overall_win_rate"]
        
        return last - first
    
    def _save_history(self):
        """Save benchmark history."""
        if self.save_dir is None:
            return
        
        path = self.save_dir / "benchmark_history.json"
        
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)

