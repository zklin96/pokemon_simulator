"""Comprehensive evaluator for VGC AI models.

This module provides multi-dimensional evaluation of agents
against various opponents and metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable, Type
from pathlib import Path
import json
from datetime import datetime
import numpy as np
from loguru import logger

from .metrics import (
    Metric, GameRecord, WinRateMetric, ELOMetric, TurnsMetric,
    KOEfficiencyMetric, ActionDistributionMetric, TeraUsageMetric,
    SwitchingMetric, HPDifferentialMetric,
)


@dataclass
class EvaluationResult:
    """Result of an evaluation run."""
    
    model_id: str
    opponent_id: str
    timestamp: str
    num_games: int
    metrics: Dict[str, Dict[str, float]]
    games: List[GameRecord] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "opponent_id": self.opponent_id,
            "timestamp": self.timestamp,
            "num_games": self.num_games,
            "metrics": self.metrics,
        }
    
    @property
    def win_rate(self) -> float:
        """Get win rate."""
        return self.metrics.get("win_rate", {}).get("win_rate", 0.0)
    
    def summary(self) -> str:
        """Get summary string."""
        lines = [
            f"Evaluation: {self.model_id} vs {self.opponent_id}",
            f"Games: {self.num_games}",
        ]
        
        for metric_name, values in self.metrics.items():
            lines.append(f"  {metric_name}:")
            for k, v in values.items():
                if isinstance(v, float):
                    lines.append(f"    {k}: {v:.4f}")
                else:
                    lines.append(f"    {k}: {v}")
        
        return "\n".join(lines)


class VGCEvaluator:
    """Multi-dimensional evaluator for VGC AI.
    
    Evaluates agents across multiple metrics against
    various opponents.
    
    Example:
        evaluator = VGCEvaluator()
        
        result = evaluator.evaluate(
            model=my_model,
            opponent=heuristic_agent,
            num_games=100,
        )
        
        print(result.summary())
    """
    
    DEFAULT_METRICS = [
        WinRateMetric,
        ELOMetric,
        TurnsMetric,
        KOEfficiencyMetric,
        ActionDistributionMetric,
        TeraUsageMetric,
        SwitchingMetric,
        HPDifferentialMetric,
    ]
    
    def __init__(
        self,
        metrics: Optional[List[Type[Metric]]] = None,
        save_dir: Optional[Path] = None,
    ):
        """Initialize evaluator.
        
        Args:
            metrics: List of metric classes to use
            save_dir: Directory to save evaluation results
        """
        metric_classes = metrics or self.DEFAULT_METRICS
        self.metrics = {m().name: m() for m in metric_classes}
        self.save_dir = save_dir
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
    
    def evaluate(
        self,
        model: Any,
        opponent: Any,
        num_games: int = 100,
        model_id: str = "model",
        opponent_id: str = "opponent",
        battle_fn: Optional[Callable] = None,
    ) -> EvaluationResult:
        """Evaluate model against opponent.
        
        Args:
            model: Model/agent to evaluate
            opponent: Opponent to play against
            num_games: Number of games to play
            model_id: Identifier for model
            opponent_id: Identifier for opponent
            battle_fn: Function to run a battle (returns GameRecord)
            
        Returns:
            EvaluationResult with all metrics
        """
        logger.info(f"Evaluating {model_id} vs {opponent_id} ({num_games} games)")
        
        # Reset metrics
        for metric in self.metrics.values():
            metric.reset()
        
        # Play games
        games: List[GameRecord] = []
        
        for i in range(num_games):
            if battle_fn:
                game = battle_fn(model, opponent)
            else:
                # Simulated game (placeholder)
                game = self._simulate_game(model, opponent)
            
            games.append(game)
            
            # Update metrics
            for metric in self.metrics.values():
                metric.update(game)
            
            if (i + 1) % 20 == 0:
                logger.debug(f"  Completed {i + 1}/{num_games} games")
        
        # Compute final metrics
        metric_results = {}
        for name, metric in self.metrics.items():
            metric_results[name] = metric.compute(games)
        
        result = EvaluationResult(
            model_id=model_id,
            opponent_id=opponent_id,
            timestamp=datetime.utcnow().isoformat(),
            num_games=num_games,
            metrics=metric_results,
            games=games,
        )
        
        logger.info(f"Evaluation complete. Win rate: {result.win_rate:.1%}")
        
        # Save if configured
        if self.save_dir:
            self._save_result(result)
        
        return result
    
    def evaluate_vs_multiple(
        self,
        model: Any,
        opponents: Dict[str, Any],
        num_games_per: int = 50,
        model_id: str = "model",
    ) -> Dict[str, EvaluationResult]:
        """Evaluate model against multiple opponents.
        
        Args:
            model: Model to evaluate
            opponents: Dict of opponent_id -> opponent
            num_games_per: Games per opponent
            model_id: Model identifier
            
        Returns:
            Dict of opponent_id -> EvaluationResult
        """
        results = {}
        
        for opp_id, opponent in opponents.items():
            result = self.evaluate(
                model=model,
                opponent=opponent,
                num_games=num_games_per,
                model_id=model_id,
                opponent_id=opp_id,
            )
            results[opp_id] = result
        
        return results
    
    def _simulate_game(self, model: Any, opponent: Any) -> GameRecord:
        """Simulate a game (placeholder).
        
        In production, this would run an actual battle.
        """
        # Random simulation
        won = np.random.random() > 0.5
        turns = np.random.randint(10, 40)
        
        return GameRecord(
            won=won,
            turns=turns,
            my_kos=np.random.randint(1, 5),
            opp_kos=np.random.randint(1, 5),
            my_final_hp=np.random.random() if won else 0.0,
            opp_final_hp=0.0 if won else np.random.random(),
            actions_taken=list(np.random.randint(0, 144, size=turns)),
            tera_used=np.random.random() > 0.5,
            switches=np.random.randint(0, 5),
        )
    
    def _save_result(self, result: EvaluationResult):
        """Save evaluation result to file."""
        filename = f"{result.model_id}_vs_{result.opponent_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        path = self.save_dir / filename
        
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        
        logger.debug(f"Saved evaluation to {path}")
    
    def compare_models(
        self,
        models: Dict[str, Any],
        opponent: Any,
        num_games: int = 100,
        opponent_id: str = "baseline",
    ) -> Dict[str, EvaluationResult]:
        """Compare multiple models against same opponent.
        
        Args:
            models: Dict of model_id -> model
            opponent: Common opponent
            num_games: Games per model
            opponent_id: Opponent identifier
            
        Returns:
            Dict of model_id -> EvaluationResult
        """
        results = {}
        
        for model_id, model in models.items():
            result = self.evaluate(
                model=model,
                opponent=opponent,
                num_games=num_games,
                model_id=model_id,
                opponent_id=opponent_id,
            )
            results[model_id] = result
        
        # Log comparison
        logger.info("Model comparison:")
        for model_id, result in sorted(results.items(), key=lambda x: -x[1].win_rate):
            logger.info(f"  {model_id}: {result.win_rate:.1%} win rate")
        
        return results
    
    def get_metric(self, name: str) -> Optional[Metric]:
        """Get a metric by name."""
        return self.metrics.get(name)
    
    def add_metric(self, metric: Metric):
        """Add a custom metric."""
        self.metrics[metric.name] = metric

