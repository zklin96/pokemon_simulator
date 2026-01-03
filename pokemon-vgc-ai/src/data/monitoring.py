"""Data quality monitoring for VGC AI.

This module provides tools to track and report on data quality metrics
during data processing pipelines.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import Counter, defaultdict
import json
from datetime import datetime
from loguru import logger


@dataclass
class DataQualityMetrics:
    """Container for data quality metrics."""
    
    # Parsing metrics
    total_battles: int = 0
    successful_parses: int = 0
    failed_parses: int = 0
    parse_errors: Dict[str, int] = field(default_factory=dict)
    
    # Trajectory metrics
    total_trajectories: int = 0
    total_transitions: int = 0
    avg_trajectory_length: float = 0.0
    min_trajectory_length: int = 0
    max_trajectory_length: int = 0
    
    # Action distribution
    action_counts: Dict[int, int] = field(default_factory=dict)
    
    # Reward statistics
    reward_mean: float = 0.0
    reward_std: float = 0.0
    reward_min: float = 0.0
    reward_max: float = 0.0
    
    # Win/loss distribution
    wins: int = 0
    losses: int = 0
    
    # State statistics
    state_mean: Optional[np.ndarray] = None
    state_std: Optional[np.ndarray] = None
    nan_states: int = 0
    inf_states: int = 0
    
    # Missing data
    missing_actions: int = 0
    missing_states: int = 0
    
    # Timestamps
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def parse_success_rate(self) -> float:
        """Get parse success rate."""
        if self.total_battles == 0:
            return 0.0
        return self.successful_parses / self.total_battles
    
    @property
    def win_rate(self) -> float:
        """Get win rate."""
        total = self.wins + self.losses
        if total == 0:
            return 0.0
        return self.wins / total
    
    @property
    def duration_seconds(self) -> float:
        """Get processing duration."""
        if self.start_time is None or self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_battles": self.total_battles,
            "successful_parses": self.successful_parses,
            "failed_parses": self.failed_parses,
            "parse_success_rate": self.parse_success_rate,
            "parse_errors": dict(self.parse_errors),
            "total_trajectories": self.total_trajectories,
            "total_transitions": self.total_transitions,
            "avg_trajectory_length": self.avg_trajectory_length,
            "min_trajectory_length": self.min_trajectory_length,
            "max_trajectory_length": self.max_trajectory_length,
            "action_distribution": dict(self.action_counts),
            "reward_mean": self.reward_mean,
            "reward_std": self.reward_std,
            "reward_min": self.reward_min,
            "reward_max": self.reward_max,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
            "nan_states": self.nan_states,
            "inf_states": self.inf_states,
            "missing_actions": self.missing_actions,
            "missing_states": self.missing_states,
            "duration_seconds": self.duration_seconds,
        }


class DataQualityMonitor:
    """Monitor data quality during processing.
    
    Example:
        monitor = DataQualityMonitor()
        
        for battle in battles:
            with monitor.track_battle():
                try:
                    trajectory = parse_battle(battle)
                    monitor.record_trajectory(trajectory)
                except Exception as e:
                    monitor.record_error(str(e))
        
        report = monitor.generate_report()
        print(report)
    """
    
    def __init__(self, name: str = "default"):
        """Initialize monitor.
        
        Args:
            name: Name for this monitoring session
        """
        self.name = name
        self.metrics = DataQualityMetrics()
        self._trajectory_lengths: List[int] = []
        self._rewards: List[float] = []
        self._states: List[np.ndarray] = []
        self._current_battle: Optional[str] = None
    
    def start(self):
        """Start monitoring session."""
        self.metrics.start_time = datetime.now()
        logger.info(f"Started data quality monitoring: {self.name}")
    
    def stop(self):
        """Stop monitoring session and compute final statistics."""
        self.metrics.end_time = datetime.now()
        self._compute_final_statistics()
        logger.info(f"Stopped monitoring. Duration: {self.metrics.duration_seconds:.1f}s")
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
    
    def track_battle(self):
        """Context manager for tracking a single battle."""
        return BattleTracker(self)
    
    def record_battle_start(self, battle_id: Optional[str] = None):
        """Record start of battle processing."""
        self.metrics.total_battles += 1
        self._current_battle = battle_id
    
    def record_battle_success(self):
        """Record successful battle parse."""
        self.metrics.successful_parses += 1
    
    def record_battle_failure(self):
        """Record failed battle parse."""
        self.metrics.failed_parses += 1
    
    def record_error(self, error_type: str):
        """Record a parse error.
        
        Args:
            error_type: Type/category of error
        """
        if error_type not in self.metrics.parse_errors:
            self.metrics.parse_errors[error_type] = 0
        self.metrics.parse_errors[error_type] += 1
    
    def record_trajectory(self, trajectory: Dict[str, Any]):
        """Record a trajectory for monitoring.
        
        Args:
            trajectory: Trajectory dictionary
        """
        self.metrics.total_trajectories += 1
        
        transitions = trajectory.get("transitions", [])
        length = len(transitions)
        self._trajectory_lengths.append(length)
        self.metrics.total_transitions += length
        
        # Track wins/losses
        player = trajectory.get("player", "p1")
        winner = trajectory.get("winner", "")
        if winner == player:
            self.metrics.wins += 1
        else:
            self.metrics.losses += 1
        
        # Track actions and rewards
        for trans in transitions:
            action = trans.get("action")
            if action is None:
                self.metrics.missing_actions += 1
            else:
                if action not in self.metrics.action_counts:
                    self.metrics.action_counts[action] = 0
                self.metrics.action_counts[action] += 1
            
            reward = trans.get("reward")
            if reward is not None:
                self._rewards.append(reward)
            
            state = trans.get("state")
            if state is None:
                self.metrics.missing_states += 1
            else:
                state_arr = np.array(state)
                if np.any(np.isnan(state_arr)):
                    self.metrics.nan_states += 1
                if np.any(np.isinf(state_arr)):
                    self.metrics.inf_states += 1
                
                # Sample states for statistics (memory efficient)
                if len(self._states) < 1000 or np.random.random() < 0.01:
                    self._states.append(state_arr)
    
    def _compute_final_statistics(self):
        """Compute final statistics from collected data."""
        # Trajectory length statistics
        if self._trajectory_lengths:
            self.metrics.avg_trajectory_length = np.mean(self._trajectory_lengths)
            self.metrics.min_trajectory_length = min(self._trajectory_lengths)
            self.metrics.max_trajectory_length = max(self._trajectory_lengths)
        
        # Reward statistics
        if self._rewards:
            self.metrics.reward_mean = np.mean(self._rewards)
            self.metrics.reward_std = np.std(self._rewards)
            self.metrics.reward_min = min(self._rewards)
            self.metrics.reward_max = max(self._rewards)
        
        # State statistics
        if self._states:
            states_array = np.stack(self._states)
            self.metrics.state_mean = np.mean(states_array, axis=0)
            self.metrics.state_std = np.std(states_array, axis=0)
    
    def generate_report(self) -> str:
        """Generate a human-readable quality report.
        
        Returns:
            Formatted report string
        """
        m = self.metrics
        
        lines = [
            "=" * 60,
            f"DATA QUALITY REPORT: {self.name}",
            "=" * 60,
            "",
            "PARSING STATISTICS",
            "-" * 40,
            f"  Total battles:      {m.total_battles:,}",
            f"  Successful parses:  {m.successful_parses:,}",
            f"  Failed parses:      {m.failed_parses:,}",
            f"  Success rate:       {m.parse_success_rate:.1%}",
            "",
        ]
        
        if m.parse_errors:
            lines.append("  Error breakdown:")
            for error, count in sorted(m.parse_errors.items(), key=lambda x: -x[1])[:10]:
                lines.append(f"    - {error}: {count}")
            lines.append("")
        
        lines.extend([
            "TRAJECTORY STATISTICS",
            "-" * 40,
            f"  Total trajectories: {m.total_trajectories:,}",
            f"  Total transitions:  {m.total_transitions:,}",
            f"  Avg length:         {m.avg_trajectory_length:.1f}",
            f"  Min length:         {m.min_trajectory_length}",
            f"  Max length:         {m.max_trajectory_length}",
            "",
            "WIN/LOSS DISTRIBUTION",
            "-" * 40,
            f"  Wins:               {m.wins:,}",
            f"  Losses:             {m.losses:,}",
            f"  Win rate:           {m.win_rate:.1%}",
            "",
            "REWARD STATISTICS",
            "-" * 40,
            f"  Mean reward:        {m.reward_mean:.4f}",
            f"  Std reward:         {m.reward_std:.4f}",
            f"  Min reward:         {m.reward_min:.4f}",
            f"  Max reward:         {m.reward_max:.4f}",
            "",
        ])
        
        # Action distribution
        if m.action_counts:
            lines.extend([
                "ACTION DISTRIBUTION (top 10)",
                "-" * 40,
            ])
            total_actions = sum(m.action_counts.values())
            top_actions = sorted(m.action_counts.items(), key=lambda x: -x[1])[:10]
            for action, count in top_actions:
                pct = count / total_actions * 100
                lines.append(f"  Action {action:3d}: {count:6,} ({pct:5.1f}%)")
            lines.append("")
        
        lines.extend([
            "DATA QUALITY ISSUES",
            "-" * 40,
            f"  Missing actions:    {m.missing_actions:,}",
            f"  Missing states:     {m.missing_states:,}",
            f"  NaN states:         {m.nan_states:,}",
            f"  Inf states:         {m.inf_states:,}",
            "",
            "TIMING",
            "-" * 40,
            f"  Duration:           {m.duration_seconds:.1f}s",
            "",
            "=" * 60,
        ])
        
        return "\n".join(lines)
    
    def save_report(self, output_path: Path):
        """Save report to file.
        
        Args:
            output_path: Output path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save text report
        report = self.generate_report()
        with open(output_path.with_suffix(".txt"), "w") as f:
            f.write(report)
        
        # Save JSON metrics
        with open(output_path.with_suffix(".json"), "w") as f:
            json.dump(self.metrics.to_dict(), f, indent=2)
        
        logger.info(f"Saved quality report to {output_path}")
    
    def get_action_entropy(self) -> float:
        """Calculate action distribution entropy.
        
        Higher entropy means more uniform action distribution.
        
        Returns:
            Entropy value
        """
        if not self.metrics.action_counts:
            return 0.0
        
        total = sum(self.metrics.action_counts.values())
        probs = [count / total for count in self.metrics.action_counts.values()]
        return -sum(p * np.log(p + 1e-10) for p in probs)
    
    def check_data_quality(self) -> Tuple[bool, List[str]]:
        """Check data quality and return issues.
        
        Returns:
            Tuple of (is_ok, list_of_issues)
        """
        issues = []
        
        # Parse rate check
        if self.metrics.parse_success_rate < 0.9:
            issues.append(
                f"Low parse success rate: {self.metrics.parse_success_rate:.1%}"
            )
        
        # Missing data checks
        if self.metrics.missing_states > 0:
            issues.append(f"Missing states: {self.metrics.missing_states}")
        
        if self.metrics.nan_states > 0:
            issues.append(f"NaN states: {self.metrics.nan_states}")
        
        if self.metrics.inf_states > 0:
            issues.append(f"Inf states: {self.metrics.inf_states}")
        
        # Win rate sanity check
        if abs(self.metrics.win_rate - 0.5) > 0.1:
            issues.append(
                f"Imbalanced win rate: {self.metrics.win_rate:.1%} (expected ~50%)"
            )
        
        # Trajectory length check
        if self.metrics.avg_trajectory_length < 5:
            issues.append(
                f"Very short trajectories: {self.metrics.avg_trajectory_length:.1f}"
            )
        
        return len(issues) == 0, issues


class BattleTracker:
    """Context manager for tracking a single battle."""
    
    def __init__(self, monitor: DataQualityMonitor):
        self.monitor = monitor
        self.success = False
    
    def __enter__(self):
        self.monitor.record_battle_start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None and self.success:
            self.monitor.record_battle_success()
        else:
            self.monitor.record_battle_failure()
            if exc_type is not None:
                self.monitor.record_error(exc_type.__name__)
        return False
    
    def mark_success(self):
        """Mark this battle as successfully processed."""
        self.success = True


def analyze_trajectory_distribution(
    trajectories: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze trajectory distribution statistics.
    
    Args:
        trajectories: List of trajectory dictionaries
        
    Returns:
        Analysis results
    """
    lengths = [len(t.get("transitions", [])) for t in trajectories]
    wins = sum(1 for t in trajectories if t.get("winner") == t.get("player"))
    
    # Action distribution
    action_counts = Counter()
    for traj in trajectories:
        for trans in traj.get("transitions", []):
            action = trans.get("action")
            if action is not None:
                action_counts[action] += 1
    
    return {
        "num_trajectories": len(trajectories),
        "length_stats": {
            "mean": np.mean(lengths) if lengths else 0,
            "std": np.std(lengths) if lengths else 0,
            "min": min(lengths) if lengths else 0,
            "max": max(lengths) if lengths else 0,
            "median": np.median(lengths) if lengths else 0,
        },
        "win_rate": wins / len(trajectories) if trajectories else 0,
        "action_entropy": -sum(
            (c / sum(action_counts.values())) * 
            np.log(c / sum(action_counts.values()) + 1e-10)
            for c in action_counts.values()
        ) if action_counts else 0,
        "unique_actions": len(action_counts),
        "top_actions": action_counts.most_common(10),
    }

