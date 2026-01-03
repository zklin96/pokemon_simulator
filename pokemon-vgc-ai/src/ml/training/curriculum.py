"""Curriculum learning for VGC AI training.

This module implements progressive training difficulty,
starting from simple scenarios and advancing to full VGC
complexity as the agent improves.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Tuple
from enum import IntEnum, auto
from pathlib import Path
import json
from loguru import logger


class DifficultyLevel(IntEnum):
    """Training difficulty levels."""
    BEGINNER = 0      # 2v2 vs random, short games
    EASY = 1          # 4v4 vs random, full games
    MEDIUM = 2        # 4v4 vs heuristic
    HARD = 3          # 4v4 vs imitation policy
    EXPERT = 4        # 6v6 vs self-play
    MASTER = 5        # Full VGC with competitive teams


@dataclass
class CurriculumStage:
    """Configuration for a curriculum stage."""
    
    name: str
    level: DifficultyLevel
    
    # Game settings
    team_size: int = 6
    bring_size: int = 4
    max_turns: int = 50
    
    # Opponent
    opponent_type: str = "random"  # random, heuristic, imitation, self
    opponent_strength: float = 1.0  # Scaling factor for opponent
    
    # Restrictions
    allow_tera: bool = True
    allow_switching: bool = True
    restricted_pokemon: List[str] = field(default_factory=list)
    
    # Progression criteria
    win_rate_threshold: float = 0.7
    games_required: int = 1000
    max_games: int = 10000  # Give up and force advance after this many
    
    # Reward scaling
    reward_scale: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "level": self.level.value,
            "team_size": self.team_size,
            "bring_size": self.bring_size,
            "max_turns": self.max_turns,
            "opponent_type": self.opponent_type,
            "opponent_strength": self.opponent_strength,
            "allow_tera": self.allow_tera,
            "allow_switching": self.allow_switching,
            "win_rate_threshold": self.win_rate_threshold,
            "games_required": self.games_required,
        }


# Default curriculum stages
DEFAULT_CURRICULUM = [
    CurriculumStage(
        name="Beginner: 2v2 Basics",
        level=DifficultyLevel.BEGINNER,
        team_size=2,
        bring_size=2,
        max_turns=20,
        opponent_type="random",
        allow_tera=False,
        allow_switching=False,
        win_rate_threshold=0.8,
        games_required=500,
    ),
    CurriculumStage(
        name="Easy: 4v4 Random",
        level=DifficultyLevel.EASY,
        team_size=4,
        bring_size=4,
        max_turns=30,
        opponent_type="random",
        allow_tera=True,
        allow_switching=True,
        win_rate_threshold=0.7,
        games_required=1000,
    ),
    CurriculumStage(
        name="Medium: vs Heuristic",
        level=DifficultyLevel.MEDIUM,
        team_size=4,
        bring_size=4,
        max_turns=40,
        opponent_type="heuristic",
        win_rate_threshold=0.6,
        games_required=2000,
    ),
    CurriculumStage(
        name="Hard: vs Imitation",
        level=DifficultyLevel.HARD,
        team_size=4,
        bring_size=4,
        max_turns=50,
        opponent_type="imitation",
        opponent_strength=0.8,
        win_rate_threshold=0.55,
        games_required=3000,
    ),
    CurriculumStage(
        name="Expert: Self-Play",
        level=DifficultyLevel.EXPERT,
        team_size=6,
        bring_size=4,
        max_turns=50,
        opponent_type="self",
        win_rate_threshold=0.52,
        games_required=5000,
    ),
    CurriculumStage(
        name="Master: Full VGC",
        level=DifficultyLevel.MASTER,
        team_size=6,
        bring_size=4,
        max_turns=50,
        opponent_type="self",
        opponent_strength=1.0,
        win_rate_threshold=0.51,
        games_required=10000,
    ),
]


@dataclass
class StageProgress:
    """Track progress within a stage."""
    
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    
    # Rolling window for recent performance
    recent_results: List[bool] = field(default_factory=list)
    window_size: int = 100
    
    # Best achieved
    best_win_rate: float = 0.0
    best_streak: int = 0
    current_streak: int = 0
    
    def record_game(self, won: bool):
        """Record a game result."""
        self.games_played += 1
        if won:
            self.wins += 1
            self.current_streak += 1
            self.best_streak = max(self.best_streak, self.current_streak)
        else:
            self.losses += 1
            self.current_streak = 0
        
        # Update rolling window
        self.recent_results.append(won)
        if len(self.recent_results) > self.window_size:
            self.recent_results.pop(0)
        
        # Update best
        self.best_win_rate = max(self.best_win_rate, self.win_rate)
    
    @property
    def win_rate(self) -> float:
        """Current overall win rate."""
        total = self.wins + self.losses
        return self.wins / total if total > 0 else 0.0
    
    @property
    def recent_win_rate(self) -> float:
        """Win rate over recent window."""
        if not self.recent_results:
            return 0.0
        return sum(self.recent_results) / len(self.recent_results)
    
    def should_advance(self, stage: CurriculumStage) -> bool:
        """Check if ready to advance to next stage."""
        if self.games_played < stage.games_required:
            return False
        
        # Use recent win rate for advancement
        if len(self.recent_results) >= 50:  # Need enough games
            return self.recent_win_rate >= stage.win_rate_threshold
        
        return self.win_rate >= stage.win_rate_threshold
    
    def should_force_advance(self, stage: CurriculumStage) -> bool:
        """Check if should force advance (giving up)."""
        return self.games_played >= stage.max_games


class CurriculumLearning:
    """Curriculum learning manager.
    
    Tracks training progress and manages stage transitions.
    
    Example:
        curriculum = CurriculumLearning()
        
        for episode in range(total_episodes):
            stage = curriculum.current_stage
            env = create_env(stage)
            
            result = train_episode(env)
            curriculum.record_result(result)
            
            if curriculum.should_advance():
                curriculum.advance()
    """
    
    def __init__(
        self,
        stages: Optional[List[CurriculumStage]] = None,
        save_path: Optional[Path] = None,
    ):
        """Initialize curriculum.
        
        Args:
            stages: List of curriculum stages (default: DEFAULT_CURRICULUM)
            save_path: Path to save/load progress
        """
        self.stages = stages or DEFAULT_CURRICULUM.copy()
        self.save_path = save_path
        
        self.current_stage_idx = 0
        self.stage_progress: Dict[int, StageProgress] = {}
        
        # Initialize progress for first stage
        self.stage_progress[0] = StageProgress()
        
        # Load saved progress if exists
        if save_path and save_path.exists():
            self.load()
    
    @property
    def current_stage(self) -> CurriculumStage:
        """Get current curriculum stage."""
        return self.stages[self.current_stage_idx]
    
    @property
    def current_progress(self) -> StageProgress:
        """Get progress for current stage."""
        return self.stage_progress[self.current_stage_idx]
    
    @property
    def is_complete(self) -> bool:
        """Check if curriculum is complete."""
        return self.current_stage_idx >= len(self.stages) - 1
    
    def record_result(self, won: bool):
        """Record a game result.
        
        Args:
            won: Whether the agent won
        """
        self.current_progress.record_game(won)
    
    def should_advance(self) -> bool:
        """Check if should advance to next stage."""
        if self.is_complete:
            return False
        
        progress = self.current_progress
        stage = self.current_stage
        
        return progress.should_advance(stage) or progress.should_force_advance(stage)
    
    def advance(self):
        """Advance to next stage."""
        if self.is_complete:
            logger.warning("Already at final stage, cannot advance")
            return
        
        old_stage = self.current_stage
        old_progress = self.current_progress
        
        logger.info(
            f"Advancing from '{old_stage.name}' "
            f"(Win rate: {old_progress.win_rate:.1%}, "
            f"Games: {old_progress.games_played})"
        )
        
        self.current_stage_idx += 1
        self.stage_progress[self.current_stage_idx] = StageProgress()
        
        new_stage = self.current_stage
        logger.info(f"Now at stage: '{new_stage.name}'")
        
        # Save progress
        if self.save_path:
            self.save()
    
    def get_opponent_factory(self) -> Callable:
        """Get opponent factory for current stage.
        
        Returns:
            Callable that creates opponent agent
        """
        stage = self.current_stage
        
        def create_opponent():
            if stage.opponent_type == "random":
                from ..battle_ai.agents import RandomAgent
                return RandomAgent(action_dim=144)
            
            elif stage.opponent_type == "heuristic":
                from ..battle_ai.agents import HeuristicAgent
                return HeuristicAgent()
            
            elif stage.opponent_type == "imitation":
                # Would load imitation model with reduced strength
                from ..battle_ai.agents import RandomAgent
                return RandomAgent(action_dim=144)  # Placeholder
            
            elif stage.opponent_type == "self":
                # Would return copy of current policy
                from ..battle_ai.agents import RandomAgent
                return RandomAgent(action_dim=144)  # Placeholder
            
            else:
                raise ValueError(f"Unknown opponent type: {stage.opponent_type}")
        
        return create_opponent
    
    def get_env_config(self) -> Dict[str, Any]:
        """Get environment configuration for current stage.
        
        Returns:
            Dict of environment parameters
        """
        stage = self.current_stage
        
        return {
            "team_size": stage.team_size,
            "bring_size": stage.bring_size,
            "max_turns": stage.max_turns,
            "allow_tera": stage.allow_tera,
            "allow_switching": stage.allow_switching,
            "reward_scale": stage.reward_scale,
        }
    
    def get_training_config(self) -> Dict[str, Any]:
        """Get training configuration for current stage.
        
        Returns:
            Dict of training parameters
        """
        stage = self.current_stage
        level = stage.level
        
        # Adjust learning rate and exploration based on stage
        configs = {
            DifficultyLevel.BEGINNER: {
                "learning_rate": 3e-4,
                "entropy_coef": 0.1,
                "n_steps": 512,
            },
            DifficultyLevel.EASY: {
                "learning_rate": 1e-4,
                "entropy_coef": 0.05,
                "n_steps": 1024,
            },
            DifficultyLevel.MEDIUM: {
                "learning_rate": 5e-5,
                "entropy_coef": 0.02,
                "n_steps": 2048,
            },
            DifficultyLevel.HARD: {
                "learning_rate": 3e-5,
                "entropy_coef": 0.01,
                "n_steps": 2048,
            },
            DifficultyLevel.EXPERT: {
                "learning_rate": 1e-5,
                "entropy_coef": 0.005,
                "n_steps": 4096,
            },
            DifficultyLevel.MASTER: {
                "learning_rate": 5e-6,
                "entropy_coef": 0.001,
                "n_steps": 4096,
            },
        }
        
        return configs.get(level, configs[DifficultyLevel.EASY])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get curriculum statistics.
        
        Returns:
            Dict of statistics
        """
        return {
            "current_stage": self.current_stage_idx,
            "stage_name": self.current_stage.name,
            "stages_completed": self.current_stage_idx,
            "total_stages": len(self.stages),
            "current_games": self.current_progress.games_played,
            "current_win_rate": self.current_progress.win_rate,
            "recent_win_rate": self.current_progress.recent_win_rate,
            "best_win_rate": self.current_progress.best_win_rate,
            "total_games": sum(p.games_played for p in self.stage_progress.values()),
        }
    
    def save(self):
        """Save progress to file."""
        if self.save_path is None:
            return
        
        data = {
            "current_stage_idx": self.current_stage_idx,
            "stage_progress": {
                str(k): {
                    "games_played": v.games_played,
                    "wins": v.wins,
                    "losses": v.losses,
                    "best_win_rate": v.best_win_rate,
                    "best_streak": v.best_streak,
                }
                for k, v in self.stage_progress.items()
            },
        }
        
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)
        
        logger.debug(f"Saved curriculum progress to {self.save_path}")
    
    def load(self):
        """Load progress from file."""
        if self.save_path is None or not self.save_path.exists():
            return
        
        with open(self.save_path) as f:
            data = json.load(f)
        
        self.current_stage_idx = data["current_stage_idx"]
        
        self.stage_progress = {}
        for k, v in data["stage_progress"].items():
            progress = StageProgress(
                games_played=v["games_played"],
                wins=v["wins"],
                losses=v["losses"],
                best_win_rate=v["best_win_rate"],
                best_streak=v["best_streak"],
            )
            self.stage_progress[int(k)] = progress
        
        logger.info(
            f"Loaded curriculum progress: Stage {self.current_stage_idx + 1}/"
            f"{len(self.stages)}"
        )
    
    def reset(self):
        """Reset curriculum to beginning."""
        self.current_stage_idx = 0
        self.stage_progress = {0: StageProgress()}
        logger.info("Reset curriculum to beginning")

