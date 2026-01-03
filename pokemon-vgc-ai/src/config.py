"""Configuration settings for Pokemon VGC AI."""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = DATA_DIR / "models"

# Ensure directories exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)


@dataclass
class ShowdownConfig:
    """Configuration for Pokemon Showdown connection."""
    
    server_url: str = "localhost:8000"
    username: str = "VGC_AI_Bot"
    password: Optional[str] = None
    # VGC format for Scarlet/Violet
    battle_format: str = "gen9vgc2024regg"
    team_size: int = 6
    battle_team_size: int = 4  # VGC uses 4 Pokemon per battle


@dataclass
class TrainingConfig:
    """Configuration for RL training."""
    
    # PPO hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    
    # Training settings
    total_timesteps: int = 1_000_000
    save_frequency: int = 10_000
    eval_frequency: int = 5_000
    n_eval_episodes: int = 10
    
    # Self-play settings
    population_size: int = 10
    elo_k_factor: float = 32.0
    initial_elo: float = 1500.0


@dataclass
class TeamBuilderConfig:
    """Configuration for evolutionary team builder."""
    
    population_size: int = 100
    generations: int = 500
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    tournament_size: int = 5
    elite_size: int = 10
    
    # Evaluation
    battles_per_evaluation: int = 10


@dataclass
class DatabaseConfig:
    """Database configuration."""
    
    db_path: Path = field(default_factory=lambda: DATA_DIR / "vgc_ai.db")
    
    @property
    def connection_string(self) -> str:
        return f"sqlite:///{self.db_path}"


@dataclass
class Config:
    """Main configuration container."""
    
    showdown: ShowdownConfig = field(default_factory=ShowdownConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    team_builder: TeamBuilderConfig = field(default_factory=TeamBuilderConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)


# Global config instance
config = Config()

