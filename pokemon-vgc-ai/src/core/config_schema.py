"""Configuration schemas for VGC AI using Hydra and OmegaConf.

This module defines structured configs that provide type safety
and validation for all configuration options.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from omegaconf import MISSING


# ====================
# Path Configuration
# ====================

@dataclass
class PathsConfig:
    """Paths configuration."""
    root: str = "."
    data: str = "${paths.root}/data"
    models: str = "${paths.data}/models"
    raw_data: str = "${paths.data}/raw"
    processed_data: str = "${paths.data}/processed"
    logs: str = "${paths.data}/logs"


# ====================
# Battle Configuration
# ====================

@dataclass
class BattleConfig:
    """Battle format configuration."""
    format: str = "gen9vgc2024regg"
    team_size: int = 6
    bring_size: int = 4


# ====================
# State Configuration
# ====================

@dataclass
class StateConfig:
    """State encoding configuration."""
    dim: int = 620
    pokemon_features: int = 50
    field_features: int = 20
    num_pokemon: int = 6


# ====================
# Action Configuration
# ====================

@dataclass
class ActionConfig:
    """Action space configuration."""
    dim: int = 144
    actions_per_slot: int = 12
    num_moves: int = 4
    num_switch_targets: int = 4


# ====================
# Model Configurations
# ====================

@dataclass
class ArchitectureConfig:
    """Neural network architecture config."""
    type: str = "mlp"
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.1
    activation: str = "relu"
    use_layer_norm: bool = True


@dataclass
class InitConfig:
    """Weight initialization config."""
    method: str = "orthogonal"
    gain: float = 1.414


@dataclass
class RegularizationConfig:
    """Regularization config."""
    weight_decay: float = 0.0001
    dropout: float = 0.1


@dataclass
class ModelConfig:
    """Model configuration."""
    name: str = "imitation"
    architecture: ArchitectureConfig = field(default_factory=ArchitectureConfig)
    state_dim: int = 620
    action_dim: int = 144
    init: InitConfig = field(default_factory=InitConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)


# ====================
# Training Configurations
# ====================

@dataclass
class SchedulerConfig:
    """Learning rate scheduler config."""
    type: str = "cosine"
    eta_min: float = 1e-6


@dataclass
class ImitationTrainingConfig:
    """Imitation learning training config."""
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_split: float = 0.1
    patience: int = 5
    winner_weight: float = 1.5
    loser_weight: float = 0.5
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class PPOTrainingConfig:
    """PPO training config."""
    total_timesteps: int = 100_000
    learning_rate: float = 1e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class OpponentSelectionConfig:
    """Self-play opponent selection config."""
    current: float = 0.6
    recent: float = 0.3
    hall_of_fame: float = 0.1


@dataclass
class SelfPlayConfig:
    """Self-play training config."""
    iterations: int = 100
    population_size: int = 10
    matches_per_iteration: int = 20
    hall_of_fame_size: int = 5
    elo_k_factor: float = 32
    initial_elo: float = 1500
    opponent_selection: OpponentSelectionConfig = field(
        default_factory=OpponentSelectionConfig
    )


@dataclass
class RewardsConfig:
    """Reward shaping config."""
    win: float = 10.0
    lose: float = -10.0
    ko: float = 2.0
    faint: float = -2.0
    hp_diff_scale: float = 0.1
    turn_penalty: float = -0.01


@dataclass
class CheckpointConfig:
    """Checkpointing config."""
    save_freq: int = 10_000
    keep_last_n: int = 5


@dataclass
class TrainingConfig:
    """Complete training configuration."""
    name: str = "full"
    imitation: ImitationTrainingConfig = field(default_factory=ImitationTrainingConfig)
    ppo: PPOTrainingConfig = field(default_factory=PPOTrainingConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    rewards: RewardsConfig = field(default_factory=RewardsConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


# ====================
# Data Configurations
# ====================

@dataclass
class DataSourceConfig:
    """Data source config."""
    type: str = "huggingface"
    dataset_id: str = "cameronangliss/vgc-battle-logs"


@dataclass
class ProcessingConfig:
    """Data processing config."""
    max_battles: Optional[int] = None
    batch_size: int = 1000
    num_workers: int = 4


@dataclass
class StorageConfig:
    """Data storage config."""
    format: str = "parquet"
    compression: str = "snappy"


@dataclass
class ValidationConfig:
    """Data validation config."""
    check_state_dims: bool = True
    check_action_range: bool = True
    check_reward_range: bool = True
    min_turns: int = 1
    max_turns: int = 50


@dataclass
class DataConfig:
    """Complete data configuration."""
    name: str = "vgc_bench"
    source: DataSourceConfig = field(default_factory=DataSourceConfig)
    default_file: str = "logs-gen9vgc2024regg.json"
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


# ====================
# Logging Configuration
# ====================

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    format: str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"


# ====================
# Main Configuration
# ====================

@dataclass
class VGCConfig:
    """Root configuration for VGC AI."""
    paths: PathsConfig = field(default_factory=PathsConfig)
    battle: BattleConfig = field(default_factory=BattleConfig)
    state: StateConfig = field(default_factory=StateConfig)
    action: ActionConfig = field(default_factory=ActionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    seed: int = 42
    device: str = "auto"


def get_device(device_str: str) -> str:
    """Resolve device string to actual device."""
    import torch
    
    if device_str == "auto":
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    return device_str

