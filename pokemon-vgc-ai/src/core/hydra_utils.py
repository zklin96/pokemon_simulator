"""Hydra utilities for loading and managing configurations.

This module provides helpers for loading configs with Hydra
and converting them to typed dataclasses.
"""

from pathlib import Path
from typing import Any, Optional, TypeVar, Type
import os

from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from loguru import logger

from .config_schema import VGCConfig


T = TypeVar("T")


def get_config_dir() -> Path:
    """Get the config directory path."""
    # Look for config relative to project root
    possible_paths = [
        Path(__file__).parent.parent.parent / "config",  # From src/core/
        Path.cwd() / "config",
        Path.cwd() / "pokemon-vgc-ai" / "config",
    ]
    
    for path in possible_paths:
        if path.exists():
            return path.resolve()
    
    raise FileNotFoundError(
        f"Could not find config directory. Searched: {possible_paths}"
    )


def load_config(
    config_name: str = "default",
    overrides: Optional[list] = None,
    return_dict: bool = False,
) -> VGCConfig | DictConfig:
    """Load configuration using Hydra.
    
    Args:
        config_name: Name of the config file (without .yaml)
        overrides: List of config overrides (e.g., ["training=quick"])
        return_dict: If True, return DictConfig instead of dataclass
        
    Returns:
        Loaded configuration as VGCConfig or DictConfig
    """
    config_dir = get_config_dir()
    
    # Clear any existing Hydra instance
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    
    # Initialize Hydra with config directory
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.3"):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    
    if return_dict:
        return cfg
    
    # Convert to structured config
    return dict_to_config(cfg)


def dict_to_config(cfg: DictConfig) -> VGCConfig:
    """Convert DictConfig to structured VGCConfig.
    
    Args:
        cfg: OmegaConf DictConfig
        
    Returns:
        Structured VGCConfig dataclass
    """
    from .config_schema import (
        VGCConfig, PathsConfig, BattleConfig, StateConfig, ActionConfig,
        ModelConfig, TrainingConfig, DataConfig, LoggingConfig,
        ArchitectureConfig, InitConfig, RegularizationConfig,
        ImitationTrainingConfig, PPOTrainingConfig, SelfPlayConfig,
        RewardsConfig, CheckpointConfig, SchedulerConfig,
        OpponentSelectionConfig, DataSourceConfig, ProcessingConfig,
        StorageConfig, ValidationConfig,
    )
    
    # Helper to safely get nested values
    def get(d, *keys, default=None):
        for key in keys:
            if isinstance(d, (dict, DictConfig)) and key in d:
                d = d[key]
            else:
                return default
        return d
    
    # Build nested configs
    paths = PathsConfig(
        root=get(cfg, "paths", "root", default="."),
        data=get(cfg, "paths", "data", default="./data"),
        models=get(cfg, "paths", "models", default="./data/models"),
        raw_data=get(cfg, "paths", "raw_data", default="./data/raw"),
        processed_data=get(cfg, "paths", "processed_data", default="./data/processed"),
        logs=get(cfg, "paths", "logs", default="./data/logs"),
    )
    
    battle = BattleConfig(
        format=get(cfg, "battle", "format", default="gen9vgc2024regg"),
        team_size=get(cfg, "battle", "team_size", default=6),
        bring_size=get(cfg, "battle", "bring_size", default=4),
    )
    
    state = StateConfig(
        dim=get(cfg, "state", "dim", default=620),
        pokemon_features=get(cfg, "state", "pokemon_features", default=50),
        field_features=get(cfg, "state", "field_features", default=20),
        num_pokemon=get(cfg, "state", "num_pokemon", default=6),
    )
    
    action = ActionConfig(
        dim=get(cfg, "action", "dim", default=144),
        actions_per_slot=get(cfg, "action", "actions_per_slot", default=12),
        num_moves=get(cfg, "action", "num_moves", default=4),
        num_switch_targets=get(cfg, "action", "num_switch_targets", default=4),
    )
    
    # Model config
    arch_cfg = get(cfg, "model", "architecture", default={})
    architecture = ArchitectureConfig(
        type=get(arch_cfg, "type", default="mlp"),
        hidden_dims=list(get(arch_cfg, "hidden_dims", default=[512, 256, 128])),
        dropout=get(arch_cfg, "dropout", default=0.1),
        activation=get(arch_cfg, "activation", default="relu"),
        use_layer_norm=get(arch_cfg, "use_layer_norm", default=True),
    )
    
    model = ModelConfig(
        name=get(cfg, "model", "name", default="imitation"),
        architecture=architecture,
        state_dim=get(cfg, "model", "state_dim", default=620),
        action_dim=get(cfg, "model", "action_dim", default=144),
    )
    
    # Training config (simplified - would need more nested handling)
    training = TrainingConfig(
        name=get(cfg, "training", "name", default="full"),
    )
    
    # Data config
    data = DataConfig(
        name=get(cfg, "data", "name", default="vgc_bench"),
        default_file=get(cfg, "data", "default_file", default="logs-gen9vgc2024regg.json"),
    )
    
    logging = LoggingConfig(
        level=get(cfg, "logging", "level", default="INFO"),
    )
    
    return VGCConfig(
        paths=paths,
        battle=battle,
        state=state,
        action=action,
        model=model,
        training=training,
        data=data,
        logging=logging,
        seed=get(cfg, "seed", default=42),
        device=get(cfg, "device", default="auto"),
    )


def save_config(cfg: VGCConfig | DictConfig, path: Path):
    """Save configuration to YAML file.
    
    Args:
        cfg: Configuration to save
        path: Output path
    """
    if isinstance(cfg, VGCConfig):
        cfg = OmegaConf.structured(cfg)
    
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, path)
    logger.info(f"Saved config to {path}")


def print_config(cfg: VGCConfig | DictConfig):
    """Pretty-print configuration."""
    if isinstance(cfg, VGCConfig):
        cfg = OmegaConf.structured(cfg)
    print(OmegaConf.to_yaml(cfg))


# Global config instance
_global_config: Optional[VGCConfig] = None


def get_global_config() -> VGCConfig:
    """Get the global configuration instance.
    
    Returns:
        Global VGCConfig instance
    """
    global _global_config
    if _global_config is None:
        _global_config = load_config()
    return _global_config


def set_global_config(cfg: VGCConfig):
    """Set the global configuration instance.
    
    Args:
        cfg: Configuration to set as global
    """
    global _global_config
    _global_config = cfg

