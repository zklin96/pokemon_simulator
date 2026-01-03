"""Unit tests for configuration management."""

import pytest
from pathlib import Path
import tempfile

from src.core.config_schema import (
    VGCConfig, PathsConfig, BattleConfig, StateConfig, ActionConfig,
    ModelConfig, TrainingConfig, DataConfig, get_device,
)


class TestVGCConfig:
    """Tests for VGCConfig dataclass."""
    
    def test_default_config(self):
        """Test creating config with defaults."""
        config = VGCConfig()
        
        assert config.seed == 42
        assert config.device == "auto"
        assert config.state.dim == 620
        assert config.action.dim == 144
    
    def test_paths_config(self):
        """Test PathsConfig defaults."""
        config = PathsConfig()
        
        assert config.root == "."
        assert "${paths.root}" in config.data
    
    def test_battle_config(self):
        """Test BattleConfig defaults."""
        config = BattleConfig()
        
        assert config.format == "gen9vgc2024regg"
        assert config.team_size == 6
        assert config.bring_size == 4
    
    def test_state_config(self):
        """Test StateConfig defaults."""
        config = StateConfig()
        
        assert config.dim == 620
        assert config.pokemon_features == 50
        assert config.num_pokemon == 6
    
    def test_action_config(self):
        """Test ActionConfig defaults."""
        config = ActionConfig()
        
        assert config.dim == 144
        assert config.actions_per_slot == 12
        assert config.num_moves == 4
    
    def test_model_config(self):
        """Test ModelConfig defaults."""
        config = ModelConfig()
        
        assert config.name == "imitation"
        assert config.state_dim == 620
        assert config.action_dim == 144
        assert config.architecture.type == "mlp"
        assert config.architecture.hidden_dims == [512, 256, 128]
    
    def test_training_config(self):
        """Test TrainingConfig defaults."""
        config = TrainingConfig()
        
        assert config.name == "full"
        assert config.imitation.epochs == 20
        assert config.ppo.total_timesteps == 100_000
        assert config.self_play.population_size == 10
    
    def test_data_config(self):
        """Test DataConfig defaults."""
        config = DataConfig()
        
        assert config.name == "vgc_bench"
        assert "vgc-battle-logs" in config.source.dataset_id
    
    def test_custom_values(self):
        """Test creating config with custom values."""
        state = StateConfig(dim=256, pokemon_features=32)
        action = ActionConfig(dim=100)
        
        config = VGCConfig(
            state=state,
            action=action,
            seed=123,
        )
        
        assert config.state.dim == 256
        assert config.action.dim == 100
        assert config.seed == 123


class TestGetDevice:
    """Tests for get_device function."""
    
    def test_explicit_device(self):
        """Test explicit device specification."""
        assert get_device("cpu") == "cpu"
        assert get_device("cuda") == "cuda"
    
    def test_auto_device(self):
        """Test auto device detection."""
        device = get_device("auto")
        assert device in ["cpu", "cuda", "mps"]

