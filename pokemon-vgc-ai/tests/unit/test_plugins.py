"""Unit tests for the plugin architecture."""

import pytest
from pathlib import Path
import tempfile
from typing import Iterator, Dict, Any, List

from src.core.plugins import (
    Plugin, DataLoaderPlugin, EncoderPlugin, TrainerPlugin, OpponentPlugin,
    PluginRegistry, DataBatch, TrainingResult,
    get_registry, set_registry, register_builtin_plugins,
)
from src.core.config_schema import VGCConfig


# ====================
# Test Plugin Implementations
# ====================

class TestPlugin(Plugin):
    """Test plugin implementation."""
    
    @property
    def name(self) -> str:
        return "test_plugin"
    
    @property
    def version(self) -> str:
        return "1.0.0"
    
    @property
    def description(self) -> str:
        return "A test plugin"


class TestDataLoader(DataLoaderPlugin):
    """Test data loader plugin."""
    
    @property
    def name(self) -> str:
        return "test_loader"
    
    @property
    def supported_formats(self) -> List[str]:
        return ["test", "json"]
    
    def can_load(self, path: Path) -> bool:
        return path.suffix in [".test", ".json"]
    
    def load(self, path: Path, config: VGCConfig) -> Iterator[DataBatch]:
        yield DataBatch(
            states=[1, 2, 3],
            actions=[0, 1, 2],
            rewards=[0.1, 0.2, 0.3],
            dones=[False, False, True],
        )
    
    def get_metadata(self, path: Path) -> Dict[str, Any]:
        return {"format": "test", "size": 100}


class TestEncoder(EncoderPlugin):
    """Test encoder plugin."""
    
    @property
    def name(self) -> str:
        return "test_encoder"
    
    @property
    def state_dim(self) -> int:
        return 256
    
    def create_encoder(self, config: VGCConfig) -> Any:
        return {"type": "test_encoder"}


class TestTrainer(TrainerPlugin):
    """Test trainer plugin."""
    
    @property
    def name(self) -> str:
        return "test_trainer"
    
    @property
    def algorithm(self) -> str:
        return "test"
    
    def train(self, data_loader, config, resume_from=None) -> TrainingResult:
        return TrainingResult(
            model_path=Path("test_model.pt"),
            metrics={"accuracy": 0.95},
            config={},
            duration_seconds=10.0,
        )
    
    def evaluate(self, model_path, data_loader, config) -> Dict[str, float]:
        return {"accuracy": 0.95}


class TestOpponent(OpponentPlugin):
    """Test opponent plugin."""
    
    @property
    def name(self) -> str:
        return "test_opponent"
    
    @property
    def difficulty(self) -> str:
        return "easy"
    
    def create_opponent(self, config: VGCConfig) -> Any:
        return {"type": "test_opponent"}


# ====================
# Tests
# ====================

class TestPluginRegistry:
    """Tests for PluginRegistry."""
    
    @pytest.fixture
    def registry(self):
        """Create a fresh registry for each test."""
        return PluginRegistry()
    
    def test_register_plugin(self, registry):
        """Test registering a plugin."""
        plugin = TestPlugin()
        registry.register(plugin)
        
        assert "test_plugin" in registry._all_plugins
    
    def test_register_data_loader(self, registry):
        """Test registering a data loader plugin."""
        loader = TestDataLoader()
        registry.register(loader)
        
        assert "test_loader" in registry.list_data_loaders()
        assert registry.get_data_loader("test_loader") is loader
    
    def test_register_encoder(self, registry):
        """Test registering an encoder plugin."""
        encoder = TestEncoder()
        registry.register(encoder)
        
        assert "test_encoder" in registry.list_encoders()
        assert registry.get_encoder("test_encoder") is encoder
    
    def test_register_trainer(self, registry):
        """Test registering a trainer plugin."""
        trainer = TestTrainer()
        registry.register(trainer)
        
        assert "test_trainer" in registry.list_trainers()
        assert registry.get_trainer("test_trainer") is trainer
    
    def test_register_opponent(self, registry):
        """Test registering an opponent plugin."""
        opponent = TestOpponent()
        registry.register(opponent)
        
        assert "test_opponent" in registry.list_opponents()
        assert registry.get_opponent("test_opponent") is opponent
    
    def test_unregister(self, registry):
        """Test unregistering a plugin."""
        plugin = TestPlugin()
        registry.register(plugin)
        
        assert "test_plugin" in registry._all_plugins
        
        registry.unregister("test_plugin")
        
        assert "test_plugin" not in registry._all_plugins
    
    def test_get_nonexistent_raises(self, registry):
        """Test that getting nonexistent plugin raises KeyError."""
        with pytest.raises(KeyError):
            registry.get_data_loader("nonexistent")
        
        with pytest.raises(KeyError):
            registry.get_encoder("nonexistent")
    
    def test_overwrite_warning(self, registry, caplog):
        """Test that overwriting plugin logs warning."""
        plugin1 = TestPlugin()
        plugin2 = TestPlugin()
        
        registry.register(plugin1)
        registry.register(plugin2)
        
        assert "already registered" in caplog.text
    
    def test_initialize_all(self, registry, sample_config):
        """Test initializing all plugins."""
        class InitializablePlugin(Plugin):
            initialized = False
            
            @property
            def name(self):
                return "initializable"
            
            def initialize(self, config):
                InitializablePlugin.initialized = True
        
        plugin = InitializablePlugin()
        registry.register(plugin)
        registry.initialize_all(sample_config)
        
        assert InitializablePlugin.initialized


class TestDataLoaderPlugin:
    """Tests for DataLoaderPlugin."""
    
    def test_can_load(self):
        """Test can_load method."""
        loader = TestDataLoader()
        
        assert loader.can_load(Path("test.json"))
        assert loader.can_load(Path("data.test"))
        assert not loader.can_load(Path("data.csv"))
    
    def test_load(self, sample_config):
        """Test loading data."""
        loader = TestDataLoader()
        batches = list(loader.load(Path("test.json"), sample_config))
        
        assert len(batches) == 1
        assert batches[0].states == [1, 2, 3]
    
    def test_get_metadata(self):
        """Test getting metadata."""
        loader = TestDataLoader()
        metadata = loader.get_metadata(Path("test.json"))
        
        assert metadata["format"] == "test"
        assert metadata["size"] == 100


class TestEncoderPlugin:
    """Tests for EncoderPlugin."""
    
    def test_create_encoder(self, sample_config):
        """Test creating encoder."""
        encoder_plugin = TestEncoder()
        encoder = encoder_plugin.create_encoder(sample_config)
        
        assert encoder["type"] == "test_encoder"
    
    def test_state_dim(self):
        """Test state dimension property."""
        encoder = TestEncoder()
        assert encoder.state_dim == 256


class TestTrainerPlugin:
    """Tests for TrainerPlugin."""
    
    def test_train(self, sample_config):
        """Test training."""
        trainer = TestTrainer()
        loader = TestDataLoader()
        
        result = trainer.train(loader, sample_config)
        
        assert result.metrics["accuracy"] == 0.95
        assert result.duration_seconds == 10.0
    
    def test_evaluate(self, sample_config):
        """Test evaluation."""
        trainer = TestTrainer()
        loader = TestDataLoader()
        
        metrics = trainer.evaluate(Path("model.pt"), loader, sample_config)
        
        assert metrics["accuracy"] == 0.95


class TestBuiltinPlugins:
    """Tests for built-in plugins."""
    
    def test_register_builtin_plugins(self):
        """Test registering built-in plugins."""
        registry = PluginRegistry()
        register_builtin_plugins(registry)
        
        assert "flat" in registry.list_encoders()
        assert "random" in registry.list_opponents()
        assert "heuristic" in registry.list_opponents()


class TestGlobalRegistry:
    """Tests for global registry functions."""
    
    def test_get_registry(self):
        """Test getting global registry."""
        registry = get_registry()
        assert isinstance(registry, PluginRegistry)
    
    def test_set_registry(self):
        """Test setting global registry."""
        new_registry = PluginRegistry()
        set_registry(new_registry)
        
        assert get_registry() is new_registry

