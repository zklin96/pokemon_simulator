"""Plugin Architecture for VGC AI.

This module provides a plugin system that enables:
- Adding new data sources without modifying core code
- Swapping state encoders for experimentation
- Registering new training algorithms
- Runtime discovery and loading of plugins

Example:
    # Define a plugin
    class MyCustomEncoder(EncoderPlugin):
        name = "my_encoder"
        
        def create_encoder(self, config):
            return MyEncoderImpl(config)
    
    # Register it
    registry = PluginRegistry()
    registry.register(MyCustomEncoder())
    
    # Use it
    encoder = registry.get_encoder("my_encoder").create_encoder(config)
"""

from abc import ABC, abstractmethod
from typing import (
    TypeVar, Type, Dict, Any, Optional, List, 
    Callable, Protocol, runtime_checkable, Iterator
)
from dataclasses import dataclass, field
from pathlib import Path
import importlib
import importlib.util
import sys
from loguru import logger

from .config_schema import VGCConfig


T = TypeVar("T")


# ====================
# Base Plugin Interface
# ====================

class Plugin(ABC):
    """Base class for all plugins.
    
    Plugins provide a way to extend the VGC AI system with
    new functionality without modifying core code.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name identifying this plugin."""
        pass
    
    @property
    def version(self) -> str:
        """Plugin version."""
        return "1.0.0"
    
    @property
    def description(self) -> str:
        """Human-readable description."""
        return ""
    
    def initialize(self, config: VGCConfig) -> None:
        """Initialize the plugin with configuration.
        
        Called when the plugin is loaded.
        
        Args:
            config: Global configuration
        """
        pass
    
    def cleanup(self) -> None:
        """Cleanup resources when plugin is unloaded."""
        pass


# ====================
# Data Loader Plugin
# ====================

@dataclass
class DataBatch:
    """A batch of data from a data loader."""
    states: Any
    actions: Any
    rewards: Any
    dones: Any
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataLoaderPlugin(Plugin):
    """Plugin for loading battle data from various sources.
    
    Implement this to add support for new data formats or sources.
    """
    
    @property
    @abstractmethod
    def supported_formats(self) -> List[str]:
        """List of supported data format identifiers."""
        pass
    
    @abstractmethod
    def can_load(self, path: Path) -> bool:
        """Check if this loader can handle the given path.
        
        Args:
            path: Path to data file or directory
            
        Returns:
            True if this loader can handle it
        """
        pass
    
    @abstractmethod
    def load(
        self, 
        path: Path, 
        config: VGCConfig
    ) -> Iterator[DataBatch]:
        """Load data and yield batches.
        
        Args:
            path: Path to data
            config: Configuration
            
        Yields:
            DataBatch objects
        """
        pass
    
    @abstractmethod
    def get_metadata(self, path: Path) -> Dict[str, Any]:
        """Get metadata about the data source.
        
        Args:
            path: Path to data
            
        Returns:
            Dict with metadata (size, format, etc.)
        """
        pass


# ====================
# Encoder Plugin
# ====================

class EncoderPlugin(Plugin):
    """Plugin for state encoding implementations.
    
    Implement this to add new ways to encode battle state.
    """
    
    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Output dimension of the encoder."""
        pass
    
    @abstractmethod
    def create_encoder(self, config: VGCConfig) -> Any:
        """Create an encoder instance.
        
        Args:
            config: Configuration
            
        Returns:
            Encoder instance
        """
        pass
    
    @property
    def requires_training(self) -> bool:
        """Whether encoder has trainable parameters."""
        return False


# ====================
# Trainer Plugin  
# ====================

@dataclass
class TrainingResult:
    """Result from a training run."""
    model_path: Path
    metrics: Dict[str, float]
    config: Dict[str, Any]
    duration_seconds: float


class TrainerPlugin(Plugin):
    """Plugin for training algorithms.
    
    Implement this to add new training methods.
    """
    
    @property
    @abstractmethod
    def algorithm(self) -> str:
        """Name of the training algorithm."""
        pass
    
    @abstractmethod
    def train(
        self,
        data_loader: DataLoaderPlugin,
        config: VGCConfig,
        resume_from: Optional[Path] = None,
    ) -> TrainingResult:
        """Run training.
        
        Args:
            data_loader: Data source
            config: Configuration
            resume_from: Optional checkpoint to resume from
            
        Returns:
            TrainingResult with model and metrics
        """
        pass
    
    @abstractmethod
    def evaluate(
        self,
        model_path: Path,
        data_loader: DataLoaderPlugin,
        config: VGCConfig,
    ) -> Dict[str, float]:
        """Evaluate a trained model.
        
        Args:
            model_path: Path to model checkpoint
            data_loader: Test data
            config: Configuration
            
        Returns:
            Dict of evaluation metrics
        """
        pass


# ====================
# Opponent Plugin
# ====================

class OpponentPlugin(Plugin):
    """Plugin for battle opponents.
    
    Implement this to add new opponent types for training/evaluation.
    """
    
    @property
    @abstractmethod
    def difficulty(self) -> str:
        """Difficulty level (easy, medium, hard, expert)."""
        pass
    
    @abstractmethod
    def create_opponent(self, config: VGCConfig) -> Any:
        """Create an opponent instance.
        
        Args:
            config: Configuration
            
        Returns:
            Opponent player instance
        """
        pass


# ====================
# Plugin Registry
# ====================

class PluginRegistry:
    """Registry for discovering and managing plugins.
    
    Example:
        registry = PluginRegistry()
        registry.discover("./plugins")
        
        encoder = registry.get_encoder("attention")
        trainer = registry.get_trainer("ppo")
    """
    
    def __init__(self):
        """Initialize the registry."""
        self._data_loaders: Dict[str, DataLoaderPlugin] = {}
        self._encoders: Dict[str, EncoderPlugin] = {}
        self._trainers: Dict[str, TrainerPlugin] = {}
        self._opponents: Dict[str, OpponentPlugin] = {}
        self._all_plugins: Dict[str, Plugin] = {}
    
    def register(self, plugin: Plugin) -> None:
        """Register a plugin.
        
        Args:
            plugin: Plugin instance to register
        """
        if plugin.name in self._all_plugins:
            logger.warning(f"Plugin '{plugin.name}' already registered, overwriting")
        
        self._all_plugins[plugin.name] = plugin
        
        # Register by type
        if isinstance(plugin, DataLoaderPlugin):
            self._data_loaders[plugin.name] = plugin
            logger.debug(f"Registered data loader: {plugin.name}")
        
        if isinstance(plugin, EncoderPlugin):
            self._encoders[plugin.name] = plugin
            logger.debug(f"Registered encoder: {plugin.name}")
        
        if isinstance(plugin, TrainerPlugin):
            self._trainers[plugin.name] = plugin
            logger.debug(f"Registered trainer: {plugin.name}")
        
        if isinstance(plugin, OpponentPlugin):
            self._opponents[plugin.name] = plugin
            logger.debug(f"Registered opponent: {plugin.name}")
    
    def unregister(self, name: str) -> None:
        """Unregister a plugin by name.
        
        Args:
            name: Plugin name
        """
        if name in self._all_plugins:
            plugin = self._all_plugins.pop(name)
            plugin.cleanup()
            
            self._data_loaders.pop(name, None)
            self._encoders.pop(name, None)
            self._trainers.pop(name, None)
            self._opponents.pop(name, None)
            
            logger.debug(f"Unregistered plugin: {name}")
    
    def get_data_loader(self, name: str) -> DataLoaderPlugin:
        """Get a data loader plugin by name."""
        if name not in self._data_loaders:
            raise KeyError(f"Data loader '{name}' not found")
        return self._data_loaders[name]
    
    def get_encoder(self, name: str) -> EncoderPlugin:
        """Get an encoder plugin by name."""
        if name not in self._encoders:
            raise KeyError(f"Encoder '{name}' not found")
        return self._encoders[name]
    
    def get_trainer(self, name: str) -> TrainerPlugin:
        """Get a trainer plugin by name."""
        if name not in self._trainers:
            raise KeyError(f"Trainer '{name}' not found")
        return self._trainers[name]
    
    def get_opponent(self, name: str) -> OpponentPlugin:
        """Get an opponent plugin by name."""
        if name not in self._opponents:
            raise KeyError(f"Opponent '{name}' not found")
        return self._opponents[name]
    
    def list_data_loaders(self) -> List[str]:
        """List registered data loader names."""
        return list(self._data_loaders.keys())
    
    def list_encoders(self) -> List[str]:
        """List registered encoder names."""
        return list(self._encoders.keys())
    
    def list_trainers(self) -> List[str]:
        """List registered trainer names."""
        return list(self._trainers.keys())
    
    def list_opponents(self) -> List[str]:
        """List registered opponent names."""
        return list(self._opponents.keys())
    
    def discover(self, plugin_dir: Path) -> int:
        """Discover and load plugins from a directory.
        
        Plugins are Python files with classes that inherit from Plugin.
        
        Args:
            plugin_dir: Directory to search for plugins
            
        Returns:
            Number of plugins discovered
        """
        plugin_dir = Path(plugin_dir)
        if not plugin_dir.exists():
            logger.warning(f"Plugin directory not found: {plugin_dir}")
            return 0
        
        count = 0
        for py_file in plugin_dir.glob("*.py"):
            if py_file.name.startswith("_"):
                continue
            
            try:
                plugins = self._load_plugins_from_file(py_file)
                for plugin in plugins:
                    self.register(plugin)
                    count += 1
            except Exception as e:
                logger.error(f"Failed to load plugin from {py_file}: {e}")
        
        logger.info(f"Discovered {count} plugins from {plugin_dir}")
        return count
    
    def _load_plugins_from_file(self, path: Path) -> List[Plugin]:
        """Load plugin classes from a Python file.
        
        Args:
            path: Path to Python file
            
        Returns:
            List of instantiated plugins
        """
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            return []
        
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        
        plugins = []
        for name in dir(module):
            obj = getattr(module, name)
            if (
                isinstance(obj, type) 
                and issubclass(obj, Plugin) 
                and obj is not Plugin
                and not obj.__name__.endswith("Plugin")  # Skip base classes
            ):
                try:
                    plugins.append(obj())
                except Exception as e:
                    logger.warning(f"Could not instantiate {name}: {e}")
        
        return plugins
    
    def initialize_all(self, config: VGCConfig) -> None:
        """Initialize all registered plugins.
        
        Args:
            config: Configuration to pass to plugins
        """
        for plugin in self._all_plugins.values():
            try:
                plugin.initialize(config)
            except Exception as e:
                logger.error(f"Failed to initialize plugin {plugin.name}: {e}")
    
    def cleanup_all(self) -> None:
        """Cleanup all registered plugins."""
        for plugin in self._all_plugins.values():
            try:
                plugin.cleanup()
            except Exception as e:
                logger.error(f"Failed to cleanup plugin {plugin.name}: {e}")


# ====================
# Global Registry
# ====================

_global_registry: Optional[PluginRegistry] = None


def get_registry() -> PluginRegistry:
    """Get the global plugin registry.
    
    Returns:
        Global PluginRegistry instance
    """
    global _global_registry
    if _global_registry is None:
        _global_registry = PluginRegistry()
    return _global_registry


def set_registry(registry: PluginRegistry) -> None:
    """Set the global plugin registry.
    
    Args:
        registry: Registry to set as global
    """
    global _global_registry
    _global_registry = registry


# ====================
# Built-in Plugins
# ====================

class FlatEncoderPlugin(EncoderPlugin):
    """Built-in flat state encoder (current 620-dim implementation)."""
    
    @property
    def name(self) -> str:
        return "flat"
    
    @property
    def description(self) -> str:
        return "Flat 620-dimensional state encoding"
    
    @property
    def state_dim(self) -> int:
        return 620
    
    def create_encoder(self, config: VGCConfig) -> Any:
        # Import here to avoid circular imports
        from ..engine.state.game_state import GameStateEncoder
        return GameStateEncoder()


class RandomOpponentPlugin(OpponentPlugin):
    """Built-in random opponent."""
    
    @property
    def name(self) -> str:
        return "random"
    
    @property
    def description(self) -> str:
        return "Random action opponent"
    
    @property
    def difficulty(self) -> str:
        return "easy"
    
    def create_opponent(self, config: VGCConfig) -> Any:
        from ..ml.battle_ai.agents import RandomAgent
        return RandomAgent(action_dim=config.action.dim)


class HeuristicOpponentPlugin(OpponentPlugin):
    """Built-in heuristic opponent."""
    
    @property
    def name(self) -> str:
        return "heuristic"
    
    @property
    def description(self) -> str:
        return "Heuristic-based opponent"
    
    @property
    def difficulty(self) -> str:
        return "medium"
    
    def create_opponent(self, config: VGCConfig) -> Any:
        from ..ml.battle_ai.agents import HeuristicAgent
        return HeuristicAgent()


def register_builtin_plugins(registry: Optional[PluginRegistry] = None) -> None:
    """Register all built-in plugins.
    
    Args:
        registry: Registry to use (defaults to global)
    """
    if registry is None:
        registry = get_registry()
    
    registry.register(FlatEncoderPlugin())
    registry.register(RandomOpponentPlugin())
    registry.register(HeuristicOpponentPlugin())
    
    logger.debug("Registered built-in plugins")

