"""Core infrastructure for VGC AI."""

from .container import Container, get_container
from .plugins import (
    Plugin,
    DataLoaderPlugin,
    EncoderPlugin,
    TrainerPlugin,
    PluginRegistry,
)

__all__ = [
    "Container",
    "get_container",
    "Plugin",
    "DataLoaderPlugin",
    "EncoderPlugin", 
    "TrainerPlugin",
    "PluginRegistry",
]

