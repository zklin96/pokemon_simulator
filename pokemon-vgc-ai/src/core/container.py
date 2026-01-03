"""Dependency Injection Container for VGC AI.

This module provides a lightweight DI container that enables:
- Loose coupling between components
- Easy swapping of implementations
- Testability via mock injection
- Configuration-driven component wiring
"""

from typing import (
    TypeVar, Type, Callable, Dict, Any, Optional, 
    Generic, Protocol, runtime_checkable
)
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import wraps
from loguru import logger


T = TypeVar("T")


class Scope(Enum):
    """Dependency lifecycle scope."""
    TRANSIENT = auto()  # New instance every time
    SINGLETON = auto()  # Single shared instance
    SCOPED = auto()     # Per-request/context instance


@dataclass
class Registration:
    """Stores registration info for a service."""
    factory: Callable[..., Any]
    scope: Scope
    instance: Optional[Any] = None
    dependencies: list = field(default_factory=list)


class ContainerError(Exception):
    """Base exception for container errors."""
    pass


class ServiceNotFoundError(ContainerError):
    """Raised when a requested service is not registered."""
    pass


class CircularDependencyError(ContainerError):
    """Raised when circular dependencies are detected."""
    pass


class Container:
    """Lightweight dependency injection container.
    
    Features:
    - Service registration with factories
    - Lifecycle management (singleton, transient, scoped)
    - Automatic dependency resolution
    - Named services
    - Hierarchical containers (child scopes)
    
    Example:
        container = Container()
        
        # Register a service
        container.register(StateEncoder, GameStateEncoder)
        
        # Register with factory
        container.register_factory(
            Policy, 
            lambda c: ImitationPolicy(c.resolve(StateEncoder))
        )
        
        # Resolve
        encoder = container.resolve(StateEncoder)
        policy = container.resolve(Policy)
    """
    
    def __init__(self, parent: Optional["Container"] = None):
        """Initialize container.
        
        Args:
            parent: Parent container for hierarchical resolution
        """
        self._registrations: Dict[str, Registration] = {}
        self._named_registrations: Dict[str, Dict[str, Registration]] = {}
        self._parent = parent
        self._resolving: set = set()  # Track currently resolving for circular detection
    
    def _get_key(self, service_type: Type[T]) -> str:
        """Get registration key for a type."""
        return f"{service_type.__module__}.{service_type.__name__}"
    
    def register(
        self,
        service_type: Type[T],
        implementation: Optional[Type[T]] = None,
        scope: Scope = Scope.SINGLETON,
        name: Optional[str] = None,
    ) -> "Container":
        """Register a service type with an implementation.
        
        Args:
            service_type: The interface/base type to register
            implementation: The concrete implementation type
            scope: Lifecycle scope
            name: Optional name for named registrations
            
        Returns:
            Self for chaining
        """
        impl = implementation or service_type
        
        def factory(container: Container) -> T:
            return impl()
        
        return self.register_factory(service_type, factory, scope, name)
    
    def register_factory(
        self,
        service_type: Type[T],
        factory: Callable[["Container"], T],
        scope: Scope = Scope.SINGLETON,
        name: Optional[str] = None,
    ) -> "Container":
        """Register a service with a factory function.
        
        Args:
            service_type: The interface/base type to register
            factory: Factory function that creates instances
            scope: Lifecycle scope
            name: Optional name for named registrations
            
        Returns:
            Self for chaining
        """
        key = self._get_key(service_type)
        registration = Registration(factory=factory, scope=scope)
        
        if name:
            if key not in self._named_registrations:
                self._named_registrations[key] = {}
            self._named_registrations[key][name] = registration
        else:
            self._registrations[key] = registration
        
        logger.debug(f"Registered {key}" + (f" as '{name}'" if name else ""))
        return self
    
    def register_instance(
        self,
        service_type: Type[T],
        instance: T,
        name: Optional[str] = None,
    ) -> "Container":
        """Register an existing instance as a singleton.
        
        Args:
            service_type: The interface/base type to register
            instance: The instance to use
            name: Optional name for named registrations
            
        Returns:
            Self for chaining
        """
        key = self._get_key(service_type)
        registration = Registration(
            factory=lambda c: instance,
            scope=Scope.SINGLETON,
            instance=instance,
        )
        
        if name:
            if key not in self._named_registrations:
                self._named_registrations[key] = {}
            self._named_registrations[key][name] = registration
        else:
            self._registrations[key] = registration
        
        return self
    
    def resolve(
        self,
        service_type: Type[T],
        name: Optional[str] = None,
    ) -> T:
        """Resolve a service instance.
        
        Args:
            service_type: The type to resolve
            name: Optional name for named registrations
            
        Returns:
            Service instance
            
        Raises:
            ServiceNotFoundError: If service is not registered
            CircularDependencyError: If circular dependencies detected
        """
        key = self._get_key(service_type)
        
        # Check for circular dependencies
        if key in self._resolving:
            raise CircularDependencyError(
                f"Circular dependency detected for {service_type.__name__}"
            )
        
        # Find registration
        registration = self._find_registration(key, name)
        if registration is None:
            raise ServiceNotFoundError(
                f"Service {service_type.__name__}" + 
                (f" named '{name}'" if name else "") +
                " is not registered"
            )
        
        # Return singleton instance if available
        if registration.scope == Scope.SINGLETON and registration.instance is not None:
            return registration.instance
        
        # Create new instance
        try:
            self._resolving.add(key)
            instance = registration.factory(self)
            
            # Store singleton
            if registration.scope == Scope.SINGLETON:
                registration.instance = instance
            
            return instance
        finally:
            self._resolving.discard(key)
    
    def _find_registration(
        self, 
        key: str, 
        name: Optional[str]
    ) -> Optional[Registration]:
        """Find a registration by key and optional name."""
        if name:
            if key in self._named_registrations:
                if name in self._named_registrations[key]:
                    return self._named_registrations[key][name]
        else:
            if key in self._registrations:
                return self._registrations[key]
        
        # Check parent
        if self._parent:
            return self._parent._find_registration(key, name)
        
        return None
    
    def is_registered(
        self, 
        service_type: Type[T], 
        name: Optional[str] = None
    ) -> bool:
        """Check if a service is registered.
        
        Args:
            service_type: The type to check
            name: Optional name for named registrations
            
        Returns:
            True if registered
        """
        key = self._get_key(service_type)
        return self._find_registration(key, name) is not None
    
    def create_scope(self) -> "Container":
        """Create a child container for scoped services.
        
        Returns:
            New child Container
        """
        return Container(parent=self)
    
    def clear(self):
        """Clear all registrations and instances."""
        self._registrations.clear()
        self._named_registrations.clear()
        self._resolving.clear()


# ====================
# Protocol Interfaces
# ====================

@runtime_checkable
class StateEncoder(Protocol):
    """Protocol for state encoding implementations."""
    
    def encode(self, battle: Any) -> Any:
        """Encode a battle state."""
        ...
    
    @property
    def state_dim(self) -> int:
        """Dimension of encoded state."""
        ...


@runtime_checkable
class ActionHandler(Protocol):
    """Protocol for action handling implementations."""
    
    def get_available_actions(self, battle: Any) -> list:
        """Get list of available actions."""
        ...
    
    def decode_action(self, action: int, battle: Any) -> Any:
        """Decode action index to battle command."""
        ...
    
    @property
    def action_dim(self) -> int:
        """Dimension of action space."""
        ...


@runtime_checkable
class RewardCalculator(Protocol):
    """Protocol for reward calculation implementations."""
    
    def compute(
        self, 
        prev_state: Any, 
        curr_state: Any, 
        done: bool, 
        won: bool
    ) -> float:
        """Compute reward for a transition."""
        ...


@runtime_checkable
class Policy(Protocol):
    """Protocol for policy implementations."""
    
    def get_action(self, state: Any) -> int:
        """Get action for a state."""
        ...
    
    def get_action_probs(self, state: Any) -> Any:
        """Get action probabilities."""
        ...


# ====================
# Global Container
# ====================

_global_container: Optional[Container] = None


def get_container() -> Container:
    """Get the global container instance.
    
    Returns:
        Global Container instance
    """
    global _global_container
    if _global_container is None:
        _global_container = Container()
    return _global_container


def set_container(container: Container):
    """Set the global container instance.
    
    Args:
        container: Container to set as global
    """
    global _global_container
    _global_container = container


def inject(service_type: Type[T], name: Optional[str] = None) -> Callable:
    """Decorator for injecting dependencies.
    
    Example:
        @inject(StateEncoder)
        def train(encoder: StateEncoder):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if service_type.__name__.lower() not in kwargs:
                container = get_container()
                kwargs[service_type.__name__.lower()] = container.resolve(
                    service_type, name
                )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def setup_default_container(config: Any) -> Container:
    """Set up the default container with standard registrations.
    
    Args:
        config: Configuration object
        
    Returns:
        Configured Container
    """
    from .config_schema import VGCConfig
    
    container = Container()
    
    # Register config
    container.register_instance(VGCConfig, config)
    
    # Note: Actual component registrations would go here
    # These would be added as we implement the actual components
    
    set_container(container)
    return container

