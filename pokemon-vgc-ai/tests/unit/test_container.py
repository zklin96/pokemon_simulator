"""Unit tests for the dependency injection container."""

import pytest
from src.core.container import (
    Container, Scope, ServiceNotFoundError, CircularDependencyError,
    get_container, set_container,
)


class TestContainer:
    """Tests for Container class."""
    
    def test_register_and_resolve(self, container):
        """Test basic registration and resolution."""
        class MyService:
            pass
        
        container.register(MyService)
        instance = container.resolve(MyService)
        
        assert isinstance(instance, MyService)
    
    def test_singleton_scope(self, container):
        """Test that singleton scope returns the same instance."""
        class SingletonService:
            pass
        
        container.register(SingletonService, scope=Scope.SINGLETON)
        
        instance1 = container.resolve(SingletonService)
        instance2 = container.resolve(SingletonService)
        
        assert instance1 is instance2
    
    def test_transient_scope(self, container):
        """Test that transient scope returns new instances."""
        class TransientService:
            pass
        
        container.register(TransientService, scope=Scope.TRANSIENT)
        
        instance1 = container.resolve(TransientService)
        instance2 = container.resolve(TransientService)
        
        assert instance1 is not instance2
    
    def test_register_factory(self, container):
        """Test registration with a factory function."""
        class ConfigurableService:
            def __init__(self, value):
                self.value = value
        
        container.register_factory(
            ConfigurableService,
            lambda c: ConfigurableService(42),
        )
        
        instance = container.resolve(ConfigurableService)
        assert instance.value == 42
    
    def test_register_instance(self, container):
        """Test registering an existing instance."""
        class MyService:
            pass
        
        existing = MyService()
        container.register_instance(MyService, existing)
        
        resolved = container.resolve(MyService)
        assert resolved is existing
    
    def test_named_registrations(self, container):
        """Test named service registrations."""
        class DatabaseConnection:
            def __init__(self, name):
                self.name = name
        
        container.register_factory(
            DatabaseConnection,
            lambda c: DatabaseConnection("primary"),
            name="primary",
        )
        container.register_factory(
            DatabaseConnection,
            lambda c: DatabaseConnection("replica"),
            name="replica",
        )
        
        primary = container.resolve(DatabaseConnection, name="primary")
        replica = container.resolve(DatabaseConnection, name="replica")
        
        assert primary.name == "primary"
        assert replica.name == "replica"
    
    def test_service_not_found(self, container):
        """Test that resolving unregistered service raises error."""
        class UnregisteredService:
            pass
        
        with pytest.raises(ServiceNotFoundError):
            container.resolve(UnregisteredService)
    
    def test_circular_dependency_detection(self, container):
        """Test that circular dependencies are detected."""
        class ServiceA:
            pass
        
        class ServiceB:
            pass
        
        # Create circular dependency
        container.register_factory(
            ServiceA,
            lambda c: (c.resolve(ServiceB), ServiceA())[1],
        )
        container.register_factory(
            ServiceB,
            lambda c: (c.resolve(ServiceA), ServiceB())[1],
        )
        
        with pytest.raises(CircularDependencyError):
            container.resolve(ServiceA)
    
    def test_is_registered(self, container):
        """Test checking if service is registered."""
        class RegisteredService:
            pass
        
        class NotRegisteredService:
            pass
        
        container.register(RegisteredService)
        
        assert container.is_registered(RegisteredService)
        assert not container.is_registered(NotRegisteredService)
    
    def test_child_scope(self, container):
        """Test child container inherits from parent."""
        class ParentService:
            pass
        
        class ChildService:
            pass
        
        container.register(ParentService)
        child = container.create_scope()
        child.register(ChildService)
        
        # Child can resolve parent services
        assert isinstance(child.resolve(ParentService), ParentService)
        assert isinstance(child.resolve(ChildService), ChildService)
        
        # Parent cannot resolve child services
        with pytest.raises(ServiceNotFoundError):
            container.resolve(ChildService)
    
    def test_dependency_injection(self, container):
        """Test dependency injection through factory."""
        class Logger:
            def log(self, msg):
                return f"LOG: {msg}"
        
        class Service:
            def __init__(self, logger):
                self.logger = logger
        
        container.register(Logger)
        container.register_factory(
            Service,
            lambda c: Service(c.resolve(Logger)),
        )
        
        service = container.resolve(Service)
        assert service.logger.log("test") == "LOG: test"
    
    def test_clear(self, container):
        """Test clearing all registrations."""
        class MyService:
            pass
        
        container.register(MyService)
        assert container.is_registered(MyService)
        
        container.clear()
        assert not container.is_registered(MyService)


class TestGlobalContainer:
    """Tests for global container functions."""
    
    def test_get_container(self):
        """Test getting global container."""
        container = get_container()
        assert isinstance(container, Container)
    
    def test_set_container(self):
        """Test setting global container."""
        new_container = Container()
        set_container(new_container)
        
        assert get_container() is new_container

