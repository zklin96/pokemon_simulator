"""Pytest configuration and shared fixtures for VGC AI tests."""

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import json
from typing import Dict, Any, List

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


# ====================
# Configuration Fixtures
# ====================

@pytest.fixture
def project_root() -> Path:
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_config():
    """Create a sample VGCConfig for testing."""
    from src.core.config_schema import VGCConfig
    return VGCConfig()


# ====================
# Container Fixtures
# ====================

@pytest.fixture
def container():
    """Create a fresh container for each test."""
    from src.core.container import Container
    return Container()


@pytest.fixture
def global_container():
    """Set up and tear down global container."""
    from src.core.container import Container, set_container, get_container
    
    original = None
    try:
        original = get_container()
    except Exception:
        pass
    
    container = Container()
    set_container(container)
    
    yield container
    
    if original:
        set_container(original)


# ====================
# Data Fixtures
# ====================

@pytest.fixture
def sample_state() -> np.ndarray:
    """Create a sample state vector."""
    return np.random.randn(620).astype(np.float32)


@pytest.fixture
def sample_state_batch() -> np.ndarray:
    """Create a batch of sample states."""
    return np.random.randn(32, 620).astype(np.float32)


@pytest.fixture
def sample_action() -> int:
    """Create a sample action."""
    return np.random.randint(0, 144)


@pytest.fixture
def sample_trajectory() -> Dict[str, Any]:
    """Create a sample trajectory."""
    num_steps = 10
    return {
        "battle_id": "test_battle_001",
        "player": "p1",
        "winner": "p1",
        "transitions": [
            {
                "state": np.random.randn(620).tolist(),
                "action": np.random.randint(0, 144),
                "reward": np.random.randn() * 0.1,
                "done": i == num_steps - 1,
            }
            for i in range(num_steps)
        ]
    }


@pytest.fixture
def sample_trajectories(sample_trajectory) -> List[Dict[str, Any]]:
    """Create multiple sample trajectories."""
    trajectories = []
    for i in range(10):
        traj = sample_trajectory.copy()
        traj["battle_id"] = f"test_battle_{i:03d}"
        traj["winner"] = "p1" if i % 2 == 0 else "p2"
        trajectories.append(traj)
    return trajectories


@pytest.fixture
def sample_battle_log() -> str:
    """Create a sample Pokemon Showdown battle log."""
    return """
|j|☆Player1
|j|☆Player2
|player|p1|Player1|
|player|p2|Player2|
|teamsize|p1|6
|teamsize|p2|6
|gametype|doubles
|gen|9
|tier|[Gen 9] VGC 2024 Reg G
|rule|Bring 6 Pick 4
|
|start
|switch|p1a: Incineroar|Incineroar, L50, M|100/100
|switch|p1b: Flutter Mane|Flutter Mane, L50|100/100
|switch|p2a: Rillaboom|Rillaboom, L50, M|100/100
|switch|p2b: Urshifu|Urshifu-Rapid-Strike, L50, M|100/100
|-ability|p2a: Rillaboom|Grassy Surge
|-fieldstart|move: Grassy Terrain
|turn|1
|
|move|p1a: Incineroar|Fake Out|p2b: Urshifu
|-damage|p2b: Urshifu|90/100
|cant|p2b: Urshifu|flinch
|move|p1b: Flutter Mane|Moonblast|p2a: Rillaboom
|-supereffective|p2a: Rillaboom
|-damage|p2a: Rillaboom|20/100
|move|p2a: Rillaboom|Grassy Glide|p1b: Flutter Mane
|-damage|p1b: Flutter Mane|60/100
|turn|2
|
|win|Player1
""".strip()


# ====================
# Model Fixtures
# ====================

@pytest.fixture
def imitation_policy():
    """Create an ImitationPolicy model for testing."""
    from src.ml.models.imitation_policy import ImitationPolicy
    return ImitationPolicy(state_dim=620, action_dim=144)


@pytest.fixture
def sample_model_state_dict(imitation_policy):
    """Get a sample model state dict."""
    return imitation_policy.state_dict()


# ====================
# Random Seed Fixture
# ====================

@pytest.fixture(autouse=True)
def set_random_seeds():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)


# ====================
# Device Fixture
# ====================

@pytest.fixture
def device() -> str:
    """Get the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

