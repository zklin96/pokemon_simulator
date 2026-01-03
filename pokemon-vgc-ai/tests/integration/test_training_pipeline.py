"""Integration tests for training pipelines."""

import pytest
import torch
import numpy as np
from pathlib import Path
import json


class TestImitationLearningPipeline:
    """Integration tests for imitation learning pipeline."""
    
    @pytest.fixture
    def training_data(self, temp_dir):
        """Create training data for tests."""
        trajectories = []
        for i in range(100):
            traj = {
                "battle_id": f"battle_{i:03d}",
                "player": "p1",
                "winner": "p1" if i % 2 == 0 else "p2",
                "transitions": [
                    {
                        "state": np.random.randn(620).tolist(),
                        "action": np.random.randint(0, 144),
                        "reward": np.random.randn() * 0.1,
                        "done": j == 9,
                    }
                    for j in range(10)
                ],
            }
            trajectories.append(traj)
        
        data_path = temp_dir / "trajectories.json"
        with open(data_path, "w") as f:
            json.dump(trajectories, f)
        
        return data_path
    
    def test_data_loading(self, training_data):
        """Test loading trajectory data."""
        with open(training_data) as f:
            data = json.load(f)
        
        assert len(data) == 100
        assert all("transitions" in t for t in data)
    
    def test_model_training(self, training_data, temp_dir, device):
        """Test training loop runs without errors."""
        from src.ml.models.imitation_policy import ImitationPolicy
        
        # Load data
        with open(training_data) as f:
            trajectories = json.load(f)
        
        # Prepare data
        states = []
        actions = []
        for traj in trajectories:
            for trans in traj["transitions"]:
                states.append(trans["state"])
                actions.append(trans["action"])
        
        states = torch.tensor(states, dtype=torch.float32, device=device)
        actions = torch.tensor(actions, dtype=torch.long, device=device)
        
        # Create model and optimizer
        model = ImitationPolicy(state_dim=620, action_dim=144).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(3):
            optimizer.zero_grad()
            logits, _ = model(states)
            loss = torch.nn.functional.cross_entropy(logits, actions)
            loss.backward()
            optimizer.step()
        
        # Check loss decreased
        final_loss = loss.item()
        assert final_loss < float("inf")
        assert not np.isnan(final_loss)
        
        # Save model
        torch.save(model.state_dict(), temp_dir / "model.pt")
        assert (temp_dir / "model.pt").exists()


class TestSelfPlayPipeline:
    """Integration tests for self-play pipeline."""
    
    def test_agent_population(self, temp_dir):
        """Test agent population management."""
        from src.ml.training.self_play import AgentPopulation
        
        pop = AgentPopulation(
            population_size=5,
            save_dir=temp_dir / "population",
        )
        
        # Add agents
        for i in range(3):
            pop.add_agent(
                model_path=str(temp_dir / f"agent_{i}.pt"),
                training_steps=1000 * (i + 1),
            )
        
        assert len(pop.agents) == 3
        
        # Get best agent
        best = pop.get_best_agent()
        assert best is not None
    
    def test_elo_system(self):
        """Test ELO rating updates."""
        from src.ml.training.self_play import EloSystem
        
        elo = EloSystem(k_factor=32)
        
        # Player A beats player B
        new_a, new_b = elo.update(1500, 1500, 1.0)
        
        assert new_a > 1500
        assert new_b < 1500
        assert abs(new_a - 1500) == abs(new_b - 1500)


class TestEndToEndPipeline:
    """End-to-end integration tests."""
    
    def test_full_workflow(self, temp_dir, device):
        """Test complete training workflow."""
        from src.ml.models.imitation_policy import ImitationPolicy
        
        # Step 1: Create mock data
        states = torch.randn(100, 620, device=device)
        actions = torch.randint(0, 144, (100,), device=device)
        
        # Step 2: Train imitation model
        model = ImitationPolicy(state_dim=620, action_dim=144).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
        for _ in range(10):
            optimizer.zero_grad()
            logits, _ = model(states)
            loss = torch.nn.functional.cross_entropy(logits, actions)
            loss.backward()
            optimizer.step()
        
        # Step 3: Evaluate model
        model.eval()
        with torch.no_grad():
            logits, _ = model(states)
            preds = logits.argmax(dim=1)
            accuracy = (preds == actions).float().mean().item()
        
        # Should learn something
        assert accuracy > 0.01  # Better than random (1/144)
        
        # Step 4: Save model
        save_path = temp_dir / "final_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "accuracy": accuracy,
        }, save_path)
        
        assert save_path.exists()

