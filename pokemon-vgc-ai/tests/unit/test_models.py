"""Unit tests for neural network models."""

import pytest
import torch
import numpy as np
from pathlib import Path


class TestImitationPolicy:
    """Tests for ImitationPolicy model."""
    
    @pytest.fixture
    def model(self):
        """Create model for testing."""
        from src.ml.models.imitation_policy import ImitationPolicy
        return ImitationPolicy(state_dim=620, action_dim=144)
    
    def test_forward(self, model, sample_state_batch, device):
        """Test forward pass."""
        states = torch.tensor(sample_state_batch).to(device)
        model = model.to(device)
        
        action_logits, values = model(states)
        
        assert action_logits.shape == (32, 144)
        assert values.shape == (32, 1)
    
    def test_get_action_probs(self, model, sample_state_batch, device):
        """Test action probability computation."""
        states = torch.tensor(sample_state_batch).to(device)
        model = model.to(device)
        
        probs = model.get_action_probs(states)
        
        assert probs.shape == (32, 144)
        # Probabilities should sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(32, device=device), atol=1e-5)
    
    def test_get_value(self, model, sample_state_batch, device):
        """Test value computation."""
        states = torch.tensor(sample_state_batch).to(device)
        model = model.to(device)
        
        values = model.get_value(states)
        
        assert values.shape == (32, 1)
    
    def test_save_load(self, model, temp_dir, device):
        """Test saving and loading model."""
        model = model.to(device)
        save_path = temp_dir / "model.pt"
        
        # Save
        torch.save(model.state_dict(), save_path)
        
        # Load
        from src.ml.models.imitation_policy import ImitationPolicy
        loaded = ImitationPolicy(state_dim=620, action_dim=144).to(device)
        loaded.load_state_dict(torch.load(save_path, map_location=device))
        
        # Compare outputs
        states = torch.randn(1, 620, device=device)
        with torch.no_grad():
            orig_logits, _ = model(states)
            loaded_logits, _ = loaded(states)
        
        assert torch.allclose(orig_logits, loaded_logits)
    
    def test_gradients(self, model, sample_state_batch, device):
        """Test that gradients flow properly."""
        states = torch.tensor(sample_state_batch, device=device, requires_grad=False)
        targets = torch.randint(0, 144, (32,), device=device)
        
        model = model.to(device)
        logits, values = model(states)
        
        loss = torch.nn.functional.cross_entropy(logits, targets)
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"


class TestPropertyBasedModels:
    """Property-based tests for models using Hypothesis."""
    
    def test_output_determinism(self, imitation_policy, device):
        """Test that same input gives same output."""
        model = imitation_policy.to(device).eval()
        states = torch.randn(1, 620, device=device)
        
        with torch.no_grad():
            out1, _ = model(states)
            out2, _ = model(states)
        
        assert torch.equal(out1, out2)
    
    def test_batch_independence(self, imitation_policy, device):
        """Test that batch items don't affect each other."""
        model = imitation_policy.to(device).eval()
        
        state1 = torch.randn(1, 620, device=device)
        state2 = torch.randn(1, 620, device=device)
        
        with torch.no_grad():
            # Process separately
            out1_single, _ = model(state1)
            out2_single, _ = model(state2)
            
            # Process as batch
            batch = torch.cat([state1, state2], dim=0)
            out_batch, _ = model(batch)
        
        assert torch.allclose(out1_single, out_batch[0:1], atol=1e-5)
        assert torch.allclose(out2_single, out_batch[1:2], atol=1e-5)

