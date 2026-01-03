"""Tests for RLPlayer class."""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.ml.training.self_play import RLPlayer


class TestRLPlayerBasics:
    """Basic tests for RLPlayer."""
    
    def test_init_without_model(self):
        """RLPlayer can be initialized without a model."""
        player = RLPlayer()
        assert player.model is None
    
    def test_random_fallback(self):
        """Without model, falls back to random action selection."""
        player = RLPlayer()
        
        # Should return valid action in range
        action = player.select_action(np.zeros(620))
        assert 0 <= action < 144
    
    def test_random_fallback_with_mask(self):
        """Random fallback respects action mask."""
        player = RLPlayer()
        
        # Create mask with only action 50 valid
        mask = np.zeros(144, dtype=bool)
        mask[50] = True
        
        action = player.select_action(np.zeros(620), action_mask=mask)
        assert action == 50
    
    def test_decode_action(self):
        """Action decoding works correctly."""
        player = RLPlayer()
        
        # Action 0 = slot_a=0, slot_b=0
        slot_a, slot_b = player.decode_action(0)
        assert slot_a == 0
        assert slot_b == 0
        
        # Action 50 = 50 // 12 = 4, 50 % 12 = 2
        slot_a, slot_b = player.decode_action(50)
        assert slot_a == 4
        assert slot_b == 2
        
        # Action 143 = 11, 11
        slot_a, slot_b = player.decode_action(143)
        assert slot_a == 11
        assert slot_b == 11


class TestRLPlayerWithMockModel:
    """Tests with mocked model."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock PPO model."""
        model = MagicMock()
        model.predict.return_value = (42, None)
        return model
    
    def test_select_action_with_model(self, mock_model):
        """Action selection with model works."""
        player = RLPlayer(model=mock_model, use_masking=False)
        
        state = np.zeros(620)
        action = player.select_action(state)
        
        assert action == 42
        mock_model.predict.assert_called_once()
    
    def test_deterministic_selection(self, mock_model):
        """Deterministic flag is passed to model."""
        player = RLPlayer(model=mock_model, deterministic=True, use_masking=False)
        
        state = np.zeros(620)
        player.select_action(state)
        
        # Check deterministic was passed
        _, kwargs = mock_model.predict.call_args
        assert kwargs.get('deterministic') == True
    
    def test_stochastic_selection(self, mock_model):
        """Stochastic selection when deterministic=False."""
        player = RLPlayer(model=mock_model, deterministic=False, use_masking=False)
        
        state = np.zeros(620)
        player.select_action(state)
        
        _, kwargs = mock_model.predict.call_args
        assert kwargs.get('deterministic') == False


class TestRLPlayerActionConversion:
    """Tests for action conversion to battle orders."""
    
    def test_action_to_order_no_battle(self):
        """Returns None when battle is None."""
        player = RLPlayer()
        result = player.action_to_order(None, 0)
        assert result is None
    
    def test_slot_action_ranges(self):
        """Slot actions are in correct ranges."""
        player = RLPlayer()
        
        # Test all 144 actions decode correctly
        for action in range(144):
            slot_a, slot_b = player.decode_action(action)
            assert 0 <= slot_a < 12
            assert 0 <= slot_b < 12
    
    def test_action_interpretation(self):
        """Actions are interpreted correctly."""
        player = RLPlayer()
        
        # Actions 0-3: Regular moves
        # Actions 4-7: Tera moves
        # Actions 8-11: Switches
        
        for i in range(4):
            slot_a, _ = player.decode_action(i * 12)
            assert slot_a == i  # Move i
        
        for i in range(4, 8):
            slot_a, _ = player.decode_action(i * 12)
            assert slot_a == i  # Tera + move (i-4)
        
        for i in range(8, 12):
            slot_a, _ = player.decode_action(i * 12)
            assert slot_a == i  # Switch to bench (i-8)


class TestRLPlayerModelLoading:
    """Tests for model loading."""
    
    def test_load_nonexistent_model(self):
        """Loading nonexistent model returns None."""
        player = RLPlayer(model_path="/nonexistent/path.zip")
        assert player.model is None
    
    def test_load_with_path_object(self):
        """Can load with Path object."""
        # This should not crash even with fake path
        player = RLPlayer(model_path=Path("/fake/path.zip"))
        assert player.model is None  # File doesn't exist


class TestRLPlayerIntegration:
    """Integration tests (require real models/environments)."""
    
    @pytest.mark.skipif(True, reason="Requires trained model")
    def test_with_real_model(self):
        """Test with a real trained model."""
        # This test would load an actual model and verify behavior
        pass
    
    def test_with_simulated_env(self):
        """Test action selection with simulated environment state."""
        player = RLPlayer()  # Random fallback
        
        # Simulate multiple turns
        for _ in range(10):
            state = np.random.randn(620).astype(np.float32)
            mask = np.random.rand(144) > 0.7  # Random valid actions
            mask[0] = True  # Ensure at least one valid
            
            action = player.select_action(state, action_mask=mask)
            
            # Verify action is valid
            assert mask[action], "Selected action should be valid according to mask"


class TestRLPlayerConsistency:
    """Tests for consistency and edge cases."""
    
    def test_consistent_action_decode_encode(self):
        """Encoding and decoding actions is consistent."""
        player = RLPlayer()
        
        for action in range(144):
            slot_a, slot_b = player.decode_action(action)
            reconstructed = slot_a * 12 + slot_b
            assert reconstructed == action
    
    def test_empty_mask_fallback(self):
        """Falls back gracefully with empty valid actions."""
        player = RLPlayer()
        
        # All actions invalid (shouldn't happen in practice)
        mask = np.zeros(144, dtype=bool)
        
        # Should still return something (or handle gracefully)
        action = player.select_action(np.zeros(620), action_mask=mask)
        assert 0 <= action < 144

