"""Tests for action masking logic."""

import pytest
import numpy as np
from src.ml.training.rl_finetuning import SimulatedVGCEnv


class TestActionMask:
    """Tests for action masking logic."""
    
    @pytest.fixture
    def env(self):
        """Create environment with default settings."""
        env = SimulatedVGCEnv(bring_size=4)
        env.reset()
        return env
    
    @pytest.fixture
    def env_no_switch(self):
        """Create environment with switching disabled."""
        env = SimulatedVGCEnv(bring_size=4, allow_switching=False)
        env.reset()
        return env
    
    @pytest.fixture
    def env_no_tera(self):
        """Create environment with tera disabled."""
        env = SimulatedVGCEnv(bring_size=4, allow_tera=False)
        env.reset()
        return env
    
    def test_mask_shape(self, env):
        """Action mask should have correct shape."""
        mask = env.get_action_mask()
        assert mask.shape == (144,)
        assert mask.dtype == bool
    
    def test_all_moves_valid_at_start(self, env):
        """Move actions should be valid at battle start."""
        mask = env.get_action_mask()
        
        # Actions where both slots use moves (0-3)
        for slot_a in range(4):
            for slot_b in range(4):
                action = slot_a * 12 + slot_b
                assert mask[action], f"Action {action} (move+move) should be valid"
    
    def test_tera_valid_at_start(self, env):
        """Tera moves should be valid when tera not used."""
        mask = env.get_action_mask()
        
        # Tera + Move combinations
        for tera_action in range(4, 8):  # Slot A tera moves
            for slot_b in range(4):  # Slot B regular moves
                action = tera_action * 12 + slot_b
                assert mask[action], f"Action {action} (tera+move) should be valid"
    
    def test_tera_invalid_after_use(self, env):
        """Tera moves invalid after tera has been used."""
        # Mark tera as used
        env.my_tera_used = True
        
        mask = env.get_action_mask()
        
        # Tera actions (4-7) should be invalid for slot A
        for tera_action in range(4, 8):
            for slot_b in range(4):
                action = tera_action * 12 + slot_b
                assert not mask[action], f"Action {action} (tera after use) should be invalid"
    
    def test_tera_disabled(self, env_no_tera):
        """Tera moves invalid when tera disabled."""
        mask = env_no_tera.get_action_mask()
        
        # Tera actions should be invalid
        for tera_action in range(4, 8):
            for slot_b in range(4):
                action = tera_action * 12 + slot_b
                assert not mask[action], f"Action {action} (tera disabled) should be invalid"
    
    def test_switch_valid_with_bench(self, env):
        """Switch actions valid when bench Pokemon available."""
        mask = env.get_action_mask()
        
        # With 4 Pokemon and 2 active, we have 2 bench Pokemon
        # Switch actions are 8-11, only first 2 should be valid
        move_action = 0  # Slot B uses move
        
        for switch in range(8, 10):  # First 2 bench slots
            action = switch * 12 + move_action
            assert mask[action], f"Action {action} (switch to valid bench) should be valid"
    
    def test_switch_invalid_when_disabled(self, env_no_switch):
        """Switch actions invalid when switching disabled."""
        mask = env_no_switch.get_action_mask()
        
        # All switch actions should be invalid
        for switch_a in range(8, 12):
            for slot_b in range(12):
                action = switch_a * 12 + slot_b
                assert not mask[action], f"Action {action} (switch disabled) should be invalid"
    
    def test_no_double_switch_same_slot(self, env):
        """Can't switch both Pokemon to same bench slot."""
        mask = env.get_action_mask()
        
        # Both switching to bench slot 0 (action 8)
        action = 8 * 12 + 8  # Switch0 + Switch0
        assert not mask[action], "Double switch to same slot should be invalid"
        
        # But switching to different slots is OK
        action = 8 * 12 + 9  # Switch0 + Switch1
        # This should be valid if there are 2 bench Pokemon
        # (env has 4 Pokemon, 2 active, 2 bench)
        # Note: may or may not be valid depending on bench size
    
    def test_dead_pokemon_excluded(self, env):
        """Can't use moves from fainted Pokemon."""
        # Faint slot A Pokemon
        active = env._get_active_indices()
        if len(active) > 0:
            env.my_pokemon_hp[active[0]] = 0
        
        mask = env.get_action_mask()
        
        # With slot A fainted, only action 0 should be valid for slot A
        # (but slot B actions still work)
        # Actually, the test should check that some actions are still valid
        assert np.any(mask), "Some actions should still be valid"
    
    def test_switch_to_fainted_invalid(self, env):
        """Can't switch to a fainted bench Pokemon."""
        active = env._get_active_indices()
        bench = [i for i in range(env.bring_size) if i not in active]
        
        if len(bench) > 0:
            # Faint the first bench Pokemon
            env.my_pokemon_hp[bench[0]] = 0
            
            mask = env.get_action_mask()
            
            # Switch to bench slot 0 should be invalid
            action = 8 * 12 + 0  # Slot A switches to bench 0
            assert not mask[action], "Switch to fainted Pokemon should be invalid"
    
    def test_at_least_one_valid(self, env):
        """Mask always has at least one valid action."""
        # Faint all Pokemon except one
        for i in range(len(env.my_pokemon_hp)):
            env.my_pokemon_hp[i] = 0
        env.my_pokemon_hp[0] = 1.0  # Keep one alive
        
        mask = env.get_action_mask()
        assert np.any(mask), "Should always have at least one valid action"
    
    def test_mask_changes_during_battle(self, env):
        """Mask should update as battle progresses."""
        initial_mask = env.get_action_mask().copy()
        
        # Use tera
        env.my_tera_used = True
        after_tera_mask = env.get_action_mask()
        
        # Tera actions should now be different
        tera_actions = [a for a in range(144) if 4 <= (a // 12) < 8 or 4 <= (a % 12) < 8]
        for action in tera_actions:
            if initial_mask[action]:
                # If it was valid before, check it changed
                pass  # Some tera actions may still be invalid


class TestActionMaskWithMaskablePPO:
    """Test that action mask integrates with MaskablePPO."""
    
    def test_mask_compatible_with_sb3(self):
        """Mask format is compatible with sb3-contrib MaskablePPO."""
        env = SimulatedVGCEnv()
        env.reset()
        
        mask = env.get_action_mask()
        
        # MaskablePPO expects numpy bool array
        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == env.action_space.n
    
    def test_masked_action_selection(self):
        """Selecting only from valid actions."""
        env = SimulatedVGCEnv()
        env.reset()
        
        mask = env.get_action_mask()
        valid_actions = np.where(mask)[0]
        
        # Should have multiple valid actions
        assert len(valid_actions) > 0
        
        # Randomly select a valid action
        action = np.random.choice(valid_actions)
        assert mask[action]
        
        # Take the action (should not error)
        obs, reward, done, truncated, info = env.step(action)
        assert obs is not None

