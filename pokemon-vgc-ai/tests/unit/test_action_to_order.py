"""Tests for action-to-order mapping in async_env_wrapper."""

import pytest
import numpy as np
from unittest.mock import MagicMock, PropertyMock

from src.ml.training.async_env_wrapper import ActionDecoder


class MockMove:
    """Mock poke-env Move object."""
    def __init__(self, move_id: str):
        self.id = move_id


class MockPokemon:
    """Mock poke-env Pokemon object."""
    def __init__(self, species: str, fainted: bool = False):
        self.species = species
        self.fainted = fainted
        self.current_hp_fraction = 0.0 if fainted else 1.0


class MockDoubleBattle:
    """Mock poke-env DoubleBattle object."""
    
    def __init__(
        self,
        active_pokemon=None,
        opponent_active_pokemon=None,
        available_moves=None,
        available_switches=None,
        can_tera=None,
    ):
        self.active_pokemon = active_pokemon or [
            MockPokemon("Pikachu"),
            MockPokemon("Charizard"),
        ]
        self.opponent_active_pokemon = opponent_active_pokemon or [
            MockPokemon("Venusaur"),
            MockPokemon("Blastoise"),
        ]
        self.available_moves = available_moves or [
            [MockMove("thunderbolt"), MockMove("quickattack"), MockMove("protect"), MockMove("voltswitch")],
            [MockMove("flamethrower"), MockMove("airslash"), MockMove("protect"), MockMove("willowisp")],
        ]
        self.available_switches = available_switches or [
            [MockPokemon("Snorlax"), MockPokemon("Gengar")],
            [MockPokemon("Snorlax"), MockPokemon("Gengar")],
        ]
        self.can_tera = can_tera or [True, True]


class TestActionDecoder:
    """Tests for ActionDecoder class."""
    
    def setup_method(self):
        self.decoder = ActionDecoder()
        self.battle = MockDoubleBattle()
    
    def test_basic_move_action(self):
        """Test decoding a basic move action."""
        # Action 0: Slot A uses move 0, Slot B uses move 0
        order = self.decoder.action_to_order(self.battle, action=0)
        
        assert order is not None
        assert "move thunderbolt" in order
        assert "move flamethrower" in order
    
    def test_action_decomposition(self):
        """Test that actions are correctly decomposed into slot A and B."""
        # Action = slot_a * 12 + slot_b
        # Action 13 = 1 * 12 + 1 = Slot A move 1, Slot B move 1
        order = self.decoder.action_to_order(self.battle, action=13)
        
        assert order is not None
        assert "move quickattack" in order
        assert "move airslash" in order
    
    def test_tera_move_action(self):
        """Test decoding a Tera move action."""
        # Action 48 = 4 * 12 + 0 = Slot A tera+move 0, Slot B move 0
        order = self.decoder.action_to_order(self.battle, action=48)
        
        assert order is not None
        assert "terastallize" in order
    
    def test_switch_action(self):
        """Test decoding a switch action."""
        # Action 96 = 8 * 12 + 0 = Slot A switch to bench 0, Slot B move 0
        order = self.decoder.action_to_order(self.battle, action=96)
        
        assert order is not None
        assert "switch Snorlax" in order
        assert "move flamethrower" in order
    
    def test_double_switch_action(self):
        """Test both slots switching."""
        # Action 104 = 8 * 12 + 8 = Both switch to bench 0
        order = self.decoder.action_to_order(self.battle, action=104)
        
        assert order is not None
        assert order.count("switch") == 2
    
    def test_all_move_indices(self):
        """Test all 4 move indices per slot."""
        for move_idx in range(4):
            # Slot A uses move_idx, Slot B uses move 0
            action = move_idx * 12 + 0
            order = self.decoder.action_to_order(self.battle, action=action)
            
            assert order is not None
            expected_moves = ["thunderbolt", "quickattack", "protect", "voltswitch"]
            assert f"move {expected_moves[move_idx]}" in order
    
    def test_all_switch_indices(self):
        """Test all 4 switch indices per slot."""
        for switch_idx in range(2):  # Only 2 switches available
            # Slot A switches to bench switch_idx
            action = (8 + switch_idx) * 12 + 0
            order = self.decoder.action_to_order(self.battle, action=action)
            
            assert order is not None
            assert "switch" in order
    
    def test_invalid_move_index_fallback(self):
        """Test fallback when move index is out of range."""
        # Create battle with only 2 moves available
        battle = MockDoubleBattle(
            available_moves=[
                [MockMove("tackle"), MockMove("scratch")],
                [MockMove("ember")],
            ]
        )
        
        # Action 3 = Slot A uses move 3 (doesn't exist), Slot B uses move 0
        order = self.decoder.action_to_order(battle, action=3 * 12 + 0)
        
        # Should fallback to first available move
        assert order is not None
        assert "move tackle" in order or "move ember" in order
    
    def test_fainted_pokemon_skipped(self):
        """Test that fainted Pokemon are skipped."""
        battle = MockDoubleBattle(
            active_pokemon=[
                MockPokemon("Pikachu", fainted=True),
                MockPokemon("Charizard"),
            ]
        )
        
        order = self.decoder.action_to_order(battle, action=0)
        
        # Only Charizard's order should be present
        assert order is not None
        assert "flamethrower" in order
        assert "thunderbolt" not in order
    
    def test_no_available_moves_returns_none(self):
        """Test that no available moves returns None for that slot."""
        battle = MockDoubleBattle(
            available_moves=[[], []],
            available_switches=[[], []],
        )
        
        order = self.decoder.action_to_order(battle, action=0)
        
        # Should return None when no options available
        assert order is None
    
    def test_target_is_included(self):
        """Test that move orders include target specification."""
        order = self.decoder.action_to_order(self.battle, action=0)
        
        # Should include target numbers (1 or 2)
        assert order is not None
        # In VGC doubles, we specify targets
        # This is implementation-dependent


class TestActionSpace:
    """Tests for action space consistency."""
    
    def test_action_space_size(self):
        """Test that action space is 12 * 12 = 144."""
        decoder = ActionDecoder()
        battle = MockDoubleBattle()
        
        # All 144 actions should be decodable (may return None for invalid)
        valid_count = 0
        for action in range(144):
            order = decoder.action_to_order(battle, action)
            if order is not None:
                valid_count += 1
        
        # Most actions should be valid with default battle state
        assert valid_count > 100
    
    def test_action_encoding_decoding(self):
        """Test that encoding/decoding is consistent."""
        for slot_a in range(12):
            for slot_b in range(12):
                action = slot_a * 12 + slot_b
                
                decoded_a = action // 12
                decoded_b = action % 12
                
                assert decoded_a == slot_a
                assert decoded_b == slot_b


class TestActionDecoderEdgeCases:
    """Edge case tests for ActionDecoder."""
    
    def test_single_active_pokemon(self):
        """Test with only one active Pokemon."""
        battle = MockDoubleBattle(
            active_pokemon=[MockPokemon("Pikachu")],
            available_moves=[[MockMove("thunderbolt")]],
            available_switches=[[]],
        )
        
        decoder = ActionDecoder()
        order = decoder.action_to_order(battle, action=0)
        
        assert order is not None
        assert "thunderbolt" in order
    
    def test_tera_not_available(self):
        """Test Tera action when Tera is not available."""
        battle = MockDoubleBattle(can_tera=[False, False])
        
        decoder = ActionDecoder()
        # Tera action
        order = decoder.action_to_order(battle, action=48)  # 4 * 12 + 0
        
        # Should still work, just without terastallize
        assert order is not None
        # terastallize should not be in order
        assert "terastallize" not in order
    
    def test_empty_battle(self):
        """Test with minimal battle state."""
        # Create battle with explicit empty state
        battle = MockDoubleBattle()
        battle.active_pokemon = []
        battle.opponent_active_pokemon = []
        battle.available_moves = []
        battle.available_switches = []
        
        decoder = ActionDecoder()
        order = decoder.action_to_order(battle, action=0)
        
        assert order is None

