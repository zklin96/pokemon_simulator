"""Tests for GameStateEncoder poke-env compatibility."""

import pytest
import numpy as np
from unittest.mock import MagicMock, PropertyMock

from src.engine.state.game_state import (
    GameStateEncoder,
    ActionSpaceHandler,
    PokemonState,
    FieldState,
    StatusCondition,
    Weather,
    Terrain,
    NUM_TYPES,
)


class TestGameStateEncoder:
    """Tests for GameStateEncoder."""
    
    def setup_method(self):
        self.encoder = GameStateEncoder()
    
    def test_observation_size(self):
        """Test observation space dimensions."""
        assert self.encoder.observation_size == 620
        assert self.encoder.STATE_DIM == 620
        assert self.encoder.observation_space_shape == (620,)
    
    def test_empty_pokemon_state(self):
        """Test empty Pokemon state creation."""
        empty = self.encoder._empty_pokemon_state()
        assert isinstance(empty, PokemonState)
        assert empty.current_hp_fraction == 0.0
        assert empty.is_fainted == True
        assert empty.is_terastallized == False
    
    def test_empty_pokemon_tensor_shape(self):
        """Test that empty Pokemon tensor has correct shape."""
        empty = self.encoder._empty_pokemon_state()
        tensor = empty.to_tensor()
        assert tensor.shape == (44,)  # 44 features per Pokemon
        assert tensor.dtype == np.float32
    
    def test_encode_pokemon_with_mock(self):
        """Test encoding a mock Pokemon object."""
        # Create mock Pokemon with poke-env-like attributes
        mock_pokemon = MagicMock()
        mock_pokemon.base_stats = {"hp": 100, "atk": 120, "def": 80, "spa": 90, "spd": 85, "spe": 110}
        
        # Create mock Type objects
        mock_type1 = MagicMock()
        mock_type1.name = "Fire"
        mock_type2 = MagicMock()
        mock_type2.name = "Flying"
        mock_pokemon.types = [mock_type1, mock_type2]
        
        mock_pokemon.current_hp_fraction = 0.75
        mock_pokemon.max_hp = 100
        mock_pokemon.current_hp = 75
        mock_pokemon.fainted = False
        mock_pokemon.active = True
        mock_pokemon.status = None
        mock_pokemon.boosts = {"atk": 1, "def": 0, "spa": 0, "spd": -1, "spe": 2, "accuracy": 0, "evasion": 0}
        mock_pokemon.is_terastallized = False
        mock_pokemon.tera_type = None
        mock_pokemon.moves = {}
        mock_pokemon.item = "lifeorb"
        
        state = self.encoder.encode_pokemon(mock_pokemon)
        
        assert isinstance(state, PokemonState)
        assert state.current_hp_fraction == 0.75
        assert state.is_active == True
        assert state.is_fainted == False
        assert state.is_terastallized == False
        assert state.can_terastallize == True
        assert state.item_active == True
    
    def test_encode_pokemon_terastallized(self):
        """Test encoding a terastallized Pokemon."""
        mock_pokemon = MagicMock()
        mock_pokemon.base_stats = {"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}
        
        mock_type = MagicMock()
        mock_type.name = "Dragon"
        mock_pokemon.types = [mock_type]
        
        mock_pokemon.current_hp_fraction = 1.0
        mock_pokemon.fainted = False
        mock_pokemon.active = True
        mock_pokemon.status = None
        mock_pokemon.boosts = {}
        
        # Terastallized to Fairy
        mock_pokemon.is_terastallized = True
        mock_tera_type = MagicMock()
        mock_tera_type.name = "Fairy"
        mock_pokemon.tera_type = mock_tera_type
        mock_pokemon.moves = {}
        mock_pokemon.item = None
        
        state = self.encoder.encode_pokemon(mock_pokemon)
        
        assert state.is_terastallized == True
        assert state.can_terastallize == False  # Already terastallized
        assert state.tera_type == 17  # Fairy is index 17
    
    def test_encode_pokemon_with_status(self):
        """Test encoding Pokemon with various status conditions."""
        for status_str, expected_status in [
            ("brn", StatusCondition.BURN),
            ("par", StatusCondition.PARALYSIS),
            ("slp", StatusCondition.SLEEP),
            ("psn", StatusCondition.POISON),
            ("tox", StatusCondition.TOXIC),
            ("frz", StatusCondition.FREEZE),
        ]:
            mock_pokemon = MagicMock()
            mock_pokemon.base_stats = {"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}
            mock_pokemon.types = []
            mock_pokemon.current_hp_fraction = 0.5
            mock_pokemon.fainted = False
            mock_pokemon.active = False
            mock_pokemon.status = status_str
            mock_pokemon.boosts = {}
            mock_pokemon.is_terastallized = False
            mock_pokemon.tera_type = None
            mock_pokemon.moves = {}
            mock_pokemon.item = None
            
            state = self.encoder.encode_pokemon(mock_pokemon)
            assert state.status == int(expected_status), f"Expected {expected_status} for {status_str}"
    
    def test_encode_pokemon_handles_none_attributes(self):
        """Test that encoder handles missing/None attributes gracefully."""
        mock_pokemon = MagicMock()
        mock_pokemon.base_stats = None  # None instead of dict
        mock_pokemon.types = None  # None instead of list
        mock_pokemon.current_hp_fraction = None
        mock_pokemon.max_hp = None
        mock_pokemon.current_hp = None
        mock_pokemon.fainted = False
        mock_pokemon.active = False
        mock_pokemon.status = None
        mock_pokemon.boosts = None
        mock_pokemon.is_terastallized = False
        mock_pokemon.tera_type = None
        mock_pokemon.moves = None
        mock_pokemon.item = None
        
        # Should not raise an exception
        state = self.encoder.encode_pokemon(mock_pokemon)
        assert isinstance(state, PokemonState)
    
    def test_encode_none_pokemon(self):
        """Test encoding None Pokemon returns empty state."""
        state = self.encoder.encode_pokemon(None)
        empty = self.encoder._empty_pokemon_state()
        
        # Should match empty state
        assert state.is_fainted == empty.is_fainted
        assert state.current_hp_fraction == empty.current_hp_fraction


class TestFieldState:
    """Tests for field state encoding."""
    
    def setup_method(self):
        self.encoder = GameStateEncoder()
    
    def test_encode_field_empty(self):
        """Test encoding field with no conditions."""
        mock_battle = MagicMock()
        mock_battle.weather = {}
        mock_battle.fields = {}
        mock_battle.side_conditions = {}
        mock_battle.opponent_side_conditions = {}
        
        field = self.encoder.encode_field(mock_battle)
        
        assert field.weather == Weather.NONE
        assert field.terrain == Terrain.NONE
        assert field.trick_room == False
        assert field.tailwind_player == False
    
    def test_encode_field_with_weather(self):
        """Test encoding field with weather."""
        mock_battle = MagicMock()
        mock_battle.weather = {"sunnyday": 3}
        mock_battle.fields = {}
        mock_battle.side_conditions = {}
        mock_battle.opponent_side_conditions = {}
        
        field = self.encoder.encode_field(mock_battle)
        
        assert field.weather == Weather.SUN
    
    def test_field_tensor_shape(self):
        """Test that field tensor has correct shape."""
        field = FieldState(
            weather=0, weather_turns=0,
            terrain=0, terrain_turns=0,
            trick_room=False, trick_room_turns=0,
            tailwind_player=False, tailwind_opponent=False,
            reflect_player=False, light_screen_player=False,
            reflect_opponent=False, light_screen_opponent=False,
        )
        tensor = field.to_tensor()
        assert tensor.shape == (12,)


class TestActionSpaceHandler:
    """Tests for action space handling."""
    
    def setup_method(self):
        self.handler = ActionSpaceHandler()
    
    def test_action_space_size(self):
        """Test action space dimensions."""
        assert self.handler.action_space_size == 144
        assert self.handler.actions_per_slot == 12
    
    def test_decode_action(self):
        """Test action decoding."""
        # Action 0: slot1=0, slot2=0
        assert self.handler.decode_action(0) == (0, 0)
        
        # Action 12: slot1=1, slot2=0
        assert self.handler.decode_action(12) == (1, 0)
        
        # Action 13: slot1=1, slot2=1
        assert self.handler.decode_action(13) == (1, 1)
        
        # Action 143: slot1=11, slot2=11
        assert self.handler.decode_action(143) == (11, 11)
    
    def test_get_slot_actions_terastallize(self):
        """Test that tera actions respect is_terastallized attribute."""
        mock_battle = MagicMock()
        mock_battle.can_tera = True
        
        mock_pokemon = MagicMock()
        mock_pokemon.fainted = False
        mock_pokemon.is_terastallized = False  # Can still tera
        
        mock_battle.active_pokemon = [mock_pokemon]
        mock_move = MagicMock()
        mock_move.id = "testmove"
        mock_battle.available_moves = [[mock_move]]
        mock_battle.available_switches = [[]]
        
        actions = self.handler._get_slot_actions(mock_battle, 0)
        
        # Should include move 0 and tera+move 4
        assert 0 in actions
        assert 4 in actions
    
    def test_get_slot_actions_already_terastallized(self):
        """Test that tera actions are not available when already terastallized."""
        mock_battle = MagicMock()
        mock_battle.can_tera = True
        
        mock_pokemon = MagicMock()
        mock_pokemon.fainted = False
        mock_pokemon.is_terastallized = True  # Already terastallized
        
        mock_battle.active_pokemon = [mock_pokemon]
        mock_move = MagicMock()
        mock_move.id = "testmove"
        mock_battle.available_moves = [[mock_move]]
        mock_battle.available_switches = [[]]
        
        actions = self.handler._get_slot_actions(mock_battle, 0)
        
        # Should include move 0 but NOT tera+move 4
        assert 0 in actions
        assert 4 not in actions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

