"""Game state representation for VGC doubles battles.

This module provides tensor representations of battle states for use
with reinforcement learning algorithms.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import IntEnum

from poke_env.battle import DoubleBattle, Pokemon, Move
from loguru import logger


class StatusCondition(IntEnum):
    """Pokemon status conditions."""
    NONE = 0
    BURN = 1
    FREEZE = 2
    PARALYSIS = 3
    POISON = 4
    TOXIC = 5
    SLEEP = 6


class Weather(IntEnum):
    """Weather conditions."""
    NONE = 0
    SUN = 1
    RAIN = 2
    SAND = 3
    SNOW = 4


class Terrain(IntEnum):
    """Terrain conditions."""
    NONE = 0
    ELECTRIC = 1
    GRASSY = 2
    MISTY = 3
    PSYCHIC = 4


# Type encoding (18 types + None)
TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic", "bug",
    "rock", "ghost", "dragon", "dark", "steel", "fairy"
]
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPES)}

# Number of features for different components
NUM_TYPES = 18
# Actual features per Pokemon: base_stats(6) + types(18) + HP/status(4) + boosts(7) + tera(3) + pp(4) + item/ability(2) = 44
# But we keep 50 for backward compatibility with pre-trained models
NUM_POKEMON_FEATURES = 50  # Per Pokemon (includes padding for compatibility)
# Actual field features: weather(2) + terrain(2) + trick_room(2) + tailwinds(2) + screens(4) = 12
# But we keep 20 for backward compatibility
NUM_FIELD_FEATURES = 20
NUM_MOVE_FEATURES = 15


@dataclass
class PokemonState:
    """Encoded state for a single Pokemon."""
    
    # Species one-hot encoding would be too large, use embedding or stats
    base_stats: np.ndarray  # [6] HP, Atk, Def, SpA, SpD, Spe
    types: np.ndarray  # [18] one-hot for primary + secondary type
    current_hp_fraction: float  # 0.0 to 1.0
    is_active: bool
    is_fainted: bool
    status: int  # StatusCondition enum value
    stat_boosts: np.ndarray  # [7] Atk, Def, SpA, SpD, Spe, Acc, Eva
    can_terastallize: bool
    is_terastallized: bool
    tera_type: int  # Type index or -1
    moves_pp: np.ndarray  # [4] PP remaining as fraction
    item_active: bool  # Whether item is still usable
    ability_index: int  # Simplified ability encoding
    
    def to_tensor(self) -> np.ndarray:
        """Convert to flat numpy array.
        
        Returns:
            Numpy array of shape [NUM_POKEMON_FEATURES]
        """
        features = []
        
        # Normalized base stats (divide by 255 for reasonable range)
        features.extend(self.base_stats / 255.0)
        
        # Type encoding
        features.extend(self.types)
        
        # HP and status
        features.append(self.current_hp_fraction)
        features.append(float(self.is_active))
        features.append(float(self.is_fainted))
        features.append(self.status / 6.0)  # Normalize status
        
        # Stat boosts (range -6 to +6, normalize to -1 to 1)
        features.extend(self.stat_boosts / 6.0)
        
        # Terastallization
        features.append(float(self.can_terastallize))
        features.append(float(self.is_terastallized))
        features.append((self.tera_type + 1) / 19.0)  # -1 to 17 -> 0 to 1
        
        # Moves PP
        features.extend(self.moves_pp)
        
        # Item and ability
        features.append(float(self.item_active))
        features.append(self.ability_index / 100.0)  # Rough normalization
        
        result = np.array(features, dtype=np.float32)
        
        # Pad to NUM_POKEMON_FEATURES for backward compatibility
        if len(result) < NUM_POKEMON_FEATURES:
            padding = np.zeros(NUM_POKEMON_FEATURES - len(result), dtype=np.float32)
            result = np.concatenate([result, padding])
        
        return result


@dataclass
class FieldState:
    """Encoded state for field conditions."""
    
    weather: int  # Weather enum value
    weather_turns: int
    terrain: int  # Terrain enum value
    terrain_turns: int
    trick_room: bool
    trick_room_turns: int
    tailwind_player: bool  # True if player has tailwind
    tailwind_opponent: bool
    reflect_player: bool
    light_screen_player: bool
    reflect_opponent: bool
    light_screen_opponent: bool
    
    def to_tensor(self) -> np.ndarray:
        """Convert to flat numpy array."""
        features = np.array([
            self.weather / 4.0,
            self.weather_turns / 5.0,
            self.terrain / 4.0,
            self.terrain_turns / 5.0,
            float(self.trick_room),
            self.trick_room_turns / 5.0,
            float(self.tailwind_player),
            float(self.tailwind_opponent),
            float(self.reflect_player),
            float(self.light_screen_player),
            float(self.reflect_opponent),
            float(self.light_screen_opponent),
        ], dtype=np.float32)
        
        # Pad to NUM_FIELD_FEATURES for backward compatibility
        if len(features) < NUM_FIELD_FEATURES:
            padding = np.zeros(NUM_FIELD_FEATURES - len(features), dtype=np.float32)
            features = np.concatenate([features, padding])
        
        return features


class GameStateEncoder:
    """Encodes Pokemon Showdown battle state to tensor representation."""
    
    def __init__(self):
        """Initialize the encoder."""
        # Calculate observation space size
        # Player: 2 active + 4 bench = 6 Pokemon
        # Opponent: 2 active + 4 bench = 6 Pokemon (may be partially known)
        # Field conditions
        self.pokemon_features = NUM_POKEMON_FEATURES
        self.player_pokemon_count = 6
        self.opponent_pokemon_count = 6
        self.field_features = NUM_FIELD_FEATURES
        
        self.observation_size = (
            self.player_pokemon_count * self.pokemon_features +
            self.opponent_pokemon_count * self.pokemon_features +
            self.field_features
        )
    
    def encode_pokemon(self, pokemon: Optional[Pokemon]) -> PokemonState:
        """Encode a Pokemon to state representation.
        
        Args:
            pokemon: Pokemon object from poke-env
            
        Returns:
            PokemonState dataclass
        """
        if pokemon is None:
            return self._empty_pokemon_state()
        
        # Base stats - use getattr for safety
        stats = getattr(pokemon, 'base_stats', {})
        if stats is None:
            stats = {}
        base_stats = np.array([
            stats.get("hp", 100),
            stats.get("atk", 100),
            stats.get("def", 100),
            stats.get("spa", 100),
            stats.get("spd", 100),
            stats.get("spe", 100),
        ], dtype=np.float32)
        
        # Types (one-hot encoding) - handle both Type enum and string
        types = np.zeros(NUM_TYPES, dtype=np.float32)
        pokemon_types = getattr(pokemon, 'types', [])
        if pokemon_types:
            for ptype in pokemon_types:
                if ptype is None:
                    continue
                # Handle both Type enum and string
                type_name = ptype.name.lower() if hasattr(ptype, 'name') else str(ptype).lower()
                if type_name in TYPE_TO_IDX:
                    types[TYPE_TO_IDX[type_name]] = 1.0
        
        # HP fraction - prefer current_hp_fraction if available
        hp_fraction = getattr(pokemon, 'current_hp_fraction', None)
        if hp_fraction is None:
            max_hp = getattr(pokemon, 'max_hp', None)
            current_hp = getattr(pokemon, 'current_hp', None)
            if max_hp and max_hp > 0 and current_hp is not None:
                hp_fraction = current_hp / max_hp
            else:
                hp_fraction = 0.0 if getattr(pokemon, 'fainted', False) else 1.0
        
        # Status - use getattr for safety
        status = StatusCondition.NONE
        pokemon_status = getattr(pokemon, 'status', None)
        if pokemon_status:
            status_map = {
                "brn": StatusCondition.BURN,
                "frz": StatusCondition.FREEZE,
                "par": StatusCondition.PARALYSIS,
                "psn": StatusCondition.POISON,
                "tox": StatusCondition.TOXIC,
                "slp": StatusCondition.SLEEP,
            }
            status = status_map.get(str(pokemon_status).lower()[:3], StatusCondition.NONE)
        
        # Stat boosts - use getattr for safety
        boosts = getattr(pokemon, 'boosts', {})
        if boosts is None:
            boosts = {}
        stat_boosts = np.array([
            boosts.get("atk", 0),
            boosts.get("def", 0),
            boosts.get("spa", 0),
            boosts.get("spd", 0),
            boosts.get("spe", 0),
            boosts.get("accuracy", 0),
            boosts.get("evasion", 0),
        ], dtype=np.float32)
        
        # Terastallization - use correct poke-env attribute names
        is_tera = getattr(pokemon, 'is_terastallized', False)
        can_tera = not is_tera  # Can tera if not already
        tera_type = -1
        pokemon_tera_type = getattr(pokemon, 'tera_type', None)
        if pokemon_tera_type:
            tera_type_name = pokemon_tera_type.name.lower() if hasattr(pokemon_tera_type, 'name') else str(pokemon_tera_type).lower()
            tera_type = TYPE_TO_IDX.get(tera_type_name, -1)
        
        # Moves PP - try to get actual PP if available
        moves_pp = np.ones(4, dtype=np.float32)
        pokemon_moves = getattr(pokemon, 'moves', {})
        if pokemon_moves:
            move_list = list(pokemon_moves.values())[:4]
            for i, move in enumerate(move_list):
                if hasattr(move, 'current_pp') and hasattr(move, 'max_pp'):
                    if move.max_pp > 0:
                        moves_pp[i] = move.current_pp / move.max_pp
        
        # Is active and fainted - use getattr for safety
        is_active = getattr(pokemon, 'active', False)
        is_fainted = getattr(pokemon, 'fainted', False)
        
        # Item - check if item exists
        item = getattr(pokemon, 'item', None)
        item_active = item is not None
        
        return PokemonState(
            base_stats=base_stats,
            types=types,
            current_hp_fraction=float(hp_fraction),
            is_active=is_active,
            is_fainted=is_fainted,
            status=int(status),
            stat_boosts=stat_boosts,
            can_terastallize=can_tera,
            is_terastallized=is_tera,
            tera_type=tera_type,
            moves_pp=moves_pp,
            item_active=item_active,
            ability_index=0,  # Simplified
        )
    
    def _empty_pokemon_state(self) -> PokemonState:
        """Create empty/unknown Pokemon state."""
        return PokemonState(
            base_stats=np.zeros(6, dtype=np.float32),
            types=np.zeros(NUM_TYPES, dtype=np.float32),
            current_hp_fraction=0.0,
            is_active=False,
            is_fainted=True,
            status=0,
            stat_boosts=np.zeros(7, dtype=np.float32),
            can_terastallize=False,
            is_terastallized=False,
            tera_type=-1,
            moves_pp=np.zeros(4, dtype=np.float32),
            item_active=False,
            ability_index=0,
        )
    
    def encode_field(self, battle: DoubleBattle) -> FieldState:
        """Encode field conditions.
        
        Args:
            battle: Current battle state
            
        Returns:
            FieldState dataclass
        """
        # Weather
        weather = Weather.NONE
        weather_turns = 0
        if battle.weather:
            weather_map = {
                "sunnyday": Weather.SUN,
                "desolateland": Weather.SUN,
                "raindance": Weather.RAIN,
                "primordialsea": Weather.RAIN,
                "sandstorm": Weather.SAND,
                "snow": Weather.SNOW,
                "hail": Weather.SNOW,
            }
            for w, turns in battle.weather.items():
                w_name = str(w).lower().replace("weather.", "")
                weather = weather_map.get(w_name, Weather.NONE)
                weather_turns = turns if isinstance(turns, int) else 0
        
        # Terrain
        terrain = Terrain.NONE
        terrain_turns = 0
        if battle.fields:
            terrain_map = {
                "electricterrain": Terrain.ELECTRIC,
                "grassyterrain": Terrain.GRASSY,
                "mistyterrain": Terrain.MISTY,
                "psychicterrain": Terrain.PSYCHIC,
            }
            for f, turns in battle.fields.items():
                f_name = str(f).lower().replace("field.", "")
                terrain = terrain_map.get(f_name, Terrain.NONE)
                terrain_turns = turns if isinstance(turns, int) else 0
        
        # Side conditions
        player_conditions = battle.side_conditions if hasattr(battle, 'side_conditions') else {}
        opp_conditions = battle.opponent_side_conditions if hasattr(battle, 'opponent_side_conditions') else {}
        
        return FieldState(
            weather=int(weather),
            weather_turns=weather_turns,
            terrain=int(terrain),
            terrain_turns=terrain_turns,
            trick_room="trickroom" in str(battle.fields).lower(),
            trick_room_turns=0,
            tailwind_player="tailwind" in str(player_conditions).lower(),
            tailwind_opponent="tailwind" in str(opp_conditions).lower(),
            reflect_player="reflect" in str(player_conditions).lower(),
            light_screen_player="lightscreen" in str(player_conditions).lower(),
            reflect_opponent="reflect" in str(opp_conditions).lower(),
            light_screen_opponent="lightscreen" in str(opp_conditions).lower(),
        )
    
    def encode_battle(self, battle: DoubleBattle) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode full battle state to tensor.
        
        Args:
            battle: Current battle state
            
        Returns:
            Tuple of (numpy array of shape [observation_size], structured_data dict)
        """
        features = []
        structured_data: Dict[str, Any] = {}
        
        # Encode player's Pokemon
        player_team = list(battle.team.values()) if battle.team else []
        for i in range(self.player_pokemon_count):
            if i < len(player_team):
                pokemon_state = self.encode_pokemon(player_team[i])
            else:
                pokemon_state = self._empty_pokemon_state()
            features.append(pokemon_state.to_tensor())
        
        # Encode opponent's Pokemon (may be partially known)
        opponent_team = list(battle.opponent_team.values()) if battle.opponent_team else []
        for i in range(self.opponent_pokemon_count):
            if i < len(opponent_team):
                pokemon_state = self.encode_pokemon(opponent_team[i])
            else:
                pokemon_state = self._empty_pokemon_state()
            features.append(pokemon_state.to_tensor())
        
        # Encode field
        field_state = self.encode_field(battle)
        features.append(field_state.to_tensor())
        
        # Combine all features
        state_vector = np.concatenate(features)
        
        # Build structured data for enhanced encoders (optional)
        structured_data = {
            "player_team_size": len(player_team),
            "opponent_team_size": len(opponent_team),
            "turn": getattr(battle, 'turn', 0),
        }
        
        return state_vector, structured_data
    
    @property
    def observation_space_shape(self) -> Tuple[int]:
        """Get the shape of the observation space."""
        return (self.observation_size,)
    
    @property
    def STATE_DIM(self) -> int:
        """Alias for observation_size for compatibility."""
        return self.observation_size


class ActionSpaceHandler:
    """Handles action encoding/decoding for VGC doubles battles."""
    
    def __init__(self):
        """Initialize action space handler.
        
        VGC Doubles Action Space:
        - Each active Pokemon can:
          - Use one of 4 moves (with targeting)
          - Switch to one of 4 bench Pokemon
          - Terastallize + move
        
        Simplified encoding:
        - Slot 1: [0-3] moves, [4-7] moves + tera, [8-11] switch
        - Slot 2: [0-3] moves, [4-7] moves + tera, [8-11] switch
        - Combined: slot1_action * 12 + slot2_action = 144 possible actions
        
        For doubles targeting, we simplify by always targeting the opponent.
        """
        self.actions_per_slot = 12  # 4 moves + 4 tera moves + 4 switches
        self.action_space_size = self.actions_per_slot ** 2  # 144
    
    def get_available_actions(self, battle: DoubleBattle) -> List[int]:
        """Get list of available action indices.
        
        Args:
            battle: Current battle state
            
        Returns:
            List of valid action indices
        """
        available = []
        
        # Get available moves and switches for each slot
        slot1_actions = self._get_slot_actions(battle, 0)
        slot2_actions = self._get_slot_actions(battle, 1)
        
        # Combine into joint actions
        for a1 in slot1_actions:
            for a2 in slot2_actions:
                action_idx = a1 * self.actions_per_slot + a2
                available.append(action_idx)
        
        return available if available else [0]  # Return at least one action
    
    def _get_slot_actions(self, battle: DoubleBattle, slot: int) -> List[int]:
        """Get available actions for a single slot."""
        actions = []
        
        # Check if Pokemon in this slot is active
        if slot >= len(battle.active_pokemon):
            return [0]
        
        pokemon = battle.active_pokemon[slot]
        if pokemon is None or pokemon.fainted:
            return [0]
        
        # Get available moves
        moves = battle.available_moves[slot] if slot < len(battle.available_moves) else []
        for i, move in enumerate(moves[:4]):
            actions.append(i)  # Move indices 0-3
            
            # Can terastallize? Use getattr for compatibility
            can_tera = getattr(battle, 'can_tera', False)
            is_terastallized = getattr(pokemon, 'is_terastallized', False)
            if can_tera and not is_terastallized:
                actions.append(i + 4)  # Tera + move indices 4-7
        
        # Get available switches
        switches = battle.available_switches[slot] if slot < len(battle.available_switches) else []
        for i, switch in enumerate(switches[:4]):
            actions.append(i + 8)  # Switch indices 8-11
        
        return actions if actions else [0]
    
    def decode_action(self, action: int) -> Tuple[int, int]:
        """Decode action index to slot actions.
        
        Args:
            action: Combined action index
            
        Returns:
            Tuple of (slot1_action, slot2_action)
        """
        slot1 = action // self.actions_per_slot
        slot2 = action % self.actions_per_slot
        return slot1, slot2
    
    def create_battle_order(self, battle: DoubleBattle, action: int) -> str:
        """Create a battle order string from action index.
        
        Args:
            battle: Current battle state
            action: Action index
            
        Returns:
            Order string for poke-env
        """
        slot1_action, slot2_action = self.decode_action(action)
        
        orders = []
        
        # Process slot 1
        order1 = self._create_slot_order(battle, 0, slot1_action)
        if order1:
            orders.append(order1)
        
        # Process slot 2
        order2 = self._create_slot_order(battle, 1, slot2_action)
        if order2:
            orders.append(order2)
        
        if not orders:
            return "/choose default"
        
        return "/choose " + ", ".join(orders)
    
    def _create_slot_order(self, battle: DoubleBattle, slot: int, action: int) -> Optional[str]:
        """Create order for a single slot."""
        if slot >= len(battle.active_pokemon):
            return None
        
        pokemon = battle.active_pokemon[slot]
        if pokemon is None or pokemon.fainted:
            return None
        
        moves = battle.available_moves[slot] if slot < len(battle.available_moves) else []
        switches = battle.available_switches[slot] if slot < len(battle.available_switches) else []
        
        if action < 4:  # Move
            if action < len(moves):
                move = moves[action]
                return f"move {move.id}"
        elif action < 8:  # Tera + Move
            move_idx = action - 4
            if move_idx < len(moves):
                move = moves[move_idx]
                return f"move {move.id} terastallize"
        else:  # Switch
            switch_idx = action - 8
            if switch_idx < len(switches):
                switch = switches[switch_idx]
                return f"switch {switch.species}"
        
        # Fallback
        if moves:
            return f"move {moves[0].id}"
        if switches:
            return f"switch {switches[0].species}"
        
        return "default"


def test_encoding():
    """Test the state encoding."""
    logger.info("Testing game state encoding...")
    
    encoder = GameStateEncoder()
    action_handler = ActionSpaceHandler()
    
    logger.info(f"Observation space size: {encoder.observation_size}")
    logger.info(f"Action space size: {action_handler.action_space_size}")
    
    # Test with empty Pokemon
    empty_state = encoder._empty_pokemon_state()
    empty_tensor = empty_state.to_tensor()
    logger.info(f"Empty Pokemon tensor shape: {empty_tensor.shape}")
    
    logger.info("State encoding test passed!")


if __name__ == "__main__":
    test_encoding()

