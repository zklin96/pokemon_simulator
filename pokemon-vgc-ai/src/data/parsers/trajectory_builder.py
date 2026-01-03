"""Build training trajectories from reconstructed game states.

This module converts game states and actions into training trajectories
suitable for imitation learning and reinforcement learning.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import json
from loguru import logger

from .replay_parser import ReplayParser, ParsedBattle, BattleEvent
from .state_reconstructor import StateReconstructor, TurnState, PokemonState
from src.data.pokemon_ids import get_move_id

# Try to import rapidfuzz for fuzzy matching
try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    logger.warning("rapidfuzz not installed. Fuzzy move matching disabled.")


@dataclass
class Transition:
    """A single state-action-reward transition."""
    state: np.ndarray  # Encoded game state (620-dim flat)
    action: int  # Combined action index (0-143)
    reward: float  # Shaped reward
    next_state: np.ndarray  # Next game state
    done: bool  # Episode terminal
    
    # Targets for each slot (for hierarchical action models)
    slot_a_target: int = 0  # Target index (0-4)
    slot_b_target: int = 0  # Target index (0-4)
    
    # Structured data for enhanced encoder (optional)
    structured_data: Optional[Dict[str, Any]] = None


@dataclass
class Trajectory:
    """A complete trajectory from one player's perspective."""
    battle_id: str
    player_perspective: str  # 'p1' or 'p2'
    won: bool
    transitions: List[Transition] = field(default_factory=list)
    
    @property
    def total_reward(self) -> float:
        """Get total reward for this trajectory."""
        return sum(t.reward for t in self.transitions)
    
    @property
    def length(self) -> int:
        """Get trajectory length."""
        return len(self.transitions)


# Move name to index mapping (simplified - would be larger in production)
MOVE_TO_IDX: Dict[str, int] = {}

# Type effectiveness (simplified)
TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic", "bug",
    "rock", "ghost", "dragon", "dark", "steel", "fairy"
]
TYPE_TO_IDX = {t: i for i, t in enumerate(TYPES)}


class StateEncoder:
    """Encode TurnState to numpy array for training."""
    
    # Feature dimensions
    POKEMON_FEATURES = 50
    FIELD_FEATURES = 20
    NUM_POKEMON = 6  # Per side
    
    # Species name to ID mapping (simplified - would be loaded from a file in production)
    SPECIES_TO_ID: Dict[str, int] = {}
    ABILITY_TO_ID: Dict[str, int] = {}
    ITEM_TO_ID: Dict[str, int] = {}
    MOVE_TO_ID: Dict[str, int] = {}
    
    def __init__(self, include_structured: bool = True):
        """Initialize the encoder.
        
        Args:
            include_structured: Whether to also output structured data for embeddings
        """
        self.observation_size = (
            self.NUM_POKEMON * 2 * self.POKEMON_FEATURES +  # Both teams
            self.FIELD_FEATURES
        )
        self.include_structured = include_structured
    
    def encode(self, state: TurnState, perspective: str = 'p1') -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """Encode a turn state to numpy array and optional structured data.
        
        Args:
            state: The turn state to encode
            perspective: 'p1' or 'p2' - whose perspective to encode from
            
        Returns:
            Tuple of (flat_features, structured_data)
        """
        features = []
        
        # Get teams from correct perspective
        if perspective == 'p1':
            my_active = state.p1_active
            my_bench = state.p1_bench
            my_side = state.p1_side
            my_can_tera = state.p1_can_tera
            opp_active = state.p2_active
            opp_bench = state.p2_bench
            opp_side = state.p2_side
        else:
            my_active = state.p2_active
            my_bench = state.p2_bench
            my_side = state.p2_side
            my_can_tera = state.p2_can_tera
            opp_active = state.p1_active
            opp_bench = state.p1_bench
            opp_side = state.p1_side
        
        # Encode my Pokemon (active first, then bench)
        my_pokemon = list(my_active) + list(my_bench)
        for i in range(self.NUM_POKEMON):
            if i < len(my_pokemon) and my_pokemon[i]:
                features.extend(self._encode_pokemon(my_pokemon[i], known=True))
            else:
                features.extend(self._empty_pokemon_features())
        
        # Encode opponent Pokemon
        opp_pokemon = list(opp_active) + list(opp_bench)
        for i in range(self.NUM_POKEMON):
            if i < len(opp_pokemon) and opp_pokemon[i]:
                features.extend(self._encode_pokemon(opp_pokemon[i], known=True))
            else:
                features.extend(self._empty_pokemon_features())
        
        # Encode field state
        features.extend(self._encode_field(state.field, my_side, opp_side, my_can_tera))
        
        flat_array = np.array(features, dtype=np.float32)
        
        # Build structured data for embeddings if requested
        structured_data = None
        if self.include_structured:
            structured_data = self._build_structured_data(
                my_pokemon, opp_pokemon, my_active, opp_active,
                state.field, my_side, opp_side, my_can_tera
            )
        
        return flat_array, structured_data
    
    def _get_pokemon_ids(self, pokemon: PokemonState) -> Dict[str, Any]:
        """Extract IDs for a Pokemon (for embeddings)."""
        # Hash species name to get a stable ID (0-1024 range)
        species_id = hash(pokemon.species.lower()) % 1024 if pokemon.species else 0
        ability_id = hash(pokemon.ability.lower()) % 310 if pokemon.ability else 0
        item_id = hash(pokemon.item.lower()) % 250 if pokemon.item else 0
        
        # Move IDs
        move_ids = []
        for move in pokemon.moves[:4]:
            move_id = hash(move.lower()) % 920 if move else 0
            move_ids.append(move_id)
        while len(move_ids) < 4:
            move_ids.append(0)
        
        # Type IDs
        type_ids = [0, 0]
        for i, t in enumerate(pokemon.types[:2]):
            if t and t.lower() in TYPE_TO_IDX:
                type_ids[i] = TYPE_TO_IDX[t.lower()] + 1  # +1 to leave 0 for padding
        
        # Tera type ID
        tera_id = 0
        if pokemon.tera_type and pokemon.tera_type.lower() in TYPE_TO_IDX:
            tera_id = TYPE_TO_IDX[pokemon.tera_type.lower()] + 1
        
        # Numerical features (normalized stats, HP, boosts, etc.)
        numerical = [
            pokemon.hp_fraction,
            1.0 if pokemon.is_active else 0.0,
            1.0 if pokemon.is_fainted else 0.0,
            1.0 if pokemon.is_terastallized else 0.0,
            pokemon.stat_boosts.get('atk', 0) / 6.0,
            pokemon.stat_boosts.get('def', 0) / 6.0,
            pokemon.stat_boosts.get('spa', 0) / 6.0,
            pokemon.stat_boosts.get('spd', 0) / 6.0,
            pokemon.stat_boosts.get('spe', 0) / 6.0,
            pokemon.stat_boosts.get('accuracy', 0) / 6.0,
            pokemon.stat_boosts.get('evasion', 0) / 6.0,
        ]
        # Pad to 20 features
        while len(numerical) < 20:
            numerical.append(0.0)
        
        return {
            'species_id': species_id,
            'ability_id': ability_id,
            'item_id': item_id,
            'move_ids': move_ids,
            'type_ids': type_ids,
            'tera_type_id': tera_id,
            'numerical': numerical[:20],
        }
    
    def _build_structured_data(
        self,
        my_pokemon: List[PokemonState],
        opp_pokemon: List[PokemonState],
        my_active: List[PokemonState],
        opp_active: List[PokemonState],
        field_state,
        my_side,
        opp_side,
        can_tera: bool,
    ) -> Dict[str, Any]:
        """Build structured data for the enhanced encoder."""
        # My team IDs
        my_species_ids = []
        my_ability_ids = []
        my_item_ids = []
        my_move_ids = []
        my_type_ids = []
        my_numerical = []
        my_active_mask = []
        my_alive_mask = []
        
        for i in range(self.NUM_POKEMON):
            if i < len(my_pokemon) and my_pokemon[i]:
                poke = my_pokemon[i]
                ids = self._get_pokemon_ids(poke)
                my_species_ids.append(ids['species_id'])
                my_ability_ids.append(ids['ability_id'])
                my_item_ids.append(ids['item_id'])
                my_move_ids.append(ids['move_ids'])
                my_type_ids.append(ids['type_ids'])
                my_numerical.append(ids['numerical'])
                my_active_mask.append(1.0 if poke.is_active else 0.0)
                my_alive_mask.append(0.0 if poke.is_fainted else 1.0)
            else:
                my_species_ids.append(0)
                my_ability_ids.append(0)
                my_item_ids.append(0)
                my_move_ids.append([0, 0, 0, 0])
                my_type_ids.append([0, 0])
                my_numerical.append([0.0] * 20)
                my_active_mask.append(0.0)
                my_alive_mask.append(0.0)
        
        # Opponent team IDs
        opp_species_ids = []
        opp_ability_ids = []
        opp_item_ids = []
        opp_move_ids = []
        opp_type_ids = []
        opp_numerical = []
        opp_active_mask = []
        opp_alive_mask = []
        
        for i in range(self.NUM_POKEMON):
            if i < len(opp_pokemon) and opp_pokemon[i]:
                poke = opp_pokemon[i]
                ids = self._get_pokemon_ids(poke)
                opp_species_ids.append(ids['species_id'])
                opp_ability_ids.append(ids['ability_id'])
                opp_item_ids.append(ids['item_id'])
                opp_move_ids.append(ids['move_ids'])
                opp_type_ids.append(ids['type_ids'])
                opp_numerical.append(ids['numerical'])
                opp_active_mask.append(1.0 if poke.is_active else 0.0)
                opp_alive_mask.append(0.0 if poke.is_fainted else 1.0)
            else:
                opp_species_ids.append(0)
                opp_ability_ids.append(0)
                opp_item_ids.append(0)
                opp_move_ids.append([0, 0, 0, 0])
                opp_type_ids.append([0, 0])
                opp_numerical.append([0.0] * 20)
                opp_active_mask.append(0.0)
                opp_alive_mask.append(0.0)
        
        # Field conditions
        weather_map = {'sun': 1, 'rain': 2, 'sand': 3, 'snow': 4, 'hail': 4}
        weather_id = 0
        if field_state.weather:
            for key, idx in weather_map.items():
                if key in field_state.weather.lower():
                    weather_id = idx
                    break
        
        terrain_map = {'electric': 1, 'grassy': 2, 'misty': 3, 'psychic': 4}
        terrain_id = 0
        if field_state.terrain and field_state.terrain.lower() in terrain_map:
            terrain_id = terrain_map[field_state.terrain.lower()]
        
        field_flags = [
            1.0 if field_state.trick_room else 0.0,
            1.0 if getattr(field_state, 'gravity', False) else 0.0,
            0.0,  # magic_room placeholder
            0.0,  # wonder_room placeholder
        ]
        
        turns_remaining = [
            getattr(field_state, 'weather_turns', 0) / 5.0,
            getattr(field_state, 'terrain_turns', 0) / 5.0,
            getattr(field_state, 'trick_room_turns', 0) / 5.0,
            0.0,
        ]
        
        # Side conditions
        my_side_conds = [
            my_side.reflect if hasattr(my_side, 'reflect') else 0,
            my_side.light_screen if hasattr(my_side, 'light_screen') else 0,
            0,  # aurora_veil placeholder
            my_side.tailwind if hasattr(my_side, 'tailwind') else 0,
            0, 0, 0, 0  # hazard placeholders
        ]
        my_side_conds = [float(x) / 5.0 if isinstance(x, int) else (1.0 if x else 0.0) for x in my_side_conds]
        
        opp_side_conds = [
            opp_side.reflect if hasattr(opp_side, 'reflect') else 0,
            opp_side.light_screen if hasattr(opp_side, 'light_screen') else 0,
            0,
            opp_side.tailwind if hasattr(opp_side, 'tailwind') else 0,
            0, 0, 0, 0
        ]
        opp_side_conds = [float(x) / 5.0 if isinstance(x, int) else (1.0 if x else 0.0) for x in opp_side_conds]
        
        return {
            'my_species_ids': my_species_ids,
            'my_ability_ids': my_ability_ids,
            'my_item_ids': my_item_ids,
            'my_move_ids': my_move_ids,
            'my_type_ids': my_type_ids,
            'my_numerical': my_numerical,
            'my_active_mask': my_active_mask,
            'my_alive_mask': my_alive_mask,
            
            'opp_species_ids': opp_species_ids,
            'opp_ability_ids': opp_ability_ids,
            'opp_item_ids': opp_item_ids,
            'opp_move_ids': opp_move_ids,
            'opp_type_ids': opp_type_ids,
            'opp_numerical': opp_numerical,
            'opp_active_mask': opp_active_mask,
            'opp_alive_mask': opp_alive_mask,
            
            'weather': weather_id,
            'terrain': terrain_id,
            'field_flags': field_flags,
            'turns_remaining': turns_remaining,
            
            'my_side_conditions': my_side_conds,
            'opp_side_conditions': opp_side_conds,
            
            'can_tera': can_tera,
        }
    
    def _encode_pokemon(self, pokemon: PokemonState, known: bool = True) -> List[float]:
        """Encode a single Pokemon."""
        features = []
        
        # HP fraction
        features.append(pokemon.hp_fraction)
        
        # Status (one-hot, 7 values: none, brn, par, slp, frz, psn, tox)
        status_vec = [0.0] * 7
        status_map = {'brn': 1, 'par': 2, 'slp': 3, 'frz': 4, 'psn': 5, 'tox': 6}
        if pokemon.status and pokemon.status.lower() in status_map:
            status_vec[status_map[pokemon.status.lower()]] = 1.0
        else:
            status_vec[0] = 1.0
        features.extend(status_vec)
        
        # Is active
        features.append(1.0 if pokemon.is_active else 0.0)
        
        # Is fainted
        features.append(1.0 if pokemon.is_fainted else 0.0)
        
        # Stat boosts (7 stats, normalized to -1 to 1)
        for stat in ['atk', 'def', 'spa', 'spd', 'spe', 'accuracy', 'evasion']:
            boost = pokemon.stat_boosts.get(stat, 0)
            features.append(boost / 6.0)
        
        # Terastallization
        features.append(1.0 if pokemon.is_terastallized else 0.0)
        
        # Tera type (one-hot, 18 types)
        tera_vec = [0.0] * 18
        if pokemon.tera_type and pokemon.tera_type.lower() in TYPE_TO_IDX:
            tera_vec[TYPE_TO_IDX[pokemon.tera_type.lower()]] = 1.0
        features.extend(tera_vec)
        
        # Pad to POKEMON_FEATURES
        while len(features) < self.POKEMON_FEATURES:
            features.append(0.0)
        
        return features[:self.POKEMON_FEATURES]
    
    def _empty_pokemon_features(self) -> List[float]:
        """Get empty Pokemon features."""
        return [0.0] * self.POKEMON_FEATURES
    
    def _encode_field(self, field_state, my_side, opp_side, can_tera: bool) -> List[float]:
        """Encode field conditions."""
        features = []
        
        # Weather (one-hot, 5 values: none, sun, rain, sand, snow)
        weather_vec = [0.0] * 5
        weather_map = {'sun': 1, 'rain': 2, 'sand': 3, 'snow': 4, 'hail': 4}
        if field_state.weather:
            for key, idx in weather_map.items():
                if key in field_state.weather.lower():
                    weather_vec[idx] = 1.0
                    break
            else:
                weather_vec[0] = 1.0
        else:
            weather_vec[0] = 1.0
        features.extend(weather_vec)
        
        # Terrain (one-hot, 5 values: none, electric, grassy, misty, psychic)
        terrain_vec = [0.0] * 5
        terrain_map = {'electric': 1, 'grassy': 2, 'misty': 3, 'psychic': 4}
        if field_state.terrain and field_state.terrain.lower() in terrain_map:
            terrain_vec[terrain_map[field_state.terrain.lower()]] = 1.0
        else:
            terrain_vec[0] = 1.0
        features.extend(terrain_vec)
        
        # Trick room
        features.append(1.0 if field_state.trick_room else 0.0)
        
        # My side conditions
        features.append(1.0 if my_side.tailwind else 0.0)
        features.append(1.0 if my_side.reflect else 0.0)
        features.append(1.0 if my_side.light_screen else 0.0)
        
        # Opponent side conditions
        features.append(1.0 if opp_side.tailwind else 0.0)
        features.append(1.0 if opp_side.reflect else 0.0)
        features.append(1.0 if opp_side.light_screen else 0.0)
        
        # Can tera
        features.append(1.0 if can_tera else 0.0)
        
        # Pad to FIELD_FEATURES
        while len(features) < self.FIELD_FEATURES:
            features.append(0.0)
        
        return features[:self.FIELD_FEATURES]


class ActionEncoder:
    """Encode actions from battle events.
    
    Action space per slot (12 actions):
        0-3:  Regular moves (move slot 0-3)
        4-7:  Terastallize + moves (move slot 0-3)
        8-11: Switch to bench slot 0-3
        
    Combined action = slot_a_action * 12 + slot_b_action (144 total)
    
    Target encoding (for hierarchical action heads):
        0: No target / Self-targeting / Spread move
        1: Opponent slot A
        2: Opponent slot B  
        3: Ally (partner)
        4: Opponent slot A or B (random in spread)
    """
    
    ACTIONS_PER_SLOT = 12  # 4 moves + 4 tera moves + 4 switches
    ACTION_SPACE_SIZE = ACTIONS_PER_SLOT ** 2  # 144
    NUM_TARGETS = 5  # Target space size
    
    # Target indices
    TARGET_NONE = 0
    TARGET_OPP_A = 1
    TARGET_OPP_B = 2
    TARGET_ALLY = 3
    TARGET_SPREAD = 4
    
    def __init__(self):
        """Initialize action encoder."""
        pass
    
    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a name for comparison."""
        return name.lower().replace(" ", "").replace("-", "").replace("'", "")
    
    def _find_move_slot(
        self, 
        move_name: str, 
        pokemon: 'PokemonState'
    ) -> int:
        """Find which slot (0-3) a move occupies in the Pokemon's moveset.
        
        Uses three-tier matching strategy:
        1. Move ID lookup (canonical matching via poke-env)
        2. Normalized string comparison
        3. Fuzzy string matching (Levenshtein similarity > 80%)
        
        Args:
            move_name: Name of the move used
            pokemon: PokemonState of the active Pokemon
            
        Returns:
            Move slot index (0-3), defaults to 0 if not found
        """
        if not pokemon or not pokemon.moves:
            return 0
        
        move_normalized = self._normalize_name(move_name)
        move_id = get_move_id(move_name)
        
        # Tier 1: Match by canonical move ID
        if move_id > 0:
            for i, known_move in enumerate(pokemon.moves[:4]):
                if get_move_id(known_move) == move_id:
                    return i
        
        # Tier 2: Match by normalized string
        for i, known_move in enumerate(pokemon.moves[:4]):
            if self._normalize_name(known_move) == move_normalized:
                return i
        
        # Tier 3: Fuzzy string matching (if available)
        if HAS_RAPIDFUZZ:
            best_match = (-1, 0.0)  # (index, similarity)
            for i, known_move in enumerate(pokemon.moves[:4]):
                known_normalized = self._normalize_name(known_move)
                # Use token_ratio for better handling of word order variations
                similarity = fuzz.ratio(move_normalized, known_normalized)
                if similarity > best_match[1]:
                    best_match = (i, similarity)
            
            # Accept match if similarity > 80%
            if best_match[1] >= 80:
                return best_match[0]
        
        # Move not found in known moveset - might be newly revealed
        # Add it to the moveset if there's room
        if len(pokemon.moves) < 4:
            pokemon.moves.append(move_name)
            return len(pokemon.moves) - 1
        
        # Default to slot 0 if move not found
        return 0
    
    def _find_switch_target(
        self, 
        target_species: str, 
        bench: List['PokemonState']
    ) -> int:
        """Find which bench slot (0-3) the switch target is at.
        
        Uses species ID and fuzzy matching for robustness.
        
        Args:
            target_species: Species name of the Pokemon being switched to
            bench: List of bench Pokemon
            
        Returns:
            Bench slot index (0-3), defaults to 0 if not found
        """
        if not bench:
            return 0
        
        target_normalized = self._normalize_name(target_species)
        
        # Tier 1: Exact normalized match (species or nickname)
        for i, pokemon in enumerate(bench[:4]):
            if self._normalize_name(pokemon.species) == target_normalized:
                return i
            # Also check nickname
            if pokemon.nickname and self._normalize_name(pokemon.nickname) == target_normalized:
                return i
        
        # Tier 2: Try species ID match (from pokemon_ids)
        from src.data.pokemon_ids import get_species_id
        target_id = get_species_id(target_species)
        if target_id > 0:
            for i, pokemon in enumerate(bench[:4]):
                if get_species_id(pokemon.species) == target_id:
                    return i
        
        # Tier 3: Fuzzy string matching
        if HAS_RAPIDFUZZ:
            best_match = (-1, 0.0)
            for i, pokemon in enumerate(bench[:4]):
                species_normalized = self._normalize_name(pokemon.species)
                similarity = fuzz.ratio(target_normalized, species_normalized)
                if similarity > best_match[1]:
                    best_match = (i, similarity)
                # Also check nickname
                if pokemon.nickname:
                    nick_normalized = self._normalize_name(pokemon.nickname)
                    nick_similarity = fuzz.ratio(target_normalized, nick_normalized)
                    if nick_similarity > best_match[1]:
                        best_match = (i, nick_similarity)
            
            if best_match[1] >= 80:
                return best_match[0]
        
        # Default to slot 0 if not found
        return 0
    
    def _get_active_pokemon(
        self, 
        slot: str, 
        state: 'TurnState', 
        player: str
    ) -> Optional['PokemonState']:
        """Get the active Pokemon in a specific slot.
        
        Args:
            slot: 'a' or 'b'
            state: Current turn state
            player: 'p1' or 'p2'
            
        Returns:
            PokemonState or None
        """
        active, _ = state.get_player_pokemon(player)
        
        for pokemon in active:
            if pokemon.slot == slot:
                return pokemon
        
        # Fall back to position-based lookup
        if slot == 'a' and len(active) > 0:
            return active[0]
        if slot == 'b' and len(active) > 1:
            return active[1]
        
        return None
    
    def _encode_target(
        self,
        event: BattleEvent,
        player: str,
    ) -> int:
        """Encode the target of a move.
        
        Args:
            event: The move event
            player: The player making the move ('p1' or 'p2')
            
        Returns:
            Target index (0-4)
        """
        # Check if spread move
        if event.details.get('is_spread', False):
            return self.TARGET_SPREAD
        
        target_player = event.details.get('target_player', '')
        target_slot = event.details.get('target_slot', '')
        
        if not target_player or not target_slot:
            return self.TARGET_NONE
        
        # Determine opponent
        opponent = 'p2' if player == 'p1' else 'p1'
        
        if target_player == opponent:
            # Targeting opponent
            if target_slot == 'a':
                return self.TARGET_OPP_A
            elif target_slot == 'b':
                return self.TARGET_OPP_B
        elif target_player == player:
            # Targeting ally
            return self.TARGET_ALLY
        
        return self.TARGET_NONE
    
    def encode_turn_actions(
        self, 
        events: List[BattleEvent], 
        player: str,
        state: 'TurnState'
    ) -> int:
        """Encode a player's actions in a turn to action index.
        
        Args:
            events: List of events in the turn
            player: 'p1' or 'p2'
            state: Current turn state
            
        Returns:
            Combined action index (0-143)
        """
        slot_a_action = 0  # Default: first move
        slot_b_action = 0
        
        # Track if we've seen tera this turn for each slot
        tera_slot = None
        
        for event in events:
            if event.event_type == 'terastallize' and event.player == player:
                tera_slot = event.slot  # Which slot terastallized
        
        # Get active and bench Pokemon
        active, bench = state.get_player_pokemon(player)
        
        # Find move/switch events for this player
        for event in events:
            if event.player != player:
                continue
            
            slot = event.slot
            
            if event.event_type == 'move':
                move_name = event.details.get('move', '')
                
                # Get the Pokemon in this slot
                pokemon = self._get_active_pokemon(slot, state, player)
                
                # Find the move slot (0-3)
                move_slot = self._find_move_slot(move_name, pokemon)
                
                # Check if this slot terastallized
                if tera_slot == slot:
                    action = 4 + move_slot  # Tera moves: 4-7
                else:
                    action = move_slot  # Regular moves: 0-3
                
                if slot == 'a':
                    slot_a_action = action
                elif slot == 'b':
                    slot_b_action = action
            
            elif event.event_type == 'switch':
                # Get the species being switched to
                target_species = event.pokemon or event.details.get('species', '')
                
                # Find which bench slot this is
                switch_slot = self._find_switch_target(target_species, bench)
                
                # Switch actions are 8-11
                action = 8 + switch_slot
                
                if slot == 'a':
                    slot_a_action = action
                elif slot == 'b':
                    slot_b_action = action
        
        # Combine into single action
        combined = slot_a_action * self.ACTIONS_PER_SLOT + slot_b_action
        return min(combined, self.ACTION_SPACE_SIZE - 1)
    
    def encode_turn_actions_with_targets(
        self, 
        events: List[BattleEvent], 
        player: str,
        state: 'TurnState'
    ) -> Tuple[int, Tuple[int, int]]:
        """Encode actions and targets for hierarchical action models.
        
        Args:
            events: List of events in the turn
            player: 'p1' or 'p2'
            state: Current turn state
            
        Returns:
            Tuple of (combined_action, (slot_a_target, slot_b_target))
        """
        slot_a_target = self.TARGET_NONE
        slot_b_target = self.TARGET_NONE
        
        # Find move events to extract targets
        for event in events:
            if event.player != player:
                continue
            
            if event.event_type == 'move':
                target = self._encode_target(event, player)
                if event.slot == 'a':
                    slot_a_target = target
                elif event.slot == 'b':
                    slot_b_target = target
        
        # Get the combined action using existing method
        combined_action = self.encode_turn_actions(events, player, state)
        
        return combined_action, (slot_a_target, slot_b_target)
    
    def decode_action(self, action: int) -> Tuple[int, int]:
        """Decode combined action to slot actions.
        
        Returns:
            Tuple of (slot_a_action, slot_b_action)
        """
        slot_a = action // self.ACTIONS_PER_SLOT
        slot_b = action % self.ACTIONS_PER_SLOT
        return slot_a, slot_b
    
    def describe_action(self, action: int) -> str:
        """Get human-readable description of an action."""
        slot_a, slot_b = self.decode_action(action)
        
        def describe_slot_action(a: int) -> str:
            if a < 4:
                return f"Move {a+1}"
            elif a < 8:
                return f"Tera+Move {a-3}"
            else:
                return f"Switch {a-7}"
        
        return f"Slot A: {describe_slot_action(slot_a)}, Slot B: {describe_slot_action(slot_b)}"


class BattleHistoryTracker:
    """Track battle history for temporal context features.
    
    Accumulates turn-by-turn data for:
    - Action history (what moves/switches were made)
    - Damage history (damage dealt and received)
    - Switch history (when switches occurred)
    - KO events
    
    This data is used to populate temporal context features in
    the enhanced state representation.
    """
    
    def __init__(self, history_len: int = 10):
        """Initialize tracker.
        
        Args:
            history_len: Number of turns to keep in history
        """
        self.history_len = history_len
        self.reset()
    
    def reset(self):
        """Clear all history for a new battle."""
        self.action_history: List[int] = []
        self.damage_dealt: List[float] = []
        self.damage_received: List[float] = []
        self.switch_events: List[bool] = []
        self.ko_dealt: List[bool] = []
        self.ko_received: List[bool] = []
        self.outcome_history: List[int] = []  # 1=good, 0=neutral, -1=bad
    
    def record_turn(
        self,
        action: int,
        damage_dealt: float = 0.0,
        damage_received: float = 0.0,
        did_switch: bool = False,
        got_ko: bool = False,
        lost_pokemon: bool = False,
    ):
        """Record data from a single turn.
        
        Args:
            action: Combined action index (0-143)
            damage_dealt: Total damage dealt this turn (as HP fraction)
            damage_received: Total damage received this turn
            did_switch: Whether a switch occurred
            got_ko: Whether we KO'd an opponent
            lost_pokemon: Whether we lost a Pokemon
        """
        self.action_history.append(action)
        self.damage_dealt.append(damage_dealt)
        self.damage_received.append(damage_received)
        self.switch_events.append(did_switch)
        self.ko_dealt.append(got_ko)
        self.ko_received.append(lost_pokemon)
        
        # Compute turn outcome
        if got_ko and not lost_pokemon:
            outcome = 1  # Good turn
        elif lost_pokemon and not got_ko:
            outcome = -1  # Bad turn
        elif damage_dealt > damage_received:
            outcome = 1
        elif damage_received > damage_dealt:
            outcome = -1
        else:
            outcome = 0  # Neutral
        self.outcome_history.append(outcome)
        
        # Trim to history length
        self._trim_history()
    
    def _trim_history(self):
        """Keep only the most recent history_len entries."""
        if len(self.action_history) > self.history_len:
            self.action_history = self.action_history[-self.history_len:]
            self.damage_dealt = self.damage_dealt[-self.history_len:]
            self.damage_received = self.damage_received[-self.history_len:]
            self.switch_events = self.switch_events[-self.history_len:]
            self.ko_dealt = self.ko_dealt[-self.history_len:]
            self.ko_received = self.ko_received[-self.history_len:]
            self.outcome_history = self.outcome_history[-self.history_len:]
    
    def _pad(self, lst: List, pad_value: Any = 0) -> List:
        """Pad a list to history_len."""
        result = list(lst)
        while len(result) < self.history_len:
            result.insert(0, pad_value)  # Pad at beginning
        return result
    
    def get_context(self) -> Dict[str, Any]:
        """Get padded history for structured data.
        
        Returns:
            Dictionary with padded history arrays
        """
        current_len = len(self.action_history)
        
        # Create mask (1.0 for real data, 0.0 for padding)
        mask = [0.0] * (self.history_len - current_len) + [1.0] * current_len
        
        # Build damage tensor [history_len, 2] (dealt, received)
        damage_history = []
        for i in range(self.history_len):
            if i < self.history_len - current_len:
                damage_history.append([0.0, 0.0])
            else:
                idx = i - (self.history_len - current_len)
                damage_history.append([
                    self.damage_dealt[idx],
                    self.damage_received[idx]
                ])
        
        # Build switch history [history_len]
        switch_history = self._pad(
            [1.0 if s else 0.0 for s in self.switch_events], 
            0.0
        )
        
        return {
            'action_history': self._pad(self.action_history, 0),
            'outcome_history': self._pad(self.outcome_history, 0),
            'damage_history': damage_history,
            'switch_history': switch_history,
            'history_mask': mask,
            'current_turn': current_len,
        }
    
    def get_summary_stats(self) -> Dict[str, float]:
        """Get summary statistics of the battle so far.
        
        Returns:
            Dictionary with aggregated stats
        """
        if not self.action_history:
            return {
                'total_damage_dealt': 0.0,
                'total_damage_received': 0.0,
                'ko_count': 0,
                'deaths': 0,
                'switch_count': 0,
                'win_rate': 0.5,
            }
        
        return {
            'total_damage_dealt': sum(self.damage_dealt),
            'total_damage_received': sum(self.damage_received),
            'ko_count': sum(self.ko_dealt),
            'deaths': sum(self.ko_received),
            'switch_count': sum(self.switch_events),
            'win_rate': sum(1 for o in self.outcome_history if o > 0) / len(self.outcome_history),
        }


class RewardShaper:
    """Calculate shaped rewards for training."""
    
    def __init__(
        self,
        win_reward: float = 10.0,
        lose_reward: float = -10.0,
        ko_reward: float = 2.0,
        faint_penalty: float = -2.0,
        hp_diff_scale: float = 0.1,
        turn_penalty: float = -0.01,
    ):
        """Initialize reward shaper.
        
        Args:
            win_reward: Reward for winning
            lose_reward: Penalty for losing
            ko_reward: Reward per opponent KO
            faint_penalty: Penalty per own Pokemon fainted
            hp_diff_scale: Scale for HP differential reward
            turn_penalty: Small penalty per turn (encourages decisive play)
        """
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.ko_reward = ko_reward
        self.faint_penalty = faint_penalty
        self.hp_diff_scale = hp_diff_scale
        self.turn_penalty = turn_penalty
    
    def compute_reward(
        self,
        prev_state: TurnState,
        curr_state: TurnState,
        player: str,
        is_terminal: bool,
        won: bool,
    ) -> float:
        """Compute shaped reward for a transition.
        
        Args:
            prev_state: State before action
            curr_state: State after action
            player: 'p1' or 'p2'
            is_terminal: Whether this is the final transition
            won: Whether the player won (only valid if is_terminal)
            
        Returns:
            Shaped reward value
        """
        reward = 0.0
        
        # Terminal reward
        if is_terminal:
            reward += self.win_reward if won else self.lose_reward
            return reward
        
        # KO rewards
        prev_my_alive = prev_state.count_alive(player)
        curr_my_alive = curr_state.count_alive(player)
        opp = 'p2' if player == 'p1' else 'p1'
        prev_opp_alive = prev_state.count_alive(opp)
        curr_opp_alive = curr_state.count_alive(opp)
        
        # Opponent KOs
        opp_kos = prev_opp_alive - curr_opp_alive
        reward += opp_kos * self.ko_reward
        
        # Own faints
        my_faints = prev_my_alive - curr_my_alive
        reward += my_faints * self.faint_penalty
        
        # HP differential
        my_hp = curr_state.total_hp_fraction(player)
        opp_hp = curr_state.total_hp_fraction(opp)
        hp_diff = my_hp - opp_hp
        reward += hp_diff * self.hp_diff_scale
        
        # Turn penalty
        reward += self.turn_penalty
        
        return reward


class TrajectoryBuilder:
    """Build training trajectories from battle data."""
    
    def __init__(self, history_len: int = 10):
        """Initialize trajectory builder.
        
        Args:
            history_len: Number of turns to keep in temporal context
        """
        self.parser = ReplayParser()
        self.reconstructor = StateReconstructor()
        self.state_encoder = StateEncoder()
        self.action_encoder = ActionEncoder()
        self.reward_shaper = RewardShaper()
        self.history_len = history_len
    
    def build_from_battle(self, battle: ParsedBattle) -> List[Trajectory]:
        """Build trajectories from a parsed battle.
        
        Args:
            battle: Parsed battle object
            
        Returns:
            List of trajectories (one per player)
        """
        # Reconstruct states
        states = self.reconstructor.reconstruct(battle)
        
        if len(states) < 2:
            return []
        
        trajectories = []
        
        # Build trajectory for each player
        for player in ['p1', 'p2']:
            won = (
                (player == 'p1' and battle.winner == battle.player1) or
                (player == 'p2' and battle.winner == battle.player2)
            )
            
            trajectory = Trajectory(
                battle_id=battle.battle_id,
                player_perspective=player,
                won=won,
            )
            
            # Initialize history tracker for temporal context
            history_tracker = BattleHistoryTracker(history_len=self.history_len)
            
            # Build transitions
            sorted_turns = sorted(states.keys())
            
            for i in range(len(sorted_turns) - 1):
                turn = sorted_turns[i]
                next_turn = sorted_turns[i + 1]
                
                # Get states
                state = states[turn]
                next_state = states[next_turn]
                
                # Encode states (now returns tuple with structured data)
                state_vec, structured = self.state_encoder.encode(state, player)
                next_state_vec, _ = self.state_encoder.encode(next_state, player)
                
                # Get actions and targets for this turn
                turn_events = battle.turn_events.get(turn, [])
                action, (slot_a_target, slot_b_target) = self.action_encoder.encode_turn_actions_with_targets(
                    turn_events, player, state
                )
                
                # Check if terminal
                is_terminal = (i == len(sorted_turns) - 2)
                
                # Compute reward
                reward = self.reward_shaper.compute_reward(
                    state, next_state, player, is_terminal, won
                )
                
                # Calculate damage and events for history tracker
                damage_dealt, damage_received, got_ko, lost_pokemon, did_switch = \
                    self._extract_turn_stats(state, next_state, player, turn_events)
                
                # Add temporal context to structured data
                if structured is not None:
                    temporal_context = history_tracker.get_context()
                    structured['action_history'] = temporal_context['action_history']
                    structured['outcome_history'] = temporal_context['outcome_history']
                    structured['damage_history'] = temporal_context['damage_history']
                    structured['switch_history'] = temporal_context['switch_history']
                    structured['history_mask'] = temporal_context['history_mask']
                    structured['current_turn'] = temporal_context['current_turn']
                
                # Record this turn in history for next iteration
                history_tracker.record_turn(
                    action=action,
                    damage_dealt=damage_dealt,
                    damage_received=damage_received,
                    did_switch=did_switch,
                    got_ko=got_ko,
                    lost_pokemon=lost_pokemon,
                )
                
                transition = Transition(
                    state=state_vec,
                    action=action,
                    reward=reward,
                    next_state=next_state_vec,
                    done=is_terminal,
                    slot_a_target=slot_a_target,
                    slot_b_target=slot_b_target,
                    structured_data=structured,
                )
                
                trajectory.transitions.append(transition)
            
            if trajectory.transitions:
                trajectories.append(trajectory)
        
        return trajectories
    
    def _extract_turn_stats(
        self,
        state: TurnState,
        next_state: TurnState,
        player: str,
        events: List[BattleEvent],
    ) -> Tuple[float, float, bool, bool, bool]:
        """Extract statistics from a turn for history tracking.
        
        Returns:
            Tuple of (damage_dealt, damage_received, got_ko, lost_pokemon, did_switch)
        """
        opp = 'p2' if player == 'p1' else 'p1'
        
        # Calculate HP changes
        my_active_prev, _ = state.get_player_pokemon(player)
        my_active_next, _ = next_state.get_player_pokemon(player)
        opp_active_prev, _ = state.get_player_pokemon(opp)
        opp_active_next, _ = next_state.get_player_pokemon(opp)
        
        # Sum HP for damage calculation
        my_hp_prev = sum(p.hp_fraction for p in my_active_prev if p)
        my_hp_next = sum(p.hp_fraction for p in my_active_next if p)
        opp_hp_prev = sum(p.hp_fraction for p in opp_active_prev if p)
        opp_hp_next = sum(p.hp_fraction for p in opp_active_next if p)
        
        damage_dealt = max(0, opp_hp_prev - opp_hp_next)
        damage_received = max(0, my_hp_prev - my_hp_next)
        
        # Check for KOs
        my_alive_prev = state.count_alive(player)
        my_alive_next = next_state.count_alive(player)
        opp_alive_prev = state.count_alive(opp)
        opp_alive_next = next_state.count_alive(opp)
        
        got_ko = opp_alive_next < opp_alive_prev
        lost_pokemon = my_alive_next < my_alive_prev
        
        # Check for switches
        did_switch = any(
            e.event_type == 'switch' and e.player == player
            for e in events
        )
        
        return damage_dealt, damage_received, got_ko, lost_pokemon, did_switch
    
    def build_from_log(self, battle_id: str, log: str, timestamp: Optional[int] = None) -> List[Trajectory]:
        """Build trajectories from a raw battle log.
        
        Args:
            battle_id: Battle identifier
            log: Raw battle log string
            timestamp: Optional timestamp
            
        Returns:
            List of trajectories
        """
        battle = self.parser.parse(battle_id, log, timestamp)
        return self.build_from_battle(battle)


class StreamingBattleProcessor:
    """Process large battle files in memory-efficient batches.
    
    This class handles processing of VGC-Bench data by:
    1. Loading battles in batches to limit memory usage
    2. Processing each batch into trajectories
    3. Saving batches incrementally to Parquet files
    4. Freeing memory between batches
    """
    
    def __init__(self, batch_size: int = 2000):
        """Initialize the streaming processor.
        
        Args:
            batch_size: Number of battles to process in each batch
        """
        self.batch_size = batch_size
        self.builder = TrajectoryBuilder()
        self.stats = {
            'total_battles': 0,
            'successful_battles': 0,
            'failed_battles': 0,
            'total_trajectories': 0,
            'total_transitions': 0,
        }
    
    def process_file_streaming(
        self, 
        input_path: Path, 
        output_dir: Path,
        max_battles: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Process a large JSON file in memory-efficient batches.
        
        Args:
            input_path: Path to input VGC-Bench JSON file
            output_dir: Directory to save Parquet batch files
            max_battles: Maximum battles to process (None for all)
            
        Returns:
            Dictionary with processing statistics
        """
        import gc
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Loading battle IDs from {input_path}")
        
        # Load the JSON file (we still load it, but process in chunks)
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        battle_ids = list(data.keys())
        if max_battles:
            battle_ids = battle_ids[:max_battles]
        
        self.stats['total_battles'] = len(battle_ids)
        num_batches = (len(battle_ids) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Processing {len(battle_ids)} battles in {num_batches} batches of {self.batch_size}")
        
        batch_files = []
        
        for batch_idx in range(0, len(battle_ids), self.batch_size):
            batch_num = batch_idx // self.batch_size
            batch_ids = battle_ids[batch_idx:batch_idx + self.batch_size]
            
            logger.info(f"Processing batch {batch_num + 1}/{num_batches} ({len(batch_ids)} battles)")
            
            # Process this batch
            trajectories = self._process_batch(data, batch_ids)
            
            if trajectories:
                # Save to Parquet
                batch_file = self._save_batch_parquet(trajectories, output_dir, batch_num)
                batch_files.append(batch_file)
                
                self.stats['total_trajectories'] += len(trajectories)
                self.stats['total_transitions'] += sum(t.length for t in trajectories)
            
            # Free memory
            del trajectories
            gc.collect()
        
        # Save metadata
        self._save_metadata(output_dir, batch_files)
        
        logger.info(f"Completed processing: {self.stats}")
        return self.stats
    
    def _process_batch(
        self, 
        data: Dict[str, Any], 
        battle_ids: List[str]
    ) -> List[Trajectory]:
        """Process a batch of battles into trajectories.
        
        Args:
            data: Full data dictionary
            battle_ids: List of battle IDs to process
            
        Returns:
            List of trajectories from this batch
        """
        trajectories = []
        
        for battle_id in battle_ids:
            try:
                timestamp, log = data[battle_id]
                batch_trajectories = self.builder.build_from_log(battle_id, log, timestamp)
                trajectories.extend(batch_trajectories)
                self.stats['successful_battles'] += 1
            except Exception as e:
                logger.debug(f"Error processing {battle_id}: {e}")
                self.stats['failed_battles'] += 1
                continue
        
        return trajectories
    
    def _save_batch_parquet(
        self, 
        trajectories: List[Trajectory], 
        output_dir: Path, 
        batch_num: int
    ) -> Path:
        """Save a batch of trajectories to Parquet format.
        
        Args:
            trajectories: List of trajectories to save
            output_dir: Output directory
            batch_num: Batch number for filename
            
        Returns:
            Path to saved Parquet file
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        # Flatten trajectories into rows for efficient storage
        rows = []
        for traj in trajectories:
            for i, t in enumerate(traj.transitions):
                row = {
                    'battle_id': traj.battle_id,
                    'player': traj.player_perspective,
                    'won': traj.won,
                    'transition_idx': i,
                    'state': t.state.tobytes(),  # Store as bytes for efficiency
                    'action': t.action,
                    'reward': t.reward,
                    'next_state': t.next_state.tobytes(),
                    'done': t.done,
                    'slot_a_target': t.slot_a_target,
                    'slot_b_target': t.slot_b_target,
                }
                rows.append(row)
        
        # Create PyArrow table
        table = pa.Table.from_pylist(rows)
        
        # Save to Parquet with compression
        output_file = output_dir / f"batch_{batch_num:04d}.parquet"
        pq.write_table(table, output_file, compression='snappy')
        
        logger.info(f"Saved batch {batch_num} to {output_file} ({len(rows)} transitions)")
        return output_file
    
    def _save_metadata(self, output_dir: Path, batch_files: List[Path]) -> None:
        """Save processing metadata.
        
        Args:
            output_dir: Output directory
            batch_files: List of batch file paths
        """
        metadata = {
            'stats': self.stats,
            'batch_files': [str(f.name) for f in batch_files],
            'num_batches': len(batch_files),
            'state_dim': 620,
            'action_dim': 144,
        }
        
        metadata_file = output_dir / 'metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_file}")


def process_vgc_bench_file(
    input_path: Path,
    output_dir: Path,
    max_battles: Optional[int] = None,
) -> int:
    """Process a VGC-Bench JSON file and save trajectories.
    
    Args:
        input_path: Path to input JSON file
        output_dir: Directory to save trajectories
        max_battles: Maximum battles to process (None for all)
        
    Returns:
        Number of trajectories created
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Loading battles from {input_path}")
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    builder = TrajectoryBuilder()
    all_trajectories = []
    
    battle_ids = list(data.keys())
    if max_battles:
        battle_ids = battle_ids[:max_battles]
    
    logger.info(f"Processing {len(battle_ids)} battles...")
    
    for i, battle_id in enumerate(battle_ids):
        if i % 1000 == 0 and i > 0:
            logger.info(f"Processed {i}/{len(battle_ids)} battles, {len(all_trajectories)} trajectories")
        
        try:
            timestamp, log = data[battle_id]
            trajectories = builder.build_from_log(battle_id, log, timestamp)
            all_trajectories.extend(trajectories)
        except Exception as e:
            logger.warning(f"Error processing {battle_id}: {e}")
            continue
    
    logger.info(f"Built {len(all_trajectories)} trajectories from {len(battle_ids)} battles")
    
    # Save trajectories
    # Convert to serializable format
    output_data = []
    for traj in all_trajectories:
        transitions_data = []
        for t in traj.transitions:
            trans_dict = {
                'state': t.state.tolist(),
                'action': t.action,
                'reward': t.reward,
                'next_state': t.next_state.tolist(),
                'done': t.done,
                'slot_a_target': t.slot_a_target,
                'slot_b_target': t.slot_b_target,
            }
            # Include structured data if available
            if t.structured_data is not None:
                trans_dict['structured'] = t.structured_data
            transitions_data.append(trans_dict)
        
        traj_dict = {
            'battle_id': traj.battle_id,
            'player': traj.player_perspective,
            'won': traj.won,
            'length': traj.length,
            'total_reward': traj.total_reward,
            'transitions': transitions_data,
        }
        output_data.append(traj_dict)
    
    # Save as JSON (can convert to parquet later for efficiency)
    output_file = output_dir / f"trajectories_{input_path.stem}.json"
    with open(output_file, 'w') as f:
        json.dump(output_data, f)
    
    logger.info(f"Saved trajectories to {output_file}")
    
    return len(all_trajectories)


def process_vgc_bench_streaming(
    input_path: Path,
    output_dir: Path,
    batch_size: int = 2000,
    max_battles: Optional[int] = None,
) -> Dict[str, Any]:
    """Process VGC-Bench file using streaming batch processing.
    
    This is the preferred method for large files (10K+ battles).
    
    Args:
        input_path: Path to input JSON file
        output_dir: Directory to save Parquet batches
        batch_size: Battles per batch (default 2000)
        max_battles: Maximum battles to process (None for all)
        
    Returns:
        Dictionary with processing statistics
    """
    processor = StreamingBattleProcessor(batch_size=batch_size)
    return processor.process_file_streaming(input_path, output_dir, max_battles)


def main():
    """Main entry point for trajectory building."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build training trajectories from VGC-Bench data")
    parser.add_argument(
        "--input", 
        type=str, 
        default="data/raw/vgc_bench/logs-gen9vgc2024regg.json",
        help="Input JSON file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/trajectories",
        help="Output directory"
    )
    parser.add_argument(
        "--max-battles",
        type=int,
        default=None,
        help="Maximum battles to process"
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Use streaming batch processing (recommended for large files)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2000,
        help="Batch size for streaming mode (default: 2000)"
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return
    
    if args.streaming:
        logger.info("Using streaming batch processing mode")
        stats = process_vgc_bench_streaming(
            input_path, 
            output_dir, 
            batch_size=args.batch_size,
            max_battles=args.max_battles
        )
        logger.info(f"Done! Stats: {stats}")
    else:
        num_trajectories = process_vgc_bench_file(input_path, output_dir, args.max_battles)
        logger.info(f"Done! Created {num_trajectories} trajectories")


if __name__ == "__main__":
    main()

