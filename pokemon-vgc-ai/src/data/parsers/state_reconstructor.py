"""Reconstruct game state from parsed battle events.

This module takes parsed battle events and reconstructs the game state
at each turn, which is necessary for building training trajectories.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger

from .replay_parser import ParsedBattle, BattleEvent, PokemonSet


@dataclass
class PokemonState:
    """State of a single Pokemon during battle."""
    species: str
    nickname: str = ""
    level: int = 50
    current_hp: int = 100
    max_hp: int = 100
    is_active: bool = False
    is_fainted: bool = False
    slot: Optional[str] = None  # 'a' or 'b' for doubles
    
    # Status
    status: Optional[str] = None  # brn, par, slp, frz, psn, tox
    
    # Stats
    stat_boosts: Dict[str, int] = field(default_factory=lambda: {
        'atk': 0, 'def': 0, 'spa': 0, 'spd': 0, 'spe': 0,
        'accuracy': 0, 'evasion': 0
    })
    
    # Terastallization
    is_terastallized: bool = False
    tera_type: Optional[str] = None
    
    # From team data
    ability: Optional[str] = None
    item: Optional[str] = None
    moves: List[str] = field(default_factory=list)
    original_tera_type: Optional[str] = None
    types: List[str] = field(default_factory=list)
    
    @property
    def hp_fraction(self) -> float:
        """Get HP as fraction 0-1."""
        if self.max_hp <= 0:
            return 0.0
        return self.current_hp / self.max_hp


@dataclass
class FieldState:
    """State of field conditions."""
    weather: Optional[str] = None
    weather_turns: int = 0
    terrain: Optional[str] = None
    terrain_turns: int = 0
    trick_room: bool = False
    trick_room_turns: int = 0


@dataclass
class SideState:
    """State of a side's conditions."""
    tailwind: bool = False
    tailwind_turns: int = 0
    reflect: bool = False
    reflect_turns: int = 0
    light_screen: bool = False
    light_screen_turns: int = 0
    aurora_veil: bool = False
    aurora_veil_turns: int = 0
    # Hazards
    stealth_rock: bool = False
    spikes: int = 0
    toxic_spikes: int = 0


@dataclass
class TurnState:
    """Complete game state at a specific turn."""
    turn: int
    
    # Player 1 state
    p1_active: List[PokemonState] = field(default_factory=list)
    p1_bench: List[PokemonState] = field(default_factory=list)
    p1_side: SideState = field(default_factory=SideState)
    p1_can_tera: bool = True
    
    # Player 2 state
    p2_active: List[PokemonState] = field(default_factory=list)
    p2_bench: List[PokemonState] = field(default_factory=list)
    p2_side: SideState = field(default_factory=SideState)
    p2_can_tera: bool = True
    
    # Field state
    field: FieldState = field(default_factory=FieldState)
    
    def get_player_pokemon(self, player: str) -> Tuple[List[PokemonState], List[PokemonState]]:
        """Get active and bench Pokemon for a player."""
        if player == 'p1':
            return self.p1_active, self.p1_bench
        else:
            return self.p2_active, self.p2_bench
    
    def get_all_pokemon(self, player: str) -> List[PokemonState]:
        """Get all Pokemon for a player."""
        active, bench = self.get_player_pokemon(player)
        return active + bench
    
    def count_alive(self, player: str) -> int:
        """Count alive Pokemon for a player."""
        all_pokemon = self.get_all_pokemon(player)
        return sum(1 for p in all_pokemon if not p.is_fainted)
    
    def total_hp_fraction(self, player: str) -> float:
        """Get total HP fraction for a player."""
        all_pokemon = self.get_all_pokemon(player)
        if not all_pokemon:
            return 0.0
        return sum(p.hp_fraction for p in all_pokemon) / len(all_pokemon)


class StateReconstructor:
    """Reconstruct game state from battle events."""
    
    # Species to approximate base stats (simplified)
    # In production, would load from a database
    DEFAULT_BASE_STATS = {
        'hp': 80, 'atk': 80, 'def': 80, 'spa': 80, 'spd': 80, 'spe': 80
    }
    
    def __init__(self):
        """Initialize the reconstructor."""
        self.current_state: Optional[TurnState] = None
        self.states_by_turn: Dict[int, TurnState] = {}
        self.pokemon_registry: Dict[str, PokemonState] = {}  # nickname -> state
    
    def reconstruct(self, battle: ParsedBattle) -> Dict[int, TurnState]:
        """Reconstruct game states for all turns.
        
        Args:
            battle: Parsed battle object
            
        Returns:
            Dictionary mapping turn number to TurnState
        """
        self.states_by_turn = {}
        self.pokemon_registry = {}
        
        # Initialize teams from team data
        self._initialize_teams(battle)
        
        # Create initial state (turn 0)
        self.current_state = TurnState(turn=0)
        self._copy_teams_to_state()
        self.states_by_turn[0] = self._copy_state(self.current_state)
        
        # Process events
        current_turn = 0
        for event in battle.events:
            # Check for turn change
            if event.event_type == 'turn':
                current_turn = event.details.get('turn_number', current_turn + 1)
                self.current_state.turn = current_turn
            
            # Apply event to state
            self._apply_event(event)
            
            # Save state at each turn
            if event.event_type == 'turn':
                self.states_by_turn[current_turn] = self._copy_state(self.current_state)
        
        return self.states_by_turn
    
    def _initialize_teams(self, battle: ParsedBattle):
        """Initialize Pokemon from team data."""
        for player in ['p1', 'p2']:
            team_data = battle.teams.get(player, [])
            for i, poke_set in enumerate(team_data):
                key = f"{player}:{poke_set.species}"
                pokemon = PokemonState(
                    species=poke_set.species,
                    nickname=poke_set.species,
                    level=poke_set.level,
                    current_hp=100,
                    max_hp=100,
                    ability=poke_set.ability,
                    item=poke_set.item,
                    moves=poke_set.moves.copy() if poke_set.moves else [],
                    original_tera_type=poke_set.tera_type,
                )
                self.pokemon_registry[key] = pokemon
    
    def _copy_teams_to_state(self):
        """Copy team Pokemon to current state benches."""
        self.current_state.p1_bench = []
        self.current_state.p2_bench = []
        
        for key, pokemon in self.pokemon_registry.items():
            player = key.split(':')[0]
            pokemon_copy = self._copy_pokemon(pokemon)
            if player == 'p1':
                self.current_state.p1_bench.append(pokemon_copy)
            else:
                self.current_state.p2_bench.append(pokemon_copy)
    
    def _apply_event(self, event: BattleEvent):
        """Apply a single event to the current state."""
        if event.event_type == 'switch':
            self._apply_switch(event)
        elif event.event_type == 'damage':
            self._apply_damage(event)
        elif event.event_type == 'heal':
            self._apply_heal(event)
        elif event.event_type == 'faint':
            self._apply_faint(event)
        elif event.event_type == 'terastallize':
            self._apply_tera(event)
        elif event.event_type == 'stat_change':
            self._apply_stat_change(event)
        elif event.event_type == 'status':
            self._apply_status(event)
        elif event.event_type == 'weather':
            self._apply_weather(event)
        elif event.event_type == 'field':
            self._apply_field(event)
        elif event.event_type == 'side_condition':
            self._apply_side_condition(event)
    
    def _apply_switch(self, event: BattleEvent):
        """Apply switch event."""
        player = event.player
        slot = event.slot
        pokemon_name = event.pokemon
        
        if not player or not slot:
            return
        
        # Get active and bench lists
        if player == 'p1':
            active = self.current_state.p1_active
            bench = self.current_state.p1_bench
        else:
            active = self.current_state.p2_active
            bench = self.current_state.p2_bench
        
        # Find Pokemon in bench or create new
        incoming = None
        for i, pokemon in enumerate(bench):
            if pokemon.species.lower() in pokemon_name.lower() or pokemon_name.lower() in pokemon.species.lower():
                incoming = bench.pop(i)
                break
        
        if incoming is None:
            # Create new Pokemon state
            incoming = PokemonState(
                species=pokemon_name,
                nickname=pokemon_name,
                current_hp=event.details.get('current_hp', 100),
                max_hp=event.details.get('max_hp', 100),
            )
        
        incoming.is_active = True
        incoming.slot = slot
        incoming.current_hp = event.details.get('current_hp', incoming.current_hp)
        incoming.max_hp = event.details.get('max_hp', incoming.max_hp)
        
        # Find slot index (a=0, b=1)
        slot_idx = 0 if slot == 'a' else 1
        
        # Move current occupant to bench if exists
        while len(active) <= slot_idx:
            active.append(None)
        
        outgoing = active[slot_idx]
        if outgoing and not outgoing.is_fainted:
            outgoing.is_active = False
            outgoing.slot = None
            # Reset stat boosts on switch out
            outgoing.stat_boosts = {k: 0 for k in outgoing.stat_boosts}
            bench.append(outgoing)
        
        active[slot_idx] = incoming
    
    def _apply_damage(self, event: BattleEvent):
        """Apply damage event."""
        pokemon = self._find_pokemon(event.player, event.pokemon)
        if pokemon:
            pokemon.current_hp = event.details.get('current_hp', pokemon.current_hp)
            pokemon.max_hp = event.details.get('max_hp', pokemon.max_hp)
            if event.details.get('fainted', False):
                pokemon.is_fainted = True
                pokemon.current_hp = 0
    
    def _apply_heal(self, event: BattleEvent):
        """Apply heal event."""
        pokemon = self._find_pokemon(event.player, event.pokemon)
        if pokemon:
            pokemon.current_hp = event.details.get('current_hp', pokemon.current_hp)
            pokemon.max_hp = event.details.get('max_hp', pokemon.max_hp)
    
    def _apply_faint(self, event: BattleEvent):
        """Apply faint event."""
        pokemon = self._find_pokemon(event.player, event.pokemon)
        if pokemon:
            pokemon.is_fainted = True
            pokemon.current_hp = 0
    
    def _apply_tera(self, event: BattleEvent):
        """Apply terastallize event."""
        pokemon = self._find_pokemon(event.player, event.pokemon)
        if pokemon:
            pokemon.is_terastallized = True
            pokemon.tera_type = event.details.get('tera_type')
        
        # Mark player as having used tera
        if event.player == 'p1':
            self.current_state.p1_can_tera = False
        else:
            self.current_state.p2_can_tera = False
    
    def _apply_stat_change(self, event: BattleEvent):
        """Apply stat boost/drop."""
        pokemon = self._find_pokemon(event.player, event.pokemon)
        if pokemon:
            stat = event.details.get('stat', '').lower()
            amount = event.details.get('amount', 0)
            if stat in pokemon.stat_boosts:
                pokemon.stat_boosts[stat] = max(-6, min(6, pokemon.stat_boosts[stat] + amount))
    
    def _apply_status(self, event: BattleEvent):
        """Apply status condition."""
        pokemon = self._find_pokemon(event.player, event.pokemon)
        if pokemon:
            pokemon.status = event.details.get('status')
    
    def _apply_weather(self, event: BattleEvent):
        """Apply weather change."""
        weather = event.details.get('weather', '')
        if 'none' in weather.lower() or 'end' in weather.lower():
            self.current_state.field.weather = None
        else:
            self.current_state.field.weather = weather
    
    def _apply_field(self, event: BattleEvent):
        """Apply field condition change."""
        condition = event.details.get('condition', '').lower()
        active = event.details.get('active', True)
        
        if 'terrain' in condition:
            if active:
                if 'electric' in condition:
                    self.current_state.field.terrain = 'electric'
                elif 'grassy' in condition:
                    self.current_state.field.terrain = 'grassy'
                elif 'misty' in condition:
                    self.current_state.field.terrain = 'misty'
                elif 'psychic' in condition:
                    self.current_state.field.terrain = 'psychic'
            else:
                self.current_state.field.terrain = None
        
        if 'trick' in condition and 'room' in condition:
            self.current_state.field.trick_room = active
    
    def _apply_side_condition(self, event: BattleEvent):
        """Apply side condition change."""
        player = event.player
        condition = event.details.get('condition', '').lower()
        active = event.details.get('active', True)
        
        if player == 'p1':
            side = self.current_state.p1_side
        else:
            side = self.current_state.p2_side
        
        if 'tailwind' in condition:
            side.tailwind = active
        elif 'reflect' in condition:
            side.reflect = active
        elif 'light' in condition and 'screen' in condition:
            side.light_screen = active
        elif 'aurora' in condition and 'veil' in condition:
            side.aurora_veil = active
    
    def _find_pokemon(self, player: Optional[str], name: Optional[str]) -> Optional[PokemonState]:
        """Find a Pokemon in the current state."""
        if not player or not name:
            return None
        
        name_lower = name.lower()
        
        # Search in active first
        if player == 'p1':
            active = self.current_state.p1_active
            bench = self.current_state.p1_bench
        else:
            active = self.current_state.p2_active
            bench = self.current_state.p2_bench
        
        for pokemon in active:
            if pokemon and (name_lower in pokemon.species.lower() or pokemon.species.lower() in name_lower):
                return pokemon
            if pokemon and (name_lower in pokemon.nickname.lower() or pokemon.nickname.lower() in name_lower):
                return pokemon
        
        for pokemon in bench:
            if name_lower in pokemon.species.lower() or pokemon.species.lower() in name_lower:
                return pokemon
            if name_lower in pokemon.nickname.lower() or pokemon.nickname.lower() in name_lower:
                return pokemon
        
        return None
    
    def _copy_state(self, state: TurnState) -> TurnState:
        """Create a deep copy of the turn state."""
        return TurnState(
            turn=state.turn,
            p1_active=[self._copy_pokemon(p) for p in state.p1_active if p],
            p1_bench=[self._copy_pokemon(p) for p in state.p1_bench],
            p1_side=SideState(
                tailwind=state.p1_side.tailwind,
                reflect=state.p1_side.reflect,
                light_screen=state.p1_side.light_screen,
            ),
            p1_can_tera=state.p1_can_tera,
            p2_active=[self._copy_pokemon(p) for p in state.p2_active if p],
            p2_bench=[self._copy_pokemon(p) for p in state.p2_bench],
            p2_side=SideState(
                tailwind=state.p2_side.tailwind,
                reflect=state.p2_side.reflect,
                light_screen=state.p2_side.light_screen,
            ),
            p2_can_tera=state.p2_can_tera,
            field=FieldState(
                weather=state.field.weather,
                terrain=state.field.terrain,
                trick_room=state.field.trick_room,
            ),
        )
    
    def _copy_pokemon(self, pokemon: PokemonState) -> PokemonState:
        """Create a copy of a Pokemon state."""
        return PokemonState(
            species=pokemon.species,
            nickname=pokemon.nickname,
            level=pokemon.level,
            current_hp=pokemon.current_hp,
            max_hp=pokemon.max_hp,
            is_active=pokemon.is_active,
            is_fainted=pokemon.is_fainted,
            slot=pokemon.slot,
            status=pokemon.status,
            stat_boosts=pokemon.stat_boosts.copy(),
            is_terastallized=pokemon.is_terastallized,
            tera_type=pokemon.tera_type,
            ability=pokemon.ability,
            item=pokemon.item,
            moves=pokemon.moves.copy(),
            original_tera_type=pokemon.original_tera_type,
            types=pokemon.types.copy(),
        )


def test_reconstructor():
    """Test the state reconstructor."""
    from .replay_parser import ReplayParser
    
    sample_log = """
|gametype|doubles
|player|p1|Player1|turo-ai|
|player|p2|Player2|trainer|
|poke|p1|Flutter Mane, L50|
|poke|p1|Rillaboom, L50, M|
|poke|p2|Urshifu-Rapid-Strike, L50, F|
|poke|p2|Incineroar, L50, M|
|turn|1
|switch|p1a: Flutter Mane|Flutter Mane, L50|100/100
|switch|p1b: Rillaboom|Rillaboom, L50, M|100/100
|switch|p2a: Urshifu|Urshifu-Rapid-Strike, L50, F|100/100
|switch|p2b: Incineroar|Incineroar, L50, M|100/100
|move|p1a: Flutter Mane|Dazzling Gleam|p2a: Urshifu
|-damage|p2a: Urshifu|65/100
|-damage|p2b: Incineroar|80/100
|move|p2a: Urshifu|Surging Strikes|p1a: Flutter Mane
|-damage|p1a: Flutter Mane|0 fnt
|faint|p1a: Flutter Mane
|turn|2
|win|Player2
"""
    
    parser = ReplayParser()
    battle = parser.parse("test-battle-001", sample_log)
    
    reconstructor = StateReconstructor()
    states = reconstructor.reconstruct(battle)
    
    logger.info(f"Reconstructed {len(states)} turn states")
    
    for turn, state in states.items():
        logger.info(f"\nTurn {turn}:")
        logger.info(f"  P1 active: {[p.species for p in state.p1_active]}")
        logger.info(f"  P1 HP: {[p.hp_fraction for p in state.p1_active]}")
        logger.info(f"  P2 active: {[p.species for p in state.p2_active]}")
        logger.info(f"  P2 HP: {[p.hp_fraction for p in state.p2_active]}")


if __name__ == "__main__":
    test_reconstructor()

