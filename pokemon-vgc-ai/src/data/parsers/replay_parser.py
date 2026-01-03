"""Parse Pokemon Showdown battle logs into structured events.

This module parses the raw Showdown protocol format into structured
BattleEvent and ParsedBattle objects.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from loguru import logger


@dataclass
class PokemonSet:
    """Full Pokemon set data from team preview."""
    species: str
    nickname: Optional[str] = None
    level: int = 50
    gender: Optional[str] = None
    ability: Optional[str] = None
    item: Optional[str] = None
    moves: List[str] = field(default_factory=list)
    tera_type: Optional[str] = None
    # IVs/EVs/Nature are rarely exposed in logs


@dataclass
class BattleEvent:
    """A single battle event from the log."""
    turn: int
    event_type: str  # move, switch, damage, faint, etc.
    player: Optional[str] = None  # p1 or p2
    slot: Optional[str] = None  # a or b (for doubles)
    pokemon: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedBattle:
    """Fully parsed battle data."""
    battle_id: str
    format: str
    player1: str
    player2: str
    winner: Optional[str] = None
    teams: Dict[str, List[PokemonSet]] = field(default_factory=dict)
    events: List[BattleEvent] = field(default_factory=list)
    turn_events: Dict[int, List[BattleEvent]] = field(default_factory=dict)
    total_turns: int = 0
    rating: Optional[int] = None
    timestamp: Optional[int] = None


class ReplayParser:
    """Parse Pokemon Showdown battle logs."""
    
    # Regex patterns for parsing
    PLAYER_PATTERN = re.compile(r'\|player\|(p[12])\|([^|]+)\|')
    POKEMON_PATTERN = re.compile(r'(p[12])([ab]): (.+)')
    HP_PATTERN = re.compile(r'(\d+)/(\d+)')
    TEAM_PATTERN = re.compile(r'\|poke\|(p[12])\|([^|]+)\|')
    SHOWTEAM_PATTERN = re.compile(r'\|showteam\|(p[12])\|(.+)')
    
    def __init__(self):
        """Initialize the parser."""
        self.current_turn = 0
    
    def parse(self, battle_id: str, log: str, timestamp: Optional[int] = None) -> ParsedBattle:
        """Parse a battle log string.
        
        Args:
            battle_id: Unique battle identifier
            log: Raw battle log string
            timestamp: Optional timestamp
            
        Returns:
            ParsedBattle object with structured data
        """
        battle = ParsedBattle(
            battle_id=battle_id,
            format="",
            player1="",
            player2="",
            timestamp=timestamp,
        )
        
        self.current_turn = 0
        events: List[BattleEvent] = []
        turn_events: Dict[int, List[BattleEvent]] = {0: []}
        
        lines = log.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or not line.startswith('|'):
                continue
            
            parts = line.split('|')
            if len(parts) < 2:
                continue
            
            event_type = parts[1]
            event = self._parse_line(event_type, parts, battle)
            
            if event:
                events.append(event)
                if self.current_turn not in turn_events:
                    turn_events[self.current_turn] = []
                turn_events[self.current_turn].append(event)
        
        battle.events = events
        battle.turn_events = turn_events
        battle.total_turns = self.current_turn
        
        return battle
    
    def _parse_line(
        self, 
        event_type: str, 
        parts: List[str], 
        battle: ParsedBattle
    ) -> Optional[BattleEvent]:
        """Parse a single line from the log.
        
        Args:
            event_type: The event type (e.g., 'move', 'switch')
            parts: Split line parts
            battle: Battle object to update metadata
            
        Returns:
            BattleEvent or None
        """
        # Handle different event types
        if event_type == 'player':
            self._parse_player(parts, battle)
            return None
        
        elif event_type == 'gametype':
            battle.format = parts[2] if len(parts) > 2 else ""
            return None
        
        elif event_type == 'tier':
            battle.format = parts[2] if len(parts) > 2 else battle.format
            return None
        
        elif event_type == 'turn':
            self.current_turn = int(parts[2]) if len(parts) > 2 else self.current_turn + 1
            return BattleEvent(
                turn=self.current_turn,
                event_type='turn',
                details={'turn_number': self.current_turn}
            )
        
        elif event_type == 'win':
            winner = parts[2] if len(parts) > 2 else ""
            battle.winner = winner
            return BattleEvent(
                turn=self.current_turn,
                event_type='win',
                details={'winner': winner}
            )
        
        elif event_type == 'poke':
            self._parse_team_pokemon(parts, battle)
            return None
        
        elif event_type == 'showteam':
            self._parse_showteam(parts, battle)
            return None
        
        elif event_type == 'switch' or event_type == 'drag':
            return self._parse_switch(parts)
        
        elif event_type == 'move':
            return self._parse_move(parts)
        
        elif event_type == '-damage' or event_type == '-heal':
            return self._parse_hp_change(parts, event_type)
        
        elif event_type == 'faint':
            return self._parse_faint(parts)
        
        elif event_type == '-terastallize':
            return self._parse_tera(parts)
        
        elif event_type == '-boost' or event_type == '-unboost':
            return self._parse_stat_change(parts, event_type)
        
        elif event_type == '-status':
            return self._parse_status(parts)
        
        elif event_type == '-weather':
            return self._parse_weather(parts)
        
        elif event_type == '-fieldstart' or event_type == '-fieldend':
            return self._parse_field(parts, event_type)
        
        elif event_type == '-sidestart' or event_type == '-sideend':
            return self._parse_side_condition(parts, event_type)
        
        elif event_type == '-ability':
            return self._parse_ability(parts)
        
        elif event_type == '-item' or event_type == '-enditem':
            return self._parse_item(parts, event_type)
        
        return None
    
    def _parse_player(self, parts: List[str], battle: ParsedBattle):
        """Parse player information."""
        if len(parts) >= 4:
            player_id = parts[2]
            username = parts[3]
            if player_id == 'p1':
                battle.player1 = username
            elif player_id == 'p2':
                battle.player2 = username
    
    def _parse_team_pokemon(self, parts: List[str], battle: ParsedBattle):
        """Parse team preview Pokemon."""
        if len(parts) >= 4:
            player = parts[2]
            pokemon_info = parts[3]
            
            # Parse species, level, gender
            species_match = pokemon_info.split(',')
            species = species_match[0].strip()
            level = 50
            gender = None
            
            for part in species_match[1:]:
                part = part.strip()
                if part.startswith('L'):
                    try:
                        level = int(part[1:])
                    except ValueError:
                        pass
                elif part in ['M', 'F']:
                    gender = part
            
            pokemon_set = PokemonSet(
                species=species,
                level=level,
                gender=gender,
            )
            
            if player not in battle.teams:
                battle.teams[player] = []
            battle.teams[player].append(pokemon_set)
    
    def _parse_showteam(self, parts: List[str], battle: ParsedBattle):
        """Parse full team data from showteam (open team sheets)."""
        if len(parts) < 4:
            return
        
        player = parts[2]
        team_data = parts[3]
        
        # Format: Species||Item|Ability|Move1,Move2,Move3,Move4||Gender|||Level|EVs,TeraType
        pokemon_strs = team_data.split(']')
        team = []
        
        for poke_str in pokemon_strs:
            if not poke_str.strip():
                continue
            
            poke_parts = poke_str.split('|')
            if len(poke_parts) < 4:
                continue
            
            species = poke_parts[0].strip()
            item = poke_parts[2] if len(poke_parts) > 2 else None
            ability = poke_parts[3] if len(poke_parts) > 3 else None
            moves_str = poke_parts[4] if len(poke_parts) > 4 else ""
            moves = [m.strip() for m in moves_str.split(',') if m.strip()]
            
            # Gender is typically at index 6
            gender = poke_parts[6] if len(poke_parts) > 6 and poke_parts[6] in ['M', 'F'] else None
            
            # Level at index 9
            level = 50
            if len(poke_parts) > 9:
                try:
                    level = int(poke_parts[9])
                except ValueError:
                    pass
            
            # Tera type is often at the end after comma
            tera_type = None
            if len(poke_parts) > 10 and ',' in poke_parts[10]:
                tera_parts = poke_parts[10].split(',')
                if len(tera_parts) > 1:
                    tera_type = tera_parts[-1].strip()
            
            pokemon_set = PokemonSet(
                species=species,
                level=level,
                gender=gender,
                ability=ability,
                item=item,
                moves=moves,
                tera_type=tera_type,
            )
            team.append(pokemon_set)
        
        battle.teams[player] = team
    
    def _parse_pokemon_identifier(self, identifier: str) -> Tuple[str, str, str]:
        """Parse a Pokemon identifier like 'p1a: Pikachu'.
        
        Returns:
            Tuple of (player, slot, pokemon_name)
        """
        match = self.POKEMON_PATTERN.match(identifier)
        if match:
            return match.group(1), match.group(2), match.group(3)
        return "", "", identifier
    
    def _parse_switch(self, parts: List[str]) -> Optional[BattleEvent]:
        """Parse switch/drag event."""
        if len(parts) < 4:
            return None
        
        player, slot, pokemon = self._parse_pokemon_identifier(parts[2])
        details_str = parts[3] if len(parts) > 3 else ""
        hp_str = parts[4] if len(parts) > 4 else "100/100"
        
        # Parse species from details
        species_parts = details_str.split(',')
        species = species_parts[0].strip()
        
        # Parse HP
        hp_match = self.HP_PATTERN.search(hp_str)
        current_hp = 100
        max_hp = 100
        if hp_match:
            current_hp = int(hp_match.group(1))
            max_hp = int(hp_match.group(2))
        
        return BattleEvent(
            turn=self.current_turn,
            event_type='switch',
            player=player,
            slot=slot,
            pokemon=pokemon,
            details={
                'species': species,
                'current_hp': current_hp,
                'max_hp': max_hp,
            }
        )
    
    def _parse_move(self, parts: List[str]) -> Optional[BattleEvent]:
        """Parse move event."""
        if len(parts) < 4:
            return None
        
        player, slot, pokemon = self._parse_pokemon_identifier(parts[2])
        move_name = parts[3]
        
        # Parse target if present
        target = None
        target_player = None
        target_slot = None
        if len(parts) > 4 and parts[4]:
            target_player, target_slot, target = self._parse_pokemon_identifier(parts[4])
        
        # Check for spread move indicator
        is_spread = '[spread]' in ' '.join(parts[4:]) if len(parts) > 4 else False
        
        return BattleEvent(
            turn=self.current_turn,
            event_type='move',
            player=player,
            slot=slot,
            pokemon=pokemon,
            details={
                'move': move_name,
                'target': target,
                'target_player': target_player,
                'target_slot': target_slot,
                'is_spread': is_spread,
            }
        )
    
    def _parse_hp_change(self, parts: List[str], event_type: str) -> Optional[BattleEvent]:
        """Parse damage or heal event."""
        if len(parts) < 4:
            return None
        
        player, slot, pokemon = self._parse_pokemon_identifier(parts[2])
        hp_str = parts[3]
        
        # Parse HP
        if hp_str == '0 fnt':
            current_hp = 0
            max_hp = 100
            fainted = True
        else:
            hp_match = self.HP_PATTERN.search(hp_str)
            current_hp = 0
            max_hp = 100
            fainted = False
            if hp_match:
                current_hp = int(hp_match.group(1))
                max_hp = int(hp_match.group(2))
        
        # Get source if present
        source = None
        if len(parts) > 4:
            source = parts[4]
        
        return BattleEvent(
            turn=self.current_turn,
            event_type='damage' if event_type == '-damage' else 'heal',
            player=player,
            slot=slot,
            pokemon=pokemon,
            details={
                'current_hp': current_hp,
                'max_hp': max_hp,
                'fainted': fainted,
                'source': source,
            }
        )
    
    def _parse_faint(self, parts: List[str]) -> Optional[BattleEvent]:
        """Parse faint event."""
        if len(parts) < 3:
            return None
        
        player, slot, pokemon = self._parse_pokemon_identifier(parts[2])
        
        return BattleEvent(
            turn=self.current_turn,
            event_type='faint',
            player=player,
            slot=slot,
            pokemon=pokemon,
        )
    
    def _parse_tera(self, parts: List[str]) -> Optional[BattleEvent]:
        """Parse terastallize event."""
        if len(parts) < 4:
            return None
        
        player, slot, pokemon = self._parse_pokemon_identifier(parts[2])
        tera_type = parts[3]
        
        return BattleEvent(
            turn=self.current_turn,
            event_type='terastallize',
            player=player,
            slot=slot,
            pokemon=pokemon,
            details={'tera_type': tera_type}
        )
    
    def _parse_stat_change(self, parts: List[str], event_type: str) -> Optional[BattleEvent]:
        """Parse stat boost/unboost event."""
        if len(parts) < 5:
            return None
        
        player, slot, pokemon = self._parse_pokemon_identifier(parts[2])
        stat = parts[3]
        amount = int(parts[4]) if parts[4].isdigit() else 1
        
        if event_type == '-unboost':
            amount = -amount
        
        return BattleEvent(
            turn=self.current_turn,
            event_type='stat_change',
            player=player,
            slot=slot,
            pokemon=pokemon,
            details={'stat': stat, 'amount': amount}
        )
    
    def _parse_status(self, parts: List[str]) -> Optional[BattleEvent]:
        """Parse status condition event."""
        if len(parts) < 4:
            return None
        
        player, slot, pokemon = self._parse_pokemon_identifier(parts[2])
        status = parts[3]
        
        return BattleEvent(
            turn=self.current_turn,
            event_type='status',
            player=player,
            slot=slot,
            pokemon=pokemon,
            details={'status': status}
        )
    
    def _parse_weather(self, parts: List[str]) -> Optional[BattleEvent]:
        """Parse weather event."""
        if len(parts) < 3:
            return None
        
        weather = parts[2]
        
        return BattleEvent(
            turn=self.current_turn,
            event_type='weather',
            details={'weather': weather}
        )
    
    def _parse_field(self, parts: List[str], event_type: str) -> Optional[BattleEvent]:
        """Parse field condition event (terrain, trick room, etc.)."""
        if len(parts) < 3:
            return None
        
        field_condition = parts[2]
        is_start = event_type == '-fieldstart'
        
        return BattleEvent(
            turn=self.current_turn,
            event_type='field',
            details={
                'condition': field_condition,
                'active': is_start,
            }
        )
    
    def _parse_side_condition(self, parts: List[str], event_type: str) -> Optional[BattleEvent]:
        """Parse side condition event (tailwind, screens, etc.)."""
        if len(parts) < 4:
            return None
        
        # Format: |-sidestart|p1: Player|move: Tailwind
        side = parts[2].split(':')[0] if ':' in parts[2] else parts[2]
        condition = parts[3]
        is_start = event_type == '-sidestart'
        
        return BattleEvent(
            turn=self.current_turn,
            event_type='side_condition',
            player=side,
            details={
                'condition': condition,
                'active': is_start,
            }
        )
    
    def _parse_ability(self, parts: List[str]) -> Optional[BattleEvent]:
        """Parse ability activation event."""
        if len(parts) < 4:
            return None
        
        player, slot, pokemon = self._parse_pokemon_identifier(parts[2])
        ability = parts[3]
        
        return BattleEvent(
            turn=self.current_turn,
            event_type='ability',
            player=player,
            slot=slot,
            pokemon=pokemon,
            details={'ability': ability}
        )
    
    def _parse_item(self, parts: List[str], event_type: str) -> Optional[BattleEvent]:
        """Parse item event."""
        if len(parts) < 4:
            return None
        
        player, slot, pokemon = self._parse_pokemon_identifier(parts[2])
        item = parts[3]
        
        return BattleEvent(
            turn=self.current_turn,
            event_type='item' if event_type == '-item' else 'item_end',
            player=player,
            slot=slot,
            pokemon=pokemon,
            details={'item': item}
        )


def test_parser():
    """Test the replay parser with a sample log."""
    sample_log = """
|j|☆Player1
|j|☆Player2
|gametype|doubles
|player|p1|Player1|turo-ai|
|player|p2|Player2|trainer|
|teamsize|p1|6
|teamsize|p2|6
|gen|9
|tier|[Gen 9] VGC 2024 Reg G
|poke|p1|Flutter Mane, L50|
|poke|p1|Rillaboom, L50, M|
|poke|p2|Urshifu-Rapid-Strike, L50, F|
|poke|p2|Incineroar, L50, M|
|turn|1
|switch|p1a: Flutter Mane|Flutter Mane, L50|100/100
|switch|p1b: Rillaboom|Rillaboom, L50, M|100/100
|switch|p2a: Urshifu|Urshifu-Rapid-Strike, L50, F|100/100
|switch|p2b: Incineroar|Incineroar, L50, M|100/100
|move|p1a: Flutter Mane|Dazzling Gleam|p2a: Urshifu|[spread]
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
    
    logger.info(f"Battle ID: {battle.battle_id}")
    logger.info(f"Format: {battle.format}")
    logger.info(f"Players: {battle.player1} vs {battle.player2}")
    logger.info(f"Winner: {battle.winner}")
    logger.info(f"Total turns: {battle.total_turns}")
    logger.info(f"Total events: {len(battle.events)}")
    
    for turn, events in battle.turn_events.items():
        logger.info(f"Turn {turn}: {len(events)} events")
        for event in events[:3]:
            logger.info(f"  - {event.event_type}: {event.pokemon} ({event.details})")


if __name__ == "__main__":
    test_parser()

