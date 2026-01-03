"""Pokemon damage calculation for VGC simulation.

This module implements the Gen 9 damage formula with type effectiveness,
STAB, and other modifiers for realistic battle simulation.

Reference: https://bulbapedia.bulbagarden.net/wiki/Damage
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


# =============================================================================
# Type Effectiveness Chart (Gen 9)
# =============================================================================

# Types in order
TYPES = [
    "normal", "fire", "water", "electric", "grass", "ice",
    "fighting", "poison", "ground", "flying", "psychic", "bug",
    "rock", "ghost", "dragon", "dark", "steel", "fairy"
]

TYPE_TO_IDX = {t: i for i, t in enumerate(TYPES)}

# Type chart: effectiveness[attack_type][defend_type]
# 0 = immune, 0.5 = resist, 1 = neutral, 2 = super effective
TYPE_CHART = {
    "normal":   {"rock": 0.5, "ghost": 0.0, "steel": 0.5},
    "fire":     {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 2.0, "bug": 2.0, 
                 "rock": 0.5, "dragon": 0.5, "steel": 2.0},
    "water":    {"fire": 2.0, "water": 0.5, "grass": 0.5, "ground": 2.0, "rock": 2.0, 
                 "dragon": 0.5},
    "electric": {"water": 2.0, "electric": 0.5, "grass": 0.5, "ground": 0.0, 
                 "flying": 2.0, "dragon": 0.5},
    "grass":    {"fire": 0.5, "water": 2.0, "grass": 0.5, "poison": 0.5, "ground": 2.0,
                 "flying": 0.5, "bug": 0.5, "rock": 2.0, "dragon": 0.5, "steel": 0.5},
    "ice":      {"fire": 0.5, "water": 0.5, "grass": 2.0, "ice": 0.5, "ground": 2.0,
                 "flying": 2.0, "dragon": 2.0, "steel": 0.5},
    "fighting": {"normal": 2.0, "ice": 2.0, "poison": 0.5, "flying": 0.5, "psychic": 0.5,
                 "bug": 0.5, "rock": 2.0, "ghost": 0.0, "dark": 2.0, "steel": 2.0, "fairy": 0.5},
    "poison":   {"grass": 2.0, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5,
                 "steel": 0.0, "fairy": 2.0},
    "ground":   {"fire": 2.0, "electric": 2.0, "grass": 0.5, "poison": 2.0, "flying": 0.0,
                 "bug": 0.5, "rock": 2.0, "steel": 2.0},
    "flying":   {"electric": 0.5, "grass": 2.0, "fighting": 2.0, "bug": 2.0, "rock": 0.5,
                 "steel": 0.5},
    "psychic":  {"fighting": 2.0, "poison": 2.0, "psychic": 0.5, "dark": 0.0, "steel": 0.5},
    "bug":      {"fire": 0.5, "grass": 2.0, "fighting": 0.5, "poison": 0.5, "flying": 0.5,
                 "psychic": 2.0, "ghost": 0.5, "dark": 2.0, "steel": 0.5, "fairy": 0.5},
    "rock":     {"fire": 2.0, "ice": 2.0, "fighting": 0.5, "ground": 0.5, "flying": 2.0,
                 "bug": 2.0, "steel": 0.5},
    "ghost":    {"normal": 0.0, "psychic": 2.0, "ghost": 2.0, "dark": 0.5},
    "dragon":   {"dragon": 2.0, "steel": 0.5, "fairy": 0.0},
    "dark":     {"fighting": 0.5, "psychic": 2.0, "ghost": 2.0, "dark": 0.5, "fairy": 0.5},
    "steel":    {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2.0, "rock": 2.0,
                 "steel": 0.5, "fairy": 2.0},
    "fairy":    {"fire": 0.5, "fighting": 2.0, "poison": 0.5, "dragon": 2.0, "dark": 2.0,
                 "steel": 0.5},
}


@dataclass
class MoveData:
    """Data for a Pokemon move."""
    name: str
    power: int
    move_type: str
    category: str  # "physical", "special", "status"
    is_spread: bool = False  # Spread moves hit multiple targets
    
    # Special move properties
    always_crit: bool = False
    priority: int = 0


# Common VGC moves with their base power
MOVE_DATABASE: Dict[str, MoveData] = {
    # High power
    "closecombat": MoveData("Close Combat", 120, "fighting", "physical"),
    "flareblitz": MoveData("Flare Blitz", 120, "fire", "physical"),
    "bravebird": MoveData("Brave Bird", 120, "flying", "physical"),
    "headsmash": MoveData("Head Smash", 150, "rock", "physical"),
    "woodhammer": MoveData("Wood Hammer", 120, "grass", "physical"),
    "hydropump": MoveData("Hydro Pump", 110, "water", "special"),
    "blizzard": MoveData("Blizzard", 110, "ice", "special", is_spread=True),
    "thunder": MoveData("Thunder", 110, "electric", "special"),
    "dracometeor": MoveData("Draco Meteor", 130, "dragon", "special"),
    "overheat": MoveData("Overheat", 130, "fire", "special"),
    
    # Medium power
    "earthquake": MoveData("Earthquake", 100, "ground", "physical", is_spread=True),
    "rockslide": MoveData("Rock Slide", 75, "rock", "physical", is_spread=True),
    "heatwave": MoveData("Heat Wave", 95, "fire", "special", is_spread=True),
    "muddywater": MoveData("Muddy Water", 90, "water", "special", is_spread=True),
    "dazzlinggleam": MoveData("Dazzling Gleam", 80, "fairy", "special", is_spread=True),
    "flamethrower": MoveData("Flamethrower", 90, "fire", "special"),
    "icebeam": MoveData("Ice Beam", 90, "ice", "special"),
    "thunderbolt": MoveData("Thunderbolt", 90, "electric", "special"),
    "moonblast": MoveData("Moonblast", 95, "fairy", "special"),
    "shadowball": MoveData("Shadow Ball", 80, "ghost", "special"),
    "psychic": MoveData("Psychic", 90, "psychic", "special"),
    "darkpulse": MoveData("Dark Pulse", 80, "dark", "special"),
    "dragonpulse": MoveData("Dragon Pulse", 85, "dragon", "special"),
    "energyball": MoveData("Energy Ball", 90, "grass", "special"),
    "flashcannon": MoveData("Flash Cannon", 80, "steel", "special"),
    "sludgebomb": MoveData("Sludge Bomb", 90, "poison", "special"),
    "focusblast": MoveData("Focus Blast", 120, "fighting", "special"),
    
    # Physical moves
    "ironhead": MoveData("Iron Head", 80, "steel", "physical"),
    "crunch": MoveData("Crunch", 80, "dark", "physical"),
    "bodyslam": MoveData("Body Slam", 85, "normal", "physical"),
    "drainpunch": MoveData("Drain Punch", 75, "fighting", "physical"),
    "icepunch": MoveData("Ice Punch", 75, "ice", "physical"),
    "firepunch": MoveData("Fire Punch", 75, "fire", "physical"),
    "thunderpunch": MoveData("Thunder Punch", 75, "electric", "physical"),
    "knockoff": MoveData("Knock Off", 65, "dark", "physical"),  # 97.5 with item
    "uturn": MoveData("U-turn", 70, "bug", "physical"),
    "wildcharge": MoveData("Wild Charge", 90, "electric", "physical"),
    "playrough": MoveData("Play Rough", 90, "fairy", "physical"),
    "superpower": MoveData("Superpower", 120, "fighting", "physical"),
    "sacredsword": MoveData("Sacred Sword", 90, "fighting", "physical"),
    "kowtowcleave": MoveData("Kowtow Cleave", 85, "dark", "physical"),
    
    # Priority moves
    "fakeout": MoveData("Fake Out", 40, "normal", "physical", priority=3),
    "extremespeed": MoveData("Extreme Speed", 80, "normal", "physical", priority=2),
    "quickattack": MoveData("Quick Attack", 40, "normal", "physical", priority=1),
    "machpunch": MoveData("Mach Punch", 40, "fighting", "physical", priority=1),
    "bulletpunch": MoveData("Bullet Punch", 40, "steel", "physical", priority=1),
    "aquajet": MoveData("Aqua Jet", 40, "water", "physical", priority=1),
    "iceshard": MoveData("Ice Shard", 40, "ice", "physical", priority=1),
    "shadowsneak": MoveData("Shadow Sneak", 40, "ghost", "physical", priority=1),
    "suckerpunch": MoveData("Sucker Punch", 70, "dark", "physical", priority=1),
    "grassyglide": MoveData("Grassy Glide", 55, "grass", "physical", priority=1),  # +1 in grassy terrain
    
    # Low power / utility
    "icywind": MoveData("Icy Wind", 55, "ice", "special", is_spread=True),
    "snarl": MoveData("Snarl", 55, "dark", "special", is_spread=True),
    "electroweb": MoveData("Electroweb", 55, "electric", "special", is_spread=True),
    "surf": MoveData("Surf", 90, "water", "special", is_spread=True),
    
    # Status (0 power)
    "protect": MoveData("Protect", 0, "normal", "status", priority=4),
    "trickroom": MoveData("Trick Room", 0, "psychic", "status", priority=-7),
    "tailwind": MoveData("Tailwind", 0, "flying", "status"),
    "willowisp": MoveData("Will-O-Wisp", 0, "fire", "status"),
    "thunderwave": MoveData("Thunder Wave", 0, "electric", "status"),
    "spore": MoveData("Spore", 0, "grass", "status"),
    "helpinghand": MoveData("Helping Hand", 0, "normal", "status", priority=5),
    "followme": MoveData("Follow Me", 0, "normal", "status", priority=2),
    "ragepowder": MoveData("Rage Powder", 0, "bug", "status", priority=2),
}


def get_move_data(move_name: str) -> Optional[MoveData]:
    """Get move data by name.
    
    Args:
        move_name: Move name (case insensitive)
        
    Returns:
        MoveData or None if not found
    """
    normalized = move_name.lower().replace(" ", "").replace("-", "").replace("'", "")
    return MOVE_DATABASE.get(normalized)


def get_type_effectiveness(attack_type: str, defend_types: List[str]) -> float:
    """Calculate type effectiveness multiplier.
    
    Args:
        attack_type: Type of the attacking move
        defend_types: List of defender's types (1-2)
        
    Returns:
        Effectiveness multiplier (0, 0.25, 0.5, 1, 2, or 4)
    """
    attack_type = attack_type.lower()
    effectiveness = 1.0
    
    for def_type in defend_types:
        if not def_type:
            continue
        def_type = def_type.lower()
        
        type_matchups = TYPE_CHART.get(attack_type, {})
        multiplier = type_matchups.get(def_type, 1.0)
        effectiveness *= multiplier
    
    return effectiveness


class DamageCalculator:
    """Pokemon damage calculator for VGC simulation.
    
    Implements Gen 9 damage formula:
    Damage = ((2*Level/5+2)*Power*A/D)/50 + 2) * Modifiers
    
    Where:
    - Level is assumed to be 50 (VGC format)
    - A = Attack/SpA depending on move category
    - D = Defense/SpD depending on move category
    - Modifiers = STAB * Type Effectiveness * Spread * Random * Other
    """
    
    LEVEL = 50  # VGC level
    
    def __init__(
        self,
        random_damage: bool = True,
        spread_modifier: float = 0.75,
        stab_multiplier: float = 1.5,
    ):
        """Initialize calculator.
        
        Args:
            random_damage: Whether to apply random damage roll (0.85-1.0)
            spread_modifier: Damage reduction for spread moves
            stab_multiplier: Same Type Attack Bonus multiplier
        """
        self.random_damage = random_damage
        self.spread_modifier = spread_modifier
        self.stab_multiplier = stab_multiplier
    
    def calculate(
        self,
        move_name: str = None,
        move_power: int = None,
        move_type: str = "normal",
        category: str = "physical",
        attacker_atk: float = 100.0,
        attacker_spa: float = 100.0,
        defender_def: float = 100.0,
        defender_spd: float = 100.0,
        defender_types: List[str] = None,
        attacker_types: List[str] = None,
        is_spread: bool = False,
        crit: bool = False,
        weather: str = None,
        terrain: str = None,
        attacker_item: str = None,
        defender_item: str = None,
        other_multiplier: float = 1.0,
    ) -> float:
        """Calculate damage as fraction of max HP.
        
        Args:
            move_name: Name of move (looks up power/type if provided)
            move_power: Base power (overrides move lookup)
            move_type: Type of move
            category: "physical" or "special"
            attacker_atk: Attacker's Attack stat
            attacker_spa: Attacker's Special Attack stat
            defender_def: Defender's Defense stat
            defender_spd: Defender's Special Defense stat
            defender_types: Defender's types for effectiveness
            attacker_types: Attacker's types for STAB
            is_spread: Whether this is a spread move
            crit: Whether this is a critical hit
            weather: Current weather
            terrain: Current terrain
            attacker_item: Attacker's held item
            defender_item: Defender's held item
            other_multiplier: Additional multiplier
            
        Returns:
            Damage as fraction of max HP (0.0 to 1.0+)
        """
        defender_types = defender_types or []
        attacker_types = attacker_types or []
        
        # Get move data if name provided
        if move_name:
            move_data = get_move_data(move_name)
            if move_data:
                move_power = move_power or move_data.power
                move_type = move_data.move_type
                category = move_data.category
                is_spread = move_data.is_spread
        
        # Status moves do no damage
        if category == "status" or move_power is None or move_power == 0:
            return 0.0
        
        # Select attack and defense stats based on category
        if category == "physical":
            attack = attacker_atk
            defense = defender_def
        else:  # special
            attack = attacker_spa
            defense = defender_spd
        
        # Critical hit ignores defense boosts and attack drops
        if crit:
            defense = min(defense, defender_def if category == "physical" else defender_spd)
            attack = max(attack, attacker_atk if category == "physical" else attacker_spa)
        
        # Base damage formula
        # ((2*Level/5+2)*Power*A/D)/50 + 2
        level_factor = (2 * self.LEVEL / 5) + 2
        base_damage = ((level_factor * move_power * attack / defense) / 50) + 2
        
        # Apply modifiers
        modifier = 1.0
        
        # STAB (Same Type Attack Bonus)
        move_type_lower = move_type.lower()
        is_stab = any(
            t and t.lower() == move_type_lower 
            for t in attacker_types
        )
        if is_stab:
            modifier *= self.stab_multiplier
        
        # Type effectiveness
        effectiveness = get_type_effectiveness(move_type, defender_types)
        modifier *= effectiveness
        
        # Immunity check
        if effectiveness == 0:
            return 0.0
        
        # Spread move reduction
        if is_spread:
            modifier *= self.spread_modifier
        
        # Critical hit
        if crit:
            modifier *= 1.5
        
        # Weather effects
        if weather:
            weather_lower = weather.lower()
            if weather_lower in ["sun", "sunnyday", "desolateland", "harshsunshine"]:
                if move_type_lower == "fire":
                    modifier *= 1.5
                elif move_type_lower == "water":
                    modifier *= 0.5
            elif weather_lower in ["rain", "raindance", "primordialsea", "heavyrain"]:
                if move_type_lower == "water":
                    modifier *= 1.5
                elif move_type_lower == "fire":
                    modifier *= 0.5
        
        # Terrain effects
        if terrain:
            terrain_lower = terrain.lower()
            if terrain_lower in ["electric", "electricterrain"]:
                if move_type_lower == "electric":
                    modifier *= 1.3
            elif terrain_lower in ["grassy", "grassyterrain"]:
                if move_type_lower == "grass":
                    modifier *= 1.3
                # Earthquake/Bulldoze reduction
                if move_name and move_name.lower().replace(" ", "") in ["earthquake", "bulldoze"]:
                    modifier *= 0.5
            elif terrain_lower in ["psychic", "psychicterrain"]:
                if move_type_lower == "psychic":
                    modifier *= 1.3
            elif terrain_lower in ["misty", "mistyterrain"]:
                if move_type_lower == "dragon":
                    modifier *= 0.5
        
        # Item effects (simplified)
        if attacker_item:
            item_lower = attacker_item.lower().replace(" ", "").replace("-", "")
            if item_lower == "lifeorb":
                modifier *= 1.3
            elif item_lower == "choiceband" and category == "physical":
                modifier *= 1.5
            elif item_lower == "choicespecs" and category == "special":
                modifier *= 1.5
            elif item_lower == "expertbelt" and effectiveness > 1:
                modifier *= 1.2
        
        # Other multiplier (for custom effects)
        modifier *= other_multiplier
        
        # Random damage roll (0.85 to 1.0)
        if self.random_damage:
            roll = np.random.uniform(0.85, 1.0)
            modifier *= roll
        
        # Calculate final damage
        damage = base_damage * modifier
        
        # Convert to HP fraction (assuming ~300 HP at level 50)
        # This is approximate - actual HP varies by species
        typical_hp = 300  # Rough average for VGC
        damage_fraction = damage / typical_hp
        
        return damage_fraction
    
    def calculate_ko_probability(
        self,
        move_name: str = None,
        move_power: int = None,
        move_type: str = "normal",
        category: str = "physical",
        attacker_atk: float = 100.0,
        attacker_spa: float = 100.0,
        defender_def: float = 100.0,
        defender_spd: float = 100.0,
        defender_types: List[str] = None,
        attacker_types: List[str] = None,
        defender_hp_fraction: float = 1.0,
        is_spread: bool = False,
        samples: int = 100,
        **kwargs,
    ) -> float:
        """Calculate probability of KO.
        
        Args:
            defender_hp_fraction: Defender's current HP as fraction of max
            samples: Number of damage rolls to simulate
            **kwargs: Additional args passed to calculate()
            
        Returns:
            Probability of KO (0.0 to 1.0)
        """
        kos = 0
        
        for _ in range(samples):
            damage = self.calculate(
                move_name=move_name,
                move_power=move_power,
                move_type=move_type,
                category=category,
                attacker_atk=attacker_atk,
                attacker_spa=attacker_spa,
                defender_def=defender_def,
                defender_spd=defender_spd,
                defender_types=defender_types,
                attacker_types=attacker_types,
                is_spread=is_spread,
                **kwargs,
            )
            
            if damage >= defender_hp_fraction:
                kos += 1
        
        return kos / samples


# Create default calculator instance
_default_calculator = DamageCalculator()


def calculate_damage(**kwargs) -> float:
    """Convenience function using default calculator."""
    return _default_calculator.calculate(**kwargs)


def calculate_ko_probability(**kwargs) -> float:
    """Convenience function using default calculator."""
    return _default_calculator.calculate_ko_probability(**kwargs)

