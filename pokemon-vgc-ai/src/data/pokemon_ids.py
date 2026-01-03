"""Pokemon ID dictionaries for consistent encoding.

This module provides collision-free mappings from species, moves, items,
abilities, and types to unique integer IDs. These are loaded from poke-env's
built-in game data to ensure accuracy.
"""

from typing import Dict, Optional
from functools import lru_cache
from loguru import logger

# Try to load from poke-env, fall back to static if unavailable
try:
    from poke_env.data import GenData
    _gen_data = GenData.from_gen(9)
    _USE_POKE_ENV = True
except Exception as e:
    logger.warning(f"Could not load poke-env GenData: {e}. Using static mappings.")
    _USE_POKE_ENV = False
    _gen_data = None


def _normalize_name(name: str) -> str:
    """Normalize a name for lookup (lowercase, no spaces/hyphens)."""
    return name.lower().replace(" ", "").replace("-", "").replace("'", "")


# =============================================================================
# Species ID Mapping
# =============================================================================

if _USE_POKE_ENV and _gen_data:
    # Build from poke-env pokedex
    _SPECIES_TO_ID: Dict[str, int] = {}
    for idx, (name, data) in enumerate(_gen_data.pokedex.items(), start=1):
        _SPECIES_TO_ID[_normalize_name(name)] = idx
        # Also add the dex number if available
        if isinstance(data, dict) and 'num' in data:
            dex_num = data['num']
            if dex_num > 0:
                _SPECIES_TO_ID[_normalize_name(name)] = dex_num
    NUM_SPECIES = max(_SPECIES_TO_ID.values()) + 1 if _SPECIES_TO_ID else 1025
else:
    # Static fallback for common VGC Pokemon (Gen 9)
    _SPECIES_TO_ID = {
        "flutter mane": 987, "gholdengo": 1000, "urshifurapidstrike": 892,
        "urshifu": 892, "rillaboom": 812, "incineroar": 727, "landorustherian": 645,
        "ogerpon": 1017, "chienpaopa": 1001, "regieleki": 894, "miraidon": 1008,
        "koraidon": 1007, "calyrexshadow": 898, "calyrexice": 898,
        "tornadustherian": 641, "amoonguss": 591, "grimmsnarl": 861,
        "ironhands": 992, "ironbundle": 991, "kingambit": 983,
        "dragonite": 149, "arcanine": 59, "arcaninetohisui": 59,
        "pelipper": 279, "talonflame": 663, "farigiraf": 981,
        "indeedee": 876, "indeedeefemale": 876, "gothitelle": 576,
        "charizard": 6, "pikachu": 25, "meowscarada": 908,
    }
    NUM_SPECIES = 1025


def get_species_id(species: str) -> int:
    """Get ID for a Pokemon species.
    
    Args:
        species: Species name (e.g., "Flutter Mane", "flutter-mane")
        
    Returns:
        Unique integer ID (0 = unknown)
    """
    normalized = _normalize_name(species)
    return _SPECIES_TO_ID.get(normalized, 0)


# =============================================================================
# Move ID Mapping
# =============================================================================

if _USE_POKE_ENV and _gen_data:
    _MOVE_TO_ID: Dict[str, int] = {
        _normalize_name(name): idx 
        for idx, name in enumerate(_gen_data.moves.keys(), start=1)
    }
    NUM_MOVES = len(_MOVE_TO_ID) + 1
else:
    # Static fallback for common VGC moves
    _MOVE_TO_ID = {
        "protect": 1, "fakeout": 2, "uturn": 3, "suckerpunch": 4,
        "moonblast": 5, "shadowball": 6, "psychic": 7, "earthquake": 8,
        "rockslide": 9, "flareblitz": 10, "closecombat": 11, "icywind": 12,
        "tailwind": 13, "trickroom": 14, "followme": 15, "ragepowder": 16,
        "snarl": 17, "knockoff": 18, "willowisp": 19, "thunderwave": 20,
        "spore": 21, "grassyglide": 22, "woodhammer": 23, "surgesurfer": 24,
        "fakeout": 25, "paraboliccharge": 26, "voltswitch": 27, "helpinghand": 28,
        "swordsdance": 29, "nastyplot": 30, "calmmind": 31, "dragondance": 32,
        "ironhead": 33, "sacredsword": 34, "kowtowcleave": 35, "makeitrain": 36,
        "dracometeor": 37, "dragonpulse": 38, "hydropump": 39, "surf": 40,
        "thunder": 41, "thunderbolt": 42, "icebeam": 43, "blizzard": 44,
    }
    NUM_MOVES = 920


def get_move_id(move: str) -> int:
    """Get ID for a move.
    
    Args:
        move: Move name (e.g., "Fake Out", "fake-out")
        
    Returns:
        Unique integer ID (0 = unknown)
    """
    normalized = _normalize_name(move)
    return _MOVE_TO_ID.get(normalized, 0)


# =============================================================================
# Item ID Mapping (poke-env doesn't include items, so always use static)
# =============================================================================

# Static mappings for common VGC items
_ITEM_TO_ID: Dict[str, int] = {
    "choicescarf": 1, "choiceband": 2, "choicespecs": 3,
    "focussash": 4, "lifeorb": 5, "assaultvest": 6,
    "leftovers": 7, "sitrusberry": 8, "lumberry": 9,
    "safetygoggles": 10, "covertcloak": 11, "clearamulet": 12,
    "boosterenergy": 13, "mirrorherb": 14, "ejectbutton": 15,
    "rockyhelmet": 16, "airballoon": 17, "widelens": 18,
    "expertbelt": 19, "scopelens": 20, "powerherb": 21,
    "whiteherb": 22, "mentalherb": 23, "protectivepads": 24,
    "lightclay": 25, "gripclaw": 26, "bindingband": 27,
    "shellbell": 28, "shedshell": 29, "redcard": 30,
    "ejectpack": 31, "heavydutyboots": 32, "throatspray": 33,
    "roomservice": 34, "blunderpolicy": 35, "weaknesspolicy": 36,
    "ironball": 37, "laggingtail": 38, "machobrace": 39,
    "kingsrock": 40, "razorfang": 41, "loadeddice": 42,
    "punchingglove": 43, "abilityshield": 44,
}
NUM_ITEMS = 250


def get_item_id(item: str) -> int:
    """Get ID for an item.
    
    Args:
        item: Item name (e.g., "Choice Scarf", "choice-scarf")
        
    Returns:
        Unique integer ID (0 = unknown/none)
    """
    if not item:
        return 0
    normalized = _normalize_name(item)
    return _ITEM_TO_ID.get(normalized, 0)


# =============================================================================
# Ability ID Mapping (poke-env doesn't include abilities, so always use static)
# =============================================================================

# Static mappings for common VGC abilities
_ABILITY_TO_ID: Dict[str, int] = {
    "intimidate": 1, "prankster": 2, "protean": 3, "libero": 4,
    "quarkdrive": 5, "protosynthesis": 6, "drizzle": 7, "drought": 8,
    "sandstream": 9, "snowwarning": 10, "electricsurge": 11,
    "psychicsurge": 12, "grassysurge": 13, "mistysurge": 14,
    "levitate": 15, "guts": 16, "multiscale": 17, "regenerator": 18,
    "contrary": 19, "unaware": 20, "magicbounce": 21, "shadowtag": 22,
    "defiant": 23, "competitive": 24, "supremeoverlord": 25,
    "embodyaspect": 26, "orichalcumpulse": 27, "hadronengine": 28,
    "moldbreaker": 29, "turboblaze": 30, "teravolt": 31,
    "innerfocus": 32, "clearbody": 33, "whitesmoke": 34,
    "fullmetalbody": 35, "lightmetal": 36, "heavymetal": 37,
    "speedboost": 38, "swordofruin": 39, "tabletsruin": 40,
    "vesselofruin": 41, "beadsofruin": 42, "goodasgold": 43,
    "thermalexchange": 44, "angershell": 45, "toxicdebris": 46,
    "armortail": 47, "eartheater": 48, "myceliumnight": 49,
    "mindseye": 50, "supersweetsyrup": 51, "hospitalityy": 52,
    "toxicchain": 53, "embodyaspectteal": 54, "embodyaspectwellspring": 55,
    "embodyaspecthearthflame": 56, "embodyaspectcornerstone": 57,
}
NUM_ABILITIES = 310


def get_ability_id(ability: str) -> int:
    """Get ID for an ability.
    
    Args:
        ability: Ability name (e.g., "Intimidate", "intimidate")
        
    Returns:
        Unique integer ID (0 = unknown)
    """
    if not ability:
        return 0
    normalized = _normalize_name(ability)
    return _ABILITY_TO_ID.get(normalized, 0)


# =============================================================================
# Type ID Mapping
# =============================================================================

TYPE_TO_ID: Dict[str, int] = {
    "normal": 1, "fire": 2, "water": 3, "electric": 4, "grass": 5,
    "ice": 6, "fighting": 7, "poison": 8, "ground": 9, "flying": 10,
    "psychic": 11, "bug": 12, "rock": 13, "ghost": 14, "dragon": 15,
    "dark": 16, "steel": 17, "fairy": 18, "stellar": 19,
}
NUM_TYPES = 20  # 0 = none/unknown, 1-19 = types


def get_type_id(type_name: str) -> int:
    """Get ID for a Pokemon type.
    
    Args:
        type_name: Type name (e.g., "Fire", "fire")
        
    Returns:
        Unique integer ID (0 = unknown)
    """
    if not type_name:
        return 0
    return TYPE_TO_ID.get(type_name.lower(), 0)


# =============================================================================
# Weather/Terrain/Status ID Mappings
# =============================================================================

WEATHER_TO_ID: Dict[str, int] = {
    "": 0, "none": 0,
    "sunnyday": 1, "sun": 1,
    "raindance": 2, "rain": 2,
    "sandstorm": 3, "sand": 3,
    "snowscape": 4, "snow": 4, "hail": 4,
    "desolateland": 5, "harshsunshine": 5,
    "primordialsea": 6, "heavyrain": 6,
}
NUM_WEATHERS = 7


TERRAIN_TO_ID: Dict[str, int] = {
    "": 0, "none": 0,
    "electricterrain": 1, "electric": 1,
    "grassyterrain": 2, "grassy": 2,
    "psychicterrain": 3, "psychic": 3,
    "mistyterrain": 4, "misty": 4,
}
NUM_TERRAINS = 5


STATUS_TO_ID: Dict[str, int] = {
    "": 0, "none": 0,
    "brn": 1, "burn": 1,
    "par": 2, "paralysis": 2,
    "slp": 3, "sleep": 3,
    "frz": 4, "freeze": 4,
    "psn": 5, "poison": 5,
    "tox": 6, "toxic": 6,
}
NUM_STATUSES = 7


def get_weather_id(weather: Optional[str]) -> int:
    """Get ID for weather condition."""
    if not weather:
        return 0
    return WEATHER_TO_ID.get(_normalize_name(weather), 0)


def get_terrain_id(terrain: Optional[str]) -> int:
    """Get ID for terrain condition."""
    if not terrain:
        return 0
    return TERRAIN_TO_ID.get(_normalize_name(terrain), 0)


def get_status_id(status: Optional[str]) -> int:
    """Get ID for status condition."""
    if not status:
        return 0
    return STATUS_TO_ID.get(_normalize_name(status), 0)


# =============================================================================
# Lookup Summary
# =============================================================================

@lru_cache(maxsize=1)
def get_vocab_sizes() -> Dict[str, int]:
    """Get vocabulary sizes for embeddings."""
    return {
        "species": NUM_SPECIES,
        "moves": NUM_MOVES,
        "items": NUM_ITEMS,
        "abilities": NUM_ABILITIES,
        "types": NUM_TYPES,
        "weathers": NUM_WEATHERS,
        "terrains": NUM_TERRAINS,
        "statuses": NUM_STATUSES,
    }


# Log what we loaded
if _USE_POKE_ENV:
    logger.info(f"Loaded Pokemon IDs from poke-env: {len(_SPECIES_TO_ID)} species, "
                f"{len(_MOVE_TO_ID)} moves; static: {len(_ITEM_TO_ID)} items, "
                f"{len(_ABILITY_TO_ID)} abilities")
else:
    logger.info("Using static Pokemon ID mappings (poke-env unavailable)")

