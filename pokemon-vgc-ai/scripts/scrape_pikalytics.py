#!/usr/bin/env python3
"""Scrape Pokemon movesets and abilities from Pikalytics for VGC 2024 Reg G."""

import requests
import json
import time
import re
from typing import Dict, List, Tuple
from pathlib import Path

# Pikalytics API endpoint for VGC 2024 Reg G
BASE_URL = "https://www.pikalytics.com/api/l/2024-12/gen9vgc2024regg"
POKEMON_URL = "https://www.pikalytics.com/api/p/2024-12/gen9vgc2024regg/{pokemon}"

# Headers to mimic a browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Accept": "application/json",
}


def get_top_pokemon(limit: int = 120) -> List[Dict]:
    """Get top Pokemon from Pikalytics usage stats."""
    print(f"Fetching top {limit} Pokemon from Pikalytics...")
    
    try:
        response = requests.get(BASE_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        pokemon_list = data if isinstance(data, list) else data.get("pokemon", [])
        return pokemon_list[:limit]
    except Exception as e:
        print(f"Error fetching Pokemon list: {e}")
        return []


def get_pokemon_details(pokemon_name: str) -> Dict:
    """Get detailed moveset data for a specific Pokemon."""
    # Convert name to URL format
    url_name = pokemon_name.lower().replace(" ", "").replace("-", "")
    
    # Special cases for forms
    form_mapping = {
        "urshifurapidstrike": "urshifurapidstrike",
        "urshifu": "urshifu",  # Single Strike
        "indeedeef": "indeedeef",
        "indeedeem": "indeedeem",
        "landorustherian": "landorustherian",
        "landorus": "landorustherian",  # Default to Therian
        "tornadustherian": "tornadustherian",
        "tornados": "tornadustherian",
        "thundurustherian": "thundurustherian",
        "thundurus": "thundurus",  # Incarnate
    }
    
    url_name = form_mapping.get(url_name, url_name)
    url = POKEMON_URL.format(pokemon=url_name)
    
    try:
        response = requests.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"  Error fetching {pokemon_name}: {e}")
        return {}


def format_pokemon_name(name: str) -> str:
    """Format Pokemon name for Python dictionary."""
    # Handle special forms
    name = name.strip()
    
    # Common form fixes
    if name.lower() == "indeedee-f" or name.lower() == "indeedeef":
        return "Indeedee-F"
    if name.lower() == "urshifu-rapid-strike":
        return "Urshifu-Rapid-Strike"
    if name.lower() == "landorus-therian" or name.lower() == "landorustherian":
        return "Landorus"  # Use base name
    if name.lower() == "tornadus-therian" or name.lower() == "tornadustherian":
        return "Tornadus"
    if name.lower() == "thundurus-incarnate":
        return "Thundurus"
    
    # Title case with hyphen handling
    parts = name.split("-")
    formatted_parts = []
    for part in parts:
        if part.lower() in ["f", "m", "galar", "alola", "hisui"]:
            formatted_parts.append(part.title())
        else:
            formatted_parts.append(part.title())
    
    return "-".join(formatted_parts)


def format_move_name(move: str) -> str:
    """Format move name properly."""
    move = move.strip()
    
    # Special move name fixes
    special_moves = {
        "uturn": "U-turn",
        "u-turn": "U-turn",
        "willowisp": "Will-O-Wisp",
        "will-o-wisp": "Will-O-Wisp",
        "trickroom": "Trick Room",
        "heatwave": "Heat Wave",
        "icebeam": "Ice Beam",
        "shadowball": "Shadow Ball",
        "thunderwave": "Thunder Wave",
        "dracometeor": "Draco Meteor",
        "closecombat": "Close Combat",
        "fakeout": "Fake Out",
        "partingshot": "Parting Shot",
        "knockoff": "Knock Off",
        "flareblitz": "Flare Blitz",
        "grassyglide": "Grassy Glide",
        "woodhammer": "Wood Hammer",
        "rockslide": "Rock Slide",
        "icywind": "Icy Wind",
        "helpinghand": "Helping Hand",
        "followme": "Follow Me",
        "ragepowder": "Rage Powder",
        "swordsdance": "Swords Dance",
        "dragondance": "Dragon Dance",
        "nastyplot": "Nasty Plot",
        "calmcind": "Calm Mind",
        "calmmind": "Calm Mind",
        "darkpulse": "Dark Pulse",
        "earthpower": "Earth Power",
        "energyball": "Energy Ball",
        "flashcannon": "Flash Cannon",
        "focusblast": "Focus Blast",
        "hypervoice": "Hyper Voice",
        "ironhead": "Iron Head",
        "leafstorm": "Leaf Storm",
        "moonblast": "Moonblast",
        "psychicterrain": "Psychic Terrain",
        "sludgebomb": "Sludge Bomb",
        "voltswitch": "Volt Switch",
        "wildcharge": "Wild Charge",
        "extremespeed": "Extreme Speed",
        "dragonpulse": "Dragon Pulse",
        "iciclespear": "Icicle Spear",
        "bodypress": "Body Press",
        "heavyslam": "Heavy Slam",
        "playrough": "Play Rough",
        "spiritbreak": "Spirit Break",
        "lightscreen": "Light Screen",
        "suckerunch": "Sucker Punch",
        "suckerpunch": "Sucker Punch",
        "wickedblow": "Wicked Blow",
        "surgingstrikes": "Surging Strikes",
        "aquajet": "Aqua Jet",
        "jetpunch": "Jet Punch",
        "wavecrash": "Wave Crash",
        "terablast": "Tera Blast",
        "electroshot": "Electro Shot",
        "kowtowcleave": "Kowtow Cleave",
        "ragefist": "Rage Fist",
        "drainpunch": "Drain Punch",
        "bulkup": "Bulk Up",
        "freezedry": "Freeze-Dry",
        "hydropump": "Hydro Pump",
        "headlongrush": "Headlong Rush",
        "icespinner": "Ice Spinner",
        "rapidspin": "Rapid Spin",
        "thunderclap": "Thunderclap",
        "makeitrain": "Make It Rain",
        "bleakwindstorm": "Bleakwind Storm",
        "sleeppowder": "Sleep Powder",
        "victorydance": "Victory Dance",
        "auroraveil": "Aurora Veil",
        "thunderbolt": "Thunderbolt",
    }
    
    move_lower = move.lower().replace(" ", "").replace("-", "")
    if move_lower in special_moves:
        return special_moves[move_lower]
    
    # Default: Title case
    return " ".join(word.title() for word in move.split())


def scrape_all_pokemon(limit: int = 120) -> Tuple[Dict[str, List[str]], Dict[str, str]]:
    """Scrape movesets and abilities for top Pokemon."""
    movesets: Dict[str, List[str]] = {}
    abilities: Dict[str, str] = {}
    
    # Get top Pokemon
    top_pokemon = get_top_pokemon(limit)
    
    if not top_pokemon:
        print("Failed to get Pokemon list. Using fallback...")
        return {}, {}
    
    print(f"Found {len(top_pokemon)} Pokemon. Fetching details...")
    
    for i, poke_data in enumerate(top_pokemon):
        name = poke_data.get("name", poke_data.get("pokemon", ""))
        if not name:
            continue
        
        formatted_name = format_pokemon_name(name)
        print(f"  [{i+1}/{len(top_pokemon)}] {formatted_name}...", end=" ")
        
        details = get_pokemon_details(name)
        
        if not details:
            print("skipped (no data)")
            continue
        
        # Get top 6 moves
        moves_data = details.get("moves", [])
        top_moves = []
        for move_entry in moves_data[:8]:  # Get top 8 to have backup
            move_name = move_entry.get("name", move_entry.get("move", ""))
            if move_name:
                formatted_move = format_move_name(move_name)
                if formatted_move not in top_moves:
                    top_moves.append(formatted_move)
            if len(top_moves) >= 6:
                break
        
        # Get top ability
        abilities_data = details.get("abilities", [])
        top_ability = ""
        if abilities_data:
            ability_entry = abilities_data[0]
            top_ability = ability_entry.get("name", ability_entry.get("ability", ""))
        
        if top_moves:
            movesets[formatted_name] = top_moves
            print(f"OK ({len(top_moves)} moves)")
        else:
            print("skipped (no moves)")
        
        if top_ability:
            abilities[formatted_name] = top_ability
        
        # Rate limiting
        time.sleep(0.3)
    
    return movesets, abilities


def generate_python_code(movesets: Dict[str, List[str]], abilities: Dict[str, str]) -> str:
    """Generate Python code for the movesets and abilities dictionaries."""
    lines = []
    
    # Movesets
    lines.append("# Pokemon-specific movesets (species -> list of typical moves)")
    lines.append("# Scraped from Pikalytics VGC 2024 Reg G data")
    lines.append("POKEMON_MOVESETS: Dict[str, List[str]] = {")
    
    for name, moves in sorted(movesets.items()):
        moves_str = ", ".join(f'"{m}"' for m in moves)
        lines.append(f'    "{name}": [{moves_str}],')
    
    lines.append("}")
    lines.append("")
    
    # Abilities
    lines.append("# Pokemon abilities (species -> primary ability)")
    lines.append("POKEMON_ABILITIES: Dict[str, str] = {")
    
    for name, ability in sorted(abilities.items()):
        lines.append(f'    "{name}": "{ability}",')
    
    lines.append("}")
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    print("=" * 60)
    print("Pikalytics VGC Moveset Scraper")
    print("=" * 60)
    
    movesets, abilities = scrape_all_pokemon(limit=120)
    
    if not movesets:
        print("\nNo data scraped. Check network connection.")
        return
    
    print(f"\nScraped {len(movesets)} Pokemon with movesets")
    print(f"Scraped {len(abilities)} Pokemon with abilities")
    
    # Generate Python code
    code = generate_python_code(movesets, abilities)
    
    # Save to file
    output_path = Path(__file__).parent.parent / "data" / "scraped_movesets.py"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write("# Auto-generated from Pikalytics VGC 2024 Reg G data\n")
        f.write("from typing import Dict, List\n\n")
        f.write(code)
    
    print(f"\nSaved to: {output_path}")
    
    # Also print summary
    print("\n" + "=" * 60)
    print("Top 10 Pokemon scraped:")
    for i, (name, moves) in enumerate(list(movesets.items())[:10]):
        print(f"  {i+1}. {name}: {moves[:4]}...")


if __name__ == "__main__":
    main()

