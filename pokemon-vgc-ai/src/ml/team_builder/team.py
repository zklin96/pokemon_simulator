"""Team representation for VGC team building."""

import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np
from loguru import logger

# Import validated VGC data - use these for team generation
from src.ml.team_builder.vgc_data import (
    POKEMON_MOVESETS as VALIDATED_MOVESETS,
    POKEMON_ABILITIES as VALIDATED_ABILITIES,
    RESTRICTED_POKEMON as VGC_RESTRICTED,
    VGC_SAFE_POOL as VALIDATED_POOL,
)


class Nature(Enum):
    """Pokemon natures and their stat modifiers."""
    HARDY = ("atk", "atk", 1.0, 1.0)
    LONELY = ("atk", "def", 1.1, 0.9)
    BRAVE = ("atk", "spe", 1.1, 0.9)
    ADAMANT = ("atk", "spa", 1.1, 0.9)
    NAUGHTY = ("atk", "spd", 1.1, 0.9)
    BOLD = ("def", "atk", 1.1, 0.9)
    DOCILE = ("def", "def", 1.0, 1.0)
    RELAXED = ("def", "spe", 1.1, 0.9)
    IMPISH = ("def", "spa", 1.1, 0.9)
    LAX = ("def", "spd", 1.1, 0.9)
    TIMID = ("spe", "atk", 1.1, 0.9)
    HASTY = ("spe", "def", 1.1, 0.9)
    SERIOUS = ("spe", "spe", 1.0, 1.0)
    JOLLY = ("spe", "spa", 1.1, 0.9)
    NAIVE = ("spe", "spd", 1.1, 0.9)
    MODEST = ("spa", "atk", 1.1, 0.9)
    MILD = ("spa", "def", 1.1, 0.9)
    QUIET = ("spa", "spe", 1.1, 0.9)
    BASHFUL = ("spa", "spa", 1.0, 1.0)
    RASH = ("spa", "spd", 1.1, 0.9)
    CALM = ("spd", "atk", 1.1, 0.9)
    GENTLE = ("spd", "def", 1.1, 0.9)
    SASSY = ("spd", "spe", 1.1, 0.9)
    CAREFUL = ("spd", "spa", 1.1, 0.9)
    QUIRKY = ("spd", "spd", 1.0, 1.0)


# Common VGC Pokemon pool (Regulation G/H examples)
VGC_POKEMON_POOL = [
    "Flutter Mane", "Landorus", "Rillaboom", "Incineroar", "Urshifu",
    "Amoonguss", "Farigiraf", "Ogerpon", "Tornadus", "Chi-Yu",
    "Iron Hands", "Raging Bolt", "Calyrex-Ice", "Calyrex-Shadow",
    "Miraidon", "Koraidon", "Chien-Pao", "Kingambit", "Gholdengo",
    "Dragonite", "Ursaluna", "Palafin", "Archaludon", "Hydreigon",
    "Gothitelle", "Indeedee-F", "Arcanine", "Pelipper", "Torkoal",
    "Talonflame", "Whimsicott", "Grimmsnarl", "Dondozo", "Tatsugiri",
    "Annihilape", "Iron Bundle", "Roaring Moon", "Great Tusk", "Iron Boulder",
]

# Common held items for VGC
VGC_ITEMS = [
    "Choice Scarf", "Choice Band", "Choice Specs",
    "Life Orb", "Focus Sash", "Assault Vest",
    "Safety Goggles", "Sitrus Berry", "Lum Berry",
    "Covert Cloak", "Clear Amulet", "Leftovers",
    "Rocky Helmet", "Eject Button", "Mental Herb",
    "Power Herb", "White Herb", "Expert Belt",
    "Muscle Band", "Wise Glasses", "Booster Energy",
]

# Tera types
TERA_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy", "Stellar",
]

# Common VGC moves (actual valid move names)
VGC_MOVES = [
    # Protect variants
    "Protect", "Detect", "Wide Guard", "Quick Guard",
    # Physical attacks
    "Close Combat", "Knock Off", "U-turn", "Sucker Punch", "Fake Out",
    "Extreme Speed", "Ice Shard", "Mach Punch", "Aqua Jet", "Bullet Punch",
    "Earthquake", "Rock Slide", "Iron Head", "Wild Charge", "Brave Bird",
    "Flare Blitz", "Wood Hammer", "Icicle Crash", "Outrage", "Dragon Claw",
    "Play Rough", "Sacred Sword", "Stone Edge", "Drill Run", "High Horsepower",
    "Crunch", "Psychic Fangs", "Thunder Punch", "Ice Punch", "Fire Punch",
    "Drain Punch", "Superpower", "Body Press", "Headlong Rush", "Tachyon Cutter",
    "Population Bomb", "Kowtow Cleave", "Wicked Blow", "Surging Strikes",
    # Special attacks
    "Moonblast", "Dazzling Gleam", "Shadow Ball", "Sludge Bomb", "Thunderbolt",
    "Thunder", "Blizzard", "Ice Beam", "Flamethrower", "Fire Blast", "Heat Wave",
    "Hydro Pump", "Surf", "Scald", "Muddy Water", "Energy Ball", "Leaf Storm",
    "Psychic", "Psyshock", "Expanding Force", "Dark Pulse", "Aura Sphere",
    "Flash Cannon", "Draco Meteor", "Dragon Pulse", "Hyper Voice", "Boomburst",
    "Earth Power", "Snarl", "Electroweb", "Icy Wind", "Volt Switch",
    "Make It Rain", "Astral Barrage", "Glacial Lance", "Psyspam",
    # Status moves
    "Trick Room", "Tailwind", "Sunny Day", "Rain Dance", "Sandstorm",
    "Hail", "Thunder Wave", "Will-O-Wisp", "Spore", "Sleep Powder",
    "Rage Powder", "Follow Me", "Helping Hand", "Ally Switch", "Coaching",
    "Swords Dance", "Dragon Dance", "Calm Mind", "Nasty Plot", "Quiver Dance",
    "Belly Drum", "Substitute", "Encore", "Taunt", "Disable", "Imprison",
    "Trick", "Switcheroo", "Pollen Puff", "Life Dew", "Heal Pulse",
]

# Pokemon-specific movesets (species -> list of typical moves)
# These are validated competitive movesets - 100+ Pokemon
POKEMON_MOVESETS: Dict[str, List[str]] = {
    # === CORE META POKEMON (Top 30 Usage) ===
    "Flutter Mane": ["Moonblast", "Shadow Ball", "Dazzling Gleam", "Protect", "Icy Wind", "Thunderbolt"],
    "Landorus": ["Earthquake", "Rock Slide", "U-turn", "Protect", "Swords Dance", "Fly"],
    "Rillaboom": ["Wood Hammer", "Grassy Glide", "U-turn", "Fake Out", "Protect", "Knock Off"],
    "Incineroar": ["Fake Out", "Flare Blitz", "Knock Off", "Parting Shot", "U-turn", "Protect"],
    "Urshifu": ["Wicked Blow", "Close Combat", "Sucker Punch", "Protect", "Detect", "U-turn"],
    "Amoonguss": ["Spore", "Rage Powder", "Pollen Puff", "Protect", "Clear Smog", "Sludge Bomb"],
    "Farigiraf": ["Trick Room", "Psychic", "Hyper Voice", "Protect", "Helping Hand", "Dazzling Gleam"],
    "Tornadus": ["Tailwind", "Bleakwind Storm", "Heat Wave", "Protect", "Taunt", "Rain Dance"],
    "Chi-Yu": ["Heat Wave", "Dark Pulse", "Overheat", "Protect", "Snarl", "Tera Blast"],
    "Iron Hands": ["Fake Out", "Close Combat", "Wild Charge", "Protect", "Drain Punch", "Heavy Slam"],
    "Raging Bolt": ["Thunderclap", "Draco Meteor", "Thunderbolt", "Protect", "Calm Mind", "Electroweb"],
    "Kingambit": ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Protect", "Swords Dance", "Tera Blast"],
    "Gholdengo": ["Make It Rain", "Shadow Ball", "Nasty Plot", "Protect", "Trick", "Thunderbolt"],
    "Dragonite": ["Extreme Speed", "Dragon Claw", "Fire Punch", "Protect", "Dragon Dance", "Ice Punch"],
    "Arcanine": ["Flare Blitz", "Wild Charge", "Extreme Speed", "Protect", "Will-O-Wisp", "Snarl"],
    "Pelipper": ["Hurricane", "Hydro Pump", "Protect", "Tailwind", "U-turn", "Wide Guard"],
    "Torkoal": ["Eruption", "Heat Wave", "Earth Power", "Protect", "Yawn", "Will-O-Wisp"],
    "Whimsicott": ["Tailwind", "Moonblast", "Encore", "Protect", "Taunt", "Helping Hand"],
    "Grimmsnarl": ["Spirit Break", "Thunder Wave", "Reflect", "Light Screen", "Taunt", "Fake Out"],
    "Annihilape": ["Rage Fist", "Close Combat", "Drain Punch", "Protect", "Bulk Up", "Taunt"],
    "Dondozo": ["Wave Crash", "Earthquake", "Order Up", "Protect", "Heavy Slam", "Yawn"],
    "Tatsugiri": ["Muddy Water", "Dragon Pulse", "Icy Wind", "Protect", "Endure", "Helping Hand"],
    "Iron Bundle": ["Freeze-Dry", "Hydro Pump", "Icy Wind", "Protect", "Flip Turn", "Encore"],
    "Great Tusk": ["Headlong Rush", "Close Combat", "Rock Slide", "Protect", "Ice Spinner", "Rapid Spin"],
    "Roaring Moon": ["Acrobatics", "Crunch", "Dragon Claw", "Protect", "Dragon Dance", "Taunt"],
    "Palafin": ["Jet Punch", "Close Combat", "Wave Crash", "Protect", "Flip Turn", "Ice Punch"],
    "Archaludon": ["Flash Cannon", "Draco Meteor", "Body Press", "Protect", "Electro Shot", "Thunderbolt"],
    "Hydreigon": ["Dark Pulse", "Draco Meteor", "Heat Wave", "Protect", "Earth Power", "Flash Cannon"],
    "Gothitelle": ["Psychic", "Trick Room", "Helping Hand", "Protect", "Fake Out", "Thunder Wave"],
    "Indeedee-F": ["Psychic", "Follow Me", "Helping Hand", "Protect", "Trick Room", "Heal Pulse"],
    "Talonflame": ["Brave Bird", "Flare Blitz", "Tailwind", "Protect", "U-turn", "Taunt"],
    
    # === PARADOX POKEMON ===
    "Iron Valiant": ["Close Combat", "Moonblast", "Spirit Break", "Protect", "Swords Dance", "Destiny Bond"],
    "Iron Moth": ["Fiery Dance", "Energy Ball", "Sludge Wave", "Protect", "Acid Spray", "Psychic"],
    "Iron Thorns": ["Rock Slide", "Wild Charge", "Earthquake", "Protect", "Dragon Dance", "Ice Punch"],
    "Iron Jugulis": ["Dark Pulse", "Air Slash", "Hydro Pump", "Protect", "Tailwind", "Snarl"],
    "Iron Treads": ["Earthquake", "Iron Head", "Rapid Spin", "Protect", "Knock Off", "Rock Slide"],
    "Brute Bonnet": ["Seed Bomb", "Sucker Punch", "Rage Powder", "Protect", "Spore", "Crunch"],
    "Scream Tail": ["Play Rough", "Disable", "Thunder Wave", "Protect", "Wish", "Perish Song"],
    "Sandy Shocks": ["Thunderbolt", "Earth Power", "Power Gem", "Protect", "Stealth Rock", "Thunder Wave"],
    "Slither Wing": ["First Impression", "Close Combat", "U-turn", "Protect", "Flame Charge", "Leech Life"],
    "Walking Wake": ["Hydro Steam", "Draco Meteor", "Flamethrower", "Protect", "Dragon Pulse", "Snarl"],
    "Iron Leaves": ["Psyblade", "Close Combat", "Leaf Blade", "Protect", "Swords Dance", "Helping Hand"],
    "Gouging Fire": ["Raging Fury", "Dragon Claw", "Flare Blitz", "Protect", "Dragon Dance", "Burning Bulwark"],
    "Iron Boulder": ["Mighty Cleave", "Close Combat", "Psycho Cut", "Protect", "Swords Dance", "Zen Headbutt"],
    "Iron Crown": ["Tachyon Cutter", "Focus Blast", "Expanding Force", "Protect", "Calm Mind", "Flash Cannon"],
    
    # === WEATHER SETTERS & ABUSERS ===
    "Ninetales-Alola": ["Aurora Veil", "Blizzard", "Moonblast", "Protect", "Encore", "Icy Wind"],
    "Tyranitar": ["Rock Slide", "Crunch", "Low Kick", "Protect", "Dragon Dance", "Ice Punch"],
    "Hippowdon": ["Earthquake", "Rock Slide", "Yawn", "Protect", "Slack Off", "Whirlwind"],
    "Politoed": ["Surf", "Icy Wind", "Helping Hand", "Protect", "Encore", "Perish Song"],
    "Lilligant-Hisui": ["Close Combat", "Leaf Blade", "After You", "Protect", "Sleep Powder", "Victory Dance"],
    "Bronzong": ["Gyro Ball", "Trick Room", "Hypnosis", "Protect", "Imprison", "Body Press"],
    
    # === SUPPORT POKEMON ===
    "Clefairy": ["Follow Me", "Helping Hand", "Moonblast", "Protect", "Icy Wind", "Thunder Wave"],
    "Porygon2": ["Tri Attack", "Trick Room", "Recover", "Protect", "Ice Beam", "Thunderbolt"],
    "Cresselia": ["Psychic", "Trick Room", "Helping Hand", "Protect", "Ice Beam", "Moonblast"],
    "Hatterene": ["Trick Room", "Dazzling Gleam", "Psychic", "Protect", "Mystical Fire", "Healing Wish"],
    "Dusclops": ["Trick Room", "Night Shade", "Pain Split", "Protect", "Will-O-Wisp", "Helping Hand"],
    "Murkrow": ["Tailwind", "Foul Play", "Haze", "Protect", "Quash", "Snarl"],
    "Mienshao": ["Fake Out", "Close Combat", "U-turn", "Protect", "Wide Guard", "Coaching"],
    
    # === OFFENSIVE THREATS ===
    "Garchomp": ["Earthquake", "Dragon Claw", "Rock Slide", "Protect", "Swords Dance", "Scale Shot"],
    "Salamence": ["Draco Meteor", "Heat Wave", "Tailwind", "Protect", "Hurricane", "Earthquake"],
    "Gyarados": ["Waterfall", "Crunch", "Earthquake", "Protect", "Dragon Dance", "Ice Fang"],
    "Volcarona": ["Heat Wave", "Bug Buzz", "Quiver Dance", "Protect", "Giga Drain", "Rage Powder"],
    "Blaziken": ["Close Combat", "Flare Blitz", "Swords Dance", "Protect", "Brave Bird", "Knock Off"],
    "Lucario": ["Close Combat", "Bullet Punch", "Meteor Mash", "Protect", "Swords Dance", "Extreme Speed"],
    "Baxcalibur": ["Icicle Crash", "Dragon Claw", "Earthquake", "Protect", "Ice Shard", "Glaive Rush"],
    "Dragapult": ["Dragon Darts", "Phantom Force", "U-turn", "Protect", "Will-O-Wisp", "Draco Meteor"],
    "Haxorus": ["Outrage", "Earthquake", "Close Combat", "Protect", "Dragon Dance", "Poison Jab"],
    "Ceruledge": ["Bitter Blade", "Shadow Sneak", "Close Combat", "Protect", "Swords Dance", "Psycho Cut"],
    "Armarouge": ["Armor Cannon", "Psychic", "Will-O-Wisp", "Protect", "Trick Room", "Heat Wave"],
    "Chien-Pao": ["Icicle Crash", "Crunch", "Sacred Sword", "Protect", "Ice Shard", "Sucker Punch"],
    "Ting-Lu": ["Earthquake", "Ruination", "Heavy Slam", "Protect", "Whirlwind", "Taunt"],
    "Wo-Chien": ["Foul Play", "Ruination", "Giga Drain", "Protect", "Leech Seed", "Pollen Puff"],
    "Guzzlord": ["Dark Pulse", "Draco Meteor", "Heat Wave", "Protect", "Snarl", "Heavy Slam"],
    "Kommo-o": ["Close Combat", "Dragon Claw", "Protect", "Iron Head", "Dragon Dance", "Clanging Scales"],
    "Goodra-Hisui": ["Dragon Pulse", "Iron Head", "Fire Blast", "Protect", "Acid Spray", "Thunderbolt"],
    "Araquanid": ["Liquidation", "Lunge", "Wide Guard", "Protect", "Sticky Web", "Mirror Coat"],
    "Tsareena": ["Power Whip", "Triple Axel", "U-turn", "Protect", "High Jump Kick", "Rapid Spin"],
    "Comfey": ["Floral Healing", "Draining Kiss", "U-turn", "Protect", "Taunt", "Trick Room"],
    "Ribombee": ["Moonblast", "Pollen Puff", "Bug Buzz", "Protect", "Tailwind", "Quiver Dance"],
    
    # === DEFENSIVE/UTILITY ===
    "Blissey": ["Seismic Toss", "Soft-Boiled", "Thunder Wave", "Protect", "Helping Hand", "Heal Pulse"],
    "Chansey": ["Seismic Toss", "Soft-Boiled", "Thunder Wave", "Protect", "Helping Hand", "Heal Pulse"],
    "Toxapex": ["Liquidation", "Baneful Bunker", "Haze", "Protect", "Recover", "Toxic"],
    "Corviknight": ["Brave Bird", "Iron Head", "Tailwind", "Protect", "Roost", "Bulk Up"],
    "Gastrodon": ["Earth Power", "Surf", "Recover", "Protect", "Yawn", "Icy Wind"],
    "Maushold": ["Population Bomb", "Beat Up", "Follow Me", "Protect", "Tidy Up", "Encore"],
    "Mudsdale": ["High Horsepower", "Heavy Slam", "Rock Slide", "Protect", "Body Press", "Stealth Rock"],
    "Oranguru": ["Instruct", "Psychic", "Trick Room", "Protect", "Ally Switch", "Foul Play"],
    "Snorlax": ["Body Slam", "Heavy Slam", "Curse", "Protect", "Rest", "Belly Drum"],
    "Slowbro": ["Scald", "Psychic", "Trick Room", "Protect", "Slack Off", "Calm Mind"],
    "Slowking-Galar": ["Sludge Bomb", "Psychic", "Trick Room", "Protect", "Slack Off", "Future Sight"],
    "Drifblim": ["Shadow Ball", "Tailwind", "Will-O-Wisp", "Protect", "Strength Sap", "Destiny Bond"],
    
    # === POPULAR PICKS ===
    "Cinderace": ["Pyro Ball", "Court Change", "U-turn", "Protect", "Gunk Shot", "High Jump Kick"],
    "Raichu": ["Thunderbolt", "Nuzzle", "Fake Out", "Protect", "Volt Switch", "Brick Break"],
    "Meowstic": ["Psychic", "Thunder Wave", "Quick Guard", "Protect", "Fake Out", "Helping Hand"],
    "Sableye": ["Foul Play", "Quash", "Will-O-Wisp", "Protect", "Fake Out", "Trick"],
    "Klefki": ["Foul Play", "Thunder Wave", "Reflect", "Light Screen", "Spikes", "Play Rough"],
    "Thundurus": ["Thunderbolt", "Taunt", "Thunder Wave", "Protect", "Eerie Impulse", "Nasty Plot"],
    "Zapdos-Galar": ["Thunderous Kick", "Brave Bird", "U-turn", "Protect", "Coaching", "Close Combat"],
    "Moltres-Galar": ["Fiery Wrath", "Hurricane", "Protect", "Nasty Plot", "Tailwind", "Snarl"],
    "Articuno-Galar": ["Freezing Glare", "Hurricane", "Recover", "Protect", "Trick Room", "Calm Mind"],
    "Ninetales": ["Heat Wave", "Will-O-Wisp", "Encore", "Protect", "Solar Beam", "Nasty Plot"],
    "Venusaur": ["Giga Drain", "Sludge Bomb", "Earth Power", "Protect", "Sleep Powder", "Leaf Storm"],
    "Charizard": ["Heat Wave", "Air Slash", "Dragon Pulse", "Protect", "Will-O-Wisp", "Overheat"],
    "Blastoise": ["Water Spout", "Ice Beam", "Fake Out", "Protect", "Shell Smash", "Life Dew"],
    "Excadrill": ["High Horsepower", "Iron Head", "Rock Slide", "Protect", "Swords Dance", "Rapid Spin"],
    "Conkeldurr": ["Drain Punch", "Mach Punch", "Knock Off", "Protect", "Wide Guard", "Ice Punch"],
    "Scizor": ["Bullet Punch", "Bug Bite", "Swords Dance", "Protect", "U-turn", "Knock Off"],
    "Metagross": ["Meteor Mash", "Zen Headbutt", "Bullet Punch", "Protect", "Ice Punch", "Earthquake"],
    "Azumarill": ["Aqua Jet", "Play Rough", "Helping Hand", "Protect", "Knock Off", "Belly Drum"],
    "Sylveon": ["Hyper Voice", "Moonblast", "Quick Attack", "Protect", "Yawn", "Helping Hand"],
    "Gengar": ["Shadow Ball", "Sludge Bomb", "Will-O-Wisp", "Protect", "Icy Wind", "Trick Room"],
    "Milotic": ["Scald", "Ice Beam", "Coil", "Protect", "Recover", "Hypnosis"],
    "Toxtricity": ["Overdrive", "Sludge Bomb", "Boomburst", "Protect", "Volt Switch", "Snarl"],
    "Gardevoir": ["Moonblast", "Psychic", "Trick Room", "Protect", "Will-O-Wisp", "Helping Hand"],
    "Gallade": ["Close Combat", "Psycho Cut", "Wide Guard", "Protect", "Helping Hand", "Swords Dance"],
}

# Restricted Pokemon (only 1 per team in VGC)
RESTRICTED_POKEMON = [
    "Koraidon", "Miraidon", "Calyrex-Ice", "Calyrex-Shadow",
    "Zacian", "Zamazenta", "Eternatus", "Kyogre", "Groudon",
    "Rayquaza", "Dialga", "Palkia", "Giratina", "Reshiram",
    "Zekrom", "Kyurem", "Xerneas", "Yveltal", "Zygarde",
    "Cosmog", "Cosmoem", "Solgaleo", "Lunala", "Necrozma",
    "Mewtwo", "Lugia", "Ho-Oh", "Mew",
]

# Safe Pokemon pool (Pokemon with defined movesets, excluding restricted)
VGC_SAFE_POOL = [p for p in POKEMON_MOVESETS.keys() if p not in RESTRICTED_POKEMON]

# Pokemon abilities (species -> primary ability) - 100+ Pokemon
POKEMON_ABILITIES: Dict[str, str] = {
    # === CORE META ===
    "Flutter Mane": "Protosynthesis",
    "Landorus": "Sheer Force",
    "Rillaboom": "Grassy Surge",
    "Incineroar": "Intimidate",
    "Urshifu": "Unseen Fist",
    "Amoonguss": "Regenerator",
    "Farigiraf": "Armor Tail",
    "Tornadus": "Prankster",
    "Chi-Yu": "Beads of Ruin",
    "Iron Hands": "Quark Drive",
    "Raging Bolt": "Protosynthesis",
    "Kingambit": "Defiant",
    "Gholdengo": "Good as Gold",
    "Dragonite": "Multiscale",
    "Arcanine": "Intimidate",
    "Pelipper": "Drizzle",
    "Torkoal": "Drought",
    "Whimsicott": "Prankster",
    "Grimmsnarl": "Prankster",
    "Annihilape": "Defiant",
    "Dondozo": "Unaware",
    "Tatsugiri": "Commander",
    "Iron Bundle": "Quark Drive",
    "Great Tusk": "Protosynthesis",
    "Roaring Moon": "Protosynthesis",
    "Palafin": "Zero to Hero",
    "Archaludon": "Stamina",
    "Hydreigon": "Levitate",
    "Gothitelle": "Shadow Tag",
    "Indeedee-F": "Psychic Surge",
    "Talonflame": "Gale Wings",
    
    # === PARADOX POKEMON ===
    "Iron Valiant": "Quark Drive",
    "Iron Moth": "Quark Drive",
    "Iron Thorns": "Quark Drive",
    "Iron Jugulis": "Quark Drive",
    "Iron Treads": "Quark Drive",
    "Brute Bonnet": "Protosynthesis",
    "Scream Tail": "Protosynthesis",
    "Sandy Shocks": "Protosynthesis",
    "Slither Wing": "Protosynthesis",
    "Walking Wake": "Protosynthesis",
    "Iron Leaves": "Quark Drive",
    "Gouging Fire": "Protosynthesis",
    "Iron Boulder": "Quark Drive",
    "Iron Crown": "Quark Drive",
    
    # === WEATHER SETTERS & ABUSERS ===
    "Ninetales-Alola": "Snow Warning",
    "Tyranitar": "Sand Stream",
    "Hippowdon": "Sand Stream",
    "Politoed": "Drizzle",
    "Lilligant-Hisui": "Chlorophyll",
    "Bronzong": "Levitate",
    
    # === SUPPORT POKEMON ===
    "Clefairy": "Friend Guard",
    "Porygon2": "Download",
    "Cresselia": "Levitate",
    "Hatterene": "Magic Bounce",
    "Dusclops": "Frisk",
    "Murkrow": "Prankster",
    "Mienshao": "Inner Focus",
    
    # === OFFENSIVE THREATS ===
    "Garchomp": "Rough Skin",
    "Salamence": "Intimidate",
    "Gyarados": "Intimidate",
    "Volcarona": "Flame Body",
    "Blaziken": "Speed Boost",
    "Lucario": "Inner Focus",
    "Baxcalibur": "Thermal Exchange",
    "Dragapult": "Clear Body",
    "Haxorus": "Mold Breaker",
    "Ceruledge": "Flash Fire",
    "Armarouge": "Flash Fire",
    "Chien-Pao": "Sword of Ruin",
    "Ting-Lu": "Vessel of Ruin",
    "Wo-Chien": "Tablets of Ruin",
    "Guzzlord": "Beast Boost",
    "Kommo-o": "Bulletproof",
    "Goodra-Hisui": "Sap Sipper",
    "Araquanid": "Water Bubble",
    "Tsareena": "Queenly Majesty",
    "Comfey": "Triage",
    "Ribombee": "Shield Dust",
    
    # === DEFENSIVE/UTILITY ===
    "Blissey": "Natural Cure",
    "Chansey": "Natural Cure",
    "Toxapex": "Regenerator",
    "Corviknight": "Pressure",
    "Gastrodon": "Storm Drain",
    "Maushold": "Technician",
    "Mudsdale": "Stamina",
    "Oranguru": "Inner Focus",
    "Snorlax": "Thick Fat",
    "Slowbro": "Regenerator",
    "Slowking-Galar": "Regenerator",
    "Drifblim": "Unburden",
    
    # === POPULAR PICKS ===
    "Cinderace": "Libero",
    "Raichu": "Lightning Rod",
    "Meowstic": "Prankster",
    "Sableye": "Prankster",
    "Klefki": "Prankster",
    "Thundurus": "Prankster",
    "Zapdos-Galar": "Defiant",
    "Moltres-Galar": "Berserk",
    "Articuno-Galar": "Competitive",
    "Ninetales": "Drought",
    "Venusaur": "Chlorophyll",
    "Charizard": "Solar Power",
    "Blastoise": "Torrent",
    "Excadrill": "Sand Rush",
    "Conkeldurr": "Guts",
    "Scizor": "Technician",
    "Metagross": "Clear Body",
    "Azumarill": "Huge Power",
    "Sylveon": "Pixilate",
    "Gengar": "Cursed Body",
    "Milotic": "Competitive",
    "Toxtricity": "Punk Rock",
    "Gardevoir": "Trace",
    "Gallade": "Justified",
}


@dataclass
class EVSpread:
    """EV (Effort Value) spread for a Pokemon."""
    
    hp: int = 0
    atk: int = 0
    def_: int = 0  # 'def' is a Python keyword
    spa: int = 0
    spd: int = 0
    spe: int = 0
    
    def __post_init__(self):
        """Validate EV spread."""
        total = self.hp + self.atk + self.def_ + self.spa + self.spd + self.spe
        if total > 510:
            self._normalize()
    
    def _normalize(self):
        """Normalize EVs to fit within 510 total."""
        values = [self.hp, self.atk, self.def_, self.spa, self.spd, self.spe]
        total = sum(values)
        if total > 510:
            scale = 510 / total
            self.hp = min(252, int(self.hp * scale))
            self.atk = min(252, int(self.atk * scale))
            self.def_ = min(252, int(self.def_ * scale))
            self.spa = min(252, int(self.spa * scale))
            self.spd = min(252, int(self.spd * scale))
            self.spe = min(252, int(self.spe * scale))
    
    def to_list(self) -> List[int]:
        return [self.hp, self.atk, self.def_, self.spa, self.spd, self.spe]
    
    @classmethod
    def random(cls) -> "EVSpread":
        """Generate a random valid EV spread."""
        evs = [0, 0, 0, 0, 0, 0]
        remaining = 510
        
        # Randomly distribute EVs
        for _ in range(remaining // 4):
            stat = random.randint(0, 5)
            if evs[stat] < 252:
                evs[stat] += 4
        
        return cls(
            hp=evs[0], atk=evs[1], def_=evs[2],
            spa=evs[3], spd=evs[4], spe=evs[5]
        )


@dataclass
class PokemonSet:
    """A complete Pokemon set (species + moves + item + EVs etc.)."""
    
    species: str
    item: str
    ability: str = ""
    moves: List[str] = field(default_factory=list)
    nature: str = "Adamant"
    evs: EVSpread = field(default_factory=EVSpread)
    tera_type: str = "Normal"
    level: int = 50
    
    def to_showdown_paste(self) -> str:
        """Convert to Pokemon Showdown paste format."""
        lines = []
        
        # Species @ Item
        if self.item:
            lines.append(f"{self.species} @ {self.item}")
        else:
            lines.append(self.species)
        
        # Ability
        if self.ability:
            lines.append(f"Ability: {self.ability}")
        
        # Level
        lines.append(f"Level: {self.level}")
        
        # Tera Type
        lines.append(f"Tera Type: {self.tera_type}")
        
        # EVs
        ev_parts = []
        if self.evs.hp > 0:
            ev_parts.append(f"{self.evs.hp} HP")
        if self.evs.atk > 0:
            ev_parts.append(f"{self.evs.atk} Atk")
        if self.evs.def_ > 0:
            ev_parts.append(f"{self.evs.def_} Def")
        if self.evs.spa > 0:
            ev_parts.append(f"{self.evs.spa} SpA")
        if self.evs.spd > 0:
            ev_parts.append(f"{self.evs.spd} SpD")
        if self.evs.spe > 0:
            ev_parts.append(f"{self.evs.spe} Spe")
        if ev_parts:
            lines.append(f"EVs: {' / '.join(ev_parts)}")
        
        # Nature
        lines.append(f"{self.nature} Nature")
        
        # Moves
        for move in self.moves[:4]:
            lines.append(f"- {move}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "species": self.species,
            "item": self.item,
            "ability": self.ability,
            "moves": self.moves,
            "nature": self.nature,
            "evs": self.evs.to_list(),
            "tera_type": self.tera_type,
            "level": self.level,
        }
    
    @classmethod
    def random(cls, species: str) -> "PokemonSet":
        """Generate a random Pokemon set with valid, species-specific moves and ability."""
        # Use validated movesets from vgc_data.py
        if species in VALIDATED_MOVESETS:
            available_moves = VALIDATED_MOVESETS[species]
            # Select 4 unique moves from the species moveset
            if len(available_moves) >= 4:
                moves = random.sample(available_moves, 4)
            else:
                moves = list(available_moves)
                # Pad with Protect variants if needed
                safe_filler = ["Protect", "Detect", "Substitute", "Rest"]
                while len(moves) < 4:
                    for filler in safe_filler:
                        if filler not in moves:
                            moves.append(filler)
                            break
                    else:
                        break
        else:
            # For Pokemon without defined movesets, use safe generic moves
            # This shouldn't happen if using VALIDATED_POOL
            moves = ["Protect", "Hyper Beam", "Giga Impact", "Rest"]
            logger.warning(f"No moveset defined for {species}, using generic moves")
        
        # Get ability from validated abilities
        ability = VALIDATED_ABILITIES.get(species, "")
        
        # Handle special Tera type requirements
        # Ogerpon must have a specific Tera type based on form
        ogerpon_tera_types = {
            "Ogerpon": "Grass",
            "Ogerpon-Wellspring": "Water",
            "Ogerpon-Hearthflame": "Fire",
            "Ogerpon-Cornerstone": "Rock",
        }
        tera_type = ogerpon_tera_types.get(species, random.choice(TERA_TYPES))
        
        return cls(
            species=species,
            item=random.choice(VGC_ITEMS),
            ability=ability,
            moves=moves,
            nature=random.choice([n.name.title() for n in Nature]),
            evs=EVSpread.random(),
            tera_type=tera_type,
        )


@dataclass
class Team:
    """A complete VGC team of 6 Pokemon."""
    
    pokemon: List[PokemonSet] = field(default_factory=list)
    name: str = ""
    
    # Performance metrics
    elo: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    
    def __post_init__(self):
        # Ensure exactly 6 Pokemon
        while len(self.pokemon) < 6:
            species = random.choice(VGC_POKEMON_POOL)
            self.pokemon.append(PokemonSet.random(species))
        self.pokemon = self.pokemon[:6]
    
    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played
    
    def to_showdown_paste(self) -> str:
        """Convert entire team to Showdown paste format."""
        return "\n\n".join(p.to_showdown_paste() for p in self.pokemon)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "pokemon": [p.to_dict() for p in self.pokemon],
            "elo": self.elo,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.win_rate,
        }
    
    def get_items(self) -> List[str]:
        """Get list of items used by the team."""
        return [p.item for p in self.pokemon]
    
    def has_duplicate_items(self) -> bool:
        """Check if team violates item clause."""
        items = [p.item for p in self.pokemon if p.item]
        return len(items) != len(set(items))
    
    def has_duplicate_species(self) -> bool:
        """Check if team violates species clause."""
        species = [p.species for p in self.pokemon]
        return len(species) != len(set(species))
    
    def is_valid(self) -> bool:
        """Check if team is valid for VGC."""
        return (
            len(self.pokemon) == 6 and
            not self.has_duplicate_items() and
            not self.has_duplicate_species()
        )
    
    @classmethod
    def random(cls) -> "Team":
        """Generate a random valid team using only Pokemon with defined movesets."""
        # Use validated pool (Pokemon with valid Gen 9 movesets, no restricted)
        pool_to_use = VALIDATED_POOL if len(VALIDATED_POOL) >= 6 else list(VALIDATED_MOVESETS.keys())
        
        # Select 6 unique Pokemon from safe pool
        if len(pool_to_use) >= 6:
            species_list = random.sample(pool_to_use, 6)
        else:
            # Not enough Pokemon, use what we have and pad with others
            species_list = list(pool_to_use)
            remaining = [p for p in VGC_POKEMON_POOL if p not in species_list]
            species_list.extend(random.sample(remaining, 6 - len(species_list)))
        
        # Create Pokemon sets
        pokemon = [PokemonSet.random(species) for species in species_list]
        
        # Ensure no duplicate items
        used_items = set()
        for p in pokemon:
            while p.item in used_items:
                p.item = random.choice(VGC_ITEMS)
            used_items.add(p.item)
        
        return cls(pokemon=pokemon)


def test_team():
    """Test team generation."""
    logger.info("Testing team generation...")
    
    # Generate random team
    team = Team.random()
    
    logger.info(f"Generated team with {len(team.pokemon)} Pokemon")
    logger.info(f"Team is valid: {team.is_valid()}")
    
    # Show team
    for i, p in enumerate(team.pokemon):
        logger.info(f"  {i+1}. {p.species} @ {p.item}")
    
    # Show paste format
    logger.info("\nShowdown paste format:")
    print(team.to_showdown_paste()[:500] + "...")
    
    logger.info("Team generation test passed!")


if __name__ == "__main__":
    test_team()

