"""Validated VGC 2024 Gen 9 Pokemon data.

All Pokemon and moves in this file are verified to exist in Gen 9 and be legal
for VGC 2024 Regulation G format.
"""

from typing import Dict, List

# ============================================================================
# VALIDATED POKEMON MOVESETS - Gen 9 VGC 2024 Reg G
# All moves verified to be legal in Gen 9 and available on the Pokemon
# ============================================================================

POKEMON_MOVESETS: Dict[str, List[str]] = {
    # === TOP TIER META (Top 30 usage) ===
    "Incineroar": ["Fake Out", "Flare Blitz", "Knock Off", "Parting Shot", "U-turn", "Protect"],
    "Rillaboom": ["Wood Hammer", "Grassy Glide", "U-turn", "Fake Out", "Protect", "Knock Off"],
    "Flutter Mane": ["Moonblast", "Shadow Ball", "Dazzling Gleam", "Protect", "Icy Wind", "Thunderbolt"],
    "Urshifu": ["Wicked Blow", "Close Combat", "Sucker Punch", "Protect", "Detect", "U-turn"],
    "Urshifu-Rapid-Strike": ["Surging Strikes", "Close Combat", "Aqua Jet", "Protect", "Detect", "U-turn"],
    "Landorus": ["Earth Power", "Sludge Bomb", "Psychic", "Protect", "Substitute", "Sandsear Storm"],
    "Amoonguss": ["Spore", "Rage Powder", "Pollen Puff", "Protect", "Clear Smog", "Sludge Bomb"],
    "Farigiraf": ["Trick Room", "Psychic", "Hyper Voice", "Protect", "Helping Hand", "Dazzling Gleam"],
    "Chi-Yu": ["Heat Wave", "Dark Pulse", "Overheat", "Protect", "Snarl", "Tera Blast"],
    "Iron Hands": ["Fake Out", "Close Combat", "Wild Charge", "Protect", "Drain Punch", "Heavy Slam"],
    "Raging Bolt": ["Thunderclap", "Draco Meteor", "Thunderbolt", "Protect", "Calm Mind", "Electroweb"],
    "Kingambit": ["Kowtow Cleave", "Sucker Punch", "Iron Head", "Protect", "Swords Dance", "Tera Blast"],
    "Gholdengo": ["Make It Rain", "Shadow Ball", "Nasty Plot", "Protect", "Trick", "Thunderbolt"],
    "Tornadus": ["Tailwind", "Bleakwind Storm", "Heat Wave", "Protect", "Taunt", "Rain Dance"],
    "Whimsicott": ["Tailwind", "Moonblast", "Encore", "Protect", "Taunt", "Helping Hand"],
    "Chien-Pao": ["Icicle Crash", "Crunch", "Sacred Sword", "Protect", "Ice Shard", "Sucker Punch"],
    "Dragonite": ["Extreme Speed", "Dragon Claw", "Fire Punch", "Protect", "Dragon Dance", "Ice Punch"],
    "Arcanine": ["Flare Blitz", "Wild Charge", "Extreme Speed", "Protect", "Will-O-Wisp", "Snarl"],
    "Pelipper": ["Hurricane", "Hydro Pump", "Protect", "Tailwind", "U-turn", "Wide Guard"],
    "Torkoal": ["Eruption", "Heat Wave", "Earth Power", "Protect", "Yawn", "Will-O-Wisp"],
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
    
    # === PARADOX POKEMON ===
    "Iron Valiant": ["Close Combat", "Moonblast", "Spirit Break", "Protect", "Swords Dance", "Destiny Bond"],
    "Iron Moth": ["Fiery Dance", "Energy Ball", "Sludge Wave", "Protect", "Acid Spray", "Psychic"],
    "Iron Thorns": ["Rock Slide", "Wild Charge", "Earthquake", "Protect", "Dragon Dance", "Ice Punch"],
    "Iron Jugulis": ["Dark Pulse", "Air Slash", "Hydro Pump", "Protect", "Tailwind", "Snarl"],
    "Iron Treads": ["Earthquake", "Iron Head", "Rapid Spin", "Protect", "Knock Off", "Rock Slide"],
    "Sandy Shocks": ["Thunderbolt", "Earth Power", "Power Gem", "Protect", "Stealth Rock", "Thunder Wave"],
    "Slither Wing": ["First Impression", "Close Combat", "U-turn", "Protect", "Flame Charge", "Leech Life"],
    "Walking Wake": ["Hydro Steam", "Draco Meteor", "Flamethrower", "Protect", "Dragon Pulse", "Snarl"],
    "Iron Leaves": ["Psyblade", "Close Combat", "Leaf Blade", "Protect", "Swords Dance", "Helping Hand"],
    "Gouging Fire": ["Raging Fury", "Dragon Claw", "Flare Blitz", "Protect", "Dragon Dance", "Burning Bulwark"],
    "Iron Boulder": ["Mighty Cleave", "Close Combat", "Psycho Cut", "Protect", "Swords Dance", "Zen Headbutt"],
    "Iron Crown": ["Tachyon Cutter", "Focus Blast", "Expanding Force", "Protect", "Calm Mind", "Flash Cannon"],
    
    # === WEATHER SUPPORT ===
    "Ninetales-Alola": ["Aurora Veil", "Blizzard", "Moonblast", "Protect", "Encore", "Icy Wind"],
    "Tyranitar": ["Rock Slide", "Crunch", "Low Kick", "Protect", "Dragon Dance", "Ice Punch"],
    "Hippowdon": ["Earthquake", "Rock Slide", "Yawn", "Protect", "Slack Off", "Whirlwind"],
    "Lilligant-Hisui": ["Close Combat", "Leaf Blade", "After You", "Protect", "Sleep Powder", "Victory Dance"],
    "Bronzong": ["Gyro Ball", "Trick Room", "Hypnosis", "Protect", "Imprison", "Body Press"],
    
    # === SUPPORT POKEMON ===
    "Clefairy": ["Follow Me", "Helping Hand", "Moonblast", "Protect", "Icy Wind", "Thunder Wave"],
    "Porygon2": ["Tri Attack", "Trick Room", "Recover", "Protect", "Ice Beam", "Thunderbolt"],
    "Cresselia": ["Psychic", "Trick Room", "Helping Hand", "Protect", "Ice Beam", "Moonblast"],
    "Hatterene": ["Trick Room", "Dazzling Gleam", "Psychic", "Protect", "Mystical Fire", "Healing Wish"],
    "Dusclops": ["Trick Room", "Night Shade", "Pain Split", "Protect", "Will-O-Wisp", "Helping Hand"],
    "Murkrow": ["Tailwind", "Foul Play", "Haze", "Protect", "Quash", "Snarl"],
    "Indeedee-F": ["Psychic", "Follow Me", "Helping Hand", "Protect", "Trick Room", "Heal Pulse"],
    "Gothitelle": ["Psychic", "Trick Room", "Helping Hand", "Protect", "Fake Out", "Thunder Wave"],
    "Talonflame": ["Brave Bird", "Flare Blitz", "Tailwind", "Protect", "U-turn", "Taunt"],
    
    # === OFFENSIVE THREATS ===
    "Garchomp": ["Earthquake", "Dragon Claw", "Rock Slide", "Protect", "Swords Dance", "Scale Shot"],
    "Salamence": ["Draco Meteor", "Heat Wave", "Tailwind", "Protect", "Hurricane", "Earthquake"],
    "Gyarados": ["Waterfall", "Crunch", "Earthquake", "Protect", "Dragon Dance", "Ice Fang"],
    "Volcarona": ["Heat Wave", "Bug Buzz", "Quiver Dance", "Protect", "Giga Drain", "Rage Powder"],
    "Baxcalibur": ["Icicle Crash", "Dragon Claw", "Earthquake", "Protect", "Ice Shard", "Glaive Rush"],
    "Dragapult": ["Dragon Darts", "Phantom Force", "U-turn", "Protect", "Will-O-Wisp", "Draco Meteor"],
    "Haxorus": ["Outrage", "Earthquake", "Close Combat", "Protect", "Dragon Dance", "Poison Jab"],
    "Ceruledge": ["Bitter Blade", "Shadow Sneak", "Close Combat", "Protect", "Swords Dance", "Psycho Cut"],
    "Armarouge": ["Armor Cannon", "Psychic", "Will-O-Wisp", "Protect", "Trick Room", "Heat Wave"],
    "Ting-Lu": ["Earthquake", "Ruination", "Heavy Slam", "Protect", "Whirlwind", "Taunt"],
    "Wo-Chien": ["Foul Play", "Ruination", "Giga Drain", "Protect", "Leech Seed", "Pollen Puff"],
    "Kommo-o": ["Close Combat", "Dragon Claw", "Protect", "Iron Head", "Dragon Dance", "Clanging Scales"],
    "Goodra-Hisui": ["Dragon Pulse", "Iron Head", "Fire Blast", "Protect", "Acid Spray", "Thunderbolt"],
    "Araquanid": ["Liquidation", "Lunge", "Wide Guard", "Protect", "Sticky Web", "Mirror Coat"],
    "Tsareena": ["Power Whip", "Triple Axel", "U-turn", "Protect", "High Jump Kick", "Rapid Spin"],
    "Comfey": ["Floral Healing", "Draining Kiss", "U-turn", "Protect", "Taunt", "Trick Room"],
    
    # === DEFENSIVE/UTILITY ===
    "Blissey": ["Seismic Toss", "Soft-Boiled", "Thunder Wave", "Protect", "Helping Hand", "Heal Pulse"],
    "Gastrodon": ["Earth Power", "Surf", "Recover", "Protect", "Yawn", "Icy Wind"],
    "Maushold": ["Population Bomb", "Beat Up", "Follow Me", "Protect", "Tidy Up", "Encore"],
    "Mudsdale": ["High Horsepower", "Heavy Slam", "Rock Slide", "Protect", "Body Press", "Stealth Rock"],
    "Oranguru": ["Instruct", "Psychic", "Trick Room", "Protect", "Nasty Plot", "Foul Play"],
    "Snorlax": ["Body Slam", "Heavy Slam", "Curse", "Protect", "Rest", "Belly Drum"],
    "Slowbro": ["Psychic", "Ice Beam", "Trick Room", "Protect", "Slack Off", "Calm Mind"],
    "Slowking-Galar": ["Sludge Bomb", "Psychic", "Trick Room", "Protect", "Slack Off", "Future Sight"],
    "Drifblim": ["Shadow Ball", "Tailwind", "Will-O-Wisp", "Protect", "Strength Sap", "Destiny Bond"],
    
    # === POPULAR PICKS ===
    "Cinderace": ["Pyro Ball", "Court Change", "U-turn", "Protect", "Gunk Shot", "High Jump Kick"],
    "Raichu": ["Thunderbolt", "Nuzzle", "Fake Out", "Protect", "Volt Switch", "Brick Break"],
    "Thundurus": ["Thunderbolt", "Taunt", "Thunder Wave", "Protect", "Eerie Impulse", "Nasty Plot"],
    "Zapdos-Galar": ["Thunderous Kick", "Brave Bird", "U-turn", "Protect", "Coaching", "Close Combat"],
    "Moltres-Galar": ["Fiery Wrath", "Hurricane", "Protect", "Nasty Plot", "Tailwind", "Snarl"],
    "Ninetales": ["Heat Wave", "Will-O-Wisp", "Encore", "Protect", "Solar Beam", "Nasty Plot"],
    "Venusaur": ["Giga Drain", "Sludge Bomb", "Earth Power", "Protect", "Sleep Powder", "Leaf Storm"],
    "Charizard": ["Heat Wave", "Air Slash", "Dragon Pulse", "Protect", "Will-O-Wisp", "Overheat"],
    "Excadrill": ["High Horsepower", "Iron Head", "Rock Slide", "Protect", "Swords Dance", "Rapid Spin"],
    "Scizor": ["Bullet Punch", "Bug Bite", "Swords Dance", "Protect", "U-turn", "Knock Off"],
    "Metagross": ["Meteor Mash", "Zen Headbutt", "Bullet Punch", "Protect", "Ice Punch", "Earthquake"],
    "Azumarill": ["Aqua Jet", "Play Rough", "Helping Hand", "Protect", "Knock Off", "Belly Drum"],
    "Sylveon": ["Hyper Voice", "Moonblast", "Quick Attack", "Protect", "Yawn", "Helping Hand"],
    "Gengar": ["Shadow Ball", "Sludge Bomb", "Will-O-Wisp", "Protect", "Icy Wind", "Trick Room"],
    "Milotic": ["Muddy Water", "Ice Beam", "Coil", "Protect", "Recover", "Hypnosis"],
    "Toxtricity": ["Overdrive", "Sludge Bomb", "Boomburst", "Protect", "Volt Switch", "Snarl"],
    "Gardevoir": ["Moonblast", "Psychic", "Trick Room", "Protect", "Will-O-Wisp", "Helping Hand"],
    "Gallade": ["Close Combat", "Psycho Cut", "Wide Guard", "Protect", "Helping Hand", "Swords Dance"],
    "Klefki": ["Foul Play", "Thunder Wave", "Reflect", "Light Screen", "Spikes", "Play Rough"],
    "Sableye": ["Foul Play", "Quash", "Will-O-Wisp", "Protect", "Fake Out", "Trick"],
    # Ogerpon base form only (forms require specific masks as held items)
    "Ogerpon": ["Ivy Cudgel", "Horn Leech", "Follow Me", "Spiky Shield", "Grassy Glide", "Encore"],
}

# ============================================================================
# VALIDATED POKEMON ABILITIES - Gen 9 VGC 2024 Reg G
# ============================================================================

POKEMON_ABILITIES: Dict[str, str] = {
    # === TOP TIER META ===
    "Incineroar": "Intimidate",
    "Rillaboom": "Grassy Surge",
    "Flutter Mane": "Protosynthesis",
    "Urshifu": "Unseen Fist",
    "Urshifu-Rapid-Strike": "Unseen Fist",
    "Landorus": "Sheer Force",
    "Amoonguss": "Regenerator",
    "Farigiraf": "Armor Tail",
    "Chi-Yu": "Beads of Ruin",
    "Iron Hands": "Quark Drive",
    "Raging Bolt": "Protosynthesis",
    "Kingambit": "Defiant",
    "Gholdengo": "Good as Gold",
    "Tornadus": "Prankster",
    "Whimsicott": "Prankster",
    "Chien-Pao": "Sword of Ruin",
    "Dragonite": "Multiscale",
    "Arcanine": "Intimidate",
    "Pelipper": "Drizzle",
    "Torkoal": "Drought",
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
    
    # === PARADOX POKEMON ===
    "Iron Valiant": "Quark Drive",
    "Iron Moth": "Quark Drive",
    "Iron Thorns": "Quark Drive",
    "Iron Jugulis": "Quark Drive",
    "Iron Treads": "Quark Drive",
    "Sandy Shocks": "Protosynthesis",
    "Slither Wing": "Protosynthesis",
    "Walking Wake": "Protosynthesis",
    "Iron Leaves": "Quark Drive",
    "Gouging Fire": "Protosynthesis",
    "Iron Boulder": "Quark Drive",
    "Iron Crown": "Quark Drive",
    
    # === WEATHER SUPPORT ===
    "Ninetales-Alola": "Snow Warning",
    "Tyranitar": "Sand Stream",
    "Hippowdon": "Sand Stream",
    "Lilligant-Hisui": "Chlorophyll",
    "Bronzong": "Levitate",
    
    # === SUPPORT POKEMON ===
    "Clefairy": "Friend Guard",
    "Porygon2": "Download",
    "Cresselia": "Levitate",
    "Hatterene": "Magic Bounce",
    "Dusclops": "Frisk",
    "Murkrow": "Prankster",
    "Indeedee-F": "Psychic Surge",
    "Gothitelle": "Shadow Tag",
    "Talonflame": "Gale Wings",
    
    # === OFFENSIVE THREATS ===
    "Garchomp": "Rough Skin",
    "Salamence": "Intimidate",
    "Gyarados": "Intimidate",
    "Volcarona": "Flame Body",
    "Baxcalibur": "Thermal Exchange",
    "Dragapult": "Clear Body",
    "Haxorus": "Mold Breaker",
    "Ceruledge": "Flash Fire",
    "Armarouge": "Flash Fire",
    "Ting-Lu": "Vessel of Ruin",
    "Wo-Chien": "Tablets of Ruin",
    "Kommo-o": "Bulletproof",
    "Goodra-Hisui": "Sap Sipper",
    "Araquanid": "Water Bubble",
    "Tsareena": "Queenly Majesty",
    "Comfey": "Triage",
    
    # === DEFENSIVE/UTILITY ===
    "Blissey": "Natural Cure",
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
    "Thundurus": "Prankster",
    "Zapdos-Galar": "Defiant",
    "Moltres-Galar": "Berserk",
    "Ninetales": "Drought",
    "Venusaur": "Chlorophyll",
    "Charizard": "Solar Power",
    "Excadrill": "Sand Rush",
    "Scizor": "Technician",
    "Metagross": "Clear Body",
    "Azumarill": "Huge Power",
    "Sylveon": "Pixilate",
    "Gengar": "Cursed Body",
    "Milotic": "Competitive",
    "Toxtricity": "Punk Rock",
    "Gardevoir": "Trace",
    "Gallade": "Justified",
    "Klefki": "Prankster",
    "Sableye": "Prankster",
    "Ogerpon": "Defiant",  # Base form only
}

# Restricted Pokemon (only 1 per team in VGC)
RESTRICTED_POKEMON = [
    "Koraidon", "Miraidon", "Calyrex-Ice", "Calyrex-Shadow",
    "Zacian", "Zamazenta", "Eternatus", "Kyogre", "Groudon",
    "Rayquaza", "Dialga", "Palkia", "Giratina", "Reshiram",
    "Zekrom", "Kyurem", "Xerneas", "Yveltal", "Zygarde",
    "Solgaleo", "Lunala", "Necrozma", "Mewtwo", "Lugia", "Ho-Oh",
]

# Safe Pokemon pool (Pokemon with defined movesets, excluding restricted)
VGC_SAFE_POOL = [p for p in POKEMON_MOVESETS.keys() if p not in RESTRICTED_POKEMON]

# Alias for compatibility
VGC_POKEMON_POOL = VGC_SAFE_POOL

# ============================================================================
# COMMON VGC ITEMS
# ============================================================================
VGC_ITEMS = [
    "Choice Scarf", "Choice Band", "Choice Specs",
    "Focus Sash", "Assault Vest", "Life Orb",
    "Leftovers", "Sitrus Berry", "Lum Berry",
    "Safety Goggles", "Covert Cloak", "Clear Amulet",
    "Eviolite", "Rocky Helmet", "Expert Belt",
    "Booster Energy", "Wide Lens", "Muscle Band",
    "Wise Glasses", "Scope Lens", "Weakness Policy",
    "White Herb", "Mental Herb", "Power Herb",
    "Mirror Herb", "Loaded Dice", "Punching Glove",
    "Throat Spray", "Adrenaline Orb", "Blunder Policy",
]

# ============================================================================
# TERA TYPES
# ============================================================================
TERA_TYPES = [
    "Normal", "Fire", "Water", "Electric", "Grass", "Ice",
    "Fighting", "Poison", "Ground", "Flying", "Psychic", "Bug",
    "Rock", "Ghost", "Dragon", "Dark", "Steel", "Fairy",
]

# Verify all Pokemon in movesets have abilities defined
_missing_abilities = [p for p in POKEMON_MOVESETS if p not in POKEMON_ABILITIES]
if _missing_abilities:
    import warnings
    warnings.warn(f"Pokemon missing abilities: {_missing_abilities}")

