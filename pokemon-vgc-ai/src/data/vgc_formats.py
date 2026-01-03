"""VGC format definitions and utilities.

Supports multiple VGC formats:
- Regulation G (2024): Standard restricted (1 per team)
- Regulation H (2024): Different restricted list
- Regulation G (2025): Updated for new games/DLC
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from pathlib import Path
import json
from loguru import logger


class VGCFormat(Enum):
    """Available VGC formats."""
    REG_G_2024 = "gen9vgc2024regg"
    REG_H_2024 = "gen9vgc2024regh"
    REG_G_2025 = "gen9vgc2025regg"


@dataclass
class FormatConfig:
    """Configuration for a VGC format."""
    
    name: str
    showdown_format: str  # e.g., "gen9vgc2024regg"
    
    # Restricted Pokemon (only 1 per team)
    restricted_pokemon: List[str] = field(default_factory=list)
    
    # Banned Pokemon (cannot use)
    banned_pokemon: List[str] = field(default_factory=list)
    
    # Special rules
    team_size: int = 6
    bring_size: int = 4
    level_cap: int = 50
    allow_tera: bool = True
    
    # Data paths
    data_file: str = ""
    trajectory_dir: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "showdown_format": self.showdown_format,
            "restricted_pokemon": self.restricted_pokemon,
            "banned_pokemon": self.banned_pokemon,
            "team_size": self.team_size,
            "bring_size": self.bring_size,
            "level_cap": self.level_cap,
            "allow_tera": self.allow_tera,
            "data_file": self.data_file,
            "trajectory_dir": self.trajectory_dir,
        }


# Regulation G 2024 restricted list
REG_G_2024_RESTRICTED = [
    "Koraidon", "Miraidon",
    "Calyrex-Ice", "Calyrex-Shadow",
    "Zacian", "Zamazenta",
    "Kyogre", "Groudon", "Rayquaza",
    "Dialga", "Palkia", "Giratina",
    "Reshiram", "Zekrom", "Kyurem",
    "Xerneas", "Yveltal", "Zygarde",
    "Solgaleo", "Lunala", "Necrozma",
    "Eternatus",
    "Mewtwo", "Lugia", "Ho-Oh",
]

# Regulation H 2024 restricted list (example - may differ)
REG_H_2024_RESTRICTED = [
    "Koraidon", "Miraidon",
    "Calyrex-Ice", "Calyrex-Shadow",
    "Terapagos", "Ogerpon-Cornerstone", "Ogerpon-Wellspring", "Ogerpon-Hearthflame",
    # ... more as per official rules
]

# Regulation G 2025 restricted list
REG_G_2025_RESTRICTED = [
    # Updated for Legends ZA and future DLC
    "Koraidon", "Miraidon",
    "Calyrex-Ice", "Calyrex-Shadow",
    "Terapagos",
    # ... more as per official rules
]

# Mythical Pokemon (usually banned)
MYTHICAL_POKEMON = [
    "Mew", "Celebi", "Jirachi", "Deoxys",
    "Phione", "Manaphy", "Darkrai", "Shaymin", "Arceus",
    "Victini", "Keldeo", "Meloetta", "Genesect",
    "Diancie", "Hoopa", "Volcanion",
    "Magearna", "Marshadow", "Zeraora", "Meltan", "Melmetal",
    "Zarude", "Pecharunt",
]


# Pre-configured formats
FORMAT_CONFIGS: Dict[VGCFormat, FormatConfig] = {
    VGCFormat.REG_G_2024: FormatConfig(
        name="VGC 2024 Regulation G",
        showdown_format="gen9vgc2024regg",
        restricted_pokemon=REG_G_2024_RESTRICTED,
        banned_pokemon=MYTHICAL_POKEMON,
        data_file="logs-gen9vgc2024regg.json",
        trajectory_dir="trajectories_full",
    ),
    VGCFormat.REG_H_2024: FormatConfig(
        name="VGC 2024 Regulation H",
        showdown_format="gen9vgc2024regh",
        restricted_pokemon=REG_H_2024_RESTRICTED,
        banned_pokemon=MYTHICAL_POKEMON,
        data_file="logs-gen9vgc2024regh.json",
        trajectory_dir="trajectories_regh",
    ),
    VGCFormat.REG_G_2025: FormatConfig(
        name="VGC 2025 Regulation G",
        showdown_format="gen9vgc2025regg",
        restricted_pokemon=REG_G_2025_RESTRICTED,
        banned_pokemon=MYTHICAL_POKEMON,
        data_file="logs-gen9vgc2025regg.json",
        trajectory_dir="trajectories_2025regg",
    ),
}


class FormatManager:
    """Manages VGC format configurations."""
    
    def __init__(self, data_root: Optional[Path] = None):
        """Initialize format manager.
        
        Args:
            data_root: Root path for data files
        """
        self.data_root = data_root or Path("data")
        self.current_format: Optional[VGCFormat] = None
        self.current_config: Optional[FormatConfig] = None
    
    def get_format(self, format_id: VGCFormat) -> FormatConfig:
        """Get configuration for a format.
        
        Args:
            format_id: Format enum value
            
        Returns:
            FormatConfig for the format
        """
        return FORMAT_CONFIGS.get(format_id, FORMAT_CONFIGS[VGCFormat.REG_G_2024])
    
    def set_format(self, format_id: VGCFormat) -> FormatConfig:
        """Set the current active format.
        
        Args:
            format_id: Format to use
            
        Returns:
            FormatConfig for the format
        """
        self.current_format = format_id
        self.current_config = self.get_format(format_id)
        logger.info(f"Set VGC format to: {self.current_config.name}")
        return self.current_config
    
    def get_data_path(self, format_id: Optional[VGCFormat] = None) -> Path:
        """Get path to raw data file for a format.
        
        Args:
            format_id: Format (uses current if None)
            
        Returns:
            Path to data file
        """
        config = self.get_format(format_id or self.current_format or VGCFormat.REG_G_2024)
        return self.data_root / "raw" / "vgc_bench" / config.data_file
    
    def get_trajectory_path(self, format_id: Optional[VGCFormat] = None) -> Path:
        """Get path to processed trajectories for a format.
        
        Args:
            format_id: Format (uses current if None)
            
        Returns:
            Path to trajectories directory
        """
        config = self.get_format(format_id or self.current_format or VGCFormat.REG_G_2024)
        return self.data_root / "processed" / config.trajectory_dir
    
    def is_pokemon_allowed(
        self, 
        species: str, 
        format_id: Optional[VGCFormat] = None,
    ) -> bool:
        """Check if a Pokemon is allowed in a format.
        
        Args:
            species: Pokemon species name
            format_id: Format (uses current if None)
            
        Returns:
            True if Pokemon is allowed
        """
        config = self.get_format(format_id or self.current_format or VGCFormat.REG_G_2024)
        return species not in config.banned_pokemon
    
    def is_restricted(
        self, 
        species: str, 
        format_id: Optional[VGCFormat] = None,
    ) -> bool:
        """Check if a Pokemon is restricted (only 1 per team).
        
        Args:
            species: Pokemon species name
            format_id: Format (uses current if None)
            
        Returns:
            True if Pokemon is restricted
        """
        config = self.get_format(format_id or self.current_format or VGCFormat.REG_G_2024)
        return species in config.restricted_pokemon
    
    def validate_team(
        self, 
        team: List[str], 
        format_id: Optional[VGCFormat] = None,
    ) -> Tuple[bool, List[str]]:
        """Validate a team for a format.
        
        Args:
            team: List of Pokemon species
            format_id: Format (uses current if None)
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        config = self.get_format(format_id or self.current_format or VGCFormat.REG_G_2024)
        errors = []
        
        # Check team size
        if len(team) > config.team_size:
            errors.append(f"Team has {len(team)} Pokemon, max is {config.team_size}")
        
        # Check for banned Pokemon
        for species in team:
            if species in config.banned_pokemon:
                errors.append(f"{species} is banned")
        
        # Check restricted count
        restricted_count = sum(1 for s in team if s in config.restricted_pokemon)
        if restricted_count > 1:
            restricted_on_team = [s for s in team if s in config.restricted_pokemon]
            errors.append(f"Team has {restricted_count} restricted Pokemon: {restricted_on_team}")
        
        # Check for duplicates
        if len(team) != len(set(team)):
            errors.append("Team has duplicate Pokemon")
        
        return len(errors) == 0, errors
    
    @staticmethod
    def list_formats() -> List[str]:
        """List available format names.
        
        Returns:
            List of format display names
        """
        return [config.name for config in FORMAT_CONFIGS.values()]


def get_format_from_string(format_str: str) -> VGCFormat:
    """Parse format string to VGCFormat enum.
    
    Args:
        format_str: Format string like "gen9vgc2024regg"
        
    Returns:
        VGCFormat enum value
    """
    format_str = format_str.lower().strip()
    
    for fmt in VGCFormat:
        if fmt.value == format_str:
            return fmt
    
    # Fuzzy matching
    if "regg" in format_str and "2024" in format_str:
        return VGCFormat.REG_G_2024
    if "regh" in format_str:
        return VGCFormat.REG_H_2024
    if "2025" in format_str:
        return VGCFormat.REG_G_2025
    
    # Default
    return VGCFormat.REG_G_2024



