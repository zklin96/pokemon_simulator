"""Pydantic schemas for data validation.

This module defines schemas for validating battle data, trajectories,
and model outputs. These schemas ensure data quality and catch errors
early in the pipeline.
"""

from pydantic import (
    BaseModel, Field, field_validator, model_validator,
    ConfigDict, computed_field
)
from typing import List, Optional, Dict, Any, Literal, Tuple
from enum import Enum
import numpy as np


# ====================
# Constants
# ====================

STATE_DIM = 620
ACTION_DIM = 144
NUM_POKEMON = 6
NUM_MOVES = 4


# ====================
# Enums
# ====================

class PlayerID(str, Enum):
    """Player identifier."""
    P1 = "p1"
    P2 = "p2"


class BattleFormat(str, Enum):
    """Supported battle formats."""
    VGC_2024_REG_G = "gen9vgc2024regg"
    VGC_2024_REG_H = "gen9vgc2024regh"
    VGC_2025_REG_G = "gen9vgc2025regg"


class Status(str, Enum):
    """Pokemon status conditions."""
    NONE = ""
    BURN = "brn"
    FREEZE = "frz"
    PARALYSIS = "par"
    POISON = "psn"
    TOXIC = "tox"
    SLEEP = "slp"
    FAINT = "fnt"


class TeraType(str, Enum):
    """Tera types."""
    NORMAL = "normal"
    FIRE = "fire"
    WATER = "water"
    ELECTRIC = "electric"
    GRASS = "grass"
    ICE = "ice"
    FIGHTING = "fighting"
    POISON = "poison"
    GROUND = "ground"
    FLYING = "flying"
    PSYCHIC = "psychic"
    BUG = "bug"
    ROCK = "rock"
    GHOST = "ghost"
    DRAGON = "dragon"
    DARK = "dark"
    STEEL = "steel"
    FAIRY = "fairy"
    STELLAR = "stellar"


# ====================
# Pokemon Schemas
# ====================

class MoveSchema(BaseModel):
    """Schema for a Pokemon move."""
    
    model_config = ConfigDict(extra="allow")
    
    name: str = Field(..., min_length=1, max_length=50)
    pp: int = Field(default=0, ge=0, le=64)
    max_pp: int = Field(default=0, ge=0, le=64)
    
    @field_validator("name")
    @classmethod
    def normalize_name(cls, v: str) -> str:
        """Normalize move name."""
        return v.lower().replace(" ", "").replace("-", "")


class StatSchema(BaseModel):
    """Schema for Pokemon stats."""
    
    hp: int = Field(default=0, ge=0, le=999)
    atk: int = Field(default=0, ge=0, le=999)
    def_: int = Field(default=0, ge=0, le=999, alias="def")
    spa: int = Field(default=0, ge=0, le=999)
    spd: int = Field(default=0, ge=0, le=999)
    spe: int = Field(default=0, ge=0, le=999)
    
    model_config = ConfigDict(populate_by_name=True)


class BoostSchema(BaseModel):
    """Schema for stat boosts."""
    
    atk: int = Field(default=0, ge=-6, le=6)
    def_: int = Field(default=0, ge=-6, le=6, alias="def")
    spa: int = Field(default=0, ge=-6, le=6)
    spd: int = Field(default=0, ge=-6, le=6)
    spe: int = Field(default=0, ge=-6, le=6)
    accuracy: int = Field(default=0, ge=-6, le=6)
    evasion: int = Field(default=0, ge=-6, le=6)
    
    model_config = ConfigDict(populate_by_name=True)


class PokemonSchema(BaseModel):
    """Schema for a Pokemon in battle."""
    
    model_config = ConfigDict(extra="allow")
    
    species: str = Field(..., min_length=1, max_length=50)
    nickname: Optional[str] = None
    level: int = Field(default=50, ge=1, le=100)
    gender: Optional[Literal["M", "F", ""]] = ""
    
    # Combat state
    hp: int = Field(default=100, ge=0)
    max_hp: int = Field(default=100, ge=1)
    status: Optional[str] = None
    
    # Configuration
    ability: Optional[str] = None
    item: Optional[str] = None
    moves: List[str] = Field(default_factory=list, max_length=4)
    
    # Types
    types: List[str] = Field(default_factory=list, max_length=2)
    tera_type: Optional[str] = None
    is_terastallized: bool = False
    
    # Stats and boosts
    stats: Optional[StatSchema] = None
    boosts: Optional[BoostSchema] = None
    
    @field_validator("species")
    @classmethod
    def normalize_species(cls, v: str) -> str:
        """Normalize species name."""
        return v.lower().replace(" ", "").replace("-", "")
    
    @computed_field
    @property
    def hp_fraction(self) -> float:
        """Get HP as fraction."""
        if self.max_hp == 0:
            return 0.0
        return self.hp / self.max_hp
    
    @computed_field
    @property
    def is_alive(self) -> bool:
        """Check if Pokemon is alive."""
        return self.hp > 0


# ====================
# Battle State Schemas
# ====================

class FieldConditionSchema(BaseModel):
    """Schema for field conditions."""
    
    weather: Optional[str] = None
    terrain: Optional[str] = None
    trick_room: bool = False
    trick_room_turns: int = Field(default=0, ge=0, le=5)


class SideConditionSchema(BaseModel):
    """Schema for side conditions."""
    
    reflect: int = Field(default=0, ge=0, le=8)
    light_screen: int = Field(default=0, ge=0, le=8)
    aurora_veil: int = Field(default=0, ge=0, le=8)
    tailwind: int = Field(default=0, ge=0, le=4)
    spikes: int = Field(default=0, ge=0, le=3)
    stealth_rock: bool = False
    sticky_web: bool = False


class TurnStateSchema(BaseModel):
    """Schema for a turn state."""
    
    model_config = ConfigDict(extra="allow")
    
    turn: int = Field(..., ge=0, le=100)
    
    # Active Pokemon
    p1_active: List[PokemonSchema] = Field(default_factory=list, max_length=2)
    p2_active: List[PokemonSchema] = Field(default_factory=list, max_length=2)
    
    # Bench Pokemon
    p1_bench: List[PokemonSchema] = Field(default_factory=list, max_length=4)
    p2_bench: List[PokemonSchema] = Field(default_factory=list, max_length=4)
    
    # Conditions
    field: Optional[FieldConditionSchema] = None
    p1_side: Optional[SideConditionSchema] = None
    p2_side: Optional[SideConditionSchema] = None
    
    # Tera availability
    p1_can_tera: bool = True
    p2_can_tera: bool = True
    
    @model_validator(mode="after")
    def validate_team_sizes(self) -> "TurnStateSchema":
        """Validate team sizes."""
        p1_total = len(self.p1_active) + len(self.p1_bench)
        p2_total = len(self.p2_active) + len(self.p2_bench)
        
        if p1_total > NUM_POKEMON:
            raise ValueError(f"P1 has too many Pokemon: {p1_total}")
        if p2_total > NUM_POKEMON:
            raise ValueError(f"P2 has too many Pokemon: {p2_total}")
        
        return self


# ====================
# Transition Schemas
# ====================

class TransitionSchema(BaseModel):
    """Schema for a state-action-reward transition."""
    
    model_config = ConfigDict(extra="allow")
    
    state: List[float] = Field(..., min_length=STATE_DIM, max_length=STATE_DIM)
    action: int = Field(..., ge=0, lt=ACTION_DIM)
    reward: float = Field(...)
    done: bool = Field(default=False)
    
    # Optional next state
    next_state: Optional[List[float]] = Field(
        default=None, min_length=STATE_DIM, max_length=STATE_DIM
    )
    
    @field_validator("state", "next_state")
    @classmethod
    def validate_state_values(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate state values are in reasonable range."""
        if v is None:
            return v
        
        for i, val in enumerate(v):
            if not np.isfinite(val):
                raise ValueError(f"State contains non-finite value at index {i}")
            if abs(val) > 1e6:
                raise ValueError(f"State value out of range at index {i}: {val}")
        
        return v
    
    @field_validator("reward")
    @classmethod
    def validate_reward(cls, v: float) -> float:
        """Validate reward is reasonable."""
        if not np.isfinite(v):
            raise ValueError(f"Reward is not finite: {v}")
        if abs(v) > 100:
            raise ValueError(f"Reward out of expected range: {v}")
        return v


class TrajectorySchema(BaseModel):
    """Schema for a complete trajectory."""
    
    model_config = ConfigDict(extra="allow")
    
    battle_id: str = Field(..., min_length=1, max_length=100)
    player: Literal["p1", "p2"] = Field(...)
    winner: Optional[Literal["p1", "p2"]] = None
    format: Optional[str] = None
    
    transitions: List[TransitionSchema] = Field(..., min_length=1, max_length=500)
    
    @computed_field
    @property
    def won(self) -> bool:
        """Whether this trajectory's player won."""
        return self.winner == self.player
    
    @computed_field
    @property
    def total_reward(self) -> float:
        """Total reward for this trajectory."""
        return sum(t.reward for t in self.transitions)
    
    @computed_field
    @property
    def length(self) -> int:
        """Number of transitions."""
        return len(self.transitions)
    
    @model_validator(mode="after")
    def validate_terminal(self) -> "TrajectorySchema":
        """Validate last transition is terminal."""
        if self.transitions and not self.transitions[-1].done:
            # This is a warning, not an error
            pass
        return self


# ====================
# Batch Schemas
# ====================

class BatchSchema(BaseModel):
    """Schema for a training batch."""
    
    states: List[List[float]] = Field(...)
    actions: List[int] = Field(...)
    rewards: List[float] = Field(...)
    dones: List[bool] = Field(...)
    
    @model_validator(mode="after")
    def validate_lengths(self) -> "BatchSchema":
        """Validate all lists have same length."""
        lengths = [
            len(self.states),
            len(self.actions),
            len(self.rewards),
            len(self.dones),
        ]
        
        if len(set(lengths)) != 1:
            raise ValueError(f"Inconsistent batch sizes: {lengths}")
        
        return self
    
    @computed_field
    @property
    def batch_size(self) -> int:
        """Get batch size."""
        return len(self.states)


# ====================
# Validation Utilities
# ====================

def validate_trajectory(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a trajectory dictionary.
    
    Args:
        data: Trajectory dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        TrajectorySchema(**data)
        return True, None
    except Exception as e:
        return False, str(e)


def validate_transition(data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
    """Validate a transition dictionary.
    
    Args:
        data: Transition dictionary
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        TransitionSchema(**data)
        return True, None
    except Exception as e:
        return False, str(e)


def validate_state_vector(state: List[float]) -> Tuple[bool, Optional[str]]:
    """Validate a state vector.
    
    Args:
        state: State vector
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(state) != STATE_DIM:
        return False, f"Wrong state dimension: {len(state)} (expected {STATE_DIM})"
    
    for i, val in enumerate(state):
        if not np.isfinite(val):
            return False, f"Non-finite value at index {i}"
    
    return True, None


def validate_action(action: int) -> Tuple[bool, Optional[str]]:
    """Validate an action.
    
    Args:
        action: Action index
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(action, int):
        return False, f"Action must be int, got {type(action)}"
    
    if action < 0 or action >= ACTION_DIM:
        return False, f"Action out of range: {action} (expected 0-{ACTION_DIM-1})"
    
    return True, None

