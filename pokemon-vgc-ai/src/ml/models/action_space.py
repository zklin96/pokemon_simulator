"""Hierarchical action space for VGC doubles.

This module provides a structured action space that decomposes
actions into strategy, target, and detail levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Any
from dataclasses import dataclass
from enum import IntEnum
import numpy as np


class ActionStrategy(IntEnum):
    """High-level action strategies."""
    ATTACK = 0      # Use a damaging move
    SUPPORT = 1     # Use a status/support move
    SWITCH = 2      # Switch to a bench Pokemon
    PROTECT = 3     # Use a protection move


class ActionTarget(IntEnum):
    """Action targets for doubles."""
    OPP_SLOT_1 = 0   # Opponent's first active slot
    OPP_SLOT_2 = 1   # Opponent's second active slot
    ALLY = 2         # Your other active Pokemon
    SELF = 3         # Target self
    ALL_OPPONENTS = 4  # Both opponents (spread move)
    ALL = 5          # All Pokemon on field
    NONE = 6         # No target (status move)


@dataclass
class DecodedAction:
    """Fully decoded action."""
    slot: int           # Which of my active Pokemon (0 or 1)
    action_type: str    # "move", "switch", "tera_move"
    move_index: int     # 0-3 for moves, 0-3 for switch targets
    target: int         # Target slot
    tera: bool          # Whether to terastallize
    
    def to_showdown_command(self) -> str:
        """Convert to Pokemon Showdown command format."""
        if self.action_type == "switch":
            return f"switch {self.move_index + 1}"
        elif self.action_type == "move":
            target_str = f" {self.target + 1}" if self.target >= 0 else ""
            return f"move {self.move_index + 1}{target_str}"
        elif self.action_type == "tera_move":
            target_str = f" {self.target + 1}" if self.target >= 0 else ""
            return f"move {self.move_index + 1} terastallize{target_str}"
        else:
            return "default"


class FlatActionSpace:
    """Traditional flat action space (144 actions).
    
    Action space:
    - Slot 1: 12 actions (4 moves + 4 tera moves + 4 switches)
    - Slot 2: 12 actions
    - Combined: 12 * 12 = 144 joint actions
    """
    
    NUM_MOVES = 4
    NUM_SWITCHES = 4
    ACTIONS_PER_SLOT = 12  # 4 moves + 4 tera moves + 4 switches
    TOTAL_ACTIONS = ACTIONS_PER_SLOT ** 2  # 144
    
    @classmethod
    def decode(cls, action: int) -> Tuple[DecodedAction, DecodedAction]:
        """Decode flat action to two slot actions.
        
        Args:
            action: Flat action index (0-143)
            
        Returns:
            Tuple of DecodedAction for each slot
        """
        slot1_action = action // cls.ACTIONS_PER_SLOT
        slot2_action = action % cls.ACTIONS_PER_SLOT
        
        return (
            cls._decode_slot_action(0, slot1_action),
            cls._decode_slot_action(1, slot2_action),
        )
    
    @classmethod
    def _decode_slot_action(cls, slot: int, action: int) -> DecodedAction:
        """Decode single slot action.
        
        Args:
            slot: Slot index (0 or 1)
            action: Slot action index (0-11)
            
        Returns:
            DecodedAction
        """
        if action < cls.NUM_MOVES:
            # Regular move
            return DecodedAction(
                slot=slot,
                action_type="move",
                move_index=action,
                target=0,  # Default target
                tera=False,
            )
        elif action < cls.NUM_MOVES * 2:
            # Tera move
            return DecodedAction(
                slot=slot,
                action_type="tera_move",
                move_index=action - cls.NUM_MOVES,
                target=0,
                tera=True,
            )
        else:
            # Switch
            return DecodedAction(
                slot=slot,
                action_type="switch",
                move_index=action - cls.NUM_MOVES * 2,
                target=-1,
                tera=False,
            )
    
    @classmethod
    def encode(cls, slot1_action: DecodedAction, slot2_action: DecodedAction) -> int:
        """Encode two slot actions to flat action.
        
        Args:
            slot1_action: Action for slot 1
            slot2_action: Action for slot 2
            
        Returns:
            Flat action index
        """
        return cls._encode_slot_action(slot1_action) * cls.ACTIONS_PER_SLOT + \
               cls._encode_slot_action(slot2_action)
    
    @classmethod
    def _encode_slot_action(cls, action: DecodedAction) -> int:
        """Encode single slot action."""
        if action.action_type == "move":
            return action.move_index
        elif action.action_type == "tera_move":
            return cls.NUM_MOVES + action.move_index
        else:  # switch
            return cls.NUM_MOVES * 2 + action.move_index


class HierarchicalActionSpace:
    """Hierarchical action space with three levels.
    
    Level 1: Strategy (4 options per slot)
    Level 2: Target (7 options for applicable strategies)
    Level 3: Detail (move/switch selection)
    
    This provides a more structured way to model VGC decisions.
    """
    
    STRATEGIES = ["attack", "support", "switch", "protect"]
    NUM_STRATEGIES = 4
    NUM_TARGETS = 7
    NUM_MOVES = 4
    NUM_BENCH = 4
    
    @classmethod
    def decode(
        cls,
        strategy: int,
        target: int,
        detail: int,
        slot: int = 0,
    ) -> DecodedAction:
        """Decode hierarchical action.
        
        Args:
            strategy: Strategy index (0-3)
            target: Target index (0-6)
            detail: Detail index (move/switch)
            slot: Active slot (0 or 1)
            
        Returns:
            DecodedAction
        """
        if strategy == ActionStrategy.ATTACK:
            return DecodedAction(
                slot=slot,
                action_type="move",
                move_index=detail % cls.NUM_MOVES,
                target=target,
                tera=detail >= cls.NUM_MOVES,
            )
        elif strategy == ActionStrategy.SUPPORT:
            return DecodedAction(
                slot=slot,
                action_type="move",
                move_index=detail,
                target=target,
                tera=False,
            )
        elif strategy == ActionStrategy.SWITCH:
            return DecodedAction(
                slot=slot,
                action_type="switch",
                move_index=detail,
                target=-1,
                tera=False,
            )
        else:  # PROTECT
            return DecodedAction(
                slot=slot,
                action_type="move",
                move_index=detail,  # Index of protect move
                target=ActionTarget.SELF,
                tera=False,
            )
    
    @classmethod
    def get_valid_targets(cls, strategy: int) -> List[int]:
        """Get valid targets for a strategy.
        
        Args:
            strategy: Strategy index
            
        Returns:
            List of valid target indices
        """
        if strategy == ActionStrategy.ATTACK:
            return [
                ActionTarget.OPP_SLOT_1,
                ActionTarget.OPP_SLOT_2,
                ActionTarget.ALLY,
                ActionTarget.ALL_OPPONENTS,
            ]
        elif strategy == ActionStrategy.SUPPORT:
            return [
                ActionTarget.ALLY,
                ActionTarget.SELF,
                ActionTarget.ALL,
            ]
        elif strategy == ActionStrategy.SWITCH:
            return [ActionTarget.NONE]
        else:  # PROTECT
            return [ActionTarget.SELF]
    
    @classmethod
    def get_detail_size(cls, strategy: int) -> int:
        """Get number of detail options for a strategy.
        
        Args:
            strategy: Strategy index
            
        Returns:
            Number of detail options
        """
        if strategy == ActionStrategy.ATTACK:
            return cls.NUM_MOVES * 2  # Moves + tera moves
        elif strategy == ActionStrategy.SUPPORT:
            return cls.NUM_MOVES
        elif strategy == ActionStrategy.SWITCH:
            return cls.NUM_BENCH
        else:  # PROTECT
            return 1  # Just protect


class HierarchicalActionHead(nn.Module):
    """Neural network head for hierarchical action prediction.
    
    Produces logits for strategy, target, and detail in sequence.
    """
    
    def __init__(
        self,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_strategies: int = 4,
        num_targets: int = 7,
        max_details: int = 8,  # Max detail options
        dropout: float = 0.1,
    ):
        """Initialize hierarchical action head.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            num_strategies: Number of strategies
            num_targets: Number of possible targets
            max_details: Maximum detail options
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_strategies = num_strategies
        self.num_targets = num_targets
        self.max_details = max_details
        
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Strategy head (first decision)
        self.strategy_head = nn.Linear(hidden_dim, num_strategies)
        
        # Target head (conditioned on strategy embedding)
        self.strategy_embed = nn.Embedding(num_strategies, hidden_dim // 4)
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_targets),
        )
        
        # Detail head (conditioned on strategy and target)
        self.target_embed = nn.Embedding(num_targets, hidden_dim // 4)
        self.detail_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, max_details),
        )
    
    def forward(
        self,
        state: torch.Tensor,
        strategy_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        detail_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            state: State features (batch, input_dim)
            strategy_mask: Valid strategies mask (batch, num_strategies)
            target_mask: Valid targets mask (batch, num_targets)
            detail_mask: Valid details mask (batch, max_details)
            
        Returns:
            Tuple of (strategy_logits, target_logits, detail_logits)
        """
        # Encode state
        features = self.encoder(state)
        
        # Strategy prediction
        strategy_logits = self.strategy_head(features)
        if strategy_mask is not None:
            strategy_logits = strategy_logits.masked_fill(~strategy_mask.bool(), -1e9)
        
        return strategy_logits, features
    
    def forward_full(
        self,
        state: torch.Tensor,
        strategy: torch.Tensor,
        target: torch.Tensor,
        strategy_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        detail_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward with known strategy and target.
        
        Used during training when we know the ground truth actions.
        
        Args:
            state: State features
            strategy: Ground truth strategy indices
            target: Ground truth target indices
            *_mask: Validity masks
            
        Returns:
            Tuple of all logits
        """
        features = self.encoder(state)
        
        # Strategy
        strategy_logits = self.strategy_head(features)
        if strategy_mask is not None:
            strategy_logits = strategy_logits.masked_fill(~strategy_mask.bool(), -1e9)
        
        # Target (conditioned on strategy)
        strategy_emb = self.strategy_embed(strategy)
        target_input = torch.cat([features, strategy_emb], dim=-1)
        target_logits = self.target_head(target_input)
        if target_mask is not None:
            target_logits = target_logits.masked_fill(~target_mask.bool(), -1e9)
        
        # Detail (conditioned on strategy and target)
        target_emb = self.target_embed(target)
        detail_input = torch.cat([features, strategy_emb, target_emb], dim=-1)
        detail_logits = self.detail_head(detail_input)
        if detail_mask is not None:
            detail_logits = detail_logits.masked_fill(~detail_mask.bool(), -1e9)
        
        return strategy_logits, target_logits, detail_logits
    
    def sample_action(
        self,
        state: torch.Tensor,
        strategy_mask: Optional[torch.Tensor] = None,
        target_mask: Optional[torch.Tensor] = None,
        detail_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action hierarchically.
        
        Args:
            state: State features
            *_mask: Validity masks
            temperature: Sampling temperature
            
        Returns:
            Tuple of (strategy, target, detail) indices
        """
        features = self.encoder(state)
        
        # Sample strategy
        strategy_logits = self.strategy_head(features) / temperature
        if strategy_mask is not None:
            strategy_logits = strategy_logits.masked_fill(~strategy_mask.bool(), -1e9)
        strategy = torch.multinomial(F.softmax(strategy_logits, dim=-1), 1).squeeze(-1)
        
        # Sample target
        strategy_emb = self.strategy_embed(strategy)
        target_input = torch.cat([features, strategy_emb], dim=-1)
        target_logits = self.target_head(target_input) / temperature
        if target_mask is not None:
            target_logits = target_logits.masked_fill(~target_mask.bool(), -1e9)
        target = torch.multinomial(F.softmax(target_logits, dim=-1), 1).squeeze(-1)
        
        # Sample detail
        target_emb = self.target_embed(target)
        detail_input = torch.cat([features, strategy_emb, target_emb], dim=-1)
        detail_logits = self.detail_head(detail_input) / temperature
        if detail_mask is not None:
            detail_logits = detail_logits.masked_fill(~detail_mask.bool(), -1e9)
        detail = torch.multinomial(F.softmax(detail_logits, dim=-1), 1).squeeze(-1)
        
        return strategy, target, detail


class ActionMaskGenerator:
    """Generate action masks based on battle state.
    
    Determines which actions are valid given the current
    game state, move PP, fainted Pokemon, etc.
    """
    
    def __init__(self, action_dim: int = 144):
        """Initialize mask generator.
        
        Args:
            action_dim: Size of action space
        """
        self.action_dim = action_dim
        self.actions_per_slot = 12
    
    def get_mask(
        self,
        move_pp: np.ndarray,          # (2, 4) PP for each move of each active
        can_switch: np.ndarray,       # (4,) which bench Pokemon can switch in
        can_tera: bool,               # Whether player can terastallize
        forced_action: Optional[int] = None,  # Forced action (e.g., recharge)
    ) -> np.ndarray:
        """Generate action mask.
        
        Args:
            move_pp: PP remaining for each move
            can_switch: Boolean array for switch targets
            can_tera: Whether tera is available
            forced_action: If set, only this action is valid
            
        Returns:
            Boolean mask of shape (action_dim,)
        """
        if forced_action is not None:
            mask = np.zeros(self.action_dim, dtype=bool)
            mask[forced_action] = True
            return mask
        
        # Generate mask for each slot
        slot1_mask = self._get_slot_mask(move_pp[0], can_switch, can_tera)
        slot2_mask = self._get_slot_mask(move_pp[1], can_switch, can_tera)
        
        # Combine into joint action mask
        mask = np.outer(slot1_mask, slot2_mask).flatten()
        
        return mask
    
    def _get_slot_mask(
        self,
        move_pp: np.ndarray,
        can_switch: np.ndarray,
        can_tera: bool,
    ) -> np.ndarray:
        """Generate mask for single slot."""
        mask = np.zeros(self.actions_per_slot, dtype=bool)
        
        # Regular moves (0-3)
        for i, pp in enumerate(move_pp):
            if pp > 0:
                mask[i] = True
        
        # Tera moves (4-7)
        if can_tera:
            for i, pp in enumerate(move_pp):
                if pp > 0:
                    mask[4 + i] = True
        
        # Switches (8-11)
        for i, can_sw in enumerate(can_switch):
            if can_sw:
                mask[8 + i] = True
        
        # Ensure at least one action is valid
        if not mask.any():
            mask[0] = True  # Default to first move (struggle)
        
        return mask
    
    def apply_mask(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Apply mask to logits.
        
        Args:
            logits: Action logits (batch, action_dim)
            mask: Boolean mask (batch, action_dim)
            
        Returns:
            Masked logits
        """
        return logits.masked_fill(~mask.bool(), -1e9)

