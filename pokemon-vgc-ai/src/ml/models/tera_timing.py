"""Tera Timing Optimization Network.

Predicts optimal timing for Terastallization in VGC battles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class TeraDecision:
    """Decision about Terastallization."""
    
    should_tera: bool
    confidence: float
    best_target: int  # 0 = slot A, 1 = slot B
    expected_value: float


class TeraTiming(nn.Module):
    """Predicts optimal Tera timing.
    
    Given current battle state, predicts:
    1. Whether to Tera this turn
    2. Which Pokemon to Tera (slot A or B)
    3. Expected value change from Terastallizing
    """
    
    def __init__(
        self,
        state_dim: int = 620,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        
        # State encoder
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Tera decision head (should Tera this turn?)
        self.should_tera_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )
        
        # Target selection head (which Pokemon to Tera?)
        self.target_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 2),  # 2 active Pokemon slots
        )
        
        # Value head (expected value change from Tera)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(
        self,
        state: torch.Tensor,  # [batch, state_dim]
        can_tera_mask: Optional[torch.Tensor] = None,  # [batch, 2]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict Tera timing decision.
        
        Args:
            state: Current battle state
            can_tera_mask: Mask for which slots can Tera (1 = can, 0 = cannot)
            
        Returns:
            Tuple of (should_tera_prob, target_logits, expected_value)
        """
        # Encode
        hidden = self.encoder(state)
        
        # Predict
        should_tera = self.should_tera_head(hidden).squeeze(-1)
        target_logits = self.target_head(hidden)
        value = self.value_head(hidden).squeeze(-1)
        
        # Apply mask to target logits
        if can_tera_mask is not None:
            target_logits = target_logits.masked_fill(
                ~can_tera_mask.bool(), float('-inf')
            )
        
        return should_tera, target_logits, value
    
    def decide(
        self,
        state: np.ndarray,
        can_tera: Tuple[bool, bool] = (True, True),
        threshold: float = 0.5,
    ) -> TeraDecision:
        """Make Tera decision for current state.
        
        Args:
            state: Current battle state
            can_tera: Whether each slot can Tera (slot_a, slot_b)
            threshold: Probability threshold for Tera
            
        Returns:
            TeraDecision with recommendation
        """
        self.eval()
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            mask = torch.tensor([can_tera], dtype=torch.float32)
            
            should_prob, target_logits, value = self.forward(state_t, mask)
            
            should_tera = should_prob.item() > threshold
            target_probs = F.softmax(target_logits, dim=-1)
            best_target = torch.argmax(target_probs[0]).item()
            
            return TeraDecision(
                should_tera=should_tera,
                confidence=should_prob.item(),
                best_target=best_target,
                expected_value=value.item(),
            )


class TeraTimingLoss(nn.Module):
    """Loss function for Tera timing training."""
    
    def __init__(self, value_weight: float = 0.5):
        super().__init__()
        self.value_weight = value_weight
    
    def forward(
        self,
        should_pred: torch.Tensor,
        target_pred: torch.Tensor,
        value_pred: torch.Tensor,
        should_label: torch.Tensor,
        target_label: torch.Tensor,
        value_label: torch.Tensor,
    ) -> torch.Tensor:
        """Compute combined loss."""
        # BCE for should_tera
        should_loss = F.binary_cross_entropy(should_pred, should_label)
        
        # Cross entropy for target
        target_loss = F.cross_entropy(target_pred, target_label)
        
        # MSE for value
        value_loss = F.mse_loss(value_pred, value_label)
        
        return should_loss + target_loss + self.value_weight * value_loss


def create_tera_model(state_dim: int = 620) -> TeraTiming:
    """Create Tera timing model.
    
    Args:
        state_dim: State dimension
        
    Returns:
        TeraTiming instance
    """
    return TeraTiming(state_dim=state_dim)

