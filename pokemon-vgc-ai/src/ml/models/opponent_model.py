"""Opponent modeling for VGC battles.

Predicts opponent's likely next action based on battle history.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class BattleHistory:
    """History of actions in a battle."""
    
    # My actions [num_turns]
    my_actions: List[int]
    
    # Opponent actions [num_turns]
    opp_actions: List[int]
    
    # Battle states [num_turns, state_dim]
    states: List[np.ndarray]


class OpponentPredictor(nn.Module):
    """Predicts opponent's next action.
    
    Uses LSTM to process battle history and predicts
    distribution over opponent's possible actions.
    """
    
    def __init__(
        self,
        state_dim: int = 620,
        action_dim: int = 144,
        hidden_dim: int = 256,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
        )
        
        # Action embedding
        self.my_action_embed = nn.Embedding(action_dim, hidden_dim // 4)
        self.opp_action_embed = nn.Embedding(action_dim, hidden_dim // 4)
        
        # LSTM for sequence
        lstm_input_dim = hidden_dim // 2 + hidden_dim // 4 * 2
        self.lstm = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        
        # Prediction head
        self.pred_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
    
    def forward(
        self,
        states: torch.Tensor,        # [batch, seq_len, state_dim]
        my_actions: torch.Tensor,    # [batch, seq_len]
        opp_actions: torch.Tensor,   # [batch, seq_len]
    ) -> torch.Tensor:
        """Predict opponent's next action distribution.
        
        Args:
            states: Battle states sequence
            my_actions: My action sequence
            opp_actions: Opponent's action sequence
            
        Returns:
            Logits for opponent's next action [batch, action_dim]
        """
        batch_size, seq_len, _ = states.shape
        
        # Encode states
        state_enc = self.state_encoder(states)  # [batch, seq, hidden//2]
        
        # Embed actions
        my_act_emb = self.my_action_embed(my_actions)  # [batch, seq, hidden//4]
        opp_act_emb = self.opp_action_embed(opp_actions)  # [batch, seq, hidden//4]
        
        # Combine
        combined = torch.cat([state_enc, my_act_emb, opp_act_emb], dim=-1)
        
        # LSTM
        lstm_out, _ = self.lstm(combined)
        
        # Predict from last hidden state
        last_hidden = lstm_out[:, -1, :]
        
        return self.pred_head(last_hidden)
    
    def predict_action(
        self,
        history: BattleHistory,
        temperature: float = 1.0,
    ) -> Tuple[int, np.ndarray]:
        """Predict opponent's next action.
        
        Args:
            history: Battle history
            temperature: Sampling temperature
            
        Returns:
            Tuple of (predicted_action, action_probs)
        """
        self.eval()
        with torch.no_grad():
            # Prepare tensors
            states = torch.tensor(
                np.stack(history.states), dtype=torch.float32
            ).unsqueeze(0)
            my_actions = torch.tensor(
                history.my_actions, dtype=torch.long
            ).unsqueeze(0)
            opp_actions = torch.tensor(
                history.opp_actions, dtype=torch.long
            ).unsqueeze(0)
            
            # Forward
            logits = self.forward(states, my_actions, opp_actions)
            
            # Apply temperature
            probs = F.softmax(logits / temperature, dim=-1)
            
            # Sample or argmax
            if temperature > 0:
                action = torch.multinomial(probs[0], 1).item()
            else:
                action = torch.argmax(probs[0]).item()
            
            return action, probs[0].cpu().numpy()


def create_opponent_model(
    state_dim: int = 620,
    action_dim: int = 144,
) -> OpponentPredictor:
    """Create opponent prediction model.
    
    Args:
        state_dim: State dimension
        action_dim: Action dimension
        
    Returns:
        OpponentPredictor instance
    """
    return OpponentPredictor(state_dim=state_dim, action_dim=action_dim)
