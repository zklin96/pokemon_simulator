"""Imitation learning policy network for VGC battles.

This module defines the neural network architecture for behavioral cloning,
which learns to predict expert actions from game states.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import numpy as np


class ImitationPolicy(nn.Module):
    """Policy network for imitation learning.
    
    This network learns to predict expert actions from game states.
    It has both a policy head (for action prediction) and a value head
    (for later RL fine-tuning).
    """
    
    def __init__(
        self,
        state_dim: int = 620,
        action_dim: int = 144,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.1,
    ):
        """Initialize the policy network.
        
        Args:
            state_dim: Dimension of input state
            action_dim: Number of possible actions
            hidden_dims: Tuple of hidden layer dimensions
            dropout: Dropout probability
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Build encoder layers
        layers = []
        prev_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim),
        )
        
        # Value head (for RL fine-tuning)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # Smaller init for output layers
        nn.init.orthogonal_(self.policy_head[-1].weight, gain=0.01)
        nn.init.orthogonal_(self.value_head[-1].weight, gain=1.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            
        Returns:
            Tuple of (action_logits, value)
        """
        features = self.encoder(state)
        action_logits = self.policy_head(features)
        value = self.value_head(features)
        
        return action_logits, value
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities.
        
        Args:
            state: Batch of states
            
        Returns:
            Action probabilities [batch_size, action_dim]
        """
        logits, _ = self.forward(state)
        return F.softmax(logits, dim=-1)
    
    def get_action(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy.
        
        Args:
            state: Batch of states
            deterministic: If True, return argmax action
            
        Returns:
            Tuple of (action, log_prob)
        """
        logits, _ = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        log_prob = F.log_softmax(logits, dim=-1)
        selected_log_prob = log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return action, selected_log_prob
    
    def evaluate_actions(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO.
        
        Args:
            state: Batch of states
            action: Batch of actions
            
        Returns:
            Tuple of (log_prob, value, entropy)
        """
        logits, value = self.forward(state)
        
        log_probs = F.log_softmax(logits, dim=-1)
        action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return action_log_prob, value.squeeze(-1), entropy


class AttentionImitationPolicy(nn.Module):
    """Attention-based policy for better Pokemon understanding.
    
    Uses self-attention to model relationships between Pokemon
    and their moves/abilities.
    """
    
    def __init__(
        self,
        state_dim: int = 620,
        action_dim: int = 144,
        pokemon_features: int = 50,
        num_pokemon: int = 12,  # 6 per side
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize attention policy.
        
        Args:
            state_dim: Dimension of input state
            action_dim: Number of possible actions
            pokemon_features: Features per Pokemon
            num_pokemon: Total Pokemon (both sides)
            hidden_dim: Hidden dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.pokemon_features = pokemon_features
        self.num_pokemon = num_pokemon
        
        # Pokemon embedding
        self.pokemon_embed = nn.Linear(pokemon_features, hidden_dim)
        
        # Field state embedding (remaining features)
        field_dim = state_dim - (num_pokemon * pokemon_features)
        self.field_embed = nn.Linear(field_dim, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            state: Batch of states [batch_size, state_dim]
            
        Returns:
            Tuple of (action_logits, value)
        """
        batch_size = state.shape[0]
        
        # Split state into Pokemon and field features
        pokemon_features = state[:, :self.num_pokemon * self.pokemon_features]
        pokemon_features = pokemon_features.view(batch_size, self.num_pokemon, self.pokemon_features)
        
        field_features = state[:, self.num_pokemon * self.pokemon_features:]
        
        # Embed Pokemon
        pokemon_embeds = self.pokemon_embed(pokemon_features)  # [batch, num_pokemon, hidden]
        
        # Embed field state and add as extra token
        field_embed = self.field_embed(field_features).unsqueeze(1)  # [batch, 1, hidden]
        
        # Concatenate
        tokens = torch.cat([pokemon_embeds, field_embed], dim=1)  # [batch, num_pokemon+1, hidden]
        
        # Apply transformer
        encoded = self.transformer(tokens)
        
        # Use mean pooling for final representation
        pooled = encoded.mean(dim=1)
        
        # Output heads
        action_logits = self.policy_head(pooled)
        value = self.value_head(pooled)
        
        return action_logits, value
    
    def get_action(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy."""
        logits, _ = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        log_prob = F.log_softmax(logits, dim=-1)
        selected_log_prob = log_prob.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return action, selected_log_prob


def test_policy():
    """Test the policy networks."""
    from loguru import logger
    
    # Test basic policy
    policy = ImitationPolicy()
    logger.info(f"ImitationPolicy parameters: {sum(p.numel() for p in policy.parameters()):,}")
    
    # Test forward pass
    batch = torch.randn(32, 620)
    logits, value = policy(batch)
    logger.info(f"Output shapes: logits={logits.shape}, value={value.shape}")
    
    # Test action sampling
    action, log_prob = policy.get_action(batch)
    logger.info(f"Action shape: {action.shape}, log_prob shape: {log_prob.shape}")
    
    # Test attention policy
    attn_policy = AttentionImitationPolicy()
    logger.info(f"AttentionPolicy parameters: {sum(p.numel() for p in attn_policy.parameters()):,}")
    
    logits2, value2 = attn_policy(batch)
    logger.info(f"Attention output shapes: logits={logits2.shape}, value={value2.shape}")
    
    logger.info("Policy tests passed!")


if __name__ == "__main__":
    test_policy()

