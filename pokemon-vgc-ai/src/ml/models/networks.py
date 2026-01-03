"""Neural network architectures for VGC Battle AI."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any, List


class BattleNetwork(nn.Module):
    """Neural network for battle decision making.
    
    Architecture:
    - Input: Game state tensor (observation)
    - Hidden: Multiple fully connected layers with LayerNorm and ReLU
    - Output: Policy (action probabilities) and Value (state value)
    """
    
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: Tuple[int, ...] = (256, 256, 128),
        use_layer_norm: bool = True,
        dropout: float = 0.0,
    ):
        """Initialize the network.
        
        Args:
            observation_size: Size of observation vector
            action_size: Number of possible actions
            hidden_sizes: Sizes of hidden layers
            use_layer_norm: Whether to use layer normalization
            dropout: Dropout probability
        """
        super().__init__()
        
        self.observation_size = observation_size
        self.action_size = action_size
        
        # Build shared feature extractor
        layers = []
        prev_size = observation_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        self.shared = nn.Sequential(*layers)
        
        # Policy head (actor)
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
        )
        
        # Value head (critic)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    
    def forward(
        self, 
        observation: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            observation: Observation tensor of shape [batch, observation_size]
            action_mask: Optional boolean mask for valid actions
            
        Returns:
            Tuple of (policy_logits, value)
        """
        features = self.shared(observation)
        
        # Policy logits
        policy_logits = self.policy_head(features)
        
        # Apply action mask if provided
        if action_mask is not None:
            # Set invalid actions to very negative value
            policy_logits = policy_logits.masked_fill(~action_mask, -1e9)
        
        # Value
        value = self.value_head(features)
        
        return policy_logits, value.squeeze(-1)
    
    def get_action(
        self,
        observation: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy.
        
        Args:
            observation: Observation tensor
            action_mask: Optional action mask
            deterministic: If True, return argmax action
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        policy_logits, value = self.forward(observation, action_mask)
        
        # Convert to probabilities
        action_probs = F.softmax(policy_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(action_probs, dim=-1)
        else:
            # Sample from categorical distribution
            dist = torch.distributions.Categorical(probs=action_probs)
            action = dist.sample()
        
        # Compute log probability
        log_prob = F.log_softmax(policy_logits, dim=-1)
        action_log_prob = log_prob.gather(-1, action.unsqueeze(-1)).squeeze(-1)
        
        return action, action_log_prob, value


class DuelingNetwork(nn.Module):
    """Dueling network architecture for battle AI.
    
    Separates state value and action advantage estimation.
    """
    
    def __init__(
        self,
        observation_size: int,
        action_size: int,
        hidden_sizes: Tuple[int, ...] = (256, 256),
    ):
        """Initialize dueling network.
        
        Args:
            observation_size: Size of observation vector
            action_size: Number of possible actions
            hidden_sizes: Sizes of hidden layers
        """
        super().__init__()
        
        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], action_size),
        )
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            observation: Observation tensor
            
        Returns:
            Q-values for all actions
        """
        features = self.feature_layer(observation)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage (dueling formula)
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        
        return q_values


class AttentionBattleNetwork(nn.Module):
    """Battle network with attention mechanism.
    
    Uses self-attention to model relationships between Pokemon.
    """
    
    def __init__(
        self,
        pokemon_feature_size: int = 50,
        num_pokemon: int = 12,  # 6 player + 6 opponent
        field_feature_size: int = 20,
        action_size: int = 144,
        hidden_size: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        """Initialize attention network.
        
        Args:
            pokemon_feature_size: Features per Pokemon
            num_pokemon: Total Pokemon in observation
            field_feature_size: Field condition features
            action_size: Number of actions
            hidden_size: Hidden layer size
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
        """
        super().__init__()
        
        self.pokemon_feature_size = pokemon_feature_size
        self.num_pokemon = num_pokemon
        self.field_feature_size = field_feature_size
        
        # Pokemon embedding
        self.pokemon_embed = nn.Linear(pokemon_feature_size, hidden_size)
        
        # Field embedding
        self.field_embed = nn.Linear(field_feature_size, hidden_size)
        
        # Transformer encoder for Pokemon interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)
    
    def forward(
        self, 
        observation: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            observation: Flat observation tensor
            
        Returns:
            Tuple of (policy_logits, value)
        """
        batch_size = observation.shape[0]
        
        # Split observation into Pokemon features and field features
        pokemon_size = self.num_pokemon * self.pokemon_feature_size
        pokemon_features = observation[:, :pokemon_size].view(
            batch_size, self.num_pokemon, self.pokemon_feature_size
        )
        field_features = observation[:, pokemon_size:]
        
        # Embed Pokemon
        pokemon_embedded = self.pokemon_embed(pokemon_features)
        
        # Embed field as a single token
        field_embedded = self.field_embed(field_features).unsqueeze(1)
        
        # Combine tokens
        tokens = torch.cat([pokemon_embedded, field_embedded], dim=1)
        
        # Apply transformer
        encoded = self.transformer(tokens)
        
        # Pool across tokens (mean pooling)
        pooled = encoded.mean(dim=1)
        
        # Output
        policy_logits = self.policy_head(pooled)
        value = self.value_head(pooled)
        
        return policy_logits, value.squeeze(-1)


def test_networks():
    """Test network instantiation and forward pass."""
    from loguru import logger
    
    observation_size = 620
    action_size = 144
    batch_size = 4
    
    logger.info("Testing neural networks...")
    
    # Test BattleNetwork
    battle_net = BattleNetwork(observation_size, action_size)
    obs = torch.randn(batch_size, observation_size)
    policy, value = battle_net(obs)
    logger.info(f"BattleNetwork - Policy shape: {policy.shape}, Value shape: {value.shape}")
    
    # Test with action mask
    mask = torch.ones(batch_size, action_size, dtype=torch.bool)
    mask[:, 50:] = False  # Mask out some actions
    policy_masked, _ = battle_net(obs, mask)
    logger.info(f"Masked policy max: {policy_masked[:, 50:].max().item():.2f} (should be very negative)")
    
    # Test get_action
    action, log_prob, val = battle_net.get_action(obs)
    logger.info(f"Sampled action: {action.shape}, log_prob: {log_prob.shape}")
    
    # Test DuelingNetwork
    dueling_net = DuelingNetwork(observation_size, action_size)
    q_values = dueling_net(obs)
    logger.info(f"DuelingNetwork - Q-values shape: {q_values.shape}")
    
    # Test AttentionBattleNetwork
    attention_net = AttentionBattleNetwork(
        pokemon_feature_size=50,
        num_pokemon=12,
        field_feature_size=20,
        action_size=action_size,
    )
    policy_att, value_att = attention_net(obs)
    logger.info(f"AttentionNetwork - Policy shape: {policy_att.shape}")
    
    logger.info("All network tests passed!")


if __name__ == "__main__":
    test_networks()

