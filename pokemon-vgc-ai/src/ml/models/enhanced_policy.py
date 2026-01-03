"""Enhanced policy network using unified encoder and hierarchical actions.

This module integrates:
- UnifiedBattleEncoder: Rich state representation
- HierarchicalActionHead: Structured action output
- Action masking: Invalid action prevention
- Value head: For actor-critic methods

Supports both flat (144) and hierarchical action spaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass
import numpy as np
from loguru import logger

from .unified_encoder import UnifiedBattleEncoder, StructuredBattleState, UnifiedEncoderWithValue
from .action_space import (
    HierarchicalActionHead, ActionMaskGenerator, FlatActionSpace,
    HierarchicalActionSpace, DecodedAction
)


@dataclass
class EnhancedPolicyConfig:
    """Configuration for enhanced policy."""
    # Encoder config
    num_species: int = 1025
    num_abilities: int = 310
    num_items: int = 250
    num_moves: int = 920
    num_types: int = 19
    pokemon_embed_dim: int = 128
    team_hidden_dim: int = 256
    team_num_heads: int = 4
    team_num_layers: int = 2
    action_dim: int = 144
    history_len: int = 10
    context_dim: int = 128
    field_dim: int = 32
    side_dim: int = 16
    encoder_output_dim: int = 256
    
    # Action space config
    use_hierarchical_actions: bool = False  # Start with flat for compatibility
    num_strategies: int = 4
    num_targets: int = 7
    max_details: int = 8
    
    # Network config
    hidden_dim: int = 256
    dropout: float = 0.1
    
    # Compatibility
    flat_state_dim: int = 620
    support_flat_input: bool = True


class EnhancedPolicy(nn.Module):
    """Enhanced policy using unified encoder.
    
    Features:
    - Rich state encoding via UnifiedBattleEncoder
    - Both flat (144) and hierarchical action support
    - Action masking for invalid actions
    - Value head for actor-critic methods
    - Backward compatible with flat 620-dim states
    """
    
    def __init__(self, config: Optional[EnhancedPolicyConfig] = None):
        """Initialize enhanced policy.
        
        Args:
            config: Policy configuration
        """
        super().__init__()
        
        self.config = config or EnhancedPolicyConfig()
        cfg = self.config
        
        # === State Encoder ===
        self.encoder = UnifiedBattleEncoder(
            num_species=cfg.num_species,
            num_abilities=cfg.num_abilities,
            num_items=cfg.num_items,
            num_moves=cfg.num_moves,
            num_types=cfg.num_types,
            pokemon_embed_dim=cfg.pokemon_embed_dim,
            team_hidden_dim=cfg.team_hidden_dim,
            team_num_heads=cfg.team_num_heads,
            team_num_layers=cfg.team_num_layers,
            action_dim=cfg.action_dim,
            history_len=cfg.history_len,
            context_dim=cfg.context_dim,
            field_dim=cfg.field_dim,
            side_dim=cfg.side_dim,
            output_dim=cfg.encoder_output_dim,
            dropout=cfg.dropout,
            flat_state_dim=cfg.flat_state_dim,
            support_flat_input=cfg.support_flat_input,
        )
        
        # === Action Heads ===
        if cfg.use_hierarchical_actions:
            # Hierarchical action head
            self.action_head = HierarchicalActionHead(
                input_dim=cfg.encoder_output_dim,
                hidden_dim=cfg.hidden_dim,
                num_strategies=cfg.num_strategies,
                num_targets=cfg.num_targets,
                max_details=cfg.max_details,
                dropout=cfg.dropout,
            )
        else:
            # Flat action head (144 actions)
            self.action_head = nn.Sequential(
                nn.Linear(cfg.encoder_output_dim, cfg.hidden_dim),
                nn.LayerNorm(cfg.hidden_dim),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, cfg.action_dim),
            )
        
        # === Value Head ===
        self.value_head = nn.Sequential(
            nn.Linear(cfg.encoder_output_dim, cfg.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim // 2, 1),
        )
        
        # === Action Mask Generator ===
        self.mask_generator = ActionMaskGenerator(action_dim=cfg.action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize output layer weights with small values."""
        if isinstance(self.action_head, nn.Sequential):
            # Initialize last layer with small weights
            last_layer = self.action_head[-1]
            if isinstance(last_layer, nn.Linear):
                nn.init.orthogonal_(last_layer.weight, gain=0.01)
                nn.init.zeros_(last_layer.bias)
        
        # Value head
        for module in self.value_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        state: Union[torch.Tensor, StructuredBattleState],
        action_mask: Optional[torch.Tensor] = None,
        use_structured: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            state: Battle state (flat or structured)
            action_mask: Optional action validity mask (batch, action_dim)
            use_structured: Force structured state interpretation
            
        Returns:
            Tuple of (action_logits, value)
        """
        # Encode state
        encoded = self.encoder(state, use_structured=use_structured)
        
        # Get action logits
        if self.config.use_hierarchical_actions:
            # Hierarchical: just get strategy logits for now
            strategy_logits, _ = self.action_head(encoded)
            action_logits = strategy_logits
        else:
            # Flat action space
            action_logits = self.action_head(encoded)
            
            # Apply action mask if provided
            if action_mask is not None:
                action_logits = action_logits.masked_fill(~action_mask.bool(), -1e9)
        
        # Get value
        value = self.value_head(encoded)
        
        return action_logits, value
    
    def get_action(
        self,
        state: Union[torch.Tensor, StructuredBattleState],
        action_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        temperature: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action from policy.
        
        Args:
            state: Battle state
            action_mask: Action validity mask
            deterministic: If True, take argmax action
            temperature: Sampling temperature
            
        Returns:
            Tuple of (action, log_prob, value)
        """
        action_logits, value = self.forward(state, action_mask)
        
        # Scale by temperature
        scaled_logits = action_logits / temperature
        
        # Sample or argmax
        probs = F.softmax(scaled_logits, dim=-1)
        
        if deterministic:
            action = torch.argmax(probs, dim=-1)
        else:
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        # Get log probability
        log_probs = F.log_softmax(scaled_logits, dim=-1)
        selected_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        return action, selected_log_prob, value.squeeze(-1)
    
    def evaluate_actions(
        self,
        state: Union[torch.Tensor, StructuredBattleState],
        action: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO.
        
        Args:
            state: Battle states (batch)
            action: Actions taken (batch)
            action_mask: Action validity masks (batch)
            
        Returns:
            Tuple of (log_prob, value, entropy)
        """
        action_logits, value = self.forward(state, action_mask)
        
        # Log probabilities
        log_probs = F.log_softmax(action_logits, dim=-1)
        action_log_prob = log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)
        
        # Entropy
        probs = F.softmax(action_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        
        return action_log_prob, value.squeeze(-1), entropy
    
    def get_action_probs(
        self,
        state: Union[torch.Tensor, StructuredBattleState],
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get action probabilities.
        
        Args:
            state: Battle state
            action_mask: Action validity mask
            
        Returns:
            Action probabilities (batch, action_dim)
        """
        action_logits, _ = self.forward(state, action_mask)
        return F.softmax(action_logits, dim=-1)
    
    def get_features(
        self,
        state: Union[torch.Tensor, StructuredBattleState],
        use_structured: bool = False,
    ) -> torch.Tensor:
        """Get encoded features (for use as SB3 feature extractor).
        
        Args:
            state: Battle state
            use_structured: Force structured interpretation
            
        Returns:
            Encoded features (batch, encoder_output_dim)
        """
        return self.encoder(state, use_structured=use_structured)


class EnhancedPolicyForSB3(nn.Module):
    """Enhanced policy adapted for Stable-Baselines3.
    
    Provides the interface expected by SB3's ActorCriticPolicy.
    """
    
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        config: Optional[EnhancedPolicyConfig] = None,
        **kwargs,
    ):
        """Initialize for SB3 compatibility.
        
        Args:
            observation_space: SB3 observation space
            action_space: SB3 action space
            lr_schedule: Learning rate schedule
            config: Policy configuration
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Initialize enhanced policy
        self.policy = EnhancedPolicy(config)
        
        # Get dimensions
        self.features_dim = self.policy.config.encoder_output_dim
    
    def forward(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass for SB3.
        
        Args:
            obs: Observations
            deterministic: Whether to use deterministic actions
            
        Returns:
            Tuple of (actions, values, log_probs)
        """
        return self.policy.get_action(obs, deterministic=deterministic)
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate actions for SB3 PPO.
        
        Args:
            obs: Observations
            actions: Actions to evaluate
            
        Returns:
            Tuple of (values, log_probs, entropy)
        """
        log_prob, value, entropy = self.policy.evaluate_actions(obs, actions)
        return value, log_prob, entropy
    
    def get_distribution(self, obs: torch.Tensor):
        """Get action distribution for SB3.
        
        Args:
            obs: Observations
            
        Returns:
            Categorical distribution
        """
        logits, _ = self.policy.forward(obs)
        return torch.distributions.Categorical(logits=logits)
    
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict values for SB3.
        
        Args:
            obs: Observations
            
        Returns:
            Value predictions
        """
        _, value = self.policy.forward(obs)
        return value


class PretrainedEnhancedExtractor(nn.Module):
    """Feature extractor using pretrained EnhancedPolicy encoder.
    
    For use with SB3's custom feature extractors.
    """
    
    def __init__(
        self,
        observation_space,
        pretrained_path: Optional[str] = None,
        freeze_encoder: bool = False,
        config: Optional[EnhancedPolicyConfig] = None,
    ):
        """Initialize pretrained feature extractor.
        
        Args:
            observation_space: SB3 observation space
            pretrained_path: Path to pretrained model
            freeze_encoder: Whether to freeze encoder weights
            config: Policy config (used if no pretrained path)
        """
        super().__init__()
        
        config = config or EnhancedPolicyConfig()
        self.features_dim = config.encoder_output_dim
        
        # Create encoder
        self.encoder = UnifiedBattleEncoder(
            num_species=config.num_species,
            num_abilities=config.num_abilities,
            num_items=config.num_items,
            num_moves=config.num_moves,
            num_types=config.num_types,
            pokemon_embed_dim=config.pokemon_embed_dim,
            team_hidden_dim=config.team_hidden_dim,
            team_num_heads=config.team_num_heads,
            team_num_layers=config.team_num_layers,
            action_dim=config.action_dim,
            history_len=config.history_len,
            context_dim=config.context_dim,
            field_dim=config.field_dim,
            side_dim=config.side_dim,
            output_dim=config.encoder_output_dim,
            dropout=config.dropout,
            flat_state_dim=config.flat_state_dim,
            support_flat_input=config.support_flat_input,
        )
        
        # Load pretrained weights if provided
        if pretrained_path:
            self._load_pretrained(pretrained_path)
        
        # Optionally freeze encoder
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def _load_pretrained(self, path: str):
        """Load pretrained encoder weights.
        
        Args:
            path: Path to pretrained model checkpoint
        """
        logger.info(f"Loading pretrained encoder from {path}")
        
        checkpoint = torch.load(path, map_location="cpu")
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Extract encoder weights
        encoder_state = {}
        for key, value in state_dict.items():
            if key.startswith("encoder."):
                encoder_state[key[8:]] = value  # Remove "encoder." prefix
        
        if encoder_state:
            self.encoder.load_state_dict(encoder_state, strict=False)
            logger.info(f"Loaded {len(encoder_state)} encoder weights")
        else:
            logger.warning("No encoder weights found in checkpoint")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations.
        
        Args:
            observations: Input observations
            
        Returns:
            Encoded features
        """
        return self.encoder(observations)


def create_enhanced_policy(
    state_dim: int = 620,
    action_dim: int = 144,
    use_hierarchical: bool = False,
    pretrained_path: Optional[str] = None,
    **kwargs,
) -> EnhancedPolicy:
    """Factory function to create enhanced policy.
    
    Args:
        state_dim: State dimension (for flat input)
        action_dim: Action dimension
        use_hierarchical: Whether to use hierarchical actions
        pretrained_path: Path to pretrained weights
        **kwargs: Additional config options
        
    Returns:
        Configured EnhancedPolicy
    """
    config = EnhancedPolicyConfig(
        flat_state_dim=state_dim,
        action_dim=action_dim,
        use_hierarchical_actions=use_hierarchical,
        **kwargs,
    )
    
    policy = EnhancedPolicy(config)
    
    if pretrained_path:
        logger.info(f"Loading pretrained policy from {pretrained_path}")
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        
        if "model_state_dict" in checkpoint:
            policy.load_state_dict(checkpoint["model_state_dict"], strict=False)
        else:
            policy.load_state_dict(checkpoint, strict=False)
    
    return policy


def test_enhanced_policy():
    """Test the enhanced policy."""
    logger.info("Testing EnhancedPolicy...")
    
    batch_size = 4
    state_dim = 620
    action_dim = 144
    
    # Create policy
    policy = EnhancedPolicy()
    
    num_params = sum(p.numel() for p in policy.parameters())
    logger.info(f"Policy parameters: {num_params:,}")
    
    # Test forward pass
    state = torch.randn(batch_size, state_dim)
    action_logits, value = policy(state)
    logger.info(f"Forward: logits={action_logits.shape}, value={value.shape}")
    
    # Test get_action
    action, log_prob, val = policy.get_action(state)
    logger.info(f"Get action: action={action.shape}, log_prob={log_prob.shape}")
    
    # Test evaluate_actions
    log_prob, val, entropy = policy.evaluate_actions(state, action)
    logger.info(f"Evaluate: log_prob={log_prob.shape}, entropy={entropy.shape}")
    
    # Test with action mask
    mask = torch.ones(batch_size, action_dim)
    mask[:, 50:] = 0  # Mask out some actions
    action_logits_masked, _ = policy(state, action_mask=mask)
    logger.info(f"With mask: max logit in masked region = {action_logits_masked[:, 50:].max():.2f}")
    
    # Test hierarchical policy
    config = EnhancedPolicyConfig(use_hierarchical_actions=True)
    hierarchical_policy = EnhancedPolicy(config)
    h_logits, h_value = hierarchical_policy(state)
    logger.info(f"Hierarchical: strategy_logits={h_logits.shape}")
    
    # Test feature extraction
    features = policy.get_features(state)
    logger.info(f"Features shape: {features.shape}")
    
    logger.info("All tests passed!")


if __name__ == "__main__":
    test_enhanced_policy()

