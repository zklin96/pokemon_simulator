"""Unified encoder combining all advanced encoding components.

This module integrates:
- PokemonEmbedding: Learned embeddings for species, moves, items, abilities
- TeamAttentionEncoder: Self-attention for team synergies
- BattleContextEncoder: Temporal history (actions, damage, switches)
- FieldEmbedding: Weather, terrain, side conditions

Together they produce a rich, hierarchical battle state representation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List, Any
from dataclasses import dataclass
import numpy as np
from loguru import logger

from .embeddings import PokemonEmbedding, FieldEmbedding, SideConditionEmbedding
from .team_encoder import TeamAttentionEncoder, MultiHeadAttention
from .temporal_context import BattleContextEncoder, BattleHistoryBuffer


@dataclass
class StructuredBattleState:
    """Structured input for the unified encoder.
    
    Instead of a flat 620-dim vector, this provides structured
    access to all battle components.
    """
    # My Pokemon data (batch, 6, ...)
    my_species_ids: torch.Tensor       # (batch, 6)
    my_ability_ids: torch.Tensor       # (batch, 6)
    my_item_ids: torch.Tensor          # (batch, 6)
    my_move_ids: torch.Tensor          # (batch, 6, 4)
    my_type_ids: torch.Tensor          # (batch, 6, 2)
    my_numerical: torch.Tensor         # (batch, 6, 20) - HP, stats, boosts, status
    my_active_mask: torch.Tensor       # (batch, 6) - which are active
    my_alive_mask: torch.Tensor        # (batch, 6) - which are alive
    
    # Opponent Pokemon data
    opp_species_ids: torch.Tensor
    opp_ability_ids: torch.Tensor
    opp_item_ids: torch.Tensor
    opp_move_ids: torch.Tensor
    opp_type_ids: torch.Tensor
    opp_numerical: torch.Tensor
    opp_active_mask: torch.Tensor
    opp_alive_mask: torch.Tensor
    
    # Field conditions
    weather: torch.Tensor              # (batch,)
    terrain: torch.Tensor              # (batch,)
    field_flags: torch.Tensor          # (batch, 4) - trick room, gravity, etc.
    turns_remaining: torch.Tensor      # (batch, 4)
    
    # Side conditions
    my_side_conditions: torch.Tensor   # (batch, 8)
    opp_side_conditions: torch.Tensor  # (batch, 8)
    
    # Temporal context
    action_history: torch.Tensor       # (batch, history_len)
    outcome_history: torch.Tensor      # (batch, history_len)
    damage_history: torch.Tensor       # (batch, history_len, 5)
    switch_history: torch.Tensor       # (batch, history_len, 12)
    current_turn: torch.Tensor         # (batch,)
    history_mask: torch.Tensor         # (batch, history_len)
    
    @classmethod
    def from_flat_state(
        cls,
        flat_state: torch.Tensor,
        feature_spec: Optional[Dict] = None,
    ) -> "StructuredBattleState":
        """Create from flat 620-dim state (for backward compatibility).
        
        Args:
            flat_state: Flat state tensor (batch, 620)
            feature_spec: Feature specification mapping
            
        Returns:
            Structured battle state with default/zero values for unavailable data
        """
        batch_size = flat_state.size(0)
        device = flat_state.device
        
        # Default feature spec based on original game_state.py encoding
        if feature_spec is None:
            feature_spec = get_default_feature_spec()
        
        # Extract what we can from flat state
        # For parts not in flat state, use zeros (will be filled by trajectory builder)
        
        def extract_or_zeros(key, shape, dtype=torch.long):
            if key in feature_spec:
                start, end = feature_spec[key]
                data = flat_state[:, start:end]
                return data.reshape(batch_size, *shape).to(dtype)
            return torch.zeros(batch_size, *shape, dtype=dtype, device=device)
        
        history_len = 10
        
        return cls(
            # My Pokemon - extract from flat state where possible
            my_species_ids=extract_or_zeros("my_species", (6,)),
            my_ability_ids=extract_or_zeros("my_ability", (6,)),
            my_item_ids=extract_or_zeros("my_item", (6,)),
            my_move_ids=extract_or_zeros("my_moves", (6, 4)),
            my_type_ids=extract_or_zeros("my_types", (6, 2)),
            my_numerical=extract_or_zeros("my_numerical", (6, 20), torch.float),
            my_active_mask=extract_or_zeros("my_active", (6,), torch.float),
            my_alive_mask=extract_or_zeros("my_alive", (6,), torch.float),
            
            # Opponent Pokemon
            opp_species_ids=extract_or_zeros("opp_species", (6,)),
            opp_ability_ids=extract_or_zeros("opp_ability", (6,)),
            opp_item_ids=extract_or_zeros("opp_item", (6,)),
            opp_move_ids=extract_or_zeros("opp_moves", (6, 4)),
            opp_type_ids=extract_or_zeros("opp_types", (6, 2)),
            opp_numerical=extract_or_zeros("opp_numerical", (6, 20), torch.float),
            opp_active_mask=extract_or_zeros("opp_active", (6,), torch.float),
            opp_alive_mask=extract_or_zeros("opp_alive", (6,), torch.float),
            
            # Field
            weather=extract_or_zeros("weather", ()).squeeze(-1) if "weather" in feature_spec else torch.zeros(batch_size, dtype=torch.long, device=device),
            terrain=extract_or_zeros("terrain", ()).squeeze(-1) if "terrain" in feature_spec else torch.zeros(batch_size, dtype=torch.long, device=device),
            field_flags=extract_or_zeros("field_flags", (4,), torch.float),
            turns_remaining=extract_or_zeros("turns_remaining", (4,), torch.float),
            
            # Side conditions
            my_side_conditions=extract_or_zeros("my_side", (8,), torch.float),
            opp_side_conditions=extract_or_zeros("opp_side", (8,), torch.float),
            
            # Temporal (zeros if not available - will be filled during training)
            action_history=torch.zeros(batch_size, history_len, dtype=torch.long, device=device),
            outcome_history=torch.zeros(batch_size, history_len, dtype=torch.long, device=device),
            damage_history=torch.zeros(batch_size, history_len, 5, dtype=torch.float, device=device),
            switch_history=torch.zeros(batch_size, history_len, 12, dtype=torch.float, device=device),
            current_turn=torch.ones(batch_size, dtype=torch.long, device=device),
            history_mask=torch.zeros(batch_size, history_len, dtype=torch.float, device=device),
        )


def get_default_feature_spec() -> Dict[str, Tuple[int, int]]:
    """Get default feature specification for flat state.
    
    Based on the original 620-dim encoding from game_state.py.
    
    Returns:
        Dictionary mapping feature names to (start, end) indices
    """
    # The 620-dim encoding from GameStateEncoder:
    # Per Pokemon (6 my + 6 opp = 12 Pokemon):
    #   - HP fraction: 1
    #   - Stats (normalized): 6
    #   - Stat boosts: 7
    #   - Status: 6 (one-hot)
    #   - Types: 18 (one-hot)
    #   - Moves (one-hot or ID): variable
    #   - Active flag: 1
    #   - Alive flag: 1
    # Field: 20 (weather, terrain, turns, etc.)
    # This is simplified - actual encoding may vary
    
    # For now, return empty spec - trajectory builder will provide structured data
    return {}


class UnifiedBattleEncoder(nn.Module):
    """Unified encoder combining all advanced components.
    
    Architecture:
    1. Pokemon Embedding: Raw IDs -> dense vectors
    2. Team Attention: Capture synergies between team members
    3. Battle Context: Encode temporal history
    4. Cross-team Attention: Model matchup interactions
    5. Final projection: Combine all features
    
    Input: StructuredBattleState (or flat 620-dim for compatibility)
    Output: 256-dim battle embedding
    """
    
    def __init__(
        self,
        # Pokemon embedding config
        num_species: int = 1025,
        num_abilities: int = 310,
        num_items: int = 250,
        num_moves: int = 920,
        num_types: int = 19,
        pokemon_embed_dim: int = 128,
        
        # Team encoder config
        team_hidden_dim: int = 256,
        team_num_heads: int = 4,
        team_num_layers: int = 2,
        
        # Context encoder config
        action_dim: int = 144,
        history_len: int = 10,
        context_dim: int = 128,
        
        # Field encoder config
        field_dim: int = 32,
        side_dim: int = 16,
        
        # Output config
        output_dim: int = 256,
        dropout: float = 0.1,
        
        # Backward compatibility
        flat_state_dim: int = 620,
        support_flat_input: bool = True,
    ):
        """Initialize unified encoder.
        
        Args:
            num_species: Number of Pokemon species
            num_abilities: Number of abilities
            num_items: Number of items  
            num_moves: Number of moves
            num_types: Number of types
            pokemon_embed_dim: Pokemon embedding dimension
            team_hidden_dim: Team encoder hidden dimension
            team_num_heads: Attention heads for team encoder
            team_num_layers: Transformer layers for team encoder
            action_dim: Number of possible actions
            history_len: History length for context
            context_dim: Context encoder output dimension
            field_dim: Field condition embedding dimension
            side_dim: Side condition embedding dimension
            output_dim: Final output dimension
            dropout: Dropout rate
            flat_state_dim: Flat state dimension for compatibility
            support_flat_input: Whether to support flat 620-dim input
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.support_flat_input = support_flat_input
        self.flat_state_dim = flat_state_dim
        
        # === Pokemon Embedding ===
        self.pokemon_embed = PokemonEmbedding(
            num_species=num_species,
            num_abilities=num_abilities,
            num_items=num_items,
            num_moves=num_moves,
            num_types=num_types,
            output_dim=pokemon_embed_dim,
            dropout=dropout,
        )
        
        # === Team Attention Encoder ===
        self.team_encoder = TeamAttentionEncoder(
            pokemon_dim=pokemon_embed_dim,
            hidden_dim=team_hidden_dim,
            num_heads=team_num_heads,
            num_layers=team_num_layers,
            dropout=dropout,
            max_pokemon=6,
            pooling="attention",
        )
        
        # === Battle Context Encoder ===
        self.context_encoder = BattleContextEncoder(
            action_dim=action_dim,
            history_len=history_len,
            output_dim=context_dim,
            dropout=dropout,
        )
        
        # === Field and Side Condition Encoders ===
        self.field_embed = FieldEmbedding(output_dim=field_dim)
        self.my_side_embed = SideConditionEmbedding(output_dim=side_dim)
        self.opp_side_embed = SideConditionEmbedding(output_dim=side_dim)
        
        # === Cross-team Attention ===
        self.cross_attention = MultiHeadAttention(
            embed_dim=team_hidden_dim,
            num_heads=team_num_heads,
            dropout=dropout,
        )
        
        # === Final Combination ===
        # 2 teams + context + field + 2 sides
        combined_dim = team_hidden_dim * 2 + context_dim + field_dim + side_dim * 2
        
        self.output_projection = nn.Sequential(
            nn.Linear(combined_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
        )
        
        # === Flat state fallback encoder ===
        if support_flat_input:
            self.flat_encoder = nn.Sequential(
                nn.Linear(flat_state_dim, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim),
            )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def _embed_team(
        self,
        species_ids: torch.Tensor,
        ability_ids: torch.Tensor,
        item_ids: torch.Tensor,
        move_ids: torch.Tensor,
        type_ids: torch.Tensor,
        numerical: torch.Tensor,
        alive_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed a team of Pokemon.
        
        Args:
            species_ids: Species IDs (batch, 6)
            ability_ids: Ability IDs (batch, 6)
            item_ids: Item IDs (batch, 6)
            move_ids: Move IDs (batch, 6, 4)
            type_ids: Type IDs (batch, 6, 2)
            numerical: Numerical features (batch, 6, 20)
            alive_mask: Alive mask (batch, 6)
            
        Returns:
            Tuple of (pokemon_embeddings, team_embedding)
        """
        batch_size = species_ids.size(0)
        num_pokemon = species_ids.size(1)
        
        # Embed each Pokemon individually
        pokemon_embs = []
        for i in range(num_pokemon):
            emb = self.pokemon_embed(
                species_id=species_ids[:, i],
                ability_id=ability_ids[:, i],
                item_id=item_ids[:, i],
                move_ids=move_ids[:, i],
                type_ids=type_ids[:, i],
                numerical_features=numerical[:, i],
            )
            pokemon_embs.append(emb)
        
        # Stack into team
        pokemon_embs = torch.stack(pokemon_embs, dim=1)  # (batch, 6, pokemon_dim)
        
        # Apply team attention with alive mask
        team_emb = self.team_encoder(pokemon_embs, mask=alive_mask)  # (batch, team_dim)
        
        return pokemon_embs, team_emb
    
    def forward_structured(
        self,
        state: StructuredBattleState,
    ) -> torch.Tensor:
        """Forward pass with structured input.
        
        Args:
            state: Structured battle state
            
        Returns:
            Battle embedding (batch, output_dim)
        """
        # Embed my team
        my_pokemon_embs, my_team_emb = self._embed_team(
            state.my_species_ids,
            state.my_ability_ids,
            state.my_item_ids,
            state.my_move_ids,
            state.my_type_ids,
            state.my_numerical,
            state.my_alive_mask,
        )
        
        # Embed opponent team
        opp_pokemon_embs, opp_team_emb = self._embed_team(
            state.opp_species_ids,
            state.opp_ability_ids,
            state.opp_item_ids,
            state.opp_move_ids,
            state.opp_type_ids,
            state.opp_numerical,
            state.opp_alive_mask,
        )
        
        # Cross-team attention (my team attending to opponent)
        my_team_cross, _ = self.cross_attention(
            my_team_emb.unsqueeze(1),
            opp_team_emb.unsqueeze(1),
            opp_team_emb.unsqueeze(1),
        )
        my_team_enhanced = my_team_emb + my_team_cross.squeeze(1)
        
        # Encode battle context (temporal)
        context_emb = self.context_encoder(
            action_history=state.action_history,
            outcome_history=state.outcome_history,
            damage_history=state.damage_history,
            switch_history=state.switch_history,
            current_turn=state.current_turn,
            action_mask=state.history_mask,
        )
        
        # Encode field conditions
        field_emb = self.field_embed(
            weather=state.weather,
            terrain=state.terrain,
            field_flags=state.field_flags,
            turns_remaining=state.turns_remaining,
        )
        
        # Encode side conditions
        my_side_emb = self.my_side_embed(state.my_side_conditions)
        opp_side_emb = self.opp_side_embed(state.opp_side_conditions)
        
        # Combine all features
        combined = torch.cat([
            my_team_enhanced,
            opp_team_emb,
            context_emb,
            field_emb,
            my_side_emb,
            opp_side_emb,
        ], dim=-1)
        
        return self.output_projection(combined)
    
    def forward_flat(self, flat_state: torch.Tensor) -> torch.Tensor:
        """Forward pass with flat 620-dim input (backward compatibility).
        
        Args:
            flat_state: Flat state tensor (batch, 620)
            
        Returns:
            Battle embedding (batch, output_dim)
        """
        if not self.support_flat_input:
            raise ValueError("Flat input not supported. Set support_flat_input=True")
        
        return self.flat_encoder(flat_state)
    
    def forward(
        self,
        state: torch.Tensor | StructuredBattleState,
        use_structured: bool = False,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            state: Either flat tensor (batch, 620) or StructuredBattleState
            use_structured: Force structured interpretation
            
        Returns:
            Battle embedding (batch, output_dim)
        """
        if isinstance(state, StructuredBattleState):
            return self.forward_structured(state)
        elif use_structured:
            # Convert flat to structured
            structured = StructuredBattleState.from_flat_state(state)
            return self.forward_structured(structured)
        else:
            # Use flat encoder for backward compatibility
            return self.forward_flat(state)


class UnifiedEncoderWithValue(nn.Module):
    """Unified encoder with value head for actor-critic methods.
    
    Wraps UnifiedBattleEncoder and adds a value head for
    use with PPO and other actor-critic algorithms.
    """
    
    def __init__(
        self,
        encoder_config: Optional[Dict] = None,
        value_hidden_dim: int = 128,
    ):
        """Initialize encoder with value head.
        
        Args:
            encoder_config: Config for UnifiedBattleEncoder
            value_hidden_dim: Hidden dimension for value head
        """
        super().__init__()
        
        config = encoder_config or {}
        self.encoder = UnifiedBattleEncoder(**config)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1),
        )
    
    def forward(
        self,
        state: torch.Tensor | StructuredBattleState,
        use_structured: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            state: Battle state
            use_structured: Force structured interpretation
            
        Returns:
            Tuple of (encoded_state, value)
        """
        encoded = self.encoder(state, use_structured)
        value = self.value_head(encoded)
        
        return encoded, value


def test_unified_encoder():
    """Test the unified encoder."""
    logger.info("Testing UnifiedBattleEncoder...")
    
    batch_size = 4
    device = "cpu"
    
    # Create encoder
    encoder = UnifiedBattleEncoder(
        support_flat_input=True,
    )
    
    num_params = sum(p.numel() for p in encoder.parameters())
    logger.info(f"Encoder parameters: {num_params:,}")
    
    # Test flat input
    flat_state = torch.randn(batch_size, 620)
    output_flat = encoder(flat_state)
    logger.info(f"Flat input shape: {flat_state.shape} -> output: {output_flat.shape}")
    
    # Test structured input
    structured_state = StructuredBattleState(
        my_species_ids=torch.randint(0, 1000, (batch_size, 6)),
        my_ability_ids=torch.randint(0, 300, (batch_size, 6)),
        my_item_ids=torch.randint(0, 200, (batch_size, 6)),
        my_move_ids=torch.randint(0, 900, (batch_size, 6, 4)),
        my_type_ids=torch.randint(0, 18, (batch_size, 6, 2)),
        my_numerical=torch.randn(batch_size, 6, 20),
        my_active_mask=torch.zeros(batch_size, 6),
        my_alive_mask=torch.ones(batch_size, 6),
        
        opp_species_ids=torch.randint(0, 1000, (batch_size, 6)),
        opp_ability_ids=torch.randint(0, 300, (batch_size, 6)),
        opp_item_ids=torch.randint(0, 200, (batch_size, 6)),
        opp_move_ids=torch.randint(0, 900, (batch_size, 6, 4)),
        opp_type_ids=torch.randint(0, 18, (batch_size, 6, 2)),
        opp_numerical=torch.randn(batch_size, 6, 20),
        opp_active_mask=torch.zeros(batch_size, 6),
        opp_alive_mask=torch.ones(batch_size, 6),
        
        weather=torch.randint(0, 8, (batch_size,)),
        terrain=torch.randint(0, 6, (batch_size,)),
        field_flags=torch.zeros(batch_size, 4),
        turns_remaining=torch.zeros(batch_size, 4),
        
        my_side_conditions=torch.zeros(batch_size, 8),
        opp_side_conditions=torch.zeros(batch_size, 8),
        
        action_history=torch.randint(0, 144, (batch_size, 10)),
        outcome_history=torch.randint(0, 3, (batch_size, 10)),
        damage_history=torch.randn(batch_size, 10, 5),
        switch_history=torch.zeros(batch_size, 10, 12),
        current_turn=torch.ones(batch_size, dtype=torch.long),
        history_mask=torch.ones(batch_size, 10),
    )
    
    output_structured = encoder(structured_state)
    logger.info(f"Structured input -> output: {output_structured.shape}")
    
    # Test with value head
    encoder_with_value = UnifiedEncoderWithValue()
    encoded, value = encoder_with_value(flat_state)
    logger.info(f"With value head: encoded={encoded.shape}, value={value.shape}")
    
    logger.info("All tests passed!")


if __name__ == "__main__":
    test_unified_encoder()

