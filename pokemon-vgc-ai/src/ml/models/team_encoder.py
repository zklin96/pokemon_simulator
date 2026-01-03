"""Attention-based team encoder for VGC AI.

This module provides transformer-based encoders that capture
Pokemon relationships and team synergies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention layer."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """Initialize attention.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            query: Query tensor (batch, seq_len, embed_dim)
            key: Key tensor (batch, seq_len, embed_dim)
            value: Value tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, _ = query.shape
        
        # Project and reshape
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: (batch, heads, seq, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)
        
        return out, attn_weights


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        """Initialize layer.
        
        Args:
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            ff_dim: Feedforward dimension (default: 4 * embed_dim)
            dropout: Dropout rate
        """
        super().__init__()
        
        ff_dim = ff_dim or embed_dim * 4
        
        self.self_attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual
        attn_out, attn_weights = self.self_attn(x, x, x, mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Feedforward with residual
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        x = self.norm2(x)
        
        return x, attn_weights


class TeamAttentionEncoder(nn.Module):
    """Attention-based encoder for Pokemon teams.
    
    Uses self-attention to model relationships between Pokemon
    on a team, capturing synergies and type coverage.
    
    Example:
        encoder = TeamAttentionEncoder(pokemon_dim=128)
        
        # pokemon_embeddings: (batch, 6, 128) - 6 Pokemon per team
        team_embedding = encoder(pokemon_embeddings)  # (batch, 256)
    """
    
    def __init__(
        self,
        pokemon_dim: int = 128,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        max_pokemon: int = 6,
        pooling: str = "mean",  # "mean", "cls", "max", "attention"
    ):
        """Initialize team encoder.
        
        Args:
            pokemon_dim: Input Pokemon embedding dimension
            hidden_dim: Hidden/output dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
            max_pokemon: Maximum Pokemon per team
            pooling: Pooling method for aggregating Pokemon
        """
        super().__init__()
        
        self.pokemon_dim = pokemon_dim
        self.hidden_dim = hidden_dim
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Linear(pokemon_dim, hidden_dim)
        
        # Positional embeddings (for slot position)
        self.position_embed = nn.Embedding(max_pokemon, hidden_dim)
        
        # CLS token for pooling (if used)
        if pooling == "cls":
            self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Attention pooling (if used)
        if pooling == "attention":
            self.pool_attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1),
            )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        pokemon_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[torch.Tensor]]:
        """Encode a team of Pokemon.
        
        Args:
            pokemon_embeddings: Pokemon embeddings (batch, num_pokemon, pokemon_dim)
            mask: Optional mask for valid Pokemon (batch, num_pokemon)
            return_attention: Whether to return attention weights
            
        Returns:
            Team embedding (batch, hidden_dim) or tuple with attention weights
        """
        batch_size, num_pokemon, _ = pokemon_embeddings.shape
        device = pokemon_embeddings.device
        
        # Project input
        x = self.input_proj(pokemon_embeddings)  # (batch, pokemon, hidden)
        
        # Add positional embeddings
        positions = torch.arange(num_pokemon, device=device)
        pos_emb = self.position_embed(positions)  # (pokemon, hidden)
        x = x + pos_emb.unsqueeze(0)
        
        # Prepend CLS token if using CLS pooling
        if self.pooling == "cls":
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            
            # Extend mask for CLS token
            if mask is not None:
                cls_mask = torch.ones(batch_size, 1, device=device)
                mask = torch.cat([cls_mask, mask], dim=1)
        
        # Apply transformer layers
        attention_weights = []
        for layer in self.layers:
            x, attn = layer(x, mask)
            attention_weights.append(attn)
        
        # Pool Pokemon embeddings
        if self.pooling == "cls":
            pooled = x[:, 0]  # CLS token
        elif self.pooling == "mean":
            if mask is not None:
                # Masked mean
                mask_expanded = mask.unsqueeze(-1)
                pooled = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
            else:
                pooled = x.mean(dim=1)
        elif self.pooling == "max":
            if mask is not None:
                x = x.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
            pooled = x.max(dim=1).values
        elif self.pooling == "attention":
            attn_scores = self.pool_attention(x)  # (batch, pokemon, 1)
            if mask is not None:
                attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1) == 0, float('-inf'))
            attn_weights_pool = F.softmax(attn_scores, dim=1)
            pooled = (x * attn_weights_pool).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        # Output projection
        output = self.output_proj(pooled)
        
        if return_attention:
            return output, attention_weights
        return output


class BattleStateEncoder(nn.Module):
    """Full battle state encoder using attention.
    
    Encodes both teams and field conditions to produce
    a complete battle state representation.
    """
    
    def __init__(
        self,
        pokemon_dim: int = 128,
        team_dim: int = 256,
        field_dim: int = 32,
        output_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize battle encoder.
        
        Args:
            pokemon_dim: Pokemon embedding dimension
            team_dim: Team encoder output dimension
            field_dim: Field condition dimension
            output_dim: Final output dimension
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.output_dim = output_dim
        
        # Team encoders (shared weights for both sides)
        self.team_encoder = TeamAttentionEncoder(
            pokemon_dim=pokemon_dim,
            hidden_dim=team_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Cross-team attention
        self.cross_attention = MultiHeadAttention(
            embed_dim=team_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # Combine teams and field
        combined_dim = team_dim * 2 + field_dim
        self.output_projection = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(
        self,
        my_pokemon: torch.Tensor,      # (batch, 6, pokemon_dim)
        opp_pokemon: torch.Tensor,     # (batch, 6, pokemon_dim)
        field_features: torch.Tensor,  # (batch, field_dim)
        my_mask: Optional[torch.Tensor] = None,
        opp_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode full battle state.
        
        Args:
            my_pokemon: My team's Pokemon embeddings
            opp_pokemon: Opponent's Pokemon embeddings
            field_features: Field condition features
            my_mask: Mask for my valid Pokemon
            opp_mask: Mask for opponent's valid Pokemon
            
        Returns:
            Battle state embedding (batch, output_dim)
        """
        # Encode both teams
        my_team = self.team_encoder(my_pokemon, my_mask)  # (batch, team_dim)
        opp_team = self.team_encoder(opp_pokemon, opp_mask)  # (batch, team_dim)
        
        # Cross-team attention (how my team views opponent)
        my_team_cross, _ = self.cross_attention(
            my_team.unsqueeze(1),
            opp_team.unsqueeze(1),
            opp_team.unsqueeze(1),
        )
        my_team_enhanced = my_team + my_team_cross.squeeze(1)
        
        # Concatenate all features
        combined = torch.cat([my_team_enhanced, opp_team, field_features], dim=-1)
        
        return self.output_projection(combined)


class HierarchicalBattleEncoder(nn.Module):
    """Hierarchical encoder: Pokemon -> Team -> Battle.
    
    Uses a three-level hierarchy:
    1. Pokemon level: Encode individual Pokemon
    2. Team level: Attention over Pokemon to get team embedding
    3. Battle level: Cross-attention between teams with field
    """
    
    def __init__(
        self,
        pokemon_input_dim: int = 50,
        pokemon_embed_dim: int = 128,
        team_dim: int = 256,
        battle_dim: int = 256,
        field_dim: int = 32,
        num_heads: int = 4,
        pokemon_layers: int = 1,
        team_layers: int = 2,
        dropout: float = 0.1,
    ):
        """Initialize hierarchical encoder.
        
        Args:
            pokemon_input_dim: Raw Pokemon feature dimension
            pokemon_embed_dim: Pokemon embedding dimension
            team_dim: Team embedding dimension
            battle_dim: Battle (output) dimension
            field_dim: Field condition dimension
            num_heads: Number of attention heads
            pokemon_layers: Pokemon-level transformer layers
            team_layers: Team-level transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.output_dim = battle_dim
        
        # Level 1: Pokemon encoder
        self.pokemon_encoder = nn.Sequential(
            nn.Linear(pokemon_input_dim, pokemon_embed_dim),
            nn.LayerNorm(pokemon_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Level 2: Team encoder
        self.team_encoder = TeamAttentionEncoder(
            pokemon_dim=pokemon_embed_dim,
            hidden_dim=team_dim,
            num_heads=num_heads,
            num_layers=team_layers,
            dropout=dropout,
            pooling="attention",
        )
        
        # Level 3: Battle encoder
        self.battle_encoder = nn.Sequential(
            nn.Linear(team_dim * 2 + field_dim, battle_dim),
            nn.LayerNorm(battle_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(battle_dim, battle_dim),
        )
    
    def forward(
        self,
        my_pokemon_features: torch.Tensor,   # (batch, 6, pokemon_input_dim)
        opp_pokemon_features: torch.Tensor,  # (batch, 6, pokemon_input_dim)
        field_features: torch.Tensor,        # (batch, field_dim)
        my_mask: Optional[torch.Tensor] = None,
        opp_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode battle state hierarchically.
        
        Args:
            my_pokemon_features: My Pokemon raw features
            opp_pokemon_features: Opponent Pokemon raw features
            field_features: Field condition features
            my_mask: Mask for my valid Pokemon
            opp_mask: Mask for opponent's valid Pokemon
            
        Returns:
            Battle state embedding (batch, battle_dim)
        """
        batch_size = my_pokemon_features.size(0)
        
        # Level 1: Encode each Pokemon
        my_pokemon_emb = self.pokemon_encoder(my_pokemon_features)
        opp_pokemon_emb = self.pokemon_encoder(opp_pokemon_features)
        
        # Level 2: Encode teams
        my_team = self.team_encoder(my_pokemon_emb, my_mask)
        opp_team = self.team_encoder(opp_pokemon_emb, opp_mask)
        
        # Level 3: Encode battle
        combined = torch.cat([my_team, opp_team, field_features], dim=-1)
        battle_emb = self.battle_encoder(combined)
        
        return battle_emb

