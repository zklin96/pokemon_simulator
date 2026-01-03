"""Learned embeddings for Pokemon components.

This module provides embedding layers for Pokemon species, abilities,
items, and moves. These replace one-hot encodings with learned
dense representations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger


class PokemonEmbedding(nn.Module):
    """Learned embeddings for Pokemon.
    
    Encodes species, ability, item, and moves into dense vectors
    that can capture semantic relationships.
    
    Example:
        embedding = PokemonEmbedding()
        
        # Input: (batch, pokemon_features)
        # - species_id: int
        # - ability_id: int
        # - item_id: int
        # - move_ids: list of 4 ints
        # - type_ids: list of 2 ints
        # - tera_type_id: int
        # - numerical_features: HP, stats, boosts, etc.
        
        output = embedding(pokemon_input)  # (batch, embed_dim)
    """
    
    def __init__(
        self,
        num_species: int = 1025,    # Includes all Pokemon through Gen 9
        num_abilities: int = 310,
        num_items: int = 250,
        num_moves: int = 920,
        num_types: int = 19,        # 18 types + stellar
        species_dim: int = 64,
        ability_dim: int = 32,
        item_dim: int = 32,
        move_dim: int = 48,
        type_dim: int = 16,
        numerical_dim: int = 32,    # For HP, stats, boosts
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Initialize Pokemon embedding.
        
        Args:
            num_species: Number of species to embed
            num_abilities: Number of abilities
            num_items: Number of items
            num_moves: Number of moves
            num_types: Number of types
            species_dim: Species embedding dimension
            ability_dim: Ability embedding dimension
            item_dim: Item embedding dimension
            move_dim: Move embedding dimension
            type_dim: Type embedding dimension
            numerical_dim: Numerical features projection dim
            output_dim: Final output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.output_dim = output_dim
        
        # Categorical embeddings
        self.species_embed = nn.Embedding(num_species, species_dim, padding_idx=0)
        self.ability_embed = nn.Embedding(num_abilities, ability_dim, padding_idx=0)
        self.item_embed = nn.Embedding(num_items, item_dim, padding_idx=0)
        self.move_embed = nn.Embedding(num_moves, move_dim, padding_idx=0)
        self.type_embed = nn.Embedding(num_types, type_dim, padding_idx=0)
        
        # Move aggregation (4 moves -> 1 vector)
        self.move_attention = nn.Sequential(
            nn.Linear(move_dim, move_dim),
            nn.Tanh(),
            nn.Linear(move_dim, 1),
        )
        
        # Type aggregation (2 types -> 1 vector)
        self.type_projection = nn.Linear(type_dim * 2, type_dim)
        
        # Numerical features projection
        # Expects: HP, ATK, DEF, SPA, SPD, SPE, boosts (7), status (6), etc.
        self.numerical_features = 20  # Number of numerical features
        self.numerical_projection = nn.Sequential(
            nn.Linear(self.numerical_features, numerical_dim),
            nn.LayerNorm(numerical_dim),
            nn.ReLU(),
        )
        
        # Combine all embeddings
        combined_dim = (
            species_dim + ability_dim + item_dim + 
            move_dim + type_dim + numerical_dim
        )
        
        self.output_projection = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize embedding weights."""
        for module in [
            self.species_embed, self.ability_embed,
            self.item_embed, self.move_embed, self.type_embed
        ]:
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)
    
    def forward(
        self,
        species_id: torch.Tensor,           # (batch,)
        ability_id: torch.Tensor,           # (batch,)
        item_id: torch.Tensor,              # (batch,)
        move_ids: torch.Tensor,             # (batch, 4)
        type_ids: torch.Tensor,             # (batch, 2)
        numerical_features: torch.Tensor,   # (batch, numerical_features)
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            species_id: Species indices
            ability_id: Ability indices
            item_id: Item indices
            move_ids: Move indices (4 per Pokemon)
            type_ids: Type indices (2 per Pokemon)
            numerical_features: HP, stats, boosts, status, etc.
            
        Returns:
            Pokemon embedding of shape (batch, output_dim)
        """
        batch_size = species_id.size(0)
        
        # Embed categorical features
        species_emb = self.species_embed(species_id)  # (batch, species_dim)
        ability_emb = self.ability_embed(ability_id)  # (batch, ability_dim)
        item_emb = self.item_embed(item_id)           # (batch, item_dim)
        
        # Embed and aggregate moves with attention
        move_embs = self.move_embed(move_ids)  # (batch, 4, move_dim)
        move_attn = self.move_attention(move_embs)  # (batch, 4, 1)
        move_attn = F.softmax(move_attn, dim=1)
        move_emb = (move_embs * move_attn).sum(dim=1)  # (batch, move_dim)
        
        # Embed and combine types
        type_embs = self.type_embed(type_ids)  # (batch, 2, type_dim)
        type_emb = self.type_projection(
            type_embs.view(batch_size, -1)
        )  # (batch, type_dim)
        
        # Project numerical features
        numerical_emb = self.numerical_projection(numerical_features)
        
        # Concatenate all embeddings
        combined = torch.cat([
            species_emb, ability_emb, item_emb,
            move_emb, type_emb, numerical_emb
        ], dim=-1)
        
        # Project to output dimension
        return self.output_projection(combined)
    
    @classmethod
    def from_flat_input(
        cls,
        flat_input: torch.Tensor,
        feature_spec: Dict[str, Tuple[int, int]],
        **kwargs
    ) -> torch.Tensor:
        """Create embedding from flat input tensor.
        
        Args:
            flat_input: Flat input tensor (batch, features)
            feature_spec: Dict mapping feature name to (start_idx, end_idx)
            **kwargs: Additional arguments for __init__
            
        Returns:
            Pokemon embedding
        """
        model = cls(**kwargs)
        
        # Extract features from flat input
        species_id = flat_input[:, feature_spec["species"][0]].long()
        ability_id = flat_input[:, feature_spec["ability"][0]].long()
        item_id = flat_input[:, feature_spec["item"][0]].long()
        
        move_start, move_end = feature_spec["moves"]
        move_ids = flat_input[:, move_start:move_end].long()
        
        type_start, type_end = feature_spec["types"]
        type_ids = flat_input[:, type_start:type_end].long()
        
        num_start, num_end = feature_spec["numerical"]
        numerical = flat_input[:, num_start:num_end]
        
        return model(species_id, ability_id, item_id, move_ids, type_ids, numerical)


class MoveEmbedding(nn.Module):
    """Specialized embedding for moves with attributes.
    
    Encodes move properties like type, category, power, accuracy,
    priority, and effects.
    """
    
    def __init__(
        self,
        num_moves: int = 920,
        num_types: int = 19,
        num_categories: int = 4,  # Physical, Special, Status, None
        base_embed_dim: int = 32,
        output_dim: int = 48,
    ):
        """Initialize move embedding.
        
        Args:
            num_moves: Number of moves
            num_types: Number of types
            num_categories: Number of move categories
            base_embed_dim: Base embedding dimension
            output_dim: Output dimension
        """
        super().__init__()
        
        self.move_embed = nn.Embedding(num_moves, base_embed_dim, padding_idx=0)
        self.type_embed = nn.Embedding(num_types, base_embed_dim // 2, padding_idx=0)
        self.category_embed = nn.Embedding(num_categories, base_embed_dim // 4)
        
        # Numerical attributes: power, accuracy, pp, priority
        self.numerical_projection = nn.Linear(4, base_embed_dim // 4)
        
        combined_dim = base_embed_dim + base_embed_dim // 2 + base_embed_dim // 4 + base_embed_dim // 4
        
        self.output_projection = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        move_id: torch.Tensor,
        move_type: torch.Tensor,
        category: torch.Tensor,
        numerical: torch.Tensor,  # power, accuracy, pp, priority
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            move_id: Move indices (batch,)
            move_type: Type indices (batch,)
            category: Category indices (batch,)
            numerical: Numerical attributes (batch, 4)
            
        Returns:
            Move embedding (batch, output_dim)
        """
        move_emb = self.move_embed(move_id)
        type_emb = self.type_embed(move_type)
        cat_emb = self.category_embed(category)
        num_emb = self.numerical_projection(numerical)
        
        combined = torch.cat([move_emb, type_emb, cat_emb, num_emb], dim=-1)
        return self.output_projection(combined)


class TypeEmbedding(nn.Module):
    """Embedding for Pokemon types with effectiveness relationships.
    
    Can optionally incorporate type chart information.
    """
    
    # Standard type chart (attacker -> defender effectiveness)
    TYPE_CHART = {
        "normal": {"rock": 0.5, "ghost": 0, "steel": 0.5},
        "fire": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 2, "bug": 2, 
                 "rock": 0.5, "dragon": 0.5, "steel": 2},
        "water": {"fire": 2, "water": 0.5, "grass": 0.5, "ground": 2, 
                  "rock": 2, "dragon": 0.5},
        # ... (full chart would be here)
    }
    
    def __init__(
        self,
        num_types: int = 19,
        embed_dim: int = 16,
        use_type_chart: bool = True,
    ):
        """Initialize type embedding.
        
        Args:
            num_types: Number of types
            embed_dim: Embedding dimension
            use_type_chart: Whether to incorporate type relationships
        """
        super().__init__()
        
        self.embed = nn.Embedding(num_types, embed_dim, padding_idx=0)
        self.use_type_chart = use_type_chart
        
        if use_type_chart:
            # Pre-computed type effectiveness matrix
            self.register_buffer(
                "type_chart",
                torch.ones(num_types, num_types)  # Would be populated with actual values
            )
            
            self.chart_projection = nn.Linear(num_types, embed_dim // 2)
            self.combine = nn.Linear(embed_dim + embed_dim // 2, embed_dim)
    
    def forward(self, type_id: torch.Tensor) -> torch.Tensor:
        """Get type embedding.
        
        Args:
            type_id: Type indices (batch,)
            
        Returns:
            Type embedding (batch, embed_dim)
        """
        base_emb = self.embed(type_id)
        
        if self.use_type_chart:
            # Get effectiveness row for this type
            effectiveness = self.type_chart[type_id]  # (batch, num_types)
            chart_emb = self.chart_projection(effectiveness)
            combined = torch.cat([base_emb, chart_emb], dim=-1)
            return self.combine(combined)
        
        return base_emb


class FieldEmbedding(nn.Module):
    """Embedding for field conditions.
    
    Encodes weather, terrain, and other field effects.
    """
    
    def __init__(
        self,
        num_weathers: int = 8,      # None, Sun, Rain, Sand, Snow, etc.
        num_terrains: int = 6,       # None, Electric, Grassy, Psychic, Misty, etc.
        embed_dim: int = 32,
        output_dim: int = 32,
    ):
        """Initialize field embedding.
        
        Args:
            num_weathers: Number of weather conditions
            num_terrains: Number of terrain types
            embed_dim: Base embedding dimension
            output_dim: Output dimension
        """
        super().__init__()
        
        self.weather_embed = nn.Embedding(num_weathers, embed_dim // 2)
        self.terrain_embed = nn.Embedding(num_terrains, embed_dim // 2)
        
        # Additional field state (trick room, tailwind, etc.)
        # Binary flags: trick_room, gravity, magic_room, wonder_room
        self.field_flags_projection = nn.Linear(4, embed_dim // 4)
        
        # Turns remaining for effects
        self.turns_projection = nn.Linear(4, embed_dim // 4)
        
        combined_dim = embed_dim + embed_dim // 2
        self.output = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        weather: torch.Tensor,        # (batch,)
        terrain: torch.Tensor,        # (batch,)
        field_flags: torch.Tensor,    # (batch, 4) binary
        turns_remaining: torch.Tensor # (batch, 4) turns left
    ) -> torch.Tensor:
        """Get field embedding.
        
        Args:
            weather: Weather index
            terrain: Terrain index
            field_flags: Binary field state flags
            turns_remaining: Turns remaining for each effect
            
        Returns:
            Field embedding (batch, output_dim)
        """
        weather_emb = self.weather_embed(weather)
        terrain_emb = self.terrain_embed(terrain)
        flags_emb = self.field_flags_projection(field_flags.float())
        turns_emb = self.turns_projection(turns_remaining.float())
        
        combined = torch.cat([weather_emb, terrain_emb, flags_emb, turns_emb], dim=-1)
        return self.output(combined)


class SideConditionEmbedding(nn.Module):
    """Embedding for side conditions.
    
    Encodes screens, hazards, and other side effects.
    """
    
    def __init__(
        self,
        output_dim: int = 16,
    ):
        """Initialize side condition embedding.
        
        Args:
            output_dim: Output dimension
        """
        super().__init__()
        
        # Features: reflect, light_screen, aurora_veil turns (3)
        #           tailwind turns (1)
        #           spikes, stealth_rock, sticky_web, toxic_spikes (4)
        self.projection = nn.Sequential(
            nn.Linear(8, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
    
    def forward(self, side_conditions: torch.Tensor) -> torch.Tensor:
        """Get side condition embedding.
        
        Args:
            side_conditions: Side condition features (batch, 8)
            
        Returns:
            Side embedding (batch, output_dim)
        """
        return self.projection(side_conditions)

