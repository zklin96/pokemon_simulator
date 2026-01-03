"""Team Preview AI for VGC team selection.

In VGC, before each battle you see both teams (6 Pokemon each) and 
must select 4 to bring, with 2 leading (starting active).

This module implements a neural network to make these selections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger


@dataclass
class TeamPreviewState:
    """State for team preview decision."""
    
    # My team embeddings [6, embed_dim]
    my_team: torch.Tensor
    
    # Opponent team embeddings [6, embed_dim]
    opp_team: torch.Tensor
    
    # Format info (optional)
    format_id: Optional[str] = None


class PokemonEmbedding(nn.Module):
    """Embed a Pokemon for team preview.
    
    Uses learned embeddings for species, types, items, abilities
    combined with stat features.
    """
    
    def __init__(
        self,
        num_species: int = 1600,  # Cover all Gen 9 Pokemon
        num_types: int = 19,       # 18 types + stellar
        num_items: int = 500,
        num_abilities: int = 300,
        embed_dim: int = 64,
    ):
        super().__init__()
        
        # Embedding layers
        self.species_embed = nn.Embedding(num_species, embed_dim // 2)
        self.type_embed = nn.Embedding(num_types, embed_dim // 4)
        self.item_embed = nn.Embedding(num_items, embed_dim // 4)
        self.ability_embed = nn.Embedding(num_abilities, embed_dim // 4)
        
        # Stats projection (6 stats)
        self.stats_proj = nn.Linear(6, embed_dim // 4)
        
        # Combine embeddings
        # species (embed_dim//2) + 2 types (embed_dim//4 * 2) + item (embed_dim//4) 
        # + ability (embed_dim//4) + stats (embed_dim//4)
        # = embed_dim//2 + embed_dim//4 * 5 = embed_dim * 7 // 4
        combined_dim = embed_dim * 7 // 4
        self.combine = nn.Sequential(
            nn.Linear(combined_dim, embed_dim),
            nn.ReLU(),
            nn.LayerNorm(embed_dim),
        )
        
        self.embed_dim = embed_dim
    
    def forward(
        self,
        species_ids: torch.Tensor,       # [batch, 6]
        type_ids: torch.Tensor,          # [batch, 6, 2]
        item_ids: torch.Tensor,          # [batch, 6]
        ability_ids: torch.Tensor,       # [batch, 6]
        stats: torch.Tensor,             # [batch, 6, 6]
    ) -> torch.Tensor:
        """Embed a team of 6 Pokemon.
        
        Returns:
            Tensor of shape [batch, 6, embed_dim]
        """
        batch_size = species_ids.shape[0]
        
        # Species embedding [batch, 6, embed_dim//2]
        species_emb = self.species_embed(species_ids)
        
        # Type embeddings [batch, 6, 2, embed_dim//4] -> [batch, 6, embed_dim//2]
        type_emb = self.type_embed(type_ids)
        type_emb = type_emb.view(batch_size, 6, -1)
        
        # Item embedding [batch, 6, embed_dim//4]
        item_emb = self.item_embed(item_ids)
        
        # Ability embedding [batch, 6, embed_dim//4]
        ability_emb = self.ability_embed(ability_ids)
        
        # Stats [batch, 6, embed_dim//4]
        stats_emb = self.stats_proj(stats)
        
        # Combine all
        combined = torch.cat([
            species_emb, type_emb, item_emb, ability_emb, stats_emb
        ], dim=-1)
        
        return self.combine(combined)


class CrossTeamAttention(nn.Module):
    """Attention between my team and opponent team.
    
    Each of my Pokemon attends to all opponent Pokemon to understand
    matchup implications.
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(
        self,
        my_team: torch.Tensor,   # [batch, 6, embed_dim]
        opp_team: torch.Tensor,  # [batch, 6, embed_dim]
    ) -> torch.Tensor:
        """Apply cross-attention from my team to opponent.
        
        Returns:
            Enhanced my team embeddings [batch, 6, embed_dim]
        """
        # Cross attention: my team queries, opponent team is key/value
        attn_out, _ = self.cross_attn(my_team, opp_team, opp_team)
        return self.norm(my_team + attn_out)


class TeamPreviewNetwork(nn.Module):
    """Neural network for team preview selection.
    
    Architecture:
    1. Embed both teams
    2. Cross-attention to understand matchups
    3. Selection head: score each Pokemon for selection
    4. Lead head: score each selected Pokemon for leading
    """
    
    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Pokemon embedder
        self.pokemon_embed = PokemonEmbedding(embed_dim=embed_dim)
        
        # Cross-team attention layers
        self.cross_attn_layers = nn.ModuleList([
            CrossTeamAttention(embed_dim, num_heads)
            for _ in range(num_layers)
        ])
        
        # Self-attention within my team
        self.self_attn = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        
        # Selection head: score each Pokemon for bring-4 selection
        self.selection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )
        
        # Lead head: score each Pokemon for lead-2 selection
        self.lead_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
        )
    
    def forward(
        self,
        my_team_features: Dict[str, torch.Tensor],
        opp_team_features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for team preview.
        
        Args:
            my_team_features: Dict with species_ids, type_ids, etc.
            opp_team_features: Dict with species_ids, type_ids, etc.
            
        Returns:
            Tuple of:
            - selection_logits: [batch, 6] scores for each Pokemon
            - lead_logits: [batch, 6] lead scores for each Pokemon
        """
        # Embed both teams
        my_emb = self.pokemon_embed(
            my_team_features["species_ids"],
            my_team_features["type_ids"],
            my_team_features["item_ids"],
            my_team_features["ability_ids"],
            my_team_features["stats"],
        )
        
        opp_emb = self.pokemon_embed(
            opp_team_features["species_ids"],
            opp_team_features["type_ids"],
            opp_team_features["item_ids"],
            opp_team_features["ability_ids"],
            opp_team_features["stats"],
        )
        
        # Cross-attention layers
        for layer in self.cross_attn_layers:
            my_emb = layer(my_emb, opp_emb)
        
        # Self-attention within my team
        my_emb = self.self_attn(my_emb)
        
        # Selection scores
        selection_logits = self.selection_head(my_emb).squeeze(-1)  # [batch, 6]
        
        # Lead scores
        lead_logits = self.lead_head(my_emb).squeeze(-1)  # [batch, 6]
        
        return selection_logits, lead_logits
    
    def select_team(
        self,
        my_team_features: Dict[str, torch.Tensor],
        opp_team_features: Dict[str, torch.Tensor],
        top_k_bring: int = 4,
        top_k_lead: int = 2,
    ) -> Tuple[List[int], List[int]]:
        """Select which Pokemon to bring and lead with.
        
        Args:
            my_team_features: My team feature dict
            opp_team_features: Opponent team feature dict
            top_k_bring: Number of Pokemon to bring (default 4)
            top_k_lead: Number to lead with (default 2)
            
        Returns:
            Tuple of:
            - bring_indices: Indices of Pokemon to bring
            - lead_indices: Indices of lead Pokemon (subset of bring)
        """
        self.eval()
        with torch.no_grad():
            selection_logits, lead_logits = self.forward(
                my_team_features, opp_team_features
            )
            
            # Get top-k for selection
            selection_probs = F.softmax(selection_logits, dim=-1)
            _, bring_indices = torch.topk(selection_probs[0], k=top_k_bring)
            bring_indices = bring_indices.cpu().tolist()
            
            # Get top-k leads from brought Pokemon
            lead_probs = F.softmax(lead_logits[0, bring_indices], dim=-1)
            _, relative_lead_indices = torch.topk(lead_probs, k=top_k_lead)
            lead_indices = [bring_indices[i] for i in relative_lead_indices.cpu().tolist()]
            
            return bring_indices, lead_indices


class TeamPreviewTrainer:
    """Trainer for team preview network."""
    
    def __init__(
        self,
        model: TeamPreviewNetwork,
        learning_rate: float = 1e-4,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Loss functions
        self.selection_loss_fn = nn.BCEWithLogitsLoss()
        self.lead_loss_fn = nn.BCEWithLogitsLoss()
    
    def train_step(
        self,
        my_team_features: Dict[str, torch.Tensor],
        opp_team_features: Dict[str, torch.Tensor],
        target_selection: torch.Tensor,  # [batch, 6] binary
        target_leads: torch.Tensor,      # [batch, 6] binary
    ) -> Dict[str, float]:
        """Single training step.
        
        Args:
            my_team_features: My team features
            opp_team_features: Opponent team features
            target_selection: Ground truth selection (1 for brought)
            target_leads: Ground truth leads (1 for lead)
            
        Returns:
            Dict with loss values
        """
        self.model.train()
        
        # Move to device
        my_team_features = {k: v.to(self.device) for k, v in my_team_features.items()}
        opp_team_features = {k: v.to(self.device) for k, v in opp_team_features.items()}
        target_selection = target_selection.to(self.device)
        target_leads = target_leads.to(self.device)
        
        # Forward
        selection_logits, lead_logits = self.model(my_team_features, opp_team_features)
        
        # Losses
        selection_loss = self.selection_loss_fn(
            selection_logits, target_selection.float()
        )
        lead_loss = self.lead_loss_fn(lead_logits, target_leads.float())
        
        total_loss = selection_loss + 0.5 * lead_loss
        
        # Backward
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            "total_loss": total_loss.item(),
            "selection_loss": selection_loss.item(),
            "lead_loss": lead_loss.item(),
        }


def create_team_preview_model(embed_dim: int = 128) -> TeamPreviewNetwork:
    """Factory function to create team preview model.
    
    Args:
        embed_dim: Embedding dimension
        
    Returns:
        TeamPreviewNetwork instance
    """
    return TeamPreviewNetwork(embed_dim=embed_dim)

