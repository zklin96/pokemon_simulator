"""Temporal battle context encoding.

This module captures battle history and momentum through
temporal features like action sequences, damage trends, and
switching patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
import math
from collections import deque


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""
    
    def __init__(
        self,
        embed_dim: int,
        max_len: int = 100,
        dropout: float = 0.1,
    ):
        """Initialize positional encoding.
        
        Args:
            embed_dim: Embedding dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * 
            -(math.log(10000.0) / embed_dim)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # (1, max_len, embed_dim)
        self.register_buffer("pe", pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.
        
        Args:
            x: Input tensor (batch, seq_len, embed_dim)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class ActionHistoryEncoder(nn.Module):
    """Encode recent action history.
    
    Uses an LSTM or Transformer to encode the sequence of
    recent actions and their outcomes.
    """
    
    def __init__(
        self,
        action_dim: int = 144,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        history_len: int = 10,
        encoder_type: str = "lstm",  # "lstm" or "transformer"
        dropout: float = 0.1,
    ):
        """Initialize action history encoder.
        
        Args:
            action_dim: Number of possible actions
            embed_dim: Action embedding dimension
            hidden_dim: Hidden/output dimension
            num_layers: Number of layers
            history_len: Maximum history length
            encoder_type: Type of encoder (lstm or transformer)
            dropout: Dropout rate
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.history_len = history_len
        self.encoder_type = encoder_type
        
        # Action embedding (with padding for empty history)
        self.action_embed = nn.Embedding(action_dim + 1, embed_dim, padding_idx=action_dim)
        
        # Outcome embedding (0=no outcome, 1=success, 2=failure)
        self.outcome_embed = nn.Embedding(3, embed_dim // 4)
        
        # Combined input projection
        input_dim = embed_dim + embed_dim // 4
        self.input_proj = nn.Linear(input_dim, embed_dim)
        
        if encoder_type == "lstm":
            self.encoder = nn.LSTM(
                embed_dim,
                hidden_dim,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0,
                bidirectional=False,
            )
        else:
            self.pos_encoding = PositionalEncoding(embed_dim, history_len, dropout)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=4,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
                batch_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.output_proj = nn.Linear(embed_dim, hidden_dim)
    
    def forward(
        self,
        action_history: torch.Tensor,   # (batch, history_len)
        outcome_history: torch.Tensor,  # (batch, history_len)
        mask: Optional[torch.Tensor] = None,  # (batch, history_len)
    ) -> torch.Tensor:
        """Encode action history.
        
        Args:
            action_history: Past action indices
            outcome_history: Past action outcomes
            mask: Mask for valid history entries
            
        Returns:
            History embedding (batch, hidden_dim)
        """
        # Embed actions and outcomes
        action_emb = self.action_embed(action_history)   # (batch, history, embed)
        outcome_emb = self.outcome_embed(outcome_history)  # (batch, history, embed//4)
        
        # Combine
        combined = torch.cat([action_emb, outcome_emb], dim=-1)
        x = self.input_proj(combined)  # (batch, history, embed)
        
        if self.encoder_type == "lstm":
            # LSTM encoding
            if mask is not None:
                # Pack sequence for efficiency
                lengths = mask.sum(dim=1).cpu()
                packed = nn.utils.rnn.pack_padded_sequence(
                    x, lengths, batch_first=True, enforce_sorted=False
                )
                _, (h, _) = self.encoder(packed)
            else:
                _, (h, _) = self.encoder(x)
            
            return h[-1]  # Last layer hidden state (batch, hidden)
        else:
            # Transformer encoding
            x = self.pos_encoding(x)
            
            if mask is not None:
                # Create attention mask (True means ignore)
                attn_mask = ~mask.bool()
            else:
                attn_mask = None
            
            encoded = self.encoder(x, src_key_padding_mask=attn_mask)
            
            # Pool over sequence (use last valid position or mean)
            if mask is not None:
                # Mean pooling over valid positions
                mask_expanded = mask.unsqueeze(-1)
                pooled = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1)
            else:
                pooled = encoded.mean(dim=1)
            
            return self.output_proj(pooled)


class DamageTrendEncoder(nn.Module):
    """Encode damage and HP trends over recent turns.
    
    Captures momentum by tracking damage dealt/received
    and HP changes over time.
    """
    
    def __init__(
        self,
        history_len: int = 10,
        hidden_dim: int = 32,
        output_dim: int = 32,
    ):
        """Initialize damage trend encoder.
        
        Args:
            history_len: Number of turns to track
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        super().__init__()
        
        self.history_len = history_len
        
        # Per-turn features: my_damage_dealt, my_damage_received,
        #                    my_kos, opp_kos, hp_diff
        features_per_turn = 5
        
        # Temporal convolution for trend detection
        self.conv = nn.Conv1d(
            features_per_turn,
            hidden_dim,
            kernel_size=3,
            padding=1,
        )
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            hidden_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        damage_history: torch.Tensor,  # (batch, history_len, 5)
    ) -> torch.Tensor:
        """Encode damage trends.
        
        Args:
            damage_history: Damage/HP features per turn
            
        Returns:
            Trend embedding (batch, output_dim)
        """
        batch_size = damage_history.size(0)
        
        # Convolution (expects channels first)
        x = damage_history.transpose(1, 2)  # (batch, features, history)
        x = F.relu(self.conv(x))
        x = x.transpose(1, 2)  # (batch, history, hidden)
        
        # LSTM
        _, (h, _) = self.lstm(x)
        h = h.squeeze(0)  # (batch, hidden)
        
        return self.output(h)


class SwitchingPatternEncoder(nn.Module):
    """Encode switching patterns.
    
    Tracks which Pokemon have been switched in/out and when,
    to detect patterns and predict future switches.
    """
    
    def __init__(
        self,
        num_pokemon: int = 6,
        history_len: int = 10,
        hidden_dim: int = 32,
        output_dim: int = 32,
    ):
        """Initialize switching pattern encoder.
        
        Args:
            num_pokemon: Number of Pokemon per team
            history_len: Number of turns to track
            hidden_dim: Hidden dimension
            output_dim: Output dimension
        """
        super().__init__()
        
        self.num_pokemon = num_pokemon
        self.history_len = history_len
        
        # Per-turn: which Pokemon are active (one-hot for each slot)
        # In doubles: 2 active per side = 4 Pokemon slots
        input_dim = num_pokemon * 2  # My + opponent active indicators
        
        self.encoder = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
        )
    
    def forward(
        self,
        switch_history: torch.Tensor,  # (batch, history_len, num_pokemon*2)
    ) -> torch.Tensor:
        """Encode switching patterns.
        
        Args:
            switch_history: Active Pokemon indicators per turn
            
        Returns:
            Pattern embedding (batch, output_dim)
        """
        _, (h, _) = self.encoder(switch_history)
        return self.output(h.squeeze(0))


class BattleContextEncoder(nn.Module):
    """Complete temporal battle context encoder.
    
    Combines action history, damage trends, and switching patterns
    to provide rich temporal context for decision making.
    """
    
    def __init__(
        self,
        action_dim: int = 144,
        history_len: int = 10,
        action_hidden: int = 64,
        damage_hidden: int = 32,
        switch_hidden: int = 32,
        output_dim: int = 128,
        dropout: float = 0.1,
    ):
        """Initialize battle context encoder.
        
        Args:
            action_dim: Number of possible actions
            history_len: Number of turns of history
            action_hidden: Action encoder hidden dim
            damage_hidden: Damage encoder hidden dim
            switch_hidden: Switching encoder hidden dim
            output_dim: Output dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.history_len = history_len
        
        # Sub-encoders
        self.action_encoder = ActionHistoryEncoder(
            action_dim=action_dim,
            hidden_dim=action_hidden,
            history_len=history_len,
            encoder_type="lstm",
        )
        
        self.damage_encoder = DamageTrendEncoder(
            history_len=history_len,
            output_dim=damage_hidden,
        )
        
        self.switch_encoder = SwitchingPatternEncoder(
            history_len=history_len,
            output_dim=switch_hidden,
        )
        
        # Turn counter embedding
        self.turn_embed = nn.Embedding(100, 16)  # Max 100 turns
        
        # Combine all context
        combined_dim = action_hidden + damage_hidden + switch_hidden + 16
        
        self.combine = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
        )
    
    def forward(
        self,
        action_history: torch.Tensor,   # (batch, history_len)
        outcome_history: torch.Tensor,  # (batch, history_len)
        damage_history: torch.Tensor,   # (batch, history_len, 5)
        switch_history: torch.Tensor,   # (batch, history_len, 12)
        current_turn: torch.Tensor,     # (batch,)
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode full battle context.
        
        Args:
            action_history: Past actions
            outcome_history: Past action outcomes
            damage_history: Damage/HP history
            switch_history: Active Pokemon history
            current_turn: Current turn number
            action_mask: Valid history mask
            
        Returns:
            Context embedding (batch, output_dim)
        """
        # Encode each component
        action_ctx = self.action_encoder(action_history, outcome_history, action_mask)
        damage_ctx = self.damage_encoder(damage_history)
        switch_ctx = self.switch_encoder(switch_history)
        turn_emb = self.turn_embed(current_turn.clamp(0, 99))
        
        # Combine
        combined = torch.cat([action_ctx, damage_ctx, switch_ctx, turn_emb], dim=-1)
        
        return self.combine(combined)


class BattleHistoryBuffer:
    """Buffer for tracking battle history during play.
    
    Maintains rolling history of actions, outcomes, and state
    changes for use during inference.
    """
    
    def __init__(self, history_len: int = 10):
        """Initialize buffer.
        
        Args:
            history_len: Maximum history length
        """
        self.history_len = history_len
        
        self.actions: deque = deque(maxlen=history_len)
        self.outcomes: deque = deque(maxlen=history_len)
        self.damage: deque = deque(maxlen=history_len)
        self.active_pokemon: deque = deque(maxlen=history_len)
        self.turn = 0
    
    def reset(self):
        """Reset buffer for new battle."""
        self.actions.clear()
        self.outcomes.clear()
        self.damage.clear()
        self.active_pokemon.clear()
        self.turn = 0
    
    def record_turn(
        self,
        action: int,
        outcome: int,  # 0=unknown, 1=success, 2=failure
        my_damage_dealt: float,
        my_damage_received: float,
        my_kos: int,
        opp_kos: int,
        hp_diff: float,
        my_active: List[int],  # Active Pokemon indices
        opp_active: List[int],
    ):
        """Record a turn's data.
        
        Args:
            action: Action taken
            outcome: Outcome of action
            my_damage_dealt: Damage dealt this turn
            my_damage_received: Damage received
            my_kos: Pokemon KOed this turn
            opp_kos: Opponent Pokemon KOed
            hp_diff: HP differential
            my_active: My active Pokemon indices
            opp_active: Opponent active Pokemon indices
        """
        self.actions.append(action)
        self.outcomes.append(outcome)
        self.damage.append([
            my_damage_dealt, my_damage_received, 
            float(my_kos), float(opp_kos), hp_diff
        ])
        
        # Create one-hot for active Pokemon
        active_vec = [0] * 12  # 6 my + 6 opp
        for idx in my_active:
            active_vec[idx] = 1
        for idx in opp_active:
            active_vec[6 + idx] = 1
        self.active_pokemon.append(active_vec)
        
        self.turn += 1
    
    def to_tensors(
        self,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, ...]:
        """Convert buffer to tensors for model input.
        
        Args:
            device: Target device
            
        Returns:
            Tuple of tensors for model input
        """
        # Pad to history_len if needed
        def pad_list(lst, pad_val, length):
            result = list(lst)
            while len(result) < length:
                result.insert(0, pad_val)
            return result
        
        action_list = pad_list(self.actions, 144, self.history_len)  # 144 = padding idx
        outcome_list = pad_list(self.outcomes, 0, self.history_len)
        damage_list = pad_list(self.damage, [0, 0, 0, 0, 0], self.history_len)
        active_list = pad_list(self.active_pokemon, [0] * 12, self.history_len)
        
        # Create mask (1 for valid, 0 for padding)
        valid_len = len(self.actions)
        mask = [0] * (self.history_len - valid_len) + [1] * valid_len
        
        return (
            torch.tensor([action_list], dtype=torch.long, device=device),
            torch.tensor([outcome_list], dtype=torch.long, device=device),
            torch.tensor([damage_list], dtype=torch.float, device=device),
            torch.tensor([active_list], dtype=torch.float, device=device),
            torch.tensor([self.turn], dtype=torch.long, device=device),
            torch.tensor([mask], dtype=torch.float, device=device),
        )

