"""Imitation learning (behavioral cloning) training for VGC AI.

This module trains a policy network to predict expert actions from game states
using the trajectories extracted from VGC-Bench replays.

Supports:
- ImitationPolicy (simple MLP)
- AttentionImitationPolicy (transformer-based)
- EnhancedPolicy (unified encoder + hierarchical actions)
- JSON and Parquet data loading
- Mixed precision training
"""

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from loguru import logger
from tqdm import tqdm

from src.ml.models.imitation_policy import ImitationPolicy, AttentionImitationPolicy
from src.ml.models.enhanced_policy import EnhancedPolicy, EnhancedPolicyConfig


@dataclass
class TrainingConfig:
    """Configuration for imitation learning training."""
    # Data
    trajectory_path: str = "data/processed/trajectories/trajectories_logs-gen9vgc2024regg.json"
    data_format: str = "json"  # "json" or "parquet"
    val_split: float = 0.1
    
    # Model type: "simple", "attention", or "enhanced"
    model_type: str = "simple"
    
    # Model params for simple/attention
    state_dim: int = 620
    action_dim: int = 144
    hidden_dims: Tuple[int, ...] = (512, 256, 128)
    dropout: float = 0.1
    
    # Enhanced policy config (used if model_type == "enhanced")
    enhanced_config: Optional[Dict[str, Any]] = None
    
    # Training
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 20
    patience: int = 5  # Early stopping patience
    
    # Mixed precision
    use_amp: bool = True
    
    # Loss weighting
    winner_weight: float = 1.5
    loser_weight: float = 0.5
    
    # Output
    save_dir: str = "data/models/imitation"
    log_interval: int = 100


class TrajectoryDataset(Dataset):
    """Dataset for trajectory transitions.
    
    Supports both flat states (620-dim) and structured data for enhanced encoder.
    """
    
    def __init__(
        self, 
        trajectories: List[Dict], 
        weight_by_outcome: bool = True,
        include_structured: bool = False,
    ):
        """Initialize dataset.
        
        Args:
            trajectories: List of trajectory dictionaries
            weight_by_outcome: Whether to weight by game outcome
            include_structured: Whether to include structured data for enhanced encoder
        """
        self.transitions = []
        self.weights = []
        self.include_structured = include_structured
        
        for traj in trajectories:
            won = traj['won']
            weight = 1.5 if won else 0.5 if weight_by_outcome else 1.0
            
            for trans in traj['transitions']:
                trans_data = {
                    'state': np.array(trans['state'], dtype=np.float32),
                    'action': trans['action'],
                    'reward': trans['reward'],
                    'won': won,
                }
                
                # Include target info if available
                if 'slot_a_target' in trans:
                    trans_data['slot_a_target'] = trans['slot_a_target']
                    trans_data['slot_b_target'] = trans['slot_b_target']
                
                # Include structured data if available and requested
                if include_structured and 'structured' in trans:
                    trans_data['structured'] = trans['structured']
                
                self.transitions.append(trans_data)
                self.weights.append(weight)
        
        self.weights = np.array(self.weights, dtype=np.float32)
        
        logger.info(f"Created dataset with {len(self.transitions)} transitions")
        if include_structured:
            num_with_structured = sum(1 for t in self.transitions if 'structured' in t)
            logger.info(f"  {num_with_structured} have structured data")
    
    def __len__(self) -> int:
        return len(self.transitions)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single transition.
        
        Returns:
            Dictionary with state, action, weight, and optionally structured data
        """
        trans = self.transitions[idx]
        
        result = {
            'state': torch.tensor(trans['state']),
            'action': torch.tensor(trans['action'], dtype=torch.long),
            'weight': torch.tensor(self.weights[idx]),
        }
        
        # Add targets if available
        if 'slot_a_target' in trans:
            result['slot_a_target'] = torch.tensor(trans['slot_a_target'], dtype=torch.long)
            result['slot_b_target'] = torch.tensor(trans['slot_b_target'], dtype=torch.long)
        
        # Add structured data if available
        if self.include_structured and 'structured' in trans:
            structured = trans['structured']
            result['structured'] = self._convert_structured(structured)
        
        return result
    
    def _convert_structured(self, structured: Dict) -> Dict[str, torch.Tensor]:
        """Convert structured data dict to tensors."""
        return {
            # My Pokemon (6 per team)
            'my_species_ids': torch.tensor(structured.get('my_species_ids', [0]*6), dtype=torch.long),
            'my_ability_ids': torch.tensor(structured.get('my_ability_ids', [0]*6), dtype=torch.long),
            'my_item_ids': torch.tensor(structured.get('my_item_ids', [0]*6), dtype=torch.long),
            'my_move_ids': torch.tensor(structured.get('my_move_ids', [[0]*4]*6), dtype=torch.long),
            'my_type_ids': torch.tensor(structured.get('my_type_ids', [[0]*2]*6), dtype=torch.long),
            'my_numerical': torch.tensor(structured.get('my_numerical', [[0.0]*20]*6), dtype=torch.float32),
            'my_active_mask': torch.tensor(structured.get('my_active_mask', [0.0]*6), dtype=torch.float32),
            'my_alive_mask': torch.tensor(structured.get('my_alive_mask', [1.0]*6), dtype=torch.float32),
            
            # Opponent Pokemon
            'opp_species_ids': torch.tensor(structured.get('opp_species_ids', [0]*6), dtype=torch.long),
            'opp_ability_ids': torch.tensor(structured.get('opp_ability_ids', [0]*6), dtype=torch.long),
            'opp_item_ids': torch.tensor(structured.get('opp_item_ids', [0]*6), dtype=torch.long),
            'opp_move_ids': torch.tensor(structured.get('opp_move_ids', [[0]*4]*6), dtype=torch.long),
            'opp_type_ids': torch.tensor(structured.get('opp_type_ids', [[0]*2]*6), dtype=torch.long),
            'opp_numerical': torch.tensor(structured.get('opp_numerical', [[0.0]*20]*6), dtype=torch.float32),
            'opp_active_mask': torch.tensor(structured.get('opp_active_mask', [0.0]*6), dtype=torch.float32),
            'opp_alive_mask': torch.tensor(structured.get('opp_alive_mask', [1.0]*6), dtype=torch.float32),
            
            # Field
            'weather': torch.tensor(structured.get('weather', 0), dtype=torch.long),
            'terrain': torch.tensor(structured.get('terrain', 0), dtype=torch.long),
            'field_flags': torch.tensor(structured.get('field_flags', [0.0]*4), dtype=torch.float32),
            'turns_remaining': torch.tensor(structured.get('turns_remaining', [0.0]*4), dtype=torch.float32),
            
            # Side conditions
            'my_side_conditions': torch.tensor(structured.get('my_side_conditions', [0.0]*8), dtype=torch.float32),
            'opp_side_conditions': torch.tensor(structured.get('opp_side_conditions', [0.0]*8), dtype=torch.float32),
            
            # Temporal (simplified - use zeros if not provided)
            'current_turn': torch.tensor(structured.get('current_turn', 0), dtype=torch.long),
        }


def collate_structured_batch(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching structured data.
    
    Args:
        batch: List of sample dictionaries from dataset
        
    Returns:
        Batched dictionary with stacked tensors
    """
    result = {
        'state': torch.stack([b['state'] for b in batch]),
        'action': torch.stack([b['action'] for b in batch]),
        'weight': torch.stack([b['weight'] for b in batch]),
    }
    
    # Handle optional targets
    if 'slot_a_target' in batch[0]:
        result['slot_a_target'] = torch.stack([b['slot_a_target'] for b in batch])
        result['slot_b_target'] = torch.stack([b['slot_b_target'] for b in batch])
    
    # Handle structured data if present
    if 'structured' in batch[0]:
        structured_keys = batch[0]['structured'].keys()
        result['structured'] = {}
        for key in structured_keys:
            result['structured'][key] = torch.stack([b['structured'][key] for b in batch])
    
    return result


def build_structured_state_from_batch(
    batch_structured: Dict[str, torch.Tensor],
    device: torch.device,
) -> 'StructuredBattleState':
    """Convert batched structured dict to StructuredBattleState.
    
    Args:
        batch_structured: Dictionary of batched tensors
        device: Target device
        
    Returns:
        StructuredBattleState dataclass
    """
    from src.ml.models.unified_encoder import StructuredBattleState
    
    batch_size = batch_structured['my_species_ids'].size(0)
    history_len = 10
    
    # Create default temporal tensors if not provided
    action_history = torch.zeros(batch_size, history_len, dtype=torch.long, device=device)
    outcome_history = torch.zeros(batch_size, history_len, dtype=torch.float32, device=device)
    damage_history = torch.zeros(batch_size, history_len, 5, dtype=torch.float32, device=device)
    switch_history = torch.zeros(batch_size, history_len, 12, dtype=torch.float32, device=device)
    history_mask = torch.zeros(batch_size, history_len, dtype=torch.float32, device=device)
    
    return StructuredBattleState(
        my_species_ids=batch_structured['my_species_ids'].to(device),
        my_ability_ids=batch_structured['my_ability_ids'].to(device),
        my_item_ids=batch_structured['my_item_ids'].to(device),
        my_move_ids=batch_structured['my_move_ids'].to(device),
        my_type_ids=batch_structured['my_type_ids'].to(device),
        my_numerical=batch_structured['my_numerical'].to(device),
        my_active_mask=batch_structured['my_active_mask'].to(device),
        my_alive_mask=batch_structured['my_alive_mask'].to(device),
        
        opp_species_ids=batch_structured['opp_species_ids'].to(device),
        opp_ability_ids=batch_structured['opp_ability_ids'].to(device),
        opp_item_ids=batch_structured['opp_item_ids'].to(device),
        opp_move_ids=batch_structured['opp_move_ids'].to(device),
        opp_type_ids=batch_structured['opp_type_ids'].to(device),
        opp_numerical=batch_structured['opp_numerical'].to(device),
        opp_active_mask=batch_structured['opp_active_mask'].to(device),
        opp_alive_mask=batch_structured['opp_alive_mask'].to(device),
        
        weather=batch_structured['weather'].to(device),
        terrain=batch_structured['terrain'].to(device),
        field_flags=batch_structured['field_flags'].to(device),
        turns_remaining=batch_structured['turns_remaining'].to(device),
        
        my_side_conditions=batch_structured['my_side_conditions'].to(device),
        opp_side_conditions=batch_structured['opp_side_conditions'].to(device),
        
        action_history=action_history,
        outcome_history=outcome_history,
        damage_history=damage_history,
        switch_history=switch_history,
        current_turn=batch_structured['current_turn'].to(device),
        history_mask=history_mask,
    )


class ParquetTrajectoryDataset(Dataset):
    """Dataset for loading trajectories from Parquet format.
    
    More efficient for large datasets.
    """
    
    def __init__(
        self,
        parquet_path: Path,
        weight_by_outcome: bool = True,
        winner_weight: float = 1.5,
        loser_weight: float = 0.5,
    ):
        """Initialize Parquet dataset.
        
        Args:
            parquet_path: Path to Parquet file
            weight_by_outcome: Whether to weight by game outcome
            winner_weight: Weight for winning player transitions
            loser_weight: Weight for losing player transitions
        """
        logger.info(f"Loading Parquet dataset from {parquet_path}")
        
        self.df = pd.read_parquet(parquet_path)
        
        logger.info(f"Loaded {len(self.df)} transitions")
        
        # Compute weights
        if weight_by_outcome:
            self.weights = np.where(
                self.df['won'].values,
                winner_weight,
                loser_weight
            ).astype(np.float32)
        else:
            self.weights = np.ones(len(self.df), dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # State is stored as a list in Parquet
        state = np.array(row['state'], dtype=np.float32)
        action = int(row['action'])
        weight = self.weights[idx]
        
        return (
            torch.from_numpy(state),
            torch.tensor(action, dtype=torch.long),
            torch.tensor(weight),
        )


class StreamingParquetDataset(Dataset):
    """Dataset for loading trajectories from multiple batch Parquet files.
    
    This is designed for large datasets processed with StreamingBattleProcessor,
    which creates multiple batch files instead of one large file.
    """
    
    STATE_DIM = 620
    
    def __init__(
        self,
        batch_dir: Path,
        weight_by_outcome: bool = True,
        winner_weight: float = 1.5,
        loser_weight: float = 0.5,
        max_transitions: Optional[int] = None,
    ):
        """Initialize streaming Parquet dataset.
        
        Args:
            batch_dir: Directory containing batch_*.parquet files and metadata.json
            weight_by_outcome: Whether to weight by game outcome
            winner_weight: Weight for winning player transitions
            loser_weight: Weight for losing player transitions
            max_transitions: Maximum transitions to load (None for all)
        """
        from src.data.storage.parquet_storage import StreamingTrajectoryReader
        
        self.batch_dir = Path(batch_dir)
        self.weight_by_outcome = weight_by_outcome
        self.winner_weight = winner_weight
        self.loser_weight = loser_weight
        
        logger.info(f"Loading streaming Parquet dataset from {batch_dir}")
        
        # Use StreamingTrajectoryReader to discover files
        self.reader = StreamingTrajectoryReader(batch_dir)
        
        # Load all batch files into a single DataFrame
        # For training efficiency, we load all data into memory
        dfs = []
        total_loaded = 0
        
        for batch_file in tqdm(self.reader.batch_files, desc="Loading batches"):
            if not batch_file.exists():
                continue
            
            try:
                df = pd.read_parquet(batch_file)
                dfs.append(df)
                total_loaded += len(df)
                
                if max_transitions and total_loaded >= max_transitions:
                    # Trim last batch if needed
                    excess = total_loaded - max_transitions
                    if excess > 0:
                        dfs[-1] = dfs[-1].iloc[:-excess]
                    break
            except Exception as e:
                logger.warning(f"Error loading {batch_file}: {e}")
                continue
        
        if not dfs:
            raise ValueError(f"No valid batch files found in {batch_dir}")
        
        self.df = pd.concat(dfs, ignore_index=True)
        
        logger.info(f"Loaded {len(self.df)} transitions from {len(dfs)} batch files")
        
        # Compute weights (won column might not exist in batched format)
        if weight_by_outcome and 'won' in self.df.columns:
            self.weights = np.where(
                self.df['won'].values,
                winner_weight,
                loser_weight
            ).astype(np.float32)
        else:
            self.weights = np.ones(len(self.df), dtype=np.float32)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Handle state - could be bytes or list depending on storage format
        state_data = row['state']
        if isinstance(state_data, bytes):
            state = np.frombuffer(state_data, dtype=np.float32)
        elif isinstance(state_data, (list, np.ndarray)):
            state = np.array(state_data, dtype=np.float32)
        else:
            state = np.zeros(self.STATE_DIM, dtype=np.float32)
        
        # Ensure correct state dimension
        if len(state) < self.STATE_DIM:
            state = np.pad(state, (0, self.STATE_DIM - len(state)))
        elif len(state) > self.STATE_DIM:
            state = state[:self.STATE_DIM]
        
        action = int(row['action'])
        weight = self.weights[idx]
        
        return (
            torch.from_numpy(np.ascontiguousarray(state)),  # Make contiguous and writable
            torch.tensor(action, dtype=torch.long),
            torch.tensor(weight),
        )


class ImitationTrainer:
    """Trainer for behavioral cloning.
    
    Supports multiple model types:
    - "simple": ImitationPolicy (MLP)
    - "attention": AttentionImitationPolicy (Transformer)
    - "enhanced": EnhancedPolicy (Unified encoder + hierarchical)
    """
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Create model based on type
        self.model = self._create_model(config)
        self.model.to(self.device)
        
        num_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model type: {config.model_type}")
        logger.info(f"Model parameters: {num_params:,}")
        
        # Optimizer and scheduler
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        self.scheduler = None  # Set after knowing total steps
        
        # Mixed precision
        self.use_amp = config.use_amp and torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info("Using mixed precision training (AMP)")
        
        # Metrics
        self.best_val_loss = float('inf')
        self.patience_counter = 0
    
    def _create_model(self, config: TrainingConfig) -> nn.Module:
        """Create model based on config.
        
        Args:
            config: Training configuration
            
        Returns:
            Initialized model
        """
        if config.model_type == "simple":
            return ImitationPolicy(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                hidden_dims=config.hidden_dims,
                dropout=config.dropout,
            )
        elif config.model_type == "attention":
            return AttentionImitationPolicy(
                state_dim=config.state_dim,
                action_dim=config.action_dim,
                dropout=config.dropout,
            )
        elif config.model_type == "enhanced":
            # Use enhanced config if provided, otherwise defaults
            enhanced_cfg = EnhancedPolicyConfig(
                flat_state_dim=config.state_dim,
                action_dim=config.action_dim,
                dropout=config.dropout,
                **(config.enhanced_config or {}),
            )
            return EnhancedPolicy(enhanced_cfg)
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
    
    def load_data(self) -> Tuple[DataLoader, DataLoader]:
        """Load and split trajectory data.
        
        Supports both JSON and Parquet formats.
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        path = Path(self.config.trajectory_path)
        logger.info(f"Loading trajectories from {path}")
        
        # Check format
        if self.config.data_format == "parquet" or path.suffix == ".parquet":
            return self._load_parquet_data(path)
        else:
            return self._load_json_data(path)
    
    def _load_json_data(self, path: Path) -> Tuple[DataLoader, DataLoader]:
        """Load data from JSON format."""
        with open(path, 'r') as f:
            trajectories = json.load(f)
        
        logger.info(f"Loaded {len(trajectories)} trajectories")
        
        # Split by trajectory (not by transition) to avoid leakage
        num_val = int(len(trajectories) * self.config.val_split)
        indices = np.random.permutation(len(trajectories))
        
        val_indices = indices[:num_val]
        train_indices = indices[num_val:]
        
        train_trajectories = [trajectories[i] for i in train_indices]
        val_trajectories = [trajectories[i] for i in val_indices]
        
        logger.info(f"Train trajectories: {len(train_trajectories)}")
        logger.info(f"Val trajectories: {len(val_trajectories)}")
        
        # Create datasets
        include_structured = self.config.model_type == "enhanced"
        train_dataset = TrajectoryDataset(
            train_trajectories, 
            weight_by_outcome=True,
            include_structured=include_structured,
        )
        val_dataset = TrajectoryDataset(
            val_trajectories, 
            weight_by_outcome=False,
            include_structured=include_structured,
        )
        
        return self._create_loaders(train_dataset, val_dataset)
    
    def _load_parquet_data(self, path: Path) -> Tuple[DataLoader, DataLoader]:
        """Load data from Parquet format.
        
        Supports both single Parquet files and directories with batched files.
        """
        path = Path(path)
        
        # Check if path is a directory with batched files
        if path.is_dir():
            # Use streaming dataset for batched files
            logger.info(f"Loading from batch directory: {path}")
            full_dataset = StreamingParquetDataset(
                path,
                weight_by_outcome=True,
                winner_weight=self.config.winner_weight,
                loser_weight=self.config.loser_weight,
            )
        else:
            # Load single Parquet file
            full_dataset = ParquetTrajectoryDataset(
                path,
                weight_by_outcome=True,
                winner_weight=self.config.winner_weight,
                loser_weight=self.config.loser_weight,
            )
        
        # Split
        num_val = int(len(full_dataset) * self.config.val_split)
        num_train = len(full_dataset) - num_val
        
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [num_train, num_val],
            generator=torch.Generator().manual_seed(42),
        )
        
        logger.info(f"Train samples: {len(train_dataset)}")
        logger.info(f"Val samples: {len(val_dataset)}")
        
        return self._create_loaders(train_dataset, val_dataset)
    
    def _create_loaders(
        self, 
        train_dataset: Dataset, 
        val_dataset: Dataset,
    ) -> Tuple[DataLoader, DataLoader]:
        """Create data loaders with appropriate collate function."""
        # Use structured collate for enhanced models
        collate_fn = None
        if self.config.model_type == "enhanced" and isinstance(train_dataset, TrajectoryDataset):
            if train_dataset.include_structured:
                collate_fn = collate_structured_batch
                logger.info("Using structured collate function for enhanced model")
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Handles both flat state tensors and structured batch dictionaries.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Dictionary of metrics
        """
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_top5_correct = 0
        total_samples = 0
        use_structured = self.config.model_type == "enhanced"
        
        pbar = tqdm(train_loader, desc="Training")
        
        for batch_idx, batch in enumerate(pbar):
            # Handle both formats: tuple (flat) or dict (structured)
            if isinstance(batch, dict):
                states = batch['state'].to(self.device)
                actions = batch['action'].to(self.device)
                weights = batch['weight'].to(self.device)
                
                # Build structured state if available
                if use_structured and 'structured' in batch:
                    structured_state = build_structured_state_from_batch(
                        batch['structured'], self.device
                    )
                else:
                    structured_state = None
            else:
                # Legacy tuple format
                states, actions, weights = batch
                states = states.to(self.device)
                actions = actions.to(self.device)
                weights = weights.to(self.device)
                structured_state = None
            
            self.optimizer.zero_grad()
            
            # Forward pass with optional AMP
            if self.use_amp:
                with autocast():
                    if structured_state is not None:
                        logits, _ = self.model(structured_state, use_structured=True)
                    else:
                        logits, _ = self.model(states)
                    loss = F.cross_entropy(logits, actions, reduction='none')
                    loss = (loss * weights).mean()
                
                # Backward with scaler
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard forward/backward
                if structured_state is not None:
                    logits, _ = self.model(structured_state, use_structured=True)
                else:
                    logits, _ = self.model(states)
                loss = F.cross_entropy(logits, actions, reduction='none')
                loss = (loss * weights).mean()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
            
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Metrics
            batch_size = states.size(0)
            total_loss += loss.item() * batch_size
            
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                total_correct += (preds == actions).sum().item()
                
                # Top-5 accuracy
                _, top5_preds = logits.topk(5, dim=-1)
                top5_correct = (top5_preds == actions.unsqueeze(-1)).any(dim=-1)
                total_top5_correct += top5_correct.sum().item()
            
            total_samples += batch_size
            
            # Update progress bar
            pbar.set_postfix({
                'loss': loss.item(),
                'acc': total_correct / total_samples,
            })
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
            'top5_accuracy': total_top5_correct / total_samples,
        }
    
    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set.
        
        Handles both flat and structured data formats.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_top5_correct = 0
        total_samples = 0
        use_structured = self.config.model_type == "enhanced"
        
        for batch in val_loader:
            # Handle both formats
            if isinstance(batch, dict):
                states = batch['state'].to(self.device)
                actions = batch['action'].to(self.device)
                
                if use_structured and 'structured' in batch:
                    structured_state = build_structured_state_from_batch(
                        batch['structured'], self.device
                    )
                else:
                    structured_state = None
            else:
                states, actions, _ = batch
                states = states.to(self.device)
                actions = actions.to(self.device)
                structured_state = None
            
            # Forward pass
            if structured_state is not None:
                logits, _ = self.model(structured_state, use_structured=True)
            else:
                logits, _ = self.model(states)
            
            loss = F.cross_entropy(logits, actions)
            
            batch_size = states.size(0)
            total_loss += loss.item() * batch_size
            
            preds = logits.argmax(dim=-1)
            total_correct += (preds == actions).sum().item()
            
            _, top5_preds = logits.topk(5, dim=-1)
            top5_correct = (top5_preds == actions.unsqueeze(-1)).any(dim=-1)
            total_top5_correct += top5_correct.sum().item()
            
            total_samples += batch_size
        
        return {
            'loss': total_loss / total_samples,
            'accuracy': total_correct / total_samples,
            'top5_accuracy': total_top5_correct / total_samples,
        }
    
    def train(self) -> Path:
        """Run full training loop.
        
        Returns:
            Path to best model checkpoint
        """
        # Load data
        train_loader, val_loader = self.load_data()
        
        # Setup scheduler
        total_steps = len(train_loader) * self.config.epochs
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
            eta_min=1e-6,
        )
        
        # Create save directory
        save_dir = Path(self.config.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        best_model_path = save_dir / "best_model.pt"
        
        logger.info(f"Starting training for {self.config.epochs} epochs")
        
        for epoch in range(self.config.epochs):
            logger.info(f"\nEpoch {epoch + 1}/{self.config.epochs}")
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, "
                       f"Acc: {train_metrics['accuracy']:.4f}, "
                       f"Top5: {train_metrics['top5_accuracy']:.4f}")
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, "
                       f"Acc: {val_metrics['accuracy']:.4f}, "
                       f"Top5: {val_metrics['top5_accuracy']:.4f}")
            
            # Check for improvement
            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.patience_counter = 0
                
                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_accuracy': val_metrics['accuracy'],
                    'config': self.config,
                }, best_model_path)
                
                logger.info(f"Saved best model (val_loss: {val_metrics['loss']:.4f})")
            else:
                self.patience_counter += 1
                logger.info(f"No improvement ({self.patience_counter}/{self.config.patience})")
                
                if self.patience_counter >= self.config.patience:
                    logger.info("Early stopping triggered")
                    break
        
        # Save final model
        final_path = save_dir / "final_model.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }, final_path)
        
        logger.info(f"Training complete. Best model saved to {best_model_path}")
        
        return best_model_path


def load_model(model_path: Path, device: Optional[str] = None) -> ImitationPolicy:
    """Load a trained model.
    
    Args:
        model_path: Path to model checkpoint
        device: Device to load to
        
    Returns:
        Loaded model
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = ImitationPolicy(
        state_dim=config.state_dim,
        action_dim=config.action_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train imitation learning model")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--model-type", type=str, default="simple",
                       choices=["simple", "attention", "enhanced"],
                       help="Model type to train")
    parser.add_argument("--data-path", type=str, 
                       default="data/processed/trajectories/trajectories_logs-gen9vgc2024regg.json",
                       help="Path to trajectory data")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    
    args = parser.parse_args()
    
    config = TrainingConfig(
        trajectory_path=args.data_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        model_type=args.model_type,
        use_amp=not args.no_amp,
    )
    
    trainer = ImitationTrainer(config)
    best_model_path = trainer.train()
    
    logger.info(f"Best model saved to: {best_model_path}")


if __name__ == "__main__":
    main()

