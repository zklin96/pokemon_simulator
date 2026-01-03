"""Model distillation for VGC AI.

This module provides knowledge distillation from a large
teacher model to a smaller, faster student model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional, Dict, Any, Tuple, List, Callable
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from loguru import logger


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    
    # Temperature for softening probabilities
    temperature: float = 4.0
    
    # Weight for distillation loss vs hard target loss
    alpha: float = 0.7  # 0 = only hard targets, 1 = only distillation
    
    # Training settings
    epochs: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    
    # Value head distillation
    distill_value: bool = True
    value_weight: float = 0.3
    
    # Feature matching (intermediate layer)
    feature_matching: bool = False
    feature_weight: float = 0.1


class PolicyDistillation(nn.Module):
    """Distill policy from teacher to student.
    
    Trains a smaller student model to match the output
    distribution of a larger teacher model.
    
    Example:
        teacher = load_model("teacher.pt")
        student = StudentPolicy(state_dim=620, action_dim=144)
        
        distiller = PolicyDistillation(teacher, student)
        distiller.distill(train_loader)
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        student: nn.Module,
        config: Optional[DistillationConfig] = None,
    ):
        """Initialize distillation.
        
        Args:
            teacher: Large teacher model
            student: Small student model
            config: Distillation configuration
        """
        super().__init__()
        
        self.teacher = teacher
        self.student = student
        self.config = config or DistillationConfig()
        
        # Freeze teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval()
    
    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        hard_targets: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute distillation loss.
        
        Combines:
        - KL divergence between soft targets
        - Cross entropy with hard targets (if provided)
        
        Args:
            student_logits: Student output logits
            teacher_logits: Teacher output logits
            hard_targets: Optional ground truth labels
            
        Returns:
            Combined loss
        """
        T = self.config.temperature
        alpha = self.config.alpha
        
        # Soft target loss (KL divergence)
        soft_targets = F.softmax(teacher_logits / T, dim=-1)
        soft_student = F.log_softmax(student_logits / T, dim=-1)
        
        soft_loss = F.kl_div(
            soft_student,
            soft_targets,
            reduction='batchmean',
        ) * (T ** 2)  # Scale by T^2 for gradient magnitude
        
        if hard_targets is not None and alpha < 1.0:
            # Hard target loss
            hard_loss = F.cross_entropy(student_logits, hard_targets)
            loss = alpha * soft_loss + (1 - alpha) * hard_loss
        else:
            loss = soft_loss
        
        return loss
    
    def value_distillation_loss(
        self,
        student_value: torch.Tensor,
        teacher_value: torch.Tensor,
    ) -> torch.Tensor:
        """Compute value head distillation loss.
        
        Args:
            student_value: Student value predictions
            teacher_value: Teacher value predictions
            
        Returns:
            MSE loss
        """
        return F.mse_loss(student_value, teacher_value)
    
    def forward(
        self,
        states: torch.Tensor,
        hard_targets: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass computing all losses.
        
        Args:
            states: Input states
            hard_targets: Optional ground truth actions
            
        Returns:
            Dict of losses
        """
        # Get teacher outputs (no grad)
        with torch.no_grad():
            teacher_logits, teacher_value = self.teacher(states)
        
        # Get student outputs
        student_logits, student_value = self.student(states)
        
        # Policy distillation loss
        policy_loss = self.distillation_loss(
            student_logits, teacher_logits, hard_targets
        )
        
        losses = {"policy": policy_loss}
        total = policy_loss
        
        # Value distillation loss
        if self.config.distill_value:
            value_loss = self.value_distillation_loss(student_value, teacher_value)
            losses["value"] = value_loss
            total = total + self.config.value_weight * value_loss
        
        losses["total"] = total
        
        return losses
    
    def distill(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        save_path: Optional[Path] = None,
    ) -> Dict[str, List[float]]:
        """Run distillation training.
        
        Args:
            train_loader: Training data
            val_loader: Optional validation data
            device: Target device
            save_path: Path to save best model
            
        Returns:
            Training history
        """
        self.student.to(device)
        self.teacher.to(device)
        
        optimizer = torch.optim.AdamW(
            self.student.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.epochs,
        )
        
        history = {
            "train_loss": [],
            "val_loss": [],
            "policy_loss": [],
            "value_loss": [],
        }
        
        best_val_loss = float("inf")
        
        for epoch in range(self.config.epochs):
            # Training
            self.student.train()
            train_loss = 0.0
            policy_loss_sum = 0.0
            value_loss_sum = 0.0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                states = batch[0].to(device)
                hard_targets = batch[1].to(device) if len(batch) > 1 else None
                
                optimizer.zero_grad()
                
                losses = self(states, hard_targets)
                losses["total"].backward()
                
                optimizer.step()
                
                train_loss += losses["total"].item()
                policy_loss_sum += losses["policy"].item()
                if "value" in losses:
                    value_loss_sum += losses["value"].item()
            
            avg_train = train_loss / len(train_loader)
            avg_policy = policy_loss_sum / len(train_loader)
            avg_value = value_loss_sum / len(train_loader)
            
            history["train_loss"].append(avg_train)
            history["policy_loss"].append(avg_policy)
            history["value_loss"].append(avg_value)
            
            # Validation
            if val_loader is not None:
                val_loss = self._validate(val_loader, device)
                history["val_loss"].append(val_loss)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if save_path:
                        torch.save(self.student.state_dict(), save_path)
                        logger.info(f"Saved best model (val_loss: {val_loss:.4f})")
            
            scheduler.step()
            
            logger.info(
                f"Epoch {epoch+1}: "
                f"train_loss={avg_train:.4f}, "
                f"policy={avg_policy:.4f}, "
                f"value={avg_value:.4f}"
            )
        
        return history
    
    @torch.no_grad()
    def _validate(
        self,
        val_loader: DataLoader,
        device: str,
    ) -> float:
        """Validate on held-out data."""
        self.student.eval()
        total_loss = 0.0
        
        for batch in val_loader:
            states = batch[0].to(device)
            hard_targets = batch[1].to(device) if len(batch) > 1 else None
            
            losses = self(states, hard_targets)
            total_loss += losses["total"].item()
        
        return total_loss / len(val_loader)


class StudentPolicy(nn.Module):
    """Smaller student policy for distillation.
    
    Uses fewer parameters than the teacher for faster inference.
    """
    
    def __init__(
        self,
        state_dim: int = 620,
        action_dim: int = 144,
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
    ):
        """Initialize student policy.
        
        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
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
        self.policy_head = nn.Linear(hidden_dims[-1], action_dim)
        self.value_head = nn.Linear(hidden_dims[-1], 1)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            state: Input state
            
        Returns:
            Tuple of (action_logits, value)
        """
        features = self.encoder(state)
        return self.policy_head(features), self.value_head(features)
    
    def get_action_probs(self, state: torch.Tensor) -> torch.Tensor:
        """Get action probabilities."""
        logits, _ = self(state)
        return F.softmax(logits, dim=-1)


class ProgressiveDistillation:
    """Progressive distillation with multiple stages.
    
    Gradually reduces model size through multiple distillation steps.
    """
    
    def __init__(
        self,
        teacher: nn.Module,
        target_hidden_dims: List[Tuple[int, ...]],
        state_dim: int = 620,
        action_dim: int = 144,
    ):
        """Initialize progressive distillation.
        
        Args:
            teacher: Original large teacher
            target_hidden_dims: Hidden dims for each stage
            state_dim: State dimension
            action_dim: Action dimension
        """
        self.teacher = teacher
        self.target_hidden_dims = target_hidden_dims
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.current_teacher = teacher
        self.students: List[nn.Module] = []
    
    def distill_stage(
        self,
        stage: int,
        train_loader: DataLoader,
        config: Optional[DistillationConfig] = None,
        device: str = "cuda",
    ) -> nn.Module:
        """Run one distillation stage.
        
        Args:
            stage: Stage index (0-indexed)
            train_loader: Training data
            config: Distillation config
            device: Target device
            
        Returns:
            Trained student for this stage
        """
        hidden_dims = self.target_hidden_dims[stage]
        
        student = StudentPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims,
        )
        
        distiller = PolicyDistillation(
            self.current_teacher,
            student,
            config=config,
        )
        
        logger.info(f"Stage {stage+1}: Distilling to hidden_dims={hidden_dims}")
        distiller.distill(train_loader, device=device)
        
        # This student becomes next teacher
        self.current_teacher = student
        self.students.append(student)
        
        return student
    
    def distill_all(
        self,
        train_loader: DataLoader,
        config: Optional[DistillationConfig] = None,
        device: str = "cuda",
    ) -> nn.Module:
        """Run all distillation stages.
        
        Args:
            train_loader: Training data
            config: Distillation config
            device: Target device
            
        Returns:
            Final (smallest) student
        """
        for stage in range(len(self.target_hidden_dims)):
            self.distill_stage(stage, train_loader, config, device)
        
        return self.students[-1]


def distill_from_checkpoint(
    teacher_path: Path,
    train_data: Tuple[torch.Tensor, torch.Tensor],
    student_hidden_dims: Tuple[int, ...] = (256, 128),
    output_path: Optional[Path] = None,
    config: Optional[DistillationConfig] = None,
) -> nn.Module:
    """Convenience function to distill from a checkpoint.
    
    Args:
        teacher_path: Path to teacher model checkpoint
        train_data: Tuple of (states, actions)
        student_hidden_dims: Student hidden dimensions
        output_path: Path to save student
        config: Distillation config
        
    Returns:
        Trained student model
    """
    from ..models.imitation_policy import ImitationPolicy
    
    # Load teacher
    teacher = ImitationPolicy(state_dim=620, action_dim=144)
    teacher.load_state_dict(torch.load(teacher_path))
    
    # Create student
    student = StudentPolicy(hidden_dims=student_hidden_dims)
    
    # Create dataset
    states, actions = train_data
    dataset = TensorDataset(states, actions)
    train_loader = DataLoader(
        dataset,
        batch_size=(config or DistillationConfig()).batch_size,
        shuffle=True,
    )
    
    # Distill
    distiller = PolicyDistillation(teacher, student, config)
    history = distiller.distill(train_loader)
    
    # Save
    if output_path:
        torch.save(student.state_dict(), output_path)
        logger.info(f"Saved distilled student to {output_path}")
    
    return student

