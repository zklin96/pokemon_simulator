"""Mixed precision training utilities for VGC AI.

This module provides utilities for FP16/BF16 training
to improve training speed and reduce memory usage.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class MixedPrecisionConfig:
    """Configuration for mixed precision training."""
    
    enabled: bool = True
    dtype: str = "float16"  # float16, bfloat16
    loss_scale: str = "dynamic"  # dynamic, static
    static_loss_scale: float = 1.0
    growth_factor: float = 2.0
    backoff_factor: float = 0.5
    growth_interval: int = 2000
    
    def get_dtype(self) -> torch.dtype:
        """Get torch dtype."""
        if self.dtype == "bfloat16":
            return torch.bfloat16
        return torch.float16


class MixedPrecisionTrainer:
    """Wrapper for mixed precision training.
    
    Handles automatic mixed precision (AMP) with gradient scaling.
    
    Example:
        trainer = MixedPrecisionTrainer(model, optimizer)
        
        for batch in dataloader:
            loss = trainer.train_step(batch, loss_fn)
            
            if step % log_interval == 0:
                print(f"Loss: {loss:.4f}, Scale: {trainer.get_scale():.0f}")
    """
    
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Optional[MixedPrecisionConfig] = None,
        device: str = "cuda",
    ):
        """Initialize mixed precision trainer.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            config: Mixed precision config
            device: Target device
        """
        self.model = model
        self.optimizer = optimizer
        self.config = config or MixedPrecisionConfig()
        self.device = device
        
        # Check availability
        self.amp_available = torch.cuda.is_available() and self.config.enabled
        
        if self.amp_available:
            if self.config.loss_scale == "dynamic":
                self.scaler = GradScaler(
                    growth_factor=self.config.growth_factor,
                    backoff_factor=self.config.backoff_factor,
                    growth_interval=self.config.growth_interval,
                )
            else:
                self.scaler = GradScaler(
                    init_scale=self.config.static_loss_scale,
                    enabled=True,
                )
            
            logger.info(
                f"Mixed precision enabled ({self.config.dtype}) "
                f"with {self.config.loss_scale} loss scaling"
            )
        else:
            self.scaler = None
            if self.config.enabled:
                logger.warning("Mixed precision requested but CUDA not available")
    
    def train_step(
        self,
        batch: Tuple[torch.Tensor, ...],
        loss_fn: Callable,
        max_grad_norm: Optional[float] = None,
    ) -> float:
        """Execute a training step with mixed precision.
        
        Args:
            batch: Input batch (will be moved to device)
            loss_fn: Function that computes loss from batch
            max_grad_norm: Optional gradient clipping
            
        Returns:
            Loss value (float)
        """
        self.optimizer.zero_grad()
        
        if self.amp_available:
            with autocast(dtype=self.config.get_dtype()):
                loss = loss_fn(batch)
            
            # Scale and backward
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            if max_grad_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    max_grad_norm
                )
            
            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss = loss_fn(batch)
            loss.backward()
            
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_grad_norm
                )
            
            self.optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def eval_step(
        self,
        batch: Tuple[torch.Tensor, ...],
        eval_fn: Callable,
    ) -> Any:
        """Execute an evaluation step with mixed precision.
        
        Args:
            batch: Input batch
            eval_fn: Function that computes evaluation from batch
            
        Returns:
            Evaluation result
        """
        if self.amp_available:
            with autocast(dtype=self.config.get_dtype()):
                return eval_fn(batch)
        else:
            return eval_fn(batch)
    
    def get_scale(self) -> float:
        """Get current loss scale."""
        if self.scaler is not None:
            return self.scaler.get_scale()
        return 1.0
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        state = {
            "config": {
                "enabled": self.config.enabled,
                "dtype": self.config.dtype,
                "loss_scale": self.config.loss_scale,
            }
        }
        
        if self.scaler is not None:
            state["scaler"] = self.scaler.state_dict()
        
        return state
    
    def load_state_dict(self, state: Dict[str, Any]):
        """Load state dict for resuming."""
        if "scaler" in state and self.scaler is not None:
            self.scaler.load_state_dict(state["scaler"])


def get_autocast_context(
    enabled: bool = True,
    dtype: torch.dtype = torch.float16,
    device_type: str = "cuda",
):
    """Get autocast context manager.
    
    Args:
        enabled: Whether to enable autocast
        dtype: Data type
        device_type: Device type
        
    Returns:
        Context manager
    """
    if enabled and torch.cuda.is_available():
        return autocast(device_type=device_type, dtype=dtype)
    else:
        # No-op context manager
        import contextlib
        return contextlib.nullcontext()


class CachedStateEncoder:
    """State encoder with caching for efficiency.
    
    Caches common computations to avoid redundant processing.
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        cache_size: int = 10000,
    ):
        """Initialize cached encoder.
        
        Args:
            encoder: Base encoder module
            cache_size: Maximum cache entries
        """
        self.encoder = encoder
        self.cache_size = cache_size
        self._cache: Dict[int, torch.Tensor] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def encode(
        self,
        state: torch.Tensor,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """Encode state with caching.
        
        Args:
            state: State tensor
            use_cache: Whether to use cache
            
        Returns:
            Encoded state
        """
        if not use_cache:
            return self.encoder(state)
        
        # Create cache key from state
        key = hash(state.data_ptr())
        
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]
        
        self._cache_misses += 1
        encoded = self.encoder(state)
        
        # Add to cache if not full
        if len(self._cache) < self.cache_size:
            self._cache[key] = encoded.detach()
        
        return encoded
    
    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0
        
        return {
            "cache_size": len(self._cache),
            "max_size": self.cache_size,
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "hit_rate": hit_rate,
        }


def optimize_model_for_inference(
    model: nn.Module,
    example_input: torch.Tensor,
    use_jit: bool = True,
    use_half: bool = True,
) -> nn.Module:
    """Optimize model for inference.
    
    Applies various optimizations:
    - TorchScript compilation
    - FP16 conversion
    - Fusion of operations
    
    Args:
        model: Model to optimize
        example_input: Example input for tracing
        use_jit: Whether to use TorchScript
        use_half: Whether to use FP16
        
    Returns:
        Optimized model
    """
    model = model.eval()
    
    # FP16 conversion
    if use_half and torch.cuda.is_available():
        model = model.half()
        example_input = example_input.half()
    
    # TorchScript compilation
    if use_jit:
        try:
            model = torch.jit.trace(model, example_input)
            model = torch.jit.optimize_for_inference(model)
            logger.info("Model compiled with TorchScript")
        except Exception as e:
            logger.warning(f"TorchScript compilation failed: {e}")
    
    return model

