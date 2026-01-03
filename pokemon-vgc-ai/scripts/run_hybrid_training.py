#!/usr/bin/env python3
"""Hybrid Training Pipeline for VGC AI.

This script implements a multi-stage training pipeline:

1. Imitation Learning: Train on all 21K+ expert replays
2. PPO Pre-training: Fast training on SimulatedVGCEnv (~1800 FPS)
3. PPO Fine-tuning: Accurate training on PokeEnvVGCEnv (~10 FPS)
4. Self-Play: Population-based training with ELO tracking (basic or enhanced)
5. Distillation: Compress model for faster inference (optional)

All stages are tracked via ExperimentTracker (MLflow or local JSON).

Usage:
    python scripts/run_hybrid_training.py --stages all
    python scripts/run_hybrid_training.py --stages imitation,ppo_sim
    python scripts/run_hybrid_training.py --stages all --use-enhanced-selfplay --distill
    python scripts/run_hybrid_training.py --stages ppo_real --continue-from data/models/ppo_sim/best.zip
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
import json
import time

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
import torch
import numpy as np

from src.config import config


def check_showdown_available() -> bool:
    """Check if Pokemon Showdown server is running."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        return result == 0
    except Exception:
        return False


def run_imitation_stage(
    data_path: Path,
    output_dir: Path,
    epochs: int = 20,
    batch_size: int = 256,
    model_type: str = "enhanced",
    use_mixed_precision: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """Run imitation learning stage.
    
    Args:
        data_path: Path to trajectory data (directory or file)
        output_dir: Directory for model outputs
        epochs: Training epochs
        batch_size: Batch size
        model_type: Model type ("simple", "attention", "enhanced")
        use_mixed_precision: Whether to use FP16 training
        
    Returns:
        Tuple of (model_path, metrics_dict)
    """
    from src.ml.training.imitation_learning import TrainingConfig, ImitationTrainer
    
    logger.info("=" * 60)
    logger.info("STAGE 1: IMITATION LEARNING")
    logger.info("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    training_config = TrainingConfig(
        trajectory_path=str(data_path),
        data_format="parquet" if data_path.is_dir() or str(data_path).endswith('.parquet') else "json",
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        save_dir=str(output_dir),
        use_amp=use_mixed_precision,
    )
    
    trainer = ImitationTrainer(training_config)
    results = trainer.train()
    
    logger.info(f"Imitation learning complete: {results}")
    
    # Gather metrics
    metrics = {
        "imitation_epochs": epochs,
        "imitation_batch_size": batch_size,
    }
    if isinstance(results, dict):
        metrics.update({f"imitation_{k}": v for k, v in results.items() if isinstance(v, (int, float))})
    
    # Find best model
    best_model = output_dir / "best_model.pt"
    if best_model.exists():
        return best_model, metrics
    
    return output_dir / "final_model.pt", metrics


def run_ppo_simulated_stage(
    pretrained_path: Optional[Path],
    output_dir: Path,
    timesteps: int = 100000,
    use_curriculum: bool = True,
    use_masking: bool = True,
    use_mixed_precision: bool = False,
) -> Tuple[Path, Dict[str, Any]]:
    """Run PPO training on simulated environment.
    
    Args:
        pretrained_path: Path to pretrained imitation model (optional)
        output_dir: Directory for model outputs
        timesteps: Total training timesteps
        use_curriculum: Whether to use curriculum learning
        use_masking: Whether to use action masking
        use_mixed_precision: Whether to use FP16 training
        
    Returns:
        Tuple of (model_path, metrics_dict)
    """
    from src.ml.training.rl_finetuning import (
        SimulatedVGCEnv,
        RLConfig,
        train_with_curriculum,
        train_maskable_ppo,
        mask_fn,
    )
    
    logger.info("=" * 60)
    logger.info("STAGE 2: PPO TRAINING (SIMULATED)")
    logger.info("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "ppo_sim_timesteps": timesteps,
        "ppo_sim_use_curriculum": use_curriculum,
        "ppo_sim_use_masking": use_masking,
        "ppo_sim_use_mixed_precision": use_mixed_precision,
    }
    
    start_time = time.time()
    
    # Create RLConfig
    rl_config = RLConfig(
        imitation_model_path=str(pretrained_path) if pretrained_path else "",
        total_timesteps=timesteps,
        save_dir=str(output_dir),
        use_curriculum=use_curriculum,
    )
    
    if use_curriculum and not use_masking:
        # Use curriculum learning
        model_path = train_with_curriculum(rl_config)
    elif use_masking:
        # Use MaskablePPO
        model_path = train_maskable_ppo(rl_config)
    else:
        # Basic PPO
        from stable_baselines3 import PPO
        
        env = SimulatedVGCEnv()
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            verbose=1,
        )
        
        model.learn(total_timesteps=timesteps)
        model_path = output_dir / "ppo_basic.zip"
        model.save(str(model_path))
    
    elapsed = time.time() - start_time
    metrics["ppo_sim_training_time_seconds"] = elapsed
    metrics["ppo_sim_fps"] = timesteps / elapsed if elapsed > 0 else 0
    
    logger.info(f"PPO simulated training complete: {model_path}")
    logger.info(f"  Training time: {elapsed:.1f}s, FPS: {metrics['ppo_sim_fps']:.1f}")
    
    return Path(model_path), metrics


def run_ppo_real_stage(
    pretrained_path: Optional[Path],
    output_dir: Path,
    timesteps: int = 10000,
) -> Tuple[Optional[Path], Dict[str, Any]]:
    """Run PPO training on real Showdown environment.
    
    Args:
        pretrained_path: Path to pretrained model
        output_dir: Directory for model outputs
        timesteps: Total training timesteps
        
    Returns:
        Tuple of (model_path or None, metrics_dict)
    """
    logger.info("=" * 60)
    logger.info("STAGE 3: PPO TRAINING (REAL SHOWDOWN)")
    logger.info("=" * 60)
    
    metrics = {"ppo_real_timesteps": timesteps}
    
    if not check_showdown_available():
        logger.warning("Pokemon Showdown not available, skipping real training")
        logger.warning("Start Showdown with: ./scripts/start_showdown.sh")
        metrics["ppo_real_skipped"] = True
        return None, metrics
    
    from src.ml.training.rl_finetuning import PokeEnvVGCEnv, create_vgc_env
    from stable_baselines3 import PPO
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create real environment
    env = create_vgc_env(use_real_env=True, opponent_type="random")
    
    # Load pretrained model or create new
    if pretrained_path and pretrained_path.exists():
        logger.info(f"Loading pretrained model from {pretrained_path}")
        model = PPO.load(str(pretrained_path), env=env)
    else:
        logger.info("Creating new PPO model")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=1e-4,  # Lower LR for fine-tuning
            n_steps=512,  # Smaller batches for real env
            batch_size=32,
            verbose=1,
        )
    
    # Train
    logger.info(f"Training for {timesteps} timesteps on real Showdown...")
    
    start_time = time.time()
    
    try:
        model.learn(total_timesteps=timesteps)
        model_path = output_dir / "ppo_real.zip"
        model.save(str(model_path))
    except Exception as e:
        logger.error(f"Training error: {e}")
        # Save whatever we have
        model_path = output_dir / "ppo_real_partial.zip"
        model.save(str(model_path))
        metrics["ppo_real_error"] = str(e)
    finally:
        env.close()
    
    elapsed = time.time() - start_time
    metrics["ppo_real_training_time_seconds"] = elapsed
    metrics["ppo_real_fps"] = timesteps / elapsed if elapsed > 0 else 0
    
    logger.info(f"PPO real training complete: {model_path}")
    return model_path, metrics


def run_self_play_stage(
    model_path: Path,
    output_dir: Path,
    iterations: int = 20,
    population_size: int = 8,
    use_real_battles: bool = False,
    use_enhanced: bool = False,
    hall_of_fame_size: int = 10,
    diversity_bonus: float = 0.1,
    battle_timeout: int = 60,
    max_real_battles: int = 5,
    evolve_teams: bool = False,
    team_pool_size: int = 20,
) -> Tuple[Path, Dict[str, Any]]:
    """Run self-play training.
    
    Args:
        model_path: Path to initial model
        output_dir: Directory for outputs
        iterations: Number of self-play iterations
        population_size: Population size
        use_real_battles: Whether to use real Showdown battles
        use_enhanced: Whether to use enhanced self-play (Hall of Fame + League)
        evolve_teams: Whether to evolve teams during training
        team_pool_size: Number of teams in the evolution pool
        hall_of_fame_size: Size of Hall of Fame (enhanced only)
        diversity_bonus: Diversity reward bonus (enhanced only)
        battle_timeout: Timeout in seconds per real battle (default: 60)
        max_real_battles: Max real battles per iteration (default: 5)
        
    Returns:
        Tuple of (output_dir, metrics_dict)
    """
    logger.info("=" * 60)
    if use_enhanced:
        logger.info("STAGE 4: ENHANCED SELF-PLAY (Hall of Fame + League)")
    else:
        logger.info("STAGE 4: SELF-PLAY (Basic)")
    logger.info("=" * 60)
    
    if use_real_battles and not check_showdown_available():
        logger.warning("Showdown not available, using simulated matches")
        use_real_battles = False
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "self_play_iterations": iterations,
        "self_play_population_size": population_size,
        "self_play_use_enhanced": use_enhanced,
        "self_play_use_real_battles": use_real_battles,
    }
    
    start_time = time.time()
    
    if use_enhanced:
        from src.ml.training.enhanced_self_play import EnhancedSelfPlayTrainer
        
        trainer = EnhancedSelfPlayTrainer(
            population_size=population_size,
            hall_of_fame_size=hall_of_fame_size,
            save_dir=output_dir,
            use_league=True,
            diversity_bonus=diversity_bonus,
            use_real_battles=use_real_battles,
            battle_timeout=float(battle_timeout),
            max_real_battles_per_iter=max_real_battles,
            evolve_teams=evolve_teams,
            team_pool_size=team_pool_size,
        )
        
        # Initialize population with the initial model
        logger.info(f"Adding initial model to population: {model_path}")
        for i in range(min(population_size, 4)):  # Start with 4 agents
            trainer.population.add_agent(str(model_path), training_steps=0)
        logger.info(f"Initialized population with {len(trainer.population.agents)} agents")
        
        metrics["self_play_hall_of_fame_size"] = hall_of_fame_size
        metrics["self_play_diversity_bonus"] = diversity_bonus
        
        # Run iterations
        all_stats = []
        for i in range(iterations):
            stats = trainer.run_iteration(matches_per_iteration=10)
            all_stats.append(stats)
            
            if (i + 1) % 5 == 0:
                logger.info(f"Enhanced self-play iteration {i+1}/{iterations}")
        
        # Get final stats
        best_agent = trainer.get_best_agent()
        if best_agent:
            metrics["self_play_best_elo"] = best_agent.elo
        
        metrics["self_play_hof_entries"] = len(trainer.hall_of_fame.entries)
        metrics["self_play_final_diversity"] = trainer.diversity_tracker.get_population_diversity()
        
        results = all_stats
        
    else:
        from src.ml.training.self_play import SelfPlayConfig, SelfPlayTrainer
        
        sp_config = SelfPlayConfig(
            initial_model_path=str(model_path),
            population_size=population_size,
            matches_per_iteration=10,
            save_dir=str(output_dir),
        )
        
        trainer = SelfPlayTrainer(sp_config)
        
        # Train
        results = trainer.train(num_iterations=iterations)
    
    elapsed = time.time() - start_time
    metrics["self_play_training_time_seconds"] = elapsed
    
    logger.info(f"Self-play complete: {len(results) if results else 0} iterations")
    
    # Save results
    results_path = output_dir / "self_play_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    return output_dir, metrics


def run_distillation_stage(
    model_path: Path,
    data_path: Path,
    output_dir: Path,
    target_hidden_dims: Tuple[int, ...] = (256, 128),
    temperature: float = 4.0,
    alpha: float = 0.7,
    epochs: int = 20,
) -> Tuple[Path, Dict[str, Any]]:
    """Run model distillation to compress the trained model.
    
    Args:
        model_path: Path to teacher model
        data_path: Path to training data for distillation
        output_dir: Directory for outputs
        target_hidden_dims: Hidden dimensions for student model
        temperature: Softmax temperature for distillation
        alpha: Weight for distillation loss (0 = hard targets, 1 = soft targets)
        epochs: Training epochs
        
    Returns:
        Tuple of (student_model_path, metrics_dict)
    """
    from src.ml.training.distillation import (
        PolicyDistillation,
        DistillationConfig,
        StudentPolicy,
    )
    from torch.utils.data import DataLoader, TensorDataset
    
    logger.info("=" * 60)
    logger.info("STAGE 5: MODEL DISTILLATION")
    logger.info("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "distill_target_dims": str(target_hidden_dims),
        "distill_temperature": temperature,
        "distill_alpha": alpha,
        "distill_epochs": epochs,
    }
    
    start_time = time.time()
    
    # Load teacher model
    logger.info(f"Loading teacher model from {model_path}")
    
    # Default dimensions for VGC
    state_dim = 620
    action_dim = 144
    
    # Wrapper class to adapt SB3 policies to expected (logits, value) interface
    class SB3PolicyWrapper(torch.nn.Module):
        """Wrapper to extract action logits and value from SB3 policy."""
        def __init__(self, sb3_policy):
            super().__init__()
            self.sb3_policy = sb3_policy
            
        def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            # SB3 policies expect observations in a specific format
            # Get features
            features = self.sb3_policy.extract_features(obs)
            if isinstance(features, tuple):
                features = features[0]
            
            # Get latent from MLP extractor
            if hasattr(self.sb3_policy, 'mlp_extractor'):
                latent_pi, latent_vf = self.sb3_policy.mlp_extractor(features)
            else:
                latent_pi = features
                latent_vf = features
            
            # Get action logits (mean for continuous, logits for discrete)
            if hasattr(self.sb3_policy, 'action_net'):
                action_logits = self.sb3_policy.action_net(latent_pi)
            else:
                action_logits = latent_pi
            
            # Get value
            if hasattr(self.sb3_policy, 'value_net'):
                value = self.sb3_policy.value_net(latent_vf)
            else:
                value = torch.zeros(obs.shape[0], 1, device=obs.device)
            
            return action_logits, value
    
    # Determine model type and load appropriately
    if str(model_path).endswith('.zip'):
        # Try loading as MaskablePPO first, then regular PPO
        teacher = None
        try:
            from sb3_contrib import MaskablePPO
            from src.ml.training.rl_finetuning import SimulatedVGCEnv, mask_fn
            from sb3_contrib.common.wrappers import ActionMasker
            
            # MaskablePPO needs environment for loading
            env = SimulatedVGCEnv()
            env = ActionMasker(env, mask_fn)
            sb3_model = MaskablePPO.load(str(model_path), env=env)
            teacher = SB3PolicyWrapper(sb3_model.policy)
            state_dim = sb3_model.observation_space.shape[0]
            action_dim = sb3_model.action_space.n
            env.close()
            logger.info("Loaded teacher as MaskablePPO")
        except Exception as e:
            logger.warning(f"Could not load as MaskablePPO: {e}")
            try:
                from stable_baselines3 import PPO
                sb3_model = PPO.load(str(model_path))
                teacher = SB3PolicyWrapper(sb3_model.policy)
                state_dim = sb3_model.observation_space.shape[0]
                action_dim = sb3_model.action_space.n
                logger.info("Loaded teacher as standard PPO")
            except Exception as e2:
                logger.warning(f"Could not load as PPO: {e2}")
        
        # If SB3 loading failed, create a simple wrapper for distillation
        if teacher is None:
            logger.warning("Using simple policy wrapper for distillation")
            from src.ml.models.imitation_policy import ImitationPolicy
            teacher = ImitationPolicy(state_dim=state_dim, action_dim=action_dim)
            # Try to load weights from the checkpoint
            checkpoint = torch.load(str(model_path).replace('.zip', '.pt'), map_location='cpu')
            if 'policy' in checkpoint:
                teacher.load_state_dict(checkpoint['policy'])
    else:
        # PyTorch model
        from src.ml.models.imitation_policy import ImitationPolicy
        teacher = ImitationPolicy(state_dim=state_dim, action_dim=action_dim)
        teacher.load_state_dict(torch.load(model_path))
    
    teacher.eval()
    
    # Load training data
    logger.info(f"Loading training data from {data_path}")
    
    if data_path.is_dir():
        # Load from parquet batches
        from src.data.storage.parquet_storage import StreamingTrajectoryReader
        reader = StreamingTrajectoryReader(data_path)
        
        # Load sample of data for distillation
        states_list = []
        actions_list = []
        max_samples = 50000
        
        for states, actions, _, _ in reader.iter_batches(batch_size=5000, shuffle_files=True):
            states_list.append(states)
            actions_list.append(actions)
            if sum(len(s) for s in states_list) >= max_samples:
                break
        
        states = torch.tensor(np.concatenate(states_list)[:max_samples], dtype=torch.float32)
        actions = torch.tensor(np.concatenate(actions_list)[:max_samples], dtype=torch.long)
    else:
        # Load from single file
        data = torch.load(data_path)
        states = data['states'][:50000]
        actions = data['actions'][:50000]
    
    logger.info(f"Loaded {len(states)} samples for distillation")
    metrics["distill_num_samples"] = len(states)
    
    # Create student model
    student = StudentPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=target_hidden_dims,
    )
    
    # Calculate model sizes
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    compression_ratio = teacher_params / student_params if student_params > 0 else 0
    
    logger.info(f"Teacher params: {teacher_params:,}")
    logger.info(f"Student params: {student_params:,}")
    logger.info(f"Compression ratio: {compression_ratio:.2f}x")
    
    metrics["distill_teacher_params"] = teacher_params
    metrics["distill_student_params"] = student_params
    metrics["distill_compression_ratio"] = compression_ratio
    
    # Create data loader
    dataset = TensorDataset(states, actions)
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    # Configure distillation
    distill_config = DistillationConfig(
        temperature=temperature,
        alpha=alpha,
        epochs=epochs,
    )
    
    # Run distillation
    distiller = PolicyDistillation(teacher, student, config=distill_config)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    history = distiller.distill(train_loader, device=device)
    
    elapsed = time.time() - start_time
    metrics["distill_training_time_seconds"] = elapsed
    
    # Get final metrics from history (history is a dict with lists)
    if history and isinstance(history, dict):
        if "train_loss" in history and history["train_loss"]:
            metrics["distill_final_loss"] = history["train_loss"][-1]
        if "policy_loss" in history and history["policy_loss"]:
            metrics["distill_final_policy_loss"] = history["policy_loss"][-1]
        if "value_loss" in history and history["value_loss"]:
            metrics["distill_final_value_loss"] = history["value_loss"][-1]
    
    # Save student model
    student_path = output_dir / "distilled_model.pt"
    torch.save(student.state_dict(), student_path)
    
    logger.info(f"Distillation complete: {student_path}")
    logger.info(f"  Compression: {compression_ratio:.2f}x smaller")
    
    return student_path, metrics


def run_full_pipeline(
    data_path: Path,
    output_base: Path,
    stages: List[str],
    continue_from: Optional[Path] = None,
    use_enhanced_selfplay: bool = False,
    use_mixed_precision: bool = False,
    use_curriculum: bool = True,
    distill_dims: Tuple[int, ...] = (256, 128),
    experiment_name: str = "vgc-hybrid",
    imitation_epochs: int = 20,
    ppo_sim_timesteps: int = 100000,
    ppo_real_timesteps: int = 10000,
    self_play_iterations: int = 20,
    self_play_real_battles: bool = False,
    battle_timeout: int = 60,
    max_real_battles: int = 5,
    evolve_teams: bool = False,
    team_pool_size: int = 20,
) -> dict:
    """Run the full hybrid training pipeline.
    
    Args:
        data_path: Path to trajectory data
        output_base: Base output directory
        stages: List of stages to run
        continue_from: Path to continue training from
        use_enhanced_selfplay: Use enhanced self-play with Hall of Fame
        use_mixed_precision: Use FP16 training where supported
        use_curriculum: Use curriculum learning (progressive difficulty)
        distill_dims: Hidden dimensions for distilled model
        experiment_name: Name for experiment tracking
        imitation_epochs: Epochs for imitation learning
        ppo_sim_timesteps: Timesteps for simulated PPO
        ppo_real_timesteps: Timesteps for real PPO
        self_play_iterations: Iterations for self-play
        self_play_real_battles: Use real Showdown battles in self-play
        battle_timeout: Timeout in seconds for each real battle
        max_real_battles: Max real battles per iteration
        evolve_teams: Use team evolution during self-play
        team_pool_size: Number of teams in evolution pool
        
    Returns:
        Dictionary with results from each stage
    """
    from src.ml.training.experiment_tracking import ExperimentTracker
    
    results = {}
    all_metrics = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(experiment_name=experiment_name)
    
    current_model = continue_from
    
    with tracker.run(f"hybrid_{timestamp}"):
        # Log pipeline parameters
        tracker.log_params({
            "stages": ",".join(stages),
            "data_path": str(data_path),
            "output_base": str(output_base),
            "use_enhanced_selfplay": use_enhanced_selfplay,
            "use_mixed_precision": use_mixed_precision,
            "distill_dims": str(distill_dims),
            "imitation_epochs": imitation_epochs,
            "ppo_sim_timesteps": ppo_sim_timesteps,
            "ppo_real_timesteps": ppo_real_timesteps,
            "self_play_iterations": self_play_iterations,
        })
        
        pipeline_start = time.time()
        
        # Stage 1: Imitation Learning
        if "imitation" in stages or "all" in stages:
            try:
                imitation_dir = output_base / "imitation" / timestamp
                current_model, metrics = run_imitation_stage(
                    data_path=data_path,
                    output_dir=imitation_dir,
                    epochs=imitation_epochs,
                    use_mixed_precision=use_mixed_precision,
                )
                results["imitation"] = {"model_path": str(current_model), "status": "success"}
                all_metrics.update(metrics)
                tracker.log_metrics(metrics)
                tracker.log_artifact(str(current_model))
            except Exception as e:
                logger.error(f"Imitation learning failed: {e}")
                results["imitation"] = {"status": "failed", "error": str(e)}
                tracker.log_metrics({"imitation_error": 1})
        
        # Stage 2: PPO Simulated
        if "ppo_sim" in stages or "all" in stages:
            try:
                ppo_sim_dir = output_base / "ppo_sim" / timestamp
                current_model, metrics = run_ppo_simulated_stage(
                    pretrained_path=current_model,
                    output_dir=ppo_sim_dir,
                    timesteps=ppo_sim_timesteps,
                    use_curriculum=use_curriculum,
                    use_mixed_precision=use_mixed_precision,
                )
                results["ppo_sim"] = {"model_path": str(current_model), "status": "success"}
                all_metrics.update(metrics)
                tracker.log_metrics(metrics)
                tracker.log_artifact(str(current_model))
            except Exception as e:
                logger.error(f"PPO simulated training failed: {e}")
                results["ppo_sim"] = {"status": "failed", "error": str(e)}
                tracker.log_metrics({"ppo_sim_error": 1})
        
        # Stage 3: PPO Real
        if "ppo_real" in stages or "all" in stages:
            try:
                ppo_real_dir = output_base / "ppo_real" / timestamp
                real_model, metrics = run_ppo_real_stage(
                    pretrained_path=current_model,
                    output_dir=ppo_real_dir,
                    timesteps=ppo_real_timesteps,
                )
                all_metrics.update(metrics)
                tracker.log_metrics(metrics)
                
                if real_model:
                    current_model = real_model
                    results["ppo_real"] = {"model_path": str(current_model), "status": "success"}
                    tracker.log_artifact(str(current_model))
                else:
                    results["ppo_real"] = {"status": "skipped", "reason": "Showdown not available"}
            except Exception as e:
                logger.error(f"PPO real training failed: {e}")
                results["ppo_real"] = {"status": "failed", "error": str(e)}
                tracker.log_metrics({"ppo_real_error": 1})
        
        # Stage 4: Self-Play
        if "self_play" in stages or "all" in stages:
            if current_model:
                try:
                    self_play_dir = output_base / "self_play" / timestamp
                    sp_output, metrics = run_self_play_stage(
                        model_path=current_model,
                        output_dir=self_play_dir,
                        iterations=self_play_iterations,
                        use_enhanced=use_enhanced_selfplay,
                        use_real_battles=self_play_real_battles,
                        battle_timeout=battle_timeout,
                        max_real_battles=max_real_battles,
                        evolve_teams=evolve_teams,
                        team_pool_size=team_pool_size,
                    )
                    results["self_play"] = {"output_dir": str(sp_output), "status": "success"}
                    all_metrics.update(metrics)
                    tracker.log_metrics(metrics)
                except Exception as e:
                    logger.error(f"Self-play training failed: {e}")
                    results["self_play"] = {"status": "failed", "error": str(e)}
                    tracker.log_metrics({"self_play_error": 1})
            else:
                results["self_play"] = {"status": "skipped", "reason": "No model available"}
        
        # Stage 5: Distillation
        if "distill" in stages or ("all" in stages and "distill" in stages):
            if current_model:
                try:
                    distill_dir = output_base / "distill" / timestamp
                    distilled_model, metrics = run_distillation_stage(
                        model_path=current_model,
                        data_path=data_path,
                        output_dir=distill_dir,
                        target_hidden_dims=distill_dims,
                    )
                    results["distill"] = {"model_path": str(distilled_model), "status": "success"}
                    all_metrics.update(metrics)
                    tracker.log_metrics(metrics)
                    tracker.log_artifact(str(distilled_model))
                except Exception as e:
                    logger.error(f"Distillation failed: {e}")
                    results["distill"] = {"status": "failed", "error": str(e)}
                    tracker.log_metrics({"distill_error": 1})
            else:
                results["distill"] = {"status": "skipped", "reason": "No model available"}
        
        # Log total pipeline time
        total_time = time.time() - pipeline_start
        tracker.log_metrics({"total_pipeline_time_seconds": total_time})
        all_metrics["total_pipeline_time_seconds"] = total_time
    
    # Save pipeline results
    pipeline_results_path = output_base / f"pipeline_results_{timestamp}.json"
    output_base.mkdir(parents=True, exist_ok=True)
    with open(pipeline_results_path, 'w') as f:
        json.dump({
            "results": results,
            "metrics": all_metrics,
            "timestamp": timestamp,
            "stages": stages,
        }, f, indent=2)
    
    logger.info(f"Pipeline complete! Results saved to {pipeline_results_path}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run hybrid training pipeline for VGC AI"
    )
    
    # Stage selection
    parser.add_argument(
        "--stages",
        type=str,
        default="all",
        help="Comma-separated stages to run: imitation,ppo_sim,ppo_real,self_play,distill,all"
    )
    
    # Data and output paths
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/trajectories_batched",
        help="Path to trajectory data"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/models/hybrid",
        help="Base output directory"
    )
    parser.add_argument(
        "--continue-from",
        type=str,
        default=None,
        help="Path to model to continue training from"
    )
    
    # Stage-specific parameters
    parser.add_argument(
        "--imitation-epochs",
        type=int,
        default=20,
        help="Epochs for imitation learning"
    )
    parser.add_argument(
        "--ppo-sim-timesteps",
        type=int,
        default=100000,
        help="Timesteps for simulated PPO"
    )
    parser.add_argument(
        "--ppo-real-timesteps",
        type=int,
        default=10000,
        help="Timesteps for real PPO"
    )
    parser.add_argument(
        "--self-play-iterations",
        type=int,
        default=20,
        help="Iterations for self-play"
    )
    
    # Enhancement flags
    parser.add_argument(
        "--use-enhanced-selfplay",
        action="store_true",
        help="Use enhanced self-play with Hall of Fame and League system"
    )
    parser.add_argument(
        "--self-play-real-battles",
        action="store_true",
        help="Run actual battles on Showdown during self-play (requires Showdown server)"
    )
    parser.add_argument(
        "--battle-timeout",
        type=int,
        default=60,
        help="Timeout in seconds for each real battle (default: 60)"
    )
    parser.add_argument(
        "--max-real-battles",
        type=int,
        default=5,
        help="Max real battles per iteration before falling back to simulation (default: 5)"
    )
    parser.add_argument(
        "--evolve-teams",
        action="store_true",
        help="Enable team evolution during self-play (teams evolve alongside agents)"
    )
    parser.add_argument(
        "--team-pool-size",
        type=int,
        default=20,
        help="Number of teams in the evolution pool (default: 20)"
    )
    parser.add_argument(
        "--use-curriculum",
        action="store_true",
        help="Use curriculum learning (progressive difficulty)"
    )
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use FP16 mixed precision training for faster training"
    )
    parser.add_argument(
        "--distill",
        action="store_true",
        help="Run distillation stage to compress the model"
    )
    parser.add_argument(
        "--distill-dims",
        type=str,
        default="256,128",
        help="Comma-separated hidden dimensions for distilled model"
    )
    
    # Experiment tracking
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="vgc-hybrid",
        help="Name for experiment tracking"
    )
    
    args = parser.parse_args()
    
    # Parse stages
    stages = [s.strip() for s in args.stages.split(",")]
    
    # Add distill stage if --distill flag is set
    if args.distill and "distill" not in stages and "all" not in stages:
        stages.append("distill")
    
    # Parse distill dims
    distill_dims = tuple(int(d.strip()) for d in args.distill_dims.split(","))
    
    # Validate paths
    data_path = Path(args.data_path)
    if not data_path.exists():
        logger.error(f"Data path not found: {data_path}")
        logger.info("Run trajectory processing first:")
        logger.info("  python -m src.data.parsers.trajectory_builder --streaming")
        sys.exit(1)
    
    output_base = Path(args.output_dir)
    continue_from = Path(args.continue_from) if args.continue_from else None
    
    # Check Showdown for real stages
    if "ppo_real" in stages or "all" in stages:
        if not check_showdown_available():
            logger.warning("Pokemon Showdown server not running!")
            logger.warning("Real training will be skipped unless you start it:")
            logger.warning("  ./scripts/start_showdown.sh")
    
    # Run pipeline
    logger.info("Starting Hybrid Training Pipeline")
    logger.info(f"  Stages: {stages}")
    logger.info(f"  Data: {data_path}")
    logger.info(f"  Output: {output_base}")
    logger.info(f"  Enhanced Self-Play: {args.use_enhanced_selfplay}")
    logger.info(f"  Self-Play Real Battles: {args.self_play_real_battles}")
    logger.info(f"  Evolve Teams: {args.evolve_teams}")
    logger.info(f"  Mixed Precision: {args.mixed_precision}")
    logger.info(f"  Distillation: {args.distill}")
    logger.info(f"  Experiment: {args.experiment_name}")
    
    results = run_full_pipeline(
        data_path=data_path,
        output_base=output_base,
        stages=stages,
        continue_from=continue_from,
        use_enhanced_selfplay=args.use_enhanced_selfplay,
        use_mixed_precision=args.mixed_precision,
        use_curriculum=args.use_curriculum,
        distill_dims=distill_dims,
        experiment_name=args.experiment_name,
        imitation_epochs=args.imitation_epochs,
        ppo_sim_timesteps=args.ppo_sim_timesteps,
        ppo_real_timesteps=args.ppo_real_timesteps,
        self_play_iterations=args.self_play_iterations,
        self_play_real_battles=args.self_play_real_battles,
        battle_timeout=args.battle_timeout,
        max_real_battles=args.max_real_battles,
        evolve_teams=args.evolve_teams,
        team_pool_size=args.team_pool_size,
    )
    
    # Print summary
    logger.info("=" * 60)
    logger.info("PIPELINE SUMMARY")
    logger.info("=" * 60)
    for stage, result in results.items():
        status = result.get("status", "unknown")
        logger.info(f"  {stage}: {status}")
        if status == "success" and "model_path" in result:
            logger.info(f"    Model: {result['model_path']}")


if __name__ == "__main__":
    main()
