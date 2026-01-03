"""Training script for VGC Battle AI using PPO."""

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from loguru import logger

from src.config import config, MODELS_DIR
from src.ml.battle_ai.environment import VGCBattleEnv


class TrainingMetricsCallback(BaseCallback):
    """Callback for logging training metrics."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log rewards at end of episodes
        if self.locals.get("dones") is not None:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals.get("infos", [{}])[i]
                    if "episode" in info:
                        self.episode_rewards.append(info["episode"]["r"])
                        self.episode_lengths.append(info["episode"]["l"])
        
        # Log every 1000 steps
        if self.num_timesteps % 1000 == 0 and self.episode_rewards:
            mean_reward = np.mean(self.episode_rewards[-100:])
            mean_length = np.mean(self.episode_lengths[-100:])
            logger.info(
                f"Step {self.num_timesteps}: "
                f"Mean reward: {mean_reward:.2f}, "
                f"Mean length: {mean_length:.0f}"
            )
        
        return True


def make_env(env_id: int = 0) -> Callable[[], VGCBattleEnv]:
    """Create environment factory function.
    
    Args:
        env_id: Environment ID for parallel envs
        
    Returns:
        Environment factory function
    """
    def _init() -> VGCBattleEnv:
        env = VGCBattleEnv()
        env = Monitor(env)
        return env
    
    return _init


def train_ppo(
    total_timesteps: int = 100_000,
    n_envs: int = 4,
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    n_epochs: int = 10,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_range: float = 0.2,
    save_path: Optional[Path] = None,
    tensorboard_log: Optional[str] = None,
) -> PPO:
    """Train PPO agent for VGC battles.
    
    Args:
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        learning_rate: Learning rate
        n_steps: Steps per update
        batch_size: Minibatch size
        n_epochs: Epochs per update
        gamma: Discount factor
        gae_lambda: GAE lambda
        clip_range: PPO clip range
        save_path: Path to save model
        tensorboard_log: TensorBoard log directory
        
    Returns:
        Trained PPO model
    """
    logger.info("=" * 50)
    logger.info("Starting PPO Training for VGC Battle AI")
    logger.info("=" * 50)
    
    # Create vectorized environment
    logger.info(f"Creating {n_envs} parallel environments...")
    env_fns = [make_env(i) for i in range(n_envs)]
    env = DummyVecEnv(env_fns)  # Use SubprocVecEnv for true parallelism
    
    logger.info(f"Observation space: {env.observation_space.shape}")
    logger.info(f"Action space: {env.action_space.n}")
    
    # Create model
    logger.info("Creating PPO model...")
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        verbose=1,
        tensorboard_log=tensorboard_log,
        policy_kwargs={
            "net_arch": dict(pi=[256, 256, 128], vf=[256, 256, 128]),
            "activation_fn": torch.nn.ReLU,
        },
    )
    
    # Setup callbacks
    callbacks = [
        TrainingMetricsCallback(verbose=1),
    ]
    
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=str(save_path),
            name_prefix="vgc_ppo",
        )
        callbacks.append(checkpoint_callback)
    
    # Train
    logger.info(f"Starting training for {total_timesteps} timesteps...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )
    
    # Save final model
    if save_path:
        final_path = save_path / "vgc_ppo_final"
        model.save(final_path)
        logger.info(f"Saved final model to {final_path}")
    
    env.close()
    
    return model


def evaluate_model(
    model_path: str,
    n_eval_episodes: int = 100,
) -> Dict[str, float]:
    """Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        n_eval_episodes: Number of evaluation episodes
        
    Returns:
        Evaluation metrics
    """
    logger.info(f"Loading model from {model_path}")
    model = PPO.load(model_path)
    
    env = VGCBattleEnv()
    
    rewards = []
    lengths = []
    
    for episode in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += reward
            episode_length += 1
        
        rewards.append(episode_reward)
        lengths.append(episode_length)
    
    env.close()
    
    metrics = {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "mean_length": np.mean(lengths),
        "min_reward": np.min(rewards),
        "max_reward": np.max(rewards),
    }
    
    logger.info(f"Evaluation Results ({n_eval_episodes} episodes):")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.2f}")
    
    return metrics


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train VGC Battle AI")
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=100_000,
        help="Total training timesteps"
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=4,
        help="Number of parallel environments"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=str(MODELS_DIR / "ppo"),
        help="Path to save model"
    )
    parser.add_argument(
        "--eval-only",
        type=str,
        default=None,
        help="Path to model for evaluation only"
    )
    
    args = parser.parse_args()
    
    if args.eval_only:
        evaluate_model(args.eval_only)
    else:
        train_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.n_envs,
            save_path=Path(args.save_path),
        )


if __name__ == "__main__":
    main()

