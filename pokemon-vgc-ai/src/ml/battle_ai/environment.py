"""Gymnasium environment for VGC battles using poke-env."""

import asyncio
import numpy as np
from typing import Optional, Tuple, Dict, Any, List

import gymnasium as gym
from gymnasium import spaces
from poke_env.player import Player
from poke_env.battle import DoubleBattle
from loguru import logger

from src.engine.state.game_state import GameStateEncoder, ActionSpaceHandler


class VGCBattleEnv(gym.Env):
    """Gymnasium environment for VGC doubles battles.
    
    This environment wraps poke-env to provide a standard Gym interface
    for training reinforcement learning agents.
    """
    
    metadata = {"render_modes": ["human", "ansi"]}
    
    def __init__(
        self,
        battle_format: str = "gen9vgc2024regg",
        opponent: Optional[Player] = None,
        render_mode: Optional[str] = None,
    ):
        """Initialize the VGC battle environment.
        
        Args:
            battle_format: Pokemon Showdown battle format
            opponent: Opponent player to battle against
            render_mode: How to render the environment
        """
        super().__init__()
        
        self.battle_format = battle_format
        self.render_mode = render_mode
        
        # Initialize encoders
        self.state_encoder = GameStateEncoder()
        self.action_handler = ActionSpaceHandler()
        
        # Define spaces
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.state_encoder.observation_space_shape,
            dtype=np.float32
        )
        
        self.action_space = spaces.Discrete(self.action_handler.action_space_size)
        
        # Battle state
        self.current_battle: Optional[DoubleBattle] = None
        self.opponent = opponent
        self._last_observation: Optional[np.ndarray] = None
        
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.
        
        Args:
            seed: Random seed
            options: Additional options
            
        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)
        
        # Create initial empty observation
        # In actual implementation, this would start a new battle
        observation = np.zeros(
            self.state_encoder.observation_space_shape,
            dtype=np.float32
        )
        
        self._last_observation = observation
        
        info = {
            "battle_started": False,
            "message": "Environment reset - ready to start battle",
        }
        
        return observation, info
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Take a step in the environment.
        
        Args:
            action: Action index to take
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Default observation (no active battle in this skeleton)
        observation = np.zeros(
            self.state_encoder.observation_space_shape,
            dtype=np.float32
        )
        
        reward = 0.0
        terminated = False
        truncated = False
        
        info = {
            "action_taken": action,
            "action_decoded": self.action_handler.decode_action(action),
        }
        
        self._last_observation = observation
        
        return observation, reward, terminated, truncated, info
    
    def render(self) -> Optional[str]:
        """Render the environment.
        
        Returns:
            String representation if render_mode is "ansi"
        """
        if self.render_mode == "ansi":
            if self.current_battle:
                return self._render_battle_state()
            return "No active battle"
        return None
    
    def _render_battle_state(self) -> str:
        """Render current battle state as string."""
        if not self.current_battle:
            return "No battle"
        
        lines = ["=" * 40]
        lines.append(f"Turn: {self.current_battle.turn}")
        lines.append("-" * 40)
        
        # Player's active Pokemon
        lines.append("Your Active Pokemon:")
        for i, pokemon in enumerate(self.current_battle.active_pokemon):
            if pokemon and not pokemon.fainted:
                hp_pct = (pokemon.current_hp / pokemon.max_hp * 100) if pokemon.max_hp else 0
                lines.append(f"  [{i+1}] {pokemon.species}: {hp_pct:.0f}% HP")
        
        # Opponent's active Pokemon
        lines.append("Opponent's Active Pokemon:")
        for i, pokemon in enumerate(self.current_battle.opponent_active_pokemon):
            if pokemon and not pokemon.fainted:
                hp_pct = (pokemon.current_hp_fraction * 100) if pokemon.current_hp_fraction else 0
                lines.append(f"  [{i+1}] {pokemon.species}: {hp_pct:.0f}% HP")
        
        lines.append("=" * 40)
        return "\n".join(lines)
    
    def close(self):
        """Clean up resources."""
        self.current_battle = None
    
    def get_action_mask(self) -> np.ndarray:
        """Get mask of valid actions.
        
        Returns:
            Boolean array where True indicates valid action
        """
        mask = np.zeros(self.action_space.n, dtype=bool)
        
        if self.current_battle:
            valid_actions = self.action_handler.get_available_actions(self.current_battle)
            mask[valid_actions] = True
        else:
            # No battle, all actions technically valid
            mask[:] = True
        
        return mask


class VGCBattleEnvAsync:
    """Async wrapper for VGC battle environment.
    
    This class handles the async nature of poke-env battles while
    providing a synchronous interface for RL training.
    """
    
    def __init__(
        self,
        player: Player,
        opponent: Player,
        battle_format: str = "gen9vgc2024regg",
    ):
        """Initialize async battle environment.
        
        Args:
            player: The RL agent player
            opponent: The opponent player
            battle_format: Battle format string
        """
        self.player = player
        self.opponent = opponent
        self.battle_format = battle_format
        self.state_encoder = GameStateEncoder()
        self.action_handler = ActionSpaceHandler()
        
        self._battle: Optional[DoubleBattle] = None
        self._rewards: Dict[str, float] = {}
    
    def compute_reward(self, battle: DoubleBattle) -> float:
        """Compute reward for current battle state.
        
        Args:
            battle: Current battle state
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Win/Lose rewards
        if battle.won:
            reward += 1.0
        elif battle.lost:
            reward -= 1.0
        else:
            # Intermediate rewards based on remaining HP
            player_hp = sum(
                p.current_hp_fraction for p in battle.team.values()
                if not p.fainted
            )
            opp_hp = sum(
                p.current_hp_fraction for p in battle.opponent_team.values()
                if not p.fainted
            )
            
            # Small reward for HP advantage
            hp_diff = (player_hp - opp_hp) * 0.01
            reward += hp_diff
        
        return reward
    
    def get_observation(self, battle: DoubleBattle) -> np.ndarray:
        """Get observation from battle state.
        
        Args:
            battle: Current battle state
            
        Returns:
            Observation array
        """
        return self.state_encoder.encode_battle(battle)


def test_environment():
    """Test the VGC battle environment."""
    logger.info("Testing VGC Battle Environment...")
    
    env = VGCBattleEnv()
    
    logger.info(f"Observation space: {env.observation_space.shape}")
    logger.info(f"Action space: {env.action_space.n}")
    
    # Test reset
    obs, info = env.reset()
    logger.info(f"Initial observation shape: {obs.shape}")
    logger.info(f"Initial info: {info}")
    
    # Test step
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    logger.info(f"Step result - reward: {reward}, terminated: {terminated}")
    
    # Test action mask
    mask = env.get_action_mask()
    logger.info(f"Action mask shape: {mask.shape}, valid actions: {mask.sum()}")
    
    env.close()
    logger.info("Environment test passed!")


if __name__ == "__main__":
    test_environment()

