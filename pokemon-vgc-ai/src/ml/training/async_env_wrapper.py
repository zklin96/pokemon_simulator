"""Async-to-Sync Environment Wrapper for poke-env.

This module provides wrappers to use async poke-env battles with
synchronous Gymnasium-style RL training (e.g., Stable-Baselines3).

The key challenge is that poke-env uses asyncio for battles, but
SB3 expects synchronous step() calls. We solve this by:

1. Running battles in a background thread with its own event loop
2. Using queues to communicate actions and observations
3. Wrapping everything in a Gymnasium-compatible interface
"""

import asyncio
import threading
import queue
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from loguru import logger

# poke-env imports
try:
    from poke_env.player import Player, RandomPlayer
    from poke_env.environment.double_battle import DoubleBattle
    HAS_POKE_ENV = True
except ImportError:
    HAS_POKE_ENV = False
    # Create placeholder classes for type hints
    class Player:
        """Placeholder Player class when poke-env is not available."""
        pass
    class DoubleBattle:
        """Placeholder DoubleBattle class when poke-env is not available."""
        pass
    RandomPlayer = None


@dataclass
class BattleMessage:
    """Message passed between async battle and sync environment."""
    msg_type: str  # "state", "action", "done", "reset", "error"
    data: Any = None


class RLControlledPlayer(Player):
    """A poke-env Player that gets actions from an external source.
    
    This player receives actions through a queue and sends observations
    back through another queue, enabling synchronous RL control.
    """
    
    def __init__(
        self,
        action_queue: queue.Queue,
        observation_queue: queue.Queue,
        state_encoder: Any,
        action_decoder: Any,
        battle_format: str = "gen9vgc2024regg",
        **kwargs
    ):
        """Initialize RL-controlled player.
        
        Args:
            action_queue: Queue to receive actions from RL agent
            observation_queue: Queue to send observations to RL agent
            state_encoder: Object with encode_battle(battle) method
            action_decoder: Object with action_to_order(battle, action) method
            battle_format: Pokemon Showdown format
        """
        super().__init__(battle_format=battle_format, **kwargs)
        self.action_queue = action_queue
        self.observation_queue = observation_queue
        self.state_encoder = state_encoder
        self.action_decoder = action_decoder
        self._current_battle = None
    
    def choose_move(self, battle: DoubleBattle):
        """Get action from RL agent and convert to battle order.
        
        This method:
        1. Encodes current state and sends to observation queue
        2. Waits for action from action queue
        3. Converts action to battle order
        """
        self._current_battle = battle
        
        try:
            # Encode current state
            state_vec, structured = self.state_encoder.encode_battle(battle)
            
            # Calculate reward from battle state
            reward = self._calculate_reward(battle)
            done = battle.finished
            
            # Send observation to RL agent
            obs_msg = BattleMessage(
                msg_type="state",
                data={
                    "state": state_vec,
                    "reward": reward,
                    "done": done,
                    "truncated": False,
                    "info": {"turn": battle.turn},
                }
            )
            self.observation_queue.put(obs_msg, timeout=30)
            
            if done:
                return self.choose_random_doubles_move(battle)
            
            # Wait for action from RL agent
            action_msg = self.action_queue.get(timeout=30)
            
            if action_msg.msg_type == "action":
                action = action_msg.data
                order = self.action_decoder.action_to_order(battle, action)
                if order:
                    return order
            
            # Fallback to random move
            return self.choose_random_doubles_move(battle)
            
        except queue.Empty:
            logger.warning("Timeout waiting for action from RL agent")
            return self.choose_random_doubles_move(battle)
        except Exception as e:
            logger.error(f"Error in choose_move: {e}")
            return self.choose_random_doubles_move(battle)
    
    def _calculate_reward(self, battle: DoubleBattle) -> float:
        """Calculate reward from current battle state."""
        if battle.finished:
            if battle.won:
                return 10.0
            elif battle.lost:
                return -10.0
            else:
                return 0.0  # Draw
        
        # Intermediate reward based on HP differential
        my_hp = sum(
            mon.current_hp_fraction
            for mon in battle.team.values()
            if mon and not mon.fainted
        )
        opp_hp = sum(
            mon.current_hp_fraction
            for mon in battle.opponent_team.values()
            if mon and not mon.fainted
        )
        
        # Normalize to [-1, 1] range
        hp_diff = (my_hp - opp_hp) / 4.0  # Divide by max team size
        
        return hp_diff * 0.1 - 0.01  # Small turn penalty


class ActionDecoder:
    """Decode action indices (0-143) to poke-env battle orders.
    
    Action space for VGC doubles (per slot, 12 actions):
    - 0-3: Use move 1-4
    - 4-7: Use move 1-4 with Terastallization
    - 8-11: Switch to Pokemon 1-4 (bench)
    
    Combined action: slot_a * 12 + slot_b = 144 total
    """
    
    def action_to_order(self, battle: DoubleBattle, action: int) -> Optional[str]:
        """Convert action index to poke-env battle order.
        
        Args:
            battle: Current battle state
            action: Action index (0-143)
            
        Returns:
            Battle order string or None if invalid
        """
        # Decode combined action
        slot_a_action = action // 12
        slot_b_action = action % 12
        
        orders = []
        
        # Get orders for each active Pokemon
        for slot_idx, slot_action in enumerate([slot_a_action, slot_b_action]):
            if slot_idx >= len(battle.active_pokemon):
                continue
            
            pokemon = battle.active_pokemon[slot_idx]
            if pokemon is None or pokemon.fainted:
                continue
            
            order = self._get_slot_order(battle, slot_idx, slot_action)
            if order:
                orders.append(order)
        
        if not orders:
            return None
        
        return "/choose " + ", ".join(orders)
    
    def _get_slot_order(
        self, 
        battle: DoubleBattle, 
        slot_idx: int, 
        action: int
    ) -> Optional[str]:
        """Get order for a single slot.
        
        Args:
            battle: Current battle state
            slot_idx: Active slot index (0 or 1)
            action: Per-slot action (0-11)
            
        Returns:
            Order string for this slot or None
        """
        is_move = action < 4
        is_tera_move = 4 <= action < 8
        is_switch = action >= 8
        
        if is_move or is_tera_move:
            move_idx = action if is_move else action - 4
            
            # Get available moves
            available_moves = battle.available_moves[slot_idx] if slot_idx < len(battle.available_moves) else []
            
            if move_idx >= len(available_moves):
                # Fallback to first available move
                if available_moves:
                    move = available_moves[0]
                else:
                    return None
            else:
                move = available_moves[move_idx]
            
            # Build order
            order = f"move {move.id}"
            
            # Add tera if requested and available
            if is_tera_move and battle.can_tera[slot_idx]:
                order += " terastallize"
            
            # Add target for singles targeting in doubles
            # Target opponent's first active Pokemon by default
            if battle.opponent_active_pokemon:
                target = 1 if slot_idx == 0 else 2  # 1-indexed targets
                order += f" {target}"
            
            return order
            
        elif is_switch:
            switch_idx = action - 8
            
            # Get available switches
            available_switches = battle.available_switches[slot_idx] if slot_idx < len(battle.available_switches) else []
            
            if switch_idx >= len(available_switches):
                if available_switches:
                    switch_target = available_switches[0]
                else:
                    return None
            else:
                switch_target = available_switches[switch_idx]
            
            return f"switch {switch_target.species}"
        
        return None


class AsyncToSyncEnv(gym.Env):
    """Gymnasium environment that wraps async poke-env battles.
    
    This environment runs battles in a background thread and provides
    a synchronous step() interface for RL training.
    """
    
    def __init__(
        self,
        battle_format: str = "gen9vgc2024regg",
        opponent_type: str = "random",
        state_dim: int = 620,
        action_dim: int = 144,
    ):
        """Initialize the async-to-sync environment.
        
        Args:
            battle_format: Pokemon Showdown format
            opponent_type: Type of opponent ("random", "heuristic")
            state_dim: State vector dimension
            action_dim: Action space size
        """
        super().__init__()
        
        if not HAS_POKE_ENV:
            raise ImportError("poke-env is required for AsyncToSyncEnv")
        
        self.battle_format = battle_format
        self.opponent_type = opponent_type
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Gym spaces
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(action_dim)
        
        # Communication queues
        self.action_queue = queue.Queue()
        self.observation_queue = queue.Queue()
        
        # State encoder and action decoder
        self._init_encoder_decoder()
        
        # Battle thread
        self._battle_thread = None
        self._battle_loop = None
        self._should_stop = threading.Event()
        
        # Current state
        self._current_state = None
        self._battles_completed = 0
    
    def _init_encoder_decoder(self):
        """Initialize state encoder and action decoder."""
        try:
            from src.engine.state.game_state import GameStateEncoder
            self.state_encoder = GameStateEncoder()
        except ImportError:
            logger.warning("GameStateEncoder not available, using placeholder")
            self.state_encoder = PlaceholderEncoder(self.state_dim)
        
        self.action_decoder = ActionDecoder()
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment and start a new battle.
        
        Returns:
            Initial observation and info dict
        """
        super().reset(seed=seed)
        
        # Stop any existing battle
        self._stop_battle()
        
        # Clear queues
        self._clear_queues()
        
        # Start new battle in background
        self._start_battle()
        
        # Wait for initial observation
        try:
            obs_msg = self.observation_queue.get(timeout=60)
            if obs_msg.msg_type == "state":
                self._current_state = obs_msg.data["state"]
                return self._current_state, obs_msg.data.get("info", {})
        except queue.Empty:
            logger.error("Timeout waiting for initial state")
        
        # Fallback to zeros
        self._current_state = np.zeros(self.state_dim, dtype=np.float32)
        return self._current_state, {"error": "timeout"}
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Take action and return next state.
        
        Args:
            action: Action index (0-143)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Send action to battle
        self.action_queue.put(BattleMessage(msg_type="action", data=action))
        
        # Wait for next observation
        try:
            obs_msg = self.observation_queue.get(timeout=60)
            
            if obs_msg.msg_type == "state":
                self._current_state = obs_msg.data["state"]
                reward = obs_msg.data.get("reward", 0.0)
                done = obs_msg.data.get("done", False)
                truncated = obs_msg.data.get("truncated", False)
                info = obs_msg.data.get("info", {})
                
                if done:
                    self._battles_completed += 1
                
                return self._current_state, reward, done, truncated, info
                
            elif obs_msg.msg_type == "error":
                logger.error(f"Battle error: {obs_msg.data}")
                return self._current_state, 0.0, True, False, {"error": obs_msg.data}
                
        except queue.Empty:
            logger.error("Timeout waiting for observation")
            return self._current_state, 0.0, True, False, {"error": "timeout"}
        
        return self._current_state, 0.0, False, False, {}
    
    def close(self):
        """Clean up resources."""
        self._stop_battle()
    
    def _start_battle(self):
        """Start a battle in a background thread."""
        self._should_stop.clear()
        self._battle_thread = threading.Thread(target=self._run_battle_loop)
        self._battle_thread.daemon = True
        self._battle_thread.start()
    
    def _stop_battle(self):
        """Stop the current battle."""
        self._should_stop.set()
        
        # Send stop signal through queue
        self.action_queue.put(BattleMessage(msg_type="stop"))
        
        if self._battle_thread and self._battle_thread.is_alive():
            self._battle_thread.join(timeout=5)
    
    def _clear_queues(self):
        """Clear both queues."""
        while not self.action_queue.empty():
            try:
                self.action_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.observation_queue.empty():
            try:
                self.observation_queue.get_nowait()
            except queue.Empty:
                break
    
    def _run_battle_loop(self):
        """Run battles in an asyncio event loop."""
        self._battle_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._battle_loop)
        
        try:
            self._battle_loop.run_until_complete(self._battle_coro())
        except Exception as e:
            logger.error(f"Battle loop error: {e}")
            self.observation_queue.put(BattleMessage(msg_type="error", data=str(e)))
        finally:
            self._battle_loop.close()
    
    async def _battle_coro(self):
        """Coroutine that runs a single battle."""
        # Create RL-controlled player
        rl_player = RLControlledPlayer(
            action_queue=self.action_queue,
            observation_queue=self.observation_queue,
            state_encoder=self.state_encoder,
            action_decoder=self.action_decoder,
            battle_format=self.battle_format,
            max_concurrent_battles=1,
        )
        
        # Create opponent
        if self.opponent_type == "heuristic":
            from src.engine.showdown.player import HeuristicVGCPlayer
            opponent = HeuristicVGCPlayer(
                battle_format=self.battle_format,
                max_concurrent_battles=1,
            )
        else:
            opponent = RandomPlayer(
                battle_format=self.battle_format,
                max_concurrent_battles=1,
            )
        
        # Run battle
        await rl_player.battle_against(opponent, n_battles=1)


class PlaceholderEncoder:
    """Placeholder encoder when GameStateEncoder is not available."""
    
    def __init__(self, state_dim: int = 620):
        self.state_dim = state_dim
    
    def encode_battle(self, battle) -> Tuple[np.ndarray, Dict]:
        """Return zeros as placeholder state."""
        return np.zeros(self.state_dim, dtype=np.float32), {}


def create_async_vgc_env(
    battle_format: str = "gen9vgc2024regg",
    opponent_type: str = "random",
) -> AsyncToSyncEnv:
    """Factory function to create AsyncToSyncEnv.
    
    Args:
        battle_format: Pokemon Showdown format
        opponent_type: Opponent type ("random", "heuristic")
        
    Returns:
        AsyncToSyncEnv instance
    """
    return AsyncToSyncEnv(
        battle_format=battle_format,
        opponent_type=opponent_type,
    )

