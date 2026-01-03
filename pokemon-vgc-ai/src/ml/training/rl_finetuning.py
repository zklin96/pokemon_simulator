"""RL Fine-tuning for VGC Battle AI.

This module fine-tunes the imitation-learned policy using PPO
with shaped rewards from live battles.

Supports:
- Pretrained feature extraction from imitation policy
- Enhanced policy integration
- Curriculum learning with progressive difficulty
- Opponent scheduling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from loguru import logger
import gymnasium as gym
from gymnasium import spaces
from enum import IntEnum

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

# Try to import MaskablePPO from sb3-contrib
try:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.wrappers import ActionMasker
    HAS_MASKABLE_PPO = True
except ImportError:
    HAS_MASKABLE_PPO = False
    MaskablePPO = None
    ActionMasker = None

from src.ml.models.imitation_policy import ImitationPolicy
from src.ml.models.enhanced_policy import EnhancedPolicy, EnhancedPolicyConfig
from src.ml.training.imitation_learning import TrainingConfig
from src.ml.training.curriculum import (
    CurriculumLearning, DifficultyLevel, CurriculumStage, 
    DEFAULT_CURRICULUM
)
from src.ml.training.damage_calculator import DamageCalculator, get_type_effectiveness
from src.ml.training.reward_shaping import RewardShaper, RewardConfig, create_reward_shaper


@dataclass
class RLConfig:
    """Configuration for RL fine-tuning."""
    # Model
    imitation_model_path: str = "data/models/imitation/best_model.pt"
    model_type: str = "simple"  # "simple", "attention", or "enhanced"
    state_dim: int = 620
    action_dim: int = 144
    
    # PPO hyperparameters
    learning_rate: float = 1e-4  # Lower for fine-tuning
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    
    # Training
    total_timesteps: int = 100_000
    eval_freq: int = 10_000
    n_eval_episodes: int = 20
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: Optional[List[CurriculumStage]] = None  # None = use defaults
    
    # Output
    save_dir: str = "data/models/ppo_finetuned"
    log_dir: str = "data/logs/ppo_finetuning"


class PretrainedFeaturesExtractor(BaseFeaturesExtractor):
    """Use pretrained imitation policy as feature extractor."""
    
    def __init__(
        self, 
        observation_space: spaces.Space, 
        pretrained_model: ImitationPolicy,
        freeze_encoder: bool = False,
    ):
        """Initialize with pretrained model.
        
        Args:
            observation_space: Gym observation space
            pretrained_model: Pretrained ImitationPolicy
            freeze_encoder: Whether to freeze encoder weights
        """
        # Get the output dimension from the pretrained encoder
        # The encoder output before policy/value heads
        features_dim = 128  # Last hidden dim from ImitationPolicy
        
        super().__init__(observation_space, features_dim=features_dim)
        
        # Copy encoder from pretrained model
        self.encoder = pretrained_model.encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)


class EnhancedFeaturesExtractor(BaseFeaturesExtractor):
    """Use pretrained EnhancedPolicy encoder as feature extractor.
    
    Works with the UnifiedBattleEncoder that includes:
    - Pokemon embeddings
    - Team attention
    - Temporal context
    - Field/side condition encoding
    """
    
    def __init__(
        self, 
        observation_space: spaces.Space, 
        pretrained_model: EnhancedPolicy,
        freeze_encoder: bool = False,
    ):
        """Initialize with pretrained enhanced policy.
        
        Args:
            observation_space: Gym observation space
            pretrained_model: Pretrained EnhancedPolicy
            freeze_encoder: Whether to freeze encoder weights
        """
        # Get encoder output dimension from config
        features_dim = pretrained_model.config.encoder_output_dim
        
        super().__init__(observation_space, features_dim=features_dim)
        
        # Copy the entire UnifiedBattleEncoder
        self.encoder = pretrained_model.encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Froze enhanced encoder weights")
        
        logger.info(f"EnhancedFeaturesExtractor with output dim {features_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from flat observations.
        
        The encoder handles both flat 620-dim and structured inputs.
        For SB3 compatibility, we use flat inputs here.
        """
        # The UnifiedBattleEncoder can handle flat inputs
        return self.encoder(observations, use_structured=False)


class OpponentType(IntEnum):
    """Types of opponents for curriculum learning."""
    RANDOM = 0
    HEURISTIC = 1
    IMITATION = 2
    SELF = 3


class SimulatedVGCEnv(gym.Env):
    """Simulated VGC environment for RL training.
    
    This is a simplified environment that generates synthetic battles
    for training when we don't have a live Showdown connection.
    
    Supports curriculum learning through configurable:
    - Team size (2v2 up to 6v6)
    - Opponent difficulty (random, heuristic, imitation, self)
    - Turn limits
    - Action restrictions
    
    Uses realistic damage calculation based on Pokemon type effectiveness.
    """
    
    # Common VGC Pokemon templates for simulation
    POKEMON_TEMPLATES = [
        {"species": "Flutter Mane", "types": ["ghost", "fairy"], "spa": 135, "spd": 135},
        {"species": "Incineroar", "types": ["fire", "dark"], "atk": 115, "def": 90},
        {"species": "Urshifu", "types": ["fighting", "water"], "atk": 130, "spd": 60},
        {"species": "Rillaboom", "types": ["grass"], "atk": 125, "def": 90},
        {"species": "Gholdengo", "types": ["steel", "ghost"], "spa": 133, "spd": 91},
        {"species": "Landorus", "types": ["ground", "flying"], "atk": 145, "spd": 80},
        {"species": "Calyrex-Shadow", "types": ["psychic", "ghost"], "spa": 165, "spd": 100},
        {"species": "Tornadus", "types": ["flying"], "spa": 125, "spd": 80},
        {"species": "Amoonguss", "types": ["grass", "poison"], "def": 70, "spd": 80},
        {"species": "Kingambit", "types": ["dark", "steel"], "atk": 135, "def": 120},
        {"species": "Iron Hands", "types": ["fighting", "electric"], "atk": 140, "def": 108},
        {"species": "Dragonite", "types": ["dragon", "flying"], "atk": 134, "spd": 100},
    ]
    
    # Common move types used in VGC
    MOVE_TYPES = ["normal", "fire", "water", "electric", "grass", "ice", 
                  "fighting", "poison", "ground", "flying", "psychic", "bug",
                  "rock", "ghost", "dragon", "dark", "steel", "fairy"]
    
    def __init__(
        self,
        opponent_policy: Optional[nn.Module] = None,
        opponent_type: OpponentType = OpponentType.RANDOM,
        team_size: int = 4,
        bring_size: int = 4,
        max_turns: int = 20,
        opponent_strength: float = 1.0,
        allow_tera: bool = True,
        allow_switching: bool = True,
    ):
        """Initialize environment.
        
        Args:
            opponent_policy: Policy to use for non-random opponent
            opponent_type: Type of opponent behavior
            team_size: Number of Pokemon per team
            bring_size: Number to bring to battle (VGC = 4)
            max_turns: Maximum turns before timeout
            opponent_strength: Scaling for opponent damage (for curriculum)
            allow_tera: Whether tera is allowed
            allow_switching: Whether switching is allowed
        """
        super().__init__()
        
        self.state_dim = 620
        self.action_dim = 144
        
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.state_dim,), 
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.action_dim)
        
        # Curriculum parameters
        self.opponent_policy = opponent_policy
        self.opponent_type = opponent_type
        self.team_size = team_size
        self.bring_size = bring_size
        self.max_turns = max_turns
        self.opponent_strength = opponent_strength
        self.allow_tera = allow_tera
        self.allow_switching = allow_switching
        
        # Damage calculator
        self.damage_calculator = DamageCalculator(random_damage=True)
        
        # Reward shaper for advanced reward signals
        self.reward_shaper = create_reward_shaper()
        self.use_shaped_rewards = True  # Enable advanced shaping
        
        self.state = None
        self.turn = 0
        self.my_pokemon_hp = None
        self.opp_pokemon_hp = None
        self.prev_my_alive = 4
        self.prev_opp_alive = 4
        
        # Pokemon data for realistic damage
        self.my_pokemon = []
        self.opp_pokemon = []
        
        # Tera tracking
        self.my_tera_used = False
        self.opp_tera_used = False
        
        # Track curriculum stats
        self.episode_wins = 0
        self.episode_count = 0
    
    def configure_curriculum(self, stage: CurriculumStage):
        """Configure environment for a curriculum stage.
        
        Args:
            stage: Curriculum stage configuration
        """
        self.bring_size = stage.bring_size
        self.max_turns = stage.max_turns
        self.opponent_strength = stage.opponent_strength
        self.allow_tera = stage.allow_tera
        self.allow_switching = stage.allow_switching
        
        # Set opponent type
        if stage.opponent_type == "random":
            self.opponent_type = OpponentType.RANDOM
        elif stage.opponent_type == "heuristic":
            self.opponent_type = OpponentType.HEURISTIC
        elif stage.opponent_type == "imitation":
            self.opponent_type = OpponentType.IMITATION
        elif stage.opponent_type == "self":
            self.opponent_type = OpponentType.SELF
        
        logger.info(f"Configured for stage: {stage.name}")
    
    def get_win_rate(self) -> float:
        """Get recent win rate for curriculum progression."""
        if self.episode_count == 0:
            return 0.0
        return self.episode_wins / self.episode_count
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        
        self.turn = 0
        
        # Initialize HP based on bring size
        self.my_pokemon_hp = np.ones(self.bring_size, dtype=np.float32)
        self.opp_pokemon_hp = np.ones(self.bring_size, dtype=np.float32)
        
        # Track alive count for KO detection
        self.prev_my_alive = self.bring_size
        self.prev_opp_alive = self.bring_size
        
        # Generate random Pokemon teams
        self.my_pokemon = self._generate_team()
        self.opp_pokemon = self._generate_team()
        
        # Track tera usage
        self.my_tera_used = False
        self.opp_tera_used = False
        
        # Reset reward shaper
        self.reward_shaper.reset()
        
        # Generate initial state
        self.state = self._generate_state()
        
        return self.state, {}
    
    def _generate_team(self) -> List[Dict]:
        """Generate a random team of Pokemon."""
        templates = np.random.choice(
            len(self.POKEMON_TEMPLATES), 
            size=self.bring_size, 
            replace=False
        )
        
        team = []
        for idx in templates:
            pokemon = self.POKEMON_TEMPLATES[idx].copy()
            # Add default stats if not present
            pokemon.setdefault("atk", 100)
            pokemon.setdefault("spa", 100)
            pokemon.setdefault("def", 100)
            pokemon.setdefault("spd", 100)
            # Assign random move types (STAB + coverage)
            primary_type = pokemon["types"][0]
            secondary_type = pokemon["types"][1] if len(pokemon["types"]) > 1 else None
            move_types = [primary_type]  # STAB move
            if secondary_type:
                move_types.append(secondary_type)  # Second STAB
            # Add coverage moves
            while len(move_types) < 4:
                random_type = np.random.choice(self.MOVE_TYPES)
                if random_type not in move_types:
                    move_types.append(random_type)
            pokemon["move_types"] = move_types
            team.append(pokemon)
        
        return team
    
    def step(self, action: int):
        """Take action and return new state.
        
        Uses realistic damage calculation based on type effectiveness.
        
        Action encoding (per slot, 12 actions):
        - 0-3: Regular moves
        - 4-7: Tera + moves
        - 8-11: Switch to bench
        
        Combined: action = slot_a * 12 + slot_b
        """
        self.turn += 1
        
        # Decode action
        slot_a_action = action // 12
        slot_b_action = action % 12
        
        # Track damage for reward shaping
        damage_dealt = 0.0
        damage_taken = 0.0
        
        # Get active Pokemon indices
        my_active = self._get_active_indices()
        opp_active = self._get_active_indices(opponent=True)
        
        # Process my actions
        for slot_idx, slot_action in enumerate([slot_a_action, slot_b_action]):
            if slot_idx >= len(my_active):
                continue
            
            my_idx = my_active[slot_idx]
            if self.my_pokemon_hp[my_idx] <= 0:
                continue  # Skip fainted Pokemon
            
            is_move = slot_action < 4
            is_tera_move = 4 <= slot_action < 8
            is_switch = slot_action >= 8
            
            if is_move or is_tera_move:
                move_slot = slot_action if is_move else slot_action - 4
                
                # Handle tera
                if is_tera_move and not self.my_tera_used and self.allow_tera:
                    self.my_tera_used = True
                
                # Calculate damage with damage calculator
                damage = self._calculate_attack_damage(
                    attacker_idx=my_idx,
                    move_slot=move_slot,
                    target_indices=opp_active,
                    is_my_attack=True,
                )
                damage_dealt += damage
                
            elif is_switch and self.allow_switching:
                # Switch logic (simplified - just note the intent)
                pass
        
        # Opponent takes action
        for opp_idx in opp_active:
            if self.opp_pokemon_hp[opp_idx] <= 0:
                continue
            
            # Opponent selects random move
            move_slot = np.random.randint(0, 4)
            
            # Heuristic opponent picks better moves
            if self.opponent_type == OpponentType.HEURISTIC:
                # Pick move with best type matchup
                move_slot = self._select_best_move(opp_idx, my_active)
            
            damage = self._calculate_attack_damage(
                attacker_idx=opp_idx,
                move_slot=move_slot,
                target_indices=my_active,
                is_my_attack=False,
            )
            
            # Scale by opponent strength (for curriculum)
            damage *= self.opponent_strength
            damage_taken += damage
        
        # Check win/lose conditions
        my_alive = np.sum(self.my_pokemon_hp > 0)
        opp_alive = np.sum(self.opp_pokemon_hp > 0)
        
        # Track KOs this turn
        my_kos_this_turn = max(0, self.prev_opp_alive - opp_alive)
        opp_kos_this_turn = max(0, self.prev_my_alive - my_alive)
        self.prev_my_alive = my_alive
        self.prev_opp_alive = opp_alive
        
        done = False
        truncated = False
        reward = 0.0
        
        # Check if Tera was used this turn (action 4-7 is tera+move)
        slot1_action = action // 12
        slot2_action = action % 12
        used_tera = (4 <= slot1_action <= 7 or 4 <= slot2_action <= 7) and not self.my_tera_used
        if used_tera:
            self.my_tera_used = True
        
        # Check if we switched
        switched = (8 <= slot1_action <= 11 or 8 <= slot2_action <= 11)
        
        if opp_alive == 0:
            done = True
            self.episode_wins += 1
        elif my_alive == 0:
            done = True
        elif self.turn >= self.max_turns:
            truncated = True
            my_total_hp = np.sum(self.my_pokemon_hp)
            opp_total_hp = np.sum(self.opp_pokemon_hp)
            if my_total_hp > opp_total_hp:
                self.episode_wins += 1
        
        # Use reward shaper for advanced rewards
        if self.use_shaped_rewards:
            my_hp_total = np.sum(np.clip(self.my_pokemon_hp / 100.0, 0, 1))
            opp_hp_total = np.sum(np.clip(self.opp_pokemon_hp / 100.0, 0, 1))
            
            reward = self.reward_shaper.calculate_reward(
                battle_finished=done or truncated,
                won=done and opp_alive == 0,
                lost=done and my_alive == 0,
                my_hp=my_hp_total,
                opp_hp=opp_hp_total,
                my_kos_this_turn=my_kos_this_turn,
                opp_kos_this_turn=opp_kos_this_turn,
                action_taken=action,
                used_tera=used_tera,
                switched=switched,
                damage_dealt=damage_dealt,
                damage_taken=damage_taken,
            )
            
            # Add end-of-battle bonus
            if done or truncated:
                reward += self.reward_shaper.get_end_of_battle_bonus()
        else:
            # Fallback to simple rewards
            if opp_alive == 0:
                reward = 10.0
            elif my_alive == 0:
                reward = -10.0
            elif self.turn >= self.max_turns:
                my_total_hp = np.sum(self.my_pokemon_hp)
                opp_total_hp = np.sum(self.opp_pokemon_hp)
                reward = 2.0 if my_total_hp > opp_total_hp else -2.0
            else:
                # Simple shaped reward
                reward = damage_dealt - damage_taken
                reward += 2.0 * (self.bring_size - opp_alive - (self.bring_size - my_alive))
                reward -= 0.01
        
        if done or truncated:
            self.episode_count += 1
        
        # Generate new state
        self.state = self._generate_state()
        
        return self.state, reward, done, truncated, {"damage_dealt": damage_dealt, "damage_taken": damage_taken}
    
    def _get_active_indices(self, opponent: bool = False) -> List[int]:
        """Get indices of active (first 2 alive) Pokemon."""
        hp = self.opp_pokemon_hp if opponent else self.my_pokemon_hp
        active = []
        for i in range(len(hp)):
            if hp[i] > 0 and len(active) < 2:
                active.append(i)
        return active
    
    def _calculate_attack_damage(
        self,
        attacker_idx: int,
        move_slot: int,
        target_indices: List[int],
        is_my_attack: bool,
    ) -> float:
        """Calculate and apply attack damage using DamageCalculator."""
        if not target_indices:
            return 0.0
        
        # Get attacker and target data
        if is_my_attack:
            attacker = self.my_pokemon[attacker_idx]
            target_idx = np.random.choice(target_indices)
            target = self.opp_pokemon[target_idx]
            target_hp = self.opp_pokemon_hp
        else:
            attacker = self.opp_pokemon[attacker_idx]
            target_idx = np.random.choice(target_indices)
            target = self.my_pokemon[target_idx]
            target_hp = self.my_pokemon_hp
        
        # Get move type
        move_types = attacker.get("move_types", ["normal"])
        move_slot = min(move_slot, len(move_types) - 1)
        move_type = move_types[move_slot]
        
        # Determine category based on attacker stats
        category = "physical" if attacker.get("atk", 100) > attacker.get("spa", 100) else "special"
        
        # Calculate damage
        damage = self.damage_calculator.calculate(
            move_power=80,  # Average VGC move power
            move_type=move_type,
            category=category,
            attacker_atk=attacker.get("atk", 100),
            attacker_spa=attacker.get("spa", 100),
            defender_def=target.get("def", 100),
            defender_spd=target.get("spd", 100),
            attacker_types=attacker.get("types", []),
            defender_types=target.get("types", []),
        )
        
        # Apply damage
        target_hp[target_idx] = max(0, target_hp[target_idx] - damage)
        
        return damage
    
    def _select_best_move(self, attacker_idx: int, target_indices: List[int]) -> int:
        """Select the move with best type effectiveness (heuristic opponent)."""
        if not target_indices:
            return 0
        
        attacker = self.opp_pokemon[attacker_idx]
        move_types = attacker.get("move_types", ["normal"])
        
        best_move = 0
        best_effectiveness = 0.0
        
        for i, move_type in enumerate(move_types):
            total_eff = 0.0
            for target_idx in target_indices:
                target = self.my_pokemon[target_idx]
                eff = get_type_effectiveness(move_type, target.get("types", []))
                total_eff += eff
            
            if total_eff > best_effectiveness:
                best_effectiveness = total_eff
                best_move = i
        
        return best_move
    
    def get_action_mask(self) -> np.ndarray:
        """Generate mask for valid actions.
        
        Action encoding:
        - Per slot (12 actions): 0-3 = moves, 4-7 = tera+moves, 8-11 = switch to bench 0-3
        - Combined: action = slot_a * 12 + slot_b (144 total)
        
        Invalid actions:
        - Moves from fainted Pokemon
        - Tera if already used
        - Switch if only 2 Pokemon
        - Switch to fainted Pokemon
        - Both slots switching to same Pokemon
        
        Returns:
            Boolean mask of shape (144,) where True = valid
        """
        mask = np.zeros(self.action_dim, dtype=bool)
        
        my_active = self._get_active_indices()
        
        # Determine valid actions per slot
        slot_a_valid = self._get_slot_valid_actions(0, my_active)
        slot_b_valid = self._get_slot_valid_actions(1, my_active)
        
        # Combine into action mask
        for a in range(12):
            for b in range(12):
                action = a * 12 + b
                
                # Both slot actions must be valid
                if not slot_a_valid[a] or not slot_b_valid[b]:
                    continue
                
                # Can't both switch to the same Pokemon
                if a >= 8 and b >= 8:
                    bench_a = a - 8
                    bench_b = b - 8
                    if bench_a == bench_b:
                        continue
                
                mask[action] = True
        
        # Ensure at least one action is valid (fallback to action 0)
        if not np.any(mask):
            mask[0] = True
        
        return mask
    
    def _get_slot_valid_actions(self, slot_idx: int, active_indices: List[int]) -> np.ndarray:
        """Get valid actions for a specific slot.
        
        Args:
            slot_idx: 0 or 1 (slot A or B)
            active_indices: Indices of currently active Pokemon
            
        Returns:
            Boolean array of shape (12,) for valid actions
        """
        valid = np.zeros(12, dtype=bool)
        
        # Check if this slot has an active Pokemon
        if slot_idx >= len(active_indices):
            # No Pokemon in this slot, only valid action is 0 (pass)
            valid[0] = True
            return valid
        
        pokemon_idx = active_indices[slot_idx]
        
        # Check if Pokemon is alive
        if self.my_pokemon_hp[pokemon_idx] <= 0:
            valid[0] = True  # Fainted, can only pass
            return valid
        
        # Moves (0-3) are always valid if alive
        valid[0:4] = True
        
        # Tera moves (4-7) only if tera not used and allowed
        if not self.my_tera_used and self.allow_tera:
            valid[4:8] = True
        
        # Switches (8-11) - check bench availability
        if self.allow_switching:
            bench_indices = [i for i in range(self.bring_size) if i not in active_indices]
            
            for i, bench_idx in enumerate(bench_indices[:4]):
                if bench_idx < len(self.my_pokemon_hp) and self.my_pokemon_hp[bench_idx] > 0:
                    valid[8 + i] = True
        
        return valid
    
    def _generate_state(self) -> np.ndarray:
        """Generate state vector from current battle state."""
        state = np.zeros(self.state_dim, dtype=np.float32)
        
        # Encode Pokemon HP (simplified)
        # My active Pokemon
        for i in range(min(2, len(self.my_pokemon_hp))):
            base_idx = i * 50
            state[base_idx] = self.my_pokemon_hp[i]  # HP
            state[base_idx + 1:base_idx + 8] = [1, 0, 0, 0, 0, 0, 0]  # No status
            state[base_idx + 8] = 1.0  # Active
        
        # My bench
        for i in range(2, min(4, len(self.my_pokemon_hp))):
            base_idx = i * 50
            state[base_idx] = self.my_pokemon_hp[i]
        
        # Opponent Pokemon (indices 300-600)
        for i in range(min(2, len(self.opp_pokemon_hp))):
            base_idx = 300 + i * 50
            state[base_idx] = self.opp_pokemon_hp[i]
            state[base_idx + 8] = 1.0  # Active
        
        for i in range(2, min(4, len(self.opp_pokemon_hp))):
            base_idx = 300 + i * 50
            state[base_idx] = self.opp_pokemon_hp[i]
        
        return state


class PokeEnvVGCEnv(gym.Env):
    """Real VGC environment using poke-env and Pokemon Showdown.
    
    This connects to a local Pokemon Showdown server for actual
    battle mechanics instead of simulated battles.
    
    Requires Pokemon Showdown server running locally.
    
    Uses AsyncToSyncEnv internally to handle async poke-env battles
    with synchronous Gymnasium interface.
    """
    
    def __init__(
        self,
        battle_format: str = "gen9vgc2024regg",
        server_configuration: Optional[Dict] = None,
        opponent_type: str = "random",
        team: Optional[str] = None,
        opponent_team: Optional[str] = None,
    ):
        """Initialize the real VGC environment.
        
        Args:
            battle_format: Pokemon Showdown format
            server_configuration: Server connection settings
            opponent_type: Type of opponent ("random", "heuristic", "max_damage")
            team: Team in Showdown paste format (None = random)
            opponent_team: Opponent team (None = random)
        """
        super().__init__()
        
        self.battle_format = battle_format
        self.server_configuration = server_configuration or {}
        self.opponent_type = opponent_type
        self.team = team
        self.opponent_team = opponent_team
        
        # Space definitions
        self.state_dim = 620
        self.action_dim = 144
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(self.action_dim)
        
        # Internal async wrapper
        self._async_env = None
        self._initialized = False
        self._fallback_to_simulated = False
    
    def _initialize_async_env(self):
        """Initialize the async environment wrapper."""
        if self._initialized:
            return
        
        try:
            from src.ml.training.async_env_wrapper import AsyncToSyncEnv, HAS_POKE_ENV
            
            if not HAS_POKE_ENV:
                logger.warning("poke-env not available, will use simulated environment")
                self._fallback_to_simulated = True
                self._initialized = False
                return
            
            self._async_env = AsyncToSyncEnv(
                battle_format=self.battle_format,
                opponent_type=self.opponent_type,
            )
            self._initialized = True
            logger.info(f"Initialized PokeEnvVGCEnv with AsyncToSyncEnv ({self.opponent_type} opponent)")
            
        except Exception as e:
            logger.error(f"Failed to initialize async environment: {e}")
            logger.warning("Falling back to simulated environment")
            self._fallback_to_simulated = True
            self._initialized = False
    
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment and start a new battle."""
        super().reset(seed=seed)
        
        # Initialize async env if not done
        self._initialize_async_env()
        
        if not self._initialized or self._async_env is None:
            # Return zeros if initialization failed
            state = np.zeros(self.state_dim, dtype=np.float32)
            return state, {"initialized": False, "fallback": self._fallback_to_simulated}
        
        # Delegate to async env
        try:
            return self._async_env.reset(seed=seed, options=options)
        except Exception as e:
            logger.error(f"Error during reset: {e}")
            state = np.zeros(self.state_dim, dtype=np.float32)
            return state, {"error": str(e)}
    
    def step(
        self,
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute an action in the battle.
        
        Args:
            action: Combined action index (0-143)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if not self._initialized or self._async_env is None:
            # Return terminal state if not initialized
            return np.zeros(self.state_dim, dtype=np.float32), 0.0, True, False, {"error": "Not initialized"}
        
        # Delegate to async env
        try:
            return self._async_env.step(action)
        except Exception as e:
            logger.error(f"Error during step: {e}")
            return np.zeros(self.state_dim, dtype=np.float32), 0.0, True, False, {"error": str(e)}
    
    def close(self):
        """Clean up resources."""
        if self._async_env is not None:
            self._async_env.close()
        self._initialized = False
    
    def get_action_mask(self) -> np.ndarray:
        """Get valid action mask.
        
        For now, returns all valid since AsyncToSyncEnv handles validation.
        """
        return np.ones(self.action_dim, dtype=bool)


def create_vgc_env(
    use_real_env: bool = False,
    **kwargs
) -> gym.Env:
    """Factory function to create appropriate VGC environment.
    
    Args:
        use_real_env: Whether to use real Showdown connection
        **kwargs: Additional arguments for environment
        
    Returns:
        VGC environment (either SimulatedVGCEnv or PokeEnvVGCEnv)
    """
    if use_real_env:
        try:
            env = PokeEnvVGCEnv(**kwargs)
            env._initialize_players()
            if env._initialized:
                logger.info("Using real PokeEnvVGCEnv")
                return env
            else:
                logger.warning("Real env failed to initialize, falling back to simulated")
        except Exception as e:
            logger.warning(f"Failed to create real env: {e}")
    
    logger.info("Using SimulatedVGCEnv")
    return SimulatedVGCEnv(**kwargs)


class TrainingCallback(BaseCallback):
    """Custom callback for logging training progress."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Log episode stats when available
        if len(self.model.ep_info_buffer) > 0:
            for info in self.model.ep_info_buffer:
                self.episode_rewards.append(info['r'])
                self.episode_lengths.append(info['l'])
        
        return True
    
    def _on_rollout_end(self) -> None:
        if self.episode_rewards:
            mean_reward = np.mean(self.episode_rewards[-100:])
            mean_length = np.mean(self.episode_lengths[-100:])
            
            if self.verbose > 0:
                logger.info(f"Rollout end - Mean reward: {mean_reward:.2f}, Mean length: {mean_length:.1f}")


def detect_model_type(checkpoint: Dict) -> str:
    """Detect model type from checkpoint.
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        
    Returns:
        Model type: "simple", "attention", or "enhanced"
    """
    state_dict = checkpoint.get('model_state_dict', {})
    
    # Check for enhanced policy keys (UnifiedBattleEncoder)
    enhanced_keys = [
        'encoder.pokemon_embed', 'encoder.team_encoder', 
        'encoder.flat_encoder', 'encoder.output_projection'
    ]
    for key in state_dict.keys():
        for enhanced_key in enhanced_keys:
            if enhanced_key in key:
                return "enhanced"
    
    # Check for attention policy keys
    if 'attention_layers' in str(state_dict.keys()):
        return "attention"
    
    return "simple"


def load_pretrained_policy(
    model_path: Path, 
    device: str = "cpu",
    model_type: Optional[str] = None,
) -> nn.Module:
    """Load pretrained imitation policy with automatic type detection.
    
    Args:
        model_path: Path to saved model
        device: Device to load to
        model_type: Override model type (auto-detect if None)
        
    Returns:
        Loaded policy (ImitationPolicy or EnhancedPolicy)
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint.get('config')
    
    # Detect model type if not specified
    if model_type is None:
        if hasattr(config, 'model_type'):
            model_type = config.model_type
        else:
            model_type = detect_model_type(checkpoint)
    
    logger.info(f"Loading pretrained model of type: {model_type}")
    
    if model_type == "enhanced":
        # Load enhanced policy
        enhanced_cfg = EnhancedPolicyConfig(
            flat_state_dim=getattr(config, 'state_dim', 620),
            action_dim=getattr(config, 'action_dim', 144),
            dropout=0.0,  # No dropout for inference
        )
        
        # Try to load enhanced config if present
        if hasattr(config, 'enhanced_config') and config.enhanced_config:
            for key, value in config.enhanced_config.items():
                if hasattr(enhanced_cfg, key):
                    setattr(enhanced_cfg, key, value)
        
        model = EnhancedPolicy(enhanced_cfg)
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded EnhancedPolicy with {sum(p.numel() for p in model.parameters()):,} parameters")
    else:
        # Load simple/attention policy
        model = ImitationPolicy(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dims=config.hidden_dims,
            dropout=0.0,  # No dropout for inference
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"Loaded ImitationPolicy with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    model.eval()
    return model


def create_ppo_from_pretrained(
    pretrained_model: nn.Module,
    env: gym.Env,
    config: RLConfig,
    freeze_encoder: bool = False,
) -> PPO:
    """Create PPO model initialized from pretrained policy.
    
    Automatically detects model type and uses appropriate feature extractor.
    
    Args:
        pretrained_model: Pretrained ImitationPolicy or EnhancedPolicy
        env: Training environment
        config: RL configuration
        freeze_encoder: Whether to freeze encoder weights
        
    Returns:
        PPO model ready for training
    """
    # Detect model type
    if isinstance(pretrained_model, EnhancedPolicy):
        logger.info("Using EnhancedFeaturesExtractor for PPO")
        features_extractor_class = EnhancedFeaturesExtractor
        features_dim = pretrained_model.config.encoder_output_dim
        # Larger heads for enhanced encoder
        net_arch = dict(pi=[128, 64], vf=[128, 64])
    else:
        logger.info("Using PretrainedFeaturesExtractor for PPO")
        features_extractor_class = PretrainedFeaturesExtractor
        features_dim = 128  # Default for simple policy
        net_arch = dict(pi=[64], vf=[64])
    
    # Create PPO with custom features extractor
    policy_kwargs = dict(
        features_extractor_class=features_extractor_class,
        features_extractor_kwargs=dict(
            pretrained_model=pretrained_model,
            freeze_encoder=freeze_encoder,
        ),
        net_arch=net_arch,
    )
    
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        verbose=1,
        tensorboard_log=None,  # Disable tensorboard
        policy_kwargs=policy_kwargs,
    )
    
    logger.info(f"Created PPO with features_dim={features_dim}, freeze_encoder={freeze_encoder}")
    
    return model


def train_ppo_from_scratch(config: RLConfig) -> Path:
    """Train PPO from scratch (without pretrained weights).
    
    Args:
        config: Training configuration
        
    Returns:
        Path to saved model
    """
    logger.info("Training PPO from scratch...")
    
    # Create environment
    env = SimulatedVGCEnv()
    
    # Create PPO model
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        verbose=1,
        tensorboard_log=None,  # Disable tensorboard
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 128], vf=[256, 128]),
        ),
    )
    
    # Train
    callback = TrainingCallback(verbose=1)
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback,
        progress_bar=True,
    )
    
    # Save
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "ppo_finetuned.zip"
    model.save(model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    return model_path


def train_ppo_finetuned(config: RLConfig) -> Path:
    """Fine-tune PPO from pretrained imitation model.
    
    Args:
        config: Training configuration
        
    Returns:
        Path to saved model
    """
    logger.info("Loading pretrained model...")
    
    # Load pretrained
    pretrained = load_pretrained_policy(Path(config.imitation_model_path))
    
    # Create environment
    env = SimulatedVGCEnv()
    
    # Create PPO from pretrained
    logger.info("Creating PPO model from pretrained weights...")
    model = create_ppo_from_pretrained(pretrained, env, config)
    
    # Train
    logger.info(f"Training for {config.total_timesteps} timesteps...")
    callback = TrainingCallback(verbose=1)
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback,
        progress_bar=True,
    )
    
    # Save
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "ppo_finetuned.zip"
    model.save(model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    return model_path


def evaluate_model(model_path: Path, n_episodes: int = 100) -> Dict[str, float]:
    """Evaluate a trained model.
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary of metrics
    """
    logger.info(f"Evaluating model from {model_path}...")
    
    env = SimulatedVGCEnv()
    model = PPO.load(model_path)
    
    wins = 0
    losses = 0
    draws = 0
    total_rewards = []
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        
        if episode_reward > 5:
            wins += 1
        elif episode_reward < -5:
            losses += 1
        else:
            draws += 1
    
    metrics = {
        'win_rate': wins / n_episodes,
        'loss_rate': losses / n_episodes,
        'draw_rate': draws / n_episodes,
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
    }
    
    logger.info(f"Win rate: {metrics['win_rate']:.2%}")
    logger.info(f"Mean reward: {metrics['mean_reward']:.2f}")
    
    return metrics


class CurriculumCallback(BaseCallback):
    """Callback for curriculum learning progression."""
    
    def __init__(
        self,
        curriculum: CurriculumLearning,
        env: SimulatedVGCEnv,
        check_freq: int = 1000,
        verbose: int = 1,
    ):
        """Initialize curriculum callback.
        
        Args:
            curriculum: VGC curriculum manager
            env: Training environment
            check_freq: How often to check for stage advancement
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.curriculum = curriculum
        self.env = env
        self.check_freq = check_freq
        self.steps_since_check = 0
    
    def _on_step(self) -> bool:
        self.steps_since_check += 1
        
        if self.steps_since_check >= self.check_freq:
            self.steps_since_check = 0
            
            # Check if we should advance to next stage
            win_rate = self.env.get_win_rate()
            
            if self.curriculum.should_advance(win_rate, self.env.episode_count):
                old_stage = self.curriculum.current_stage
                self.curriculum.advance()
                new_stage = self.curriculum.current_stage
                
                if self.verbose > 0:
                    logger.info(f"Curriculum advancement: {old_stage.name} -> {new_stage.name}")
                
                # Configure environment for new stage
                self.env.configure_curriculum(new_stage)
                
                # Reset win tracking
                self.env.episode_wins = 0
                self.env.episode_count = 0
        
        return True


def train_with_curriculum(config: RLConfig) -> Path:
    """Train PPO with curriculum learning.
    
    Progressively increases difficulty as the agent improves.
    
    Args:
        config: Training configuration
        
    Returns:
        Path to saved model
    """
    logger.info("Training with curriculum learning...")
    
    # Initialize curriculum
    stages = config.curriculum_stages or DEFAULT_CURRICULUM
    curriculum = CurriculumLearning(stages=stages)
    
    # Create environment with initial stage
    initial_stage = curriculum.current_stage
    env = SimulatedVGCEnv(
        opponent_type=OpponentType.RANDOM,
        bring_size=initial_stage.bring_size,
        max_turns=initial_stage.max_turns,
        opponent_strength=initial_stage.opponent_strength,
        allow_tera=initial_stage.allow_tera,
        allow_switching=initial_stage.allow_switching,
    )
    
    logger.info(f"Starting at stage: {initial_stage.name}")
    
    # Load pretrained if available
    pretrained_path = Path(config.imitation_model_path)
    if pretrained_path.exists():
        logger.info("Loading pretrained model...")
        pretrained = load_pretrained_policy(pretrained_path)
        model = create_ppo_from_pretrained(pretrained, env, config)
    else:
        logger.info("Training from scratch (no pretrained model found)")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            verbose=1,
            policy_kwargs=dict(net_arch=dict(pi=[256, 128], vf=[256, 128])),
        )
    
    # Create callbacks
    curriculum_callback = CurriculumCallback(
        curriculum=curriculum,
        env=env,
        check_freq=1000,
        verbose=1,
    )
    training_callback = TrainingCallback(verbose=1)
    
    # Train
    logger.info(f"Training for {config.total_timesteps} timesteps...")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=[curriculum_callback, training_callback],
        progress_bar=True,
    )
    
    # Log final curriculum status
    final_stage = curriculum.current_stage
    logger.info(f"Finished at stage: {final_stage.name}")
    logger.info(f"Final win rate: {env.get_win_rate():.2%}")
    
    # Save
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "ppo_curriculum.zip"
    model.save(model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    return model_path


def mask_fn(env: SimulatedVGCEnv) -> np.ndarray:
    """Action mask function for MaskablePPO wrapper."""
    return env.get_action_mask()


def train_maskable_ppo(config: RLConfig) -> Path:
    """Train MaskablePPO with action masking.
    
    Uses sb3-contrib's MaskablePPO to prevent selecting invalid actions.
    
    Args:
        config: Training configuration
        
    Returns:
        Path to saved model
    """
    if not HAS_MASKABLE_PPO:
        logger.error("MaskablePPO not available. Install sb3-contrib: pip install sb3-contrib")
        raise ImportError("sb3-contrib not installed")
    
    logger.info("Training MaskablePPO with action masking...")
    
    # Create environment with action masker wrapper
    def make_env():
        env = SimulatedVGCEnv()
        env = ActionMasker(env, mask_fn)
        return env
    
    env = DummyVecEnv([make_env])
    
    # Load pretrained if available
    pretrained_path = Path(config.imitation_model_path)
    policy_kwargs = dict(net_arch=dict(pi=[256, 128], vf=[256, 128]))
    
    if pretrained_path.exists():
        logger.info("Loading pretrained model for feature extraction...")
        try:
            pretrained = load_pretrained_policy(pretrained_path)
            
            # Detect model type and create appropriate extractor
            if isinstance(pretrained, EnhancedPolicy):
                features_extractor_class = EnhancedFeaturesExtractor
            else:
                features_extractor_class = PretrainedFeaturesExtractor
            
            policy_kwargs["features_extractor_class"] = features_extractor_class
            policy_kwargs["features_extractor_kwargs"] = dict(
                pretrained_model=pretrained,
                freeze_encoder=False,
            )
        except Exception as e:
            logger.warning(f"Failed to load pretrained model: {e}")
            logger.info("Training from scratch instead")
    
    # Create MaskablePPO model
    model = MaskablePPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )
    
    # Train
    logger.info(f"Training for {config.total_timesteps} timesteps...")
    callback = TrainingCallback(verbose=1)
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callback,
        progress_bar=True,
    )
    
    # Save
    save_dir = Path(config.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / "maskable_ppo.zip"
    model.save(model_path)
    
    logger.info(f"MaskablePPO model saved to {model_path}")
    
    return model_path


def evaluate_maskable_model(model_path: Path, n_episodes: int = 100) -> Dict[str, float]:
    """Evaluate a MaskablePPO model.
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of evaluation episodes
        
    Returns:
        Dictionary of metrics
    """
    if not HAS_MASKABLE_PPO:
        logger.error("MaskablePPO not available")
        return {}
    
    logger.info(f"Evaluating MaskablePPO from {model_path}...")
    
    env = SimulatedVGCEnv()
    model = MaskablePPO.load(model_path)
    
    wins = 0
    losses = 0
    draws = 0
    total_rewards = []
    invalid_actions = 0
    
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        episode_reward = 0
        
        while not (done or truncated):
            # Get action mask
            action_mask = env.get_action_mask()
            
            # Predict with mask
            action, _ = model.predict(obs, action_masks=action_mask, deterministic=True)
            
            # Check if action was valid
            if not action_mask[action]:
                invalid_actions += 1
            
            obs, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
        
        total_rewards.append(episode_reward)
        
        if episode_reward > 5:
            wins += 1
        elif episode_reward < -5:
            losses += 1
        else:
            draws += 1
    
    metrics = {
        'win_rate': wins / n_episodes,
        'loss_rate': losses / n_episodes,
        'draw_rate': draws / n_episodes,
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'invalid_action_rate': invalid_actions / (n_episodes * 20),  # Approx actions per episode
    }
    
    logger.info(f"Win rate: {metrics['win_rate']:.2%}")
    logger.info(f"Mean reward: {metrics['mean_reward']:.2f}")
    logger.info(f"Invalid action rate: {metrics['invalid_action_rate']:.4f}")
    
    return metrics


def main():
    """Main entry point for RL fine-tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune VGC AI with PPO")
    parser.add_argument("--timesteps", type=int, default=100_000, help="Total training timesteps")
    parser.add_argument("--from-scratch", action="store_true", help="Train from scratch without pretrained weights")
    parser.add_argument("--curriculum", action="store_true", help="Use curriculum learning")
    parser.add_argument("--maskable", action="store_true", help="Use MaskablePPO with action masking")
    parser.add_argument("--eval-only", type=str, default=None, help="Only evaluate a saved model")
    
    args = parser.parse_args()
    
    config = RLConfig(
        total_timesteps=args.timesteps,
        use_curriculum=args.curriculum,
    )
    
    if args.eval_only:
        if args.maskable:
            evaluate_maskable_model(Path(args.eval_only))
        else:
            evaluate_model(Path(args.eval_only))
    elif args.maskable:
        model_path = train_maskable_ppo(config)
        evaluate_maskable_model(model_path)
    elif args.curriculum:
        model_path = train_with_curriculum(config)
        evaluate_model(model_path)
    elif args.from_scratch:
        model_path = train_ppo_from_scratch(config)
        evaluate_model(model_path)
    else:
        model_path = train_ppo_finetuned(config)
        evaluate_model(model_path)


if __name__ == "__main__":
    main()

