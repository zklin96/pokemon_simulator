"""Advanced reward shaping for VGC battle AI.

This module provides enhanced reward components beyond simple win/lose:
- Momentum rewards for consecutive KOs
- Tera efficiency tracking
- Positional advantages
- Switch prediction bonuses
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import numpy as np
from loguru import logger


@dataclass
class BattleTracker:
    """Tracks battle state for reward calculation."""
    
    # KO tracking
    my_kos: int = 0
    opp_kos: int = 0
    consecutive_my_kos: int = 0
    consecutive_opp_kos: int = 0
    last_ko_player: str = ""  # "me" or "opp"
    
    # Tera tracking
    my_tera_used: bool = False
    my_tera_turn: int = -1
    my_tera_effective: bool = False  # Was Tera useful?
    opp_tera_used: bool = False
    
    # HP tracking
    prev_my_hp: float = 4.0  # Total HP (fraction * 4 Pokemon)
    prev_opp_hp: float = 4.0
    
    # Turn tracking
    turn: int = 0
    
    # Action history
    my_actions: List[int] = field(default_factory=list)
    opp_actions: List[int] = field(default_factory=list)
    
    # Switch tracking
    my_switches: int = 0
    predicted_opp_switch: bool = False
    opp_actually_switched: bool = False
    
    def record_ko(self, is_my_ko: bool):
        """Record a KO event."""
        if is_my_ko:
            self.my_kos += 1
            if self.last_ko_player == "me":
                self.consecutive_my_kos += 1
            else:
                self.consecutive_my_kos = 1
                self.consecutive_opp_kos = 0
            self.last_ko_player = "me"
        else:
            self.opp_kos += 1
            if self.last_ko_player == "opp":
                self.consecutive_opp_kos += 1
            else:
                self.consecutive_opp_kos = 1
                self.consecutive_my_kos = 0
            self.last_ko_player = "opp"
    
    def record_tera(self, turn: int):
        """Record Tera usage."""
        self.my_tera_used = True
        self.my_tera_turn = turn
    
    def mark_tera_effective(self):
        """Mark Tera as effective (got a KO or survived a hit)."""
        self.my_tera_effective = True
    
    def update_hp(self, my_hp: float, opp_hp: float):
        """Update HP values."""
        self.prev_my_hp = my_hp
        self.prev_opp_hp = opp_hp


@dataclass 
class RewardConfig:
    """Configuration for reward shaping."""
    
    # Win/lose rewards
    win_reward: float = 10.0
    lose_reward: float = -10.0
    draw_reward: float = 0.0
    
    # Damage rewards
    damage_dealt_scale: float = 1.0  # Reward per 100 HP damage dealt
    damage_taken_scale: float = -0.5  # Penalty per 100 HP damage taken
    
    # KO rewards
    ko_reward: float = 2.0
    ko_penalty: float = -2.0
    
    # Momentum rewards
    consecutive_ko_bonus: float = 0.5  # Per consecutive KO
    max_consecutive_bonus: float = 2.0  # Cap
    momentum_break_penalty: float = -0.3  # Losing momentum
    
    # Tera rewards
    tera_effective_bonus: float = 1.0  # Tera led to KO or survival
    tera_waste_penalty: float = -1.0  # Tera without benefit
    tera_timing_late_penalty: float = -0.2  # Using Tera after turn 10
    
    # Switch rewards
    switch_penalty: float = -0.1  # Small penalty for switching
    predict_switch_bonus: float = 0.3  # Correctly predicted opponent switch
    
    # Positioning rewards
    speed_advantage_bonus: float = 0.1  # Having speed control
    type_advantage_bonus: float = 0.2  # Favorable matchup on field
    
    # Turn penalties
    turn_penalty: float = -0.01  # Small penalty per turn
    stall_penalty: float = -0.1  # Extra penalty for low damage turns
    
    # HP-based rewards
    hp_advantage_scale: float = 0.5  # Reward for HP lead
    low_hp_threshold: float = 0.25  # Below this = danger
    
    # Action diversity
    diversity_bonus: float = 0.05  # Using variety of actions


class RewardShaper:
    """Computes shaped rewards for VGC battles."""
    
    def __init__(self, config: Optional[RewardConfig] = None):
        """Initialize reward shaper.
        
        Args:
            config: Reward configuration (uses defaults if None)
        """
        self.config = config or RewardConfig()
        self.battle_tracker = BattleTracker()
        self.action_history: Dict[int, int] = {}  # action -> count
        
    def reset(self):
        """Reset for new battle."""
        self.battle_tracker = BattleTracker()
        self.action_history = {}
    
    def calculate_reward(
        self,
        battle_finished: bool = False,
        won: bool = False,
        lost: bool = False,
        my_hp: float = 4.0,
        opp_hp: float = 4.0,
        my_kos_this_turn: int = 0,
        opp_kos_this_turn: int = 0,
        action_taken: int = 0,
        used_tera: bool = False,
        switched: bool = False,
        opp_switched: bool = False,
        had_type_advantage: bool = False,
        had_speed_advantage: bool = False,
        damage_dealt: float = 0.0,
        damage_taken: float = 0.0,
    ) -> float:
        """Calculate shaped reward for current turn.
        
        Args:
            battle_finished: Whether battle ended
            won: Whether we won (if finished)
            lost: Whether we lost (if finished)
            my_hp: Total HP fraction (0-4 for 4 Pokemon)
            opp_hp: Opponent HP fraction (0-4)
            my_kos_this_turn: How many KOs we got this turn
            opp_kos_this_turn: How many KOs opponent got
            action_taken: Action index used
            used_tera: Whether we Terastallized this turn
            switched: Whether we switched this turn
            opp_switched: Whether opponent switched
            had_type_advantage: Field has favorable type matchup
            had_speed_advantage: We have speed control
            damage_dealt: Damage dealt this turn
            damage_taken: Damage taken this turn
            
        Returns:
            Shaped reward value
        """
        config = self.config
        reward = 0.0
        
        self.battle_tracker.turn += 1
        
        # ========== TERMINAL REWARDS ==========
        if battle_finished:
            if won:
                reward += config.win_reward
                # Bonus for decisive victory (HP remaining)
                reward += my_hp * 0.5
            elif lost:
                reward += config.lose_reward
            else:
                reward += config.draw_reward
            return reward
        
        # ========== DAMAGE REWARDS ==========
        reward += damage_dealt * config.damage_dealt_scale / 100.0
        reward += damage_taken * config.damage_taken_scale / 100.0
        
        # Stall penalty if very little happened
        if damage_dealt < 10 and damage_taken < 10 and not switched:
            reward += config.stall_penalty
        
        # ========== KO REWARDS ==========
        for _ in range(my_kos_this_turn):
            self.battle_tracker.record_ko(is_my_ko=True)
            reward += config.ko_reward
            
            # Momentum bonus for consecutive KOs
            consecutive = self.battle_tracker.consecutive_my_kos
            if consecutive > 1:
                momentum_bonus = min(
                    (consecutive - 1) * config.consecutive_ko_bonus,
                    config.max_consecutive_bonus
                )
                reward += momentum_bonus
            
            # Tera effectiveness check
            if self.battle_tracker.my_tera_used and not self.battle_tracker.my_tera_effective:
                turns_since_tera = self.battle_tracker.turn - self.battle_tracker.my_tera_turn
                if turns_since_tera <= 3:  # Got KO within 3 turns of Tera
                    self.battle_tracker.mark_tera_effective()
                    reward += config.tera_effective_bonus
        
        for _ in range(opp_kos_this_turn):
            self.battle_tracker.record_ko(is_my_ko=False)
            reward += config.ko_penalty
            
            # Momentum break penalty
            if self.battle_tracker.consecutive_my_kos > 0:
                reward += config.momentum_break_penalty
        
        # ========== TERA REWARDS ==========
        if used_tera:
            self.battle_tracker.record_tera(self.battle_tracker.turn)
            
            # Late Tera penalty
            if self.battle_tracker.turn > 10:
                reward += config.tera_timing_late_penalty
        
        # ========== SWITCH HANDLING ==========
        if switched:
            self.battle_tracker.my_switches += 1
            reward += config.switch_penalty
        
        # Switch prediction bonus
        if self.battle_tracker.predicted_opp_switch and opp_switched:
            reward += config.predict_switch_bonus
        self.battle_tracker.opp_actually_switched = opp_switched
        
        # ========== POSITIONING REWARDS ==========
        if had_type_advantage:
            reward += config.type_advantage_bonus
        
        if had_speed_advantage:
            reward += config.speed_advantage_bonus
        
        # ========== HP-BASED REWARDS ==========
        hp_diff = my_hp - opp_hp
        reward += hp_diff * config.hp_advantage_scale * 0.1  # Scale down
        
        # Track HP changes
        self.battle_tracker.update_hp(my_hp, opp_hp)
        
        # ========== TURN PENALTY ==========
        reward += config.turn_penalty
        
        # ========== ACTION DIVERSITY ==========
        if action_taken not in self.action_history:
            reward += config.diversity_bonus
        self.action_history[action_taken] = self.action_history.get(action_taken, 0) + 1
        
        # Track action
        self.battle_tracker.my_actions.append(action_taken)
        
        return reward
    
    def get_end_of_battle_bonus(self) -> float:
        """Calculate end-of-battle bonuses/penalties.
        
        Call this when battle ends to get additional rewards.
        
        Returns:
            Additional reward based on battle performance
        """
        bonus = 0.0
        config = self.config
        
        # Tera efficiency check
        if self.battle_tracker.my_tera_used and not self.battle_tracker.my_tera_effective:
            bonus += config.tera_waste_penalty
        
        # Efficient win bonus (few switches, quick game)
        if self.battle_tracker.my_kos >= 4:  # Won
            if self.battle_tracker.my_switches <= 2:
                bonus += 0.5  # Minimal switches
            if self.battle_tracker.turn <= 10:
                bonus += 0.5  # Quick win
        
        return bonus


def create_reward_shaper(
    config_dict: Optional[Dict[str, float]] = None,
) -> RewardShaper:
    """Factory function to create a reward shaper.
    
    Args:
        config_dict: Optional config overrides
        
    Returns:
        Configured RewardShaper instance
    """
    if config_dict:
        config = RewardConfig(**config_dict)
    else:
        config = RewardConfig()
    
    return RewardShaper(config)


# Pre-configured reward shapers for different training phases
def get_early_training_shaper() -> RewardShaper:
    """Get reward shaper for early training.
    
    Emphasizes survival and basic game mechanics.
    """
    config = RewardConfig(
        win_reward=10.0,
        lose_reward=-10.0,
        damage_dealt_scale=1.5,  # Encourage dealing damage
        damage_taken_scale=-0.3,  # Less penalty for taking damage
        ko_reward=3.0,  # Big KO bonus
        consecutive_ko_bonus=0.3,  # Lower momentum bonus
        tera_effective_bonus=0.5,  # Lower Tera bonus
        tera_waste_penalty=-0.5,  # Lower waste penalty
    )
    return RewardShaper(config)


def get_competitive_shaper() -> RewardShaper:
    """Get reward shaper for competitive training.
    
    Emphasizes efficient play and advanced strategies.
    """
    config = RewardConfig(
        win_reward=10.0,
        lose_reward=-10.0,
        damage_dealt_scale=0.8,
        damage_taken_scale=-0.8,  # Higher penalty
        ko_reward=2.0,
        consecutive_ko_bonus=0.7,  # Higher momentum bonus
        tera_effective_bonus=1.5,  # Bigger Tera bonus
        tera_waste_penalty=-1.5,  # Bigger waste penalty
        type_advantage_bonus=0.3,  # Higher positioning bonus
        speed_advantage_bonus=0.2,
    )
    return RewardShaper(config)

