"""Battle AI agents for VGC doubles battles."""

import random
from typing import Optional, List, Dict, Any
from abc import ABC, abstractmethod

import numpy as np
from poke_env.player import Player
from poke_env.battle import DoubleBattle, Move, Pokemon
from loguru import logger

from src.engine.state.game_state import GameStateEncoder, ActionSpaceHandler


class BaseVGCAgent(ABC):
    """Abstract base class for VGC battle agents."""
    
    def __init__(self, name: str = "BaseAgent"):
        """Initialize the agent.
        
        Args:
            name: Agent name for logging
        """
        self.name = name
        self.state_encoder = GameStateEncoder()
        self.action_handler = ActionSpaceHandler()
    
    @abstractmethod
    def select_action(self, battle: DoubleBattle) -> int:
        """Select an action given the current battle state.
        
        Args:
            battle: Current battle state
            
        Returns:
            Action index
        """
        pass
    
    def get_valid_actions(self, battle: DoubleBattle) -> List[int]:
        """Get list of valid actions for current state.
        
        Args:
            battle: Current battle state
            
        Returns:
            List of valid action indices
        """
        return self.action_handler.get_available_actions(battle)


class RandomAgent(BaseVGCAgent):
    """Agent that selects random valid actions."""
    
    def __init__(self, name: str = "RandomAgent"):
        super().__init__(name)
    
    def select_action(self, battle: DoubleBattle) -> int:
        """Select a random valid action.
        
        Args:
            battle: Current battle state
            
        Returns:
            Random valid action index
        """
        valid_actions = self.get_valid_actions(battle)
        return random.choice(valid_actions)


class MaxDamageAgent(BaseVGCAgent):
    """Agent that selects moves to maximize expected damage."""
    
    # Type effectiveness chart
    TYPE_CHART = {
        "normal": {"rock": 0.5, "ghost": 0, "steel": 0.5},
        "fire": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 2, "bug": 2, "rock": 0.5, "dragon": 0.5, "steel": 2},
        "water": {"fire": 2, "water": 0.5, "grass": 0.5, "ground": 2, "rock": 2, "dragon": 0.5},
        "electric": {"water": 2, "electric": 0.5, "grass": 0.5, "ground": 0, "flying": 2, "dragon": 0.5},
        "grass": {"fire": 0.5, "water": 2, "grass": 0.5, "poison": 0.5, "ground": 2, "flying": 0.5, "bug": 0.5, "rock": 2, "dragon": 0.5, "steel": 0.5},
        "ice": {"fire": 0.5, "water": 0.5, "grass": 2, "ice": 0.5, "ground": 2, "flying": 2, "dragon": 2, "steel": 0.5},
        "fighting": {"normal": 2, "ice": 2, "poison": 0.5, "flying": 0.5, "psychic": 0.5, "bug": 0.5, "rock": 2, "ghost": 0, "dark": 2, "steel": 2, "fairy": 0.5},
        "poison": {"grass": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "ghost": 0.5, "steel": 0, "fairy": 2},
        "ground": {"fire": 2, "electric": 2, "grass": 0.5, "poison": 2, "flying": 0, "bug": 0.5, "rock": 2, "steel": 2},
        "flying": {"electric": 0.5, "grass": 2, "fighting": 2, "bug": 2, "rock": 0.5, "steel": 0.5},
        "psychic": {"fighting": 2, "poison": 2, "psychic": 0.5, "dark": 0, "steel": 0.5},
        "bug": {"fire": 0.5, "grass": 2, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2, "ghost": 0.5, "dark": 2, "steel": 0.5, "fairy": 0.5},
        "rock": {"fire": 2, "ice": 2, "fighting": 0.5, "ground": 0.5, "flying": 2, "bug": 2, "steel": 0.5},
        "ghost": {"normal": 0, "psychic": 2, "ghost": 2, "dark": 0.5},
        "dragon": {"dragon": 2, "steel": 0.5, "fairy": 0},
        "dark": {"fighting": 0.5, "psychic": 2, "ghost": 2, "dark": 0.5, "fairy": 0.5},
        "steel": {"fire": 0.5, "water": 0.5, "electric": 0.5, "ice": 2, "rock": 2, "steel": 0.5, "fairy": 2},
        "fairy": {"fire": 0.5, "fighting": 2, "poison": 0.5, "dragon": 2, "dark": 2, "steel": 0.5},
    }
    
    def __init__(self, name: str = "MaxDamageAgent"):
        super().__init__(name)
    
    def estimate_damage(
        self, 
        move: Move, 
        attacker: Pokemon, 
        defender: Pokemon
    ) -> float:
        """Estimate damage output for a move.
        
        Args:
            move: The move being used
            attacker: Attacking Pokemon
            defender: Defending Pokemon
            
        Returns:
            Estimated damage as a score
        """
        if not move.base_power or move.base_power == 0:
            return 0.0
        
        # Base score from power
        score = float(move.base_power)
        
        # STAB bonus
        move_type = move.type.name.lower() if move.type else "normal"
        attacker_types = [t.name.lower() for t in attacker.types if t]
        if move_type in attacker_types:
            score *= 1.5
        
        # Type effectiveness
        for def_type in defender.types:
            if def_type:
                def_type_name = def_type.name.lower()
                if move_type in self.TYPE_CHART:
                    score *= self.TYPE_CHART[move_type].get(def_type_name, 1.0)
        
        # Accuracy factor
        if move.accuracy:
            score *= move.accuracy / 100.0
        
        return score
    
    def select_action(self, battle: DoubleBattle) -> int:
        """Select action that maximizes expected damage.
        
        Args:
            battle: Current battle state
            
        Returns:
            Action index for highest damage
        """
        valid_actions = self.get_valid_actions(battle)
        
        best_action = valid_actions[0]
        best_score = -1
        
        for action in valid_actions:
            slot1_action, slot2_action = self.action_handler.decode_action(action)
            score = 0.0
            
            # Evaluate slot 1 action
            score += self._evaluate_slot_action(battle, 0, slot1_action)
            
            # Evaluate slot 2 action
            score += self._evaluate_slot_action(battle, 1, slot2_action)
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def _evaluate_slot_action(
        self, 
        battle: DoubleBattle, 
        slot: int, 
        action: int
    ) -> float:
        """Evaluate a single slot action.
        
        Args:
            battle: Current battle state
            slot: Pokemon slot (0 or 1)
            action: Action index for this slot
            
        Returns:
            Evaluation score
        """
        if slot >= len(battle.active_pokemon):
            return 0.0
        
        attacker = battle.active_pokemon[slot]
        if attacker is None or attacker.fainted:
            return 0.0
        
        moves = battle.available_moves[slot] if slot < len(battle.available_moves) else []
        
        # Move action (0-3) or Tera + Move (4-7)
        if action < 8:
            move_idx = action % 4
            if move_idx >= len(moves):
                return 0.0
            
            move = moves[move_idx]
            
            # Calculate damage against all opponents
            total_damage = 0.0
            for defender in battle.opponent_active_pokemon:
                if defender and not defender.fainted:
                    total_damage += self.estimate_damage(move, attacker, defender)
            
            # Bonus for tera (action 4-7)
            if action >= 4 and battle.can_tera and not attacker.terastallized:
                total_damage *= 1.2  # Rough tera bonus
            
            return total_damage
        
        # Switch action (8-11) - small positive score for switches
        return 10.0  # Base switch score


class HeuristicAgent(BaseVGCAgent):
    """Agent using multiple heuristics for decision making."""
    
    def __init__(self, name: str = "HeuristicAgent"):
        super().__init__(name)
        self.max_damage_agent = MaxDamageAgent()
    
    def select_action(self, battle: DoubleBattle) -> int:
        """Select action using heuristics.
        
        Strategy:
        1. If low HP, consider switching
        2. Otherwise, maximize damage
        3. Consider speed tiers and priority moves
        
        Args:
            battle: Current battle state
            
        Returns:
            Action index
        """
        valid_actions = self.get_valid_actions(battle)
        
        # Check if any active Pokemon should switch
        should_switch = []
        for i, pokemon in enumerate(battle.active_pokemon):
            if pokemon and not pokemon.fainted:
                hp_pct = pokemon.current_hp / pokemon.max_hp if pokemon.max_hp else 0
                if hp_pct < 0.25:
                    should_switch.append(i)
        
        # If Pokemon should switch, look for switch actions
        if should_switch:
            for action in valid_actions:
                slot1, slot2 = self.action_handler.decode_action(action)
                
                # Check if switches are available for the slots that need it
                has_needed_switches = True
                if 0 in should_switch and slot1 < 8:
                    has_needed_switches = False
                if 1 in should_switch and slot2 < 8:
                    has_needed_switches = False
                
                if has_needed_switches:
                    return action
        
        # Otherwise use max damage strategy
        return self.max_damage_agent.select_action(battle)


class GreedyProtectAgent(BaseVGCAgent):
    """Agent that uses Protect strategically and maximizes damage otherwise."""
    
    def __init__(self, name: str = "GreedyProtectAgent"):
        super().__init__(name)
        self.max_damage_agent = MaxDamageAgent()
        self.last_protected = {}  # Track last protect usage
    
    def select_action(self, battle: DoubleBattle) -> int:
        """Select action with strategic Protect usage.
        
        Args:
            battle: Current battle state
            
        Returns:
            Action index
        """
        # Simple strategy: use max damage for now
        # Protect logic would check for threats and use Protect accordingly
        return self.max_damage_agent.select_action(battle)


def evaluate_agents(
    agent1: BaseVGCAgent,
    agent2: BaseVGCAgent,
    num_battles: int = 100
) -> Dict[str, Any]:
    """Evaluate two agents against each other.
    
    Note: This is a placeholder - actual battles require poke-env server.
    
    Args:
        agent1: First agent
        agent2: Second agent
        num_battles: Number of battles to simulate
        
    Returns:
        Evaluation results
    """
    results = {
        "agent1_name": agent1.name,
        "agent2_name": agent2.name,
        "num_battles": num_battles,
        "agent1_wins": 0,
        "agent2_wins": 0,
        "ties": 0,
    }
    
    logger.info(f"Evaluation requires running actual battles with poke-env server")
    logger.info(f"Would evaluate {agent1.name} vs {agent2.name}")
    
    return results


def test_agents():
    """Test agent instantiation and basic functionality."""
    logger.info("Testing VGC Battle Agents...")
    
    # Create agents
    random_agent = RandomAgent()
    max_damage_agent = MaxDamageAgent()
    heuristic_agent = HeuristicAgent()
    
    logger.info(f"Created {random_agent.name}")
    logger.info(f"Created {max_damage_agent.name}")
    logger.info(f"Created {heuristic_agent.name}")
    
    logger.info("Agent tests passed!")


if __name__ == "__main__":
    test_agents()

