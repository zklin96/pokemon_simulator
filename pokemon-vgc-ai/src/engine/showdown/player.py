"""Base player classes for Pokemon Showdown battles using poke-env."""

import asyncio
from typing import Optional, List, Dict, Any
import numpy as np

from poke_env.player import Player, RandomPlayer
from poke_env.battle import (
    AbstractBattle,
    DoubleBattle,
    Move,
    Pokemon,
)
from loguru import logger


class VGCPlayer(Player):
    """Base VGC player that can be extended for different strategies."""
    
    def __init__(
        self,
        battle_format: str = "gen9vgc2024regg",
        team: Optional[str] = None,
        **kwargs
    ):
        """Initialize VGC Player.
        
        Args:
            battle_format: The VGC format to play (e.g., gen9vgc2024regg)
            team: Team paste string or None for random team
            **kwargs: Additional arguments passed to Player
        """
        super().__init__(
            battle_format=battle_format,
            team=team,
            **kwargs
        )
        self.battle_format = battle_format
    
    def choose_move(self, battle: DoubleBattle) -> str:
        """Choose a move for the current turn.
        
        This is the main method to override for custom AI.
        Default implementation makes random choices.
        
        Args:
            battle: Current battle state
            
        Returns:
            Move order string for Showdown
        """
        return self.choose_random_doubles_move(battle)
    
    def teampreview(self, battle: DoubleBattle) -> str:
        """Choose team for VGC team preview.
        
        In VGC, you bring 6 Pokemon but only choose 4 for battle.
        
        Args:
            battle: Current battle state
            
        Returns:
            Team order string (e.g., "/team 1234" for first 4 Pokemon)
        """
        # Default: choose first 4 Pokemon
        return "/team 1234"


class HeuristicVGCPlayer(VGCPlayer):
    """Heuristic-based VGC player using damage calculation and type effectiveness."""
    
    # Type effectiveness chart
    TYPE_CHART: Dict[str, Dict[str, float]] = {
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
    
    def get_type_multiplier(self, move_type: str, defender: Pokemon) -> float:
        """Calculate type effectiveness multiplier.
        
        Args:
            move_type: Type of the attacking move
            defender: Defending Pokemon
            
        Returns:
            Type effectiveness multiplier
        """
        move_type = move_type.lower()
        multiplier = 1.0
        
        for def_type in defender.types:
            if def_type:
                def_type = def_type.name.lower()
                if move_type in self.TYPE_CHART:
                    multiplier *= self.TYPE_CHART[move_type].get(def_type, 1.0)
        
        return multiplier
    
    def estimate_damage(
        self,
        move: Move,
        attacker: Pokemon,
        defender: Pokemon
    ) -> float:
        """Estimate damage output for a move.
        
        Simplified damage calculation for heuristic purposes.
        
        Args:
            move: The move being used
            attacker: Attacking Pokemon
            defender: Defending Pokemon
            
        Returns:
            Estimated damage as percentage of defender's HP
        """
        if move.base_power == 0:
            return 0
        
        # Get attacking and defending stats
        if move.category.name == "PHYSICAL":
            attack = attacker.stats.get("atk", 100)
            defense = defender.stats.get("def", 100)
        elif move.category.name == "SPECIAL":
            attack = attacker.stats.get("spa", 100)
            defense = defender.stats.get("spd", 100)
        else:  # Status move
            return 0
        
        # Basic damage formula (simplified)
        base_damage = ((22 * move.base_power * attack / defense) / 50) + 2
        
        # STAB bonus
        move_type = move.type.name.lower() if move.type else "normal"
        if any(t and t.name.lower() == move_type for t in attacker.types):
            base_damage *= 1.5
        
        # Type effectiveness
        type_mult = self.get_type_multiplier(move_type, defender)
        base_damage *= type_mult
        
        # Estimate as percentage of defender's max HP
        defender_hp = defender.max_hp or 300
        damage_percent = (base_damage / defender_hp) * 100
        
        return damage_percent
    
    def choose_move(self, battle: DoubleBattle) -> str:
        """Choose moves using damage-based heuristics.
        
        Strategy:
        1. For each active Pokemon, evaluate all available moves
        2. Target the opponent Pokemon that takes the most damage
        3. Choose the highest expected damage move
        
        Args:
            battle: Current battle state
            
        Returns:
            Move order string
        """
        orders = []
        
        for i, pokemon in enumerate(battle.active_pokemon):
            if pokemon is None or pokemon.fainted:
                continue
                
            available_moves = battle.available_moves[i] if i < len(battle.available_moves) else []
            
            if not available_moves:
                # No moves available, must switch
                available_switches = battle.available_switches[i] if i < len(battle.available_switches) else []
                if available_switches:
                    orders.append(self.create_order(available_switches[0]))
                continue
            
            best_move = None
            best_damage = -1
            best_target = None
            
            # Evaluate each move against each opponent
            for move in available_moves:
                for j, opponent in enumerate(battle.opponent_active_pokemon):
                    if opponent is None or opponent.fainted:
                        continue
                    
                    damage = self.estimate_damage(move, pokemon, opponent)
                    
                    if damage > best_damage:
                        best_damage = damage
                        best_move = move
                        best_target = j + 1  # 1-indexed target
            
            if best_move:
                # Create order with target for doubles
                if best_target:
                    orders.append(self.create_order(best_move, move_target=best_target))
                else:
                    orders.append(self.create_order(best_move))
            elif available_moves:
                # Fallback to first move
                orders.append(self.create_order(available_moves[0]))
        
        if orders:
            return "/choose " + ", ".join(orders)
        
        # Fallback to random
        return self.choose_random_doubles_move(battle)
    
    def teampreview(self, battle: DoubleBattle) -> str:
        """Choose team based on type matchups against opponent's revealed team.
        
        Args:
            battle: Current battle state
            
        Returns:
            Team order string
        """
        # For now, use default order
        # TODO: Implement smart team selection based on opponent's team
        return "/team 1234"


class RandomVGCPlayer(VGCPlayer):
    """Random player for VGC format (for benchmarking)."""
    
    def choose_move(self, battle: DoubleBattle) -> str:
        """Make random move choices."""
        return self.choose_random_doubles_move(battle)


async def test_players():
    """Test that players can be instantiated and make moves."""
    logger.info("Testing player instantiation...")
    
    # Test player creation
    random_player = RandomVGCPlayer(
        battle_format="gen9vgc2024regg",
        max_concurrent_battles=1
    )
    
    heuristic_player = HeuristicVGCPlayer(
        battle_format="gen9vgc2024regg", 
        max_concurrent_battles=1
    )
    
    logger.info(f"Created RandomVGCPlayer: {random_player}")
    logger.info(f"Created HeuristicVGCPlayer: {heuristic_player}")
    
    return True


if __name__ == "__main__":
    asyncio.run(test_players())

