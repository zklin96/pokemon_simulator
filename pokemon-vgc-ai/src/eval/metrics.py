"""Evaluation metrics for VGC AI.

This module defines various metrics for evaluating agent performance
beyond simple win rate.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
import numpy as np


@dataclass
class GameRecord:
    """Record of a single game."""
    won: bool
    turns: int
    my_kos: int
    opp_kos: int
    my_final_hp: float
    opp_final_hp: float
    actions_taken: List[int]
    tera_used: bool = False
    switches: int = 0


class Metric(ABC):
    """Base class for evaluation metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name."""
        pass
    
    @abstractmethod
    def compute(self, games: List[GameRecord]) -> Dict[str, float]:
        """Compute metric from game records.
        
        Args:
            games: List of game records
            
        Returns:
            Dict of metric values
        """
        pass
    
    @abstractmethod
    def update(self, game: GameRecord):
        """Update metric with a single game."""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset metric state."""
        pass


class WinRateMetric(Metric):
    """Track win rate and related statistics."""
    
    def __init__(self):
        self.wins = 0
        self.losses = 0
        self.games = 0
    
    @property
    def name(self) -> str:
        return "win_rate"
    
    def compute(self, games: List[GameRecord]) -> Dict[str, float]:
        wins = sum(1 for g in games if g.won)
        total = len(games)
        
        if total == 0:
            return {"win_rate": 0.0, "wins": 0, "losses": 0}
        
        return {
            "win_rate": wins / total,
            "wins": wins,
            "losses": total - wins,
            "total_games": total,
        }
    
    def update(self, game: GameRecord):
        self.games += 1
        if game.won:
            self.wins += 1
        else:
            self.losses += 1
    
    def reset(self):
        self.wins = 0
        self.losses = 0
        self.games = 0
    
    def get_current(self) -> float:
        if self.games == 0:
            return 0.0
        return self.wins / self.games


class ELOMetric(Metric):
    """Track ELO rating changes."""
    
    def __init__(self, initial_elo: float = 1500.0, k_factor: float = 32.0):
        self.elo = initial_elo
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.history: List[float] = [initial_elo]
    
    @property
    def name(self) -> str:
        return "elo"
    
    def compute(self, games: List[GameRecord]) -> Dict[str, float]:
        return {
            "current_elo": self.elo,
            "initial_elo": self.initial_elo,
            "elo_change": self.elo - self.initial_elo,
            "peak_elo": max(self.history),
            "min_elo": min(self.history),
        }
    
    def update(self, game: GameRecord):
        # Simplified update (assuming opponent ELO = current ELO)
        expected = 0.5  # Equal strength
        actual = 1.0 if game.won else 0.0
        
        self.elo += self.k_factor * (actual - expected)
        self.history.append(self.elo)
    
    def update_vs(self, won: bool, opponent_elo: float):
        """Update ELO with known opponent rating."""
        expected = 1.0 / (1.0 + 10 ** ((opponent_elo - self.elo) / 400.0))
        actual = 1.0 if won else 0.0
        
        self.elo += self.k_factor * (actual - expected)
        self.history.append(self.elo)
    
    def reset(self):
        self.elo = self.initial_elo
        self.history = [self.initial_elo]


class TurnsMetric(Metric):
    """Track game length and pace."""
    
    def __init__(self):
        self.turns: List[int] = []
    
    @property
    def name(self) -> str:
        return "turns"
    
    def compute(self, games: List[GameRecord]) -> Dict[str, float]:
        turns = [g.turns for g in games]
        
        if not turns:
            return {"avg_turns": 0.0, "min_turns": 0, "max_turns": 0}
        
        return {
            "avg_turns": np.mean(turns),
            "std_turns": np.std(turns),
            "min_turns": min(turns),
            "max_turns": max(turns),
            "median_turns": np.median(turns),
        }
    
    def update(self, game: GameRecord):
        self.turns.append(game.turns)
    
    def reset(self):
        self.turns = []


class KOEfficiencyMetric(Metric):
    """Track KO efficiency (KOs per loss)."""
    
    def __init__(self):
        self.my_kos: List[int] = []
        self.opp_kos: List[int] = []
    
    @property
    def name(self) -> str:
        return "ko_efficiency"
    
    def compute(self, games: List[GameRecord]) -> Dict[str, float]:
        my_kos = [g.my_kos for g in games]
        opp_kos = [g.opp_kos for g in games]
        
        if not my_kos:
            return {"ko_efficiency": 0.0}
        
        total_my = sum(my_kos)
        total_opp = sum(opp_kos)
        
        return {
            "ko_efficiency": total_my / max(total_opp, 1),
            "avg_my_kos": np.mean(my_kos),
            "avg_opp_kos": np.mean(opp_kos),
            "total_my_kos": total_my,
            "total_opp_kos": total_opp,
            "ko_differential": total_my - total_opp,
        }
    
    def update(self, game: GameRecord):
        self.my_kos.append(game.my_kos)
        self.opp_kos.append(game.opp_kos)
    
    def reset(self):
        self.my_kos = []
        self.opp_kos = []


class ActionDistributionMetric(Metric):
    """Track action distribution and diversity."""
    
    def __init__(self, action_dim: int = 144):
        self.action_dim = action_dim
        self.action_counts = Counter()
    
    @property
    def name(self) -> str:
        return "action_distribution"
    
    def compute(self, games: List[GameRecord]) -> Dict[str, float]:
        all_actions = []
        for g in games:
            all_actions.extend(g.actions_taken)
        
        if not all_actions:
            return {"action_entropy": 0.0}
        
        counter = Counter(all_actions)
        total = len(all_actions)
        
        # Entropy
        probs = [count / total for count in counter.values()]
        entropy = -sum(p * np.log(p + 1e-10) for p in probs)
        max_entropy = np.log(self.action_dim)
        
        # Most common actions
        top_actions = counter.most_common(5)
        
        return {
            "action_entropy": entropy,
            "normalized_entropy": entropy / max_entropy,
            "unique_actions": len(counter),
            "top_action_1": top_actions[0][0] if top_actions else -1,
            "top_action_1_pct": top_actions[0][1] / total if top_actions else 0,
        }
    
    def update(self, game: GameRecord):
        self.action_counts.update(game.actions_taken)
    
    def reset(self):
        self.action_counts = Counter()


class TeraUsageMetric(Metric):
    """Track terastallization usage."""
    
    def __init__(self):
        self.tera_games = 0
        self.total_games = 0
        self.tera_wins = 0
    
    @property
    def name(self) -> str:
        return "tera_usage"
    
    def compute(self, games: List[GameRecord]) -> Dict[str, float]:
        tera_games = sum(1 for g in games if g.tera_used)
        tera_wins = sum(1 for g in games if g.tera_used and g.won)
        total = len(games)
        
        if total == 0:
            return {"tera_rate": 0.0}
        
        return {
            "tera_rate": tera_games / total,
            "tera_win_rate": tera_wins / max(tera_games, 1),
            "tera_games": tera_games,
        }
    
    def update(self, game: GameRecord):
        self.total_games += 1
        if game.tera_used:
            self.tera_games += 1
            if game.won:
                self.tera_wins += 1
    
    def reset(self):
        self.tera_games = 0
        self.total_games = 0
        self.tera_wins = 0


class SwitchingMetric(Metric):
    """Track switching behavior."""
    
    def __init__(self):
        self.switches_per_game: List[int] = []
    
    @property
    def name(self) -> str:
        return "switching"
    
    def compute(self, games: List[GameRecord]) -> Dict[str, float]:
        switches = [g.switches for g in games]
        
        if not switches:
            return {"avg_switches": 0.0}
        
        return {
            "avg_switches": np.mean(switches),
            "std_switches": np.std(switches),
            "max_switches": max(switches),
            "switch_rate": np.mean(switches) / max(np.mean([g.turns for g in games]), 1),
        }
    
    def update(self, game: GameRecord):
        self.switches_per_game.append(game.switches)
    
    def reset(self):
        self.switches_per_game = []


class HPDifferentialMetric(Metric):
    """Track HP differential at end of games."""
    
    def __init__(self):
        self.differentials: List[float] = []
    
    @property
    def name(self) -> str:
        return "hp_differential"
    
    def compute(self, games: List[GameRecord]) -> Dict[str, float]:
        diffs = [g.my_final_hp - g.opp_final_hp for g in games]
        
        if not diffs:
            return {"avg_hp_diff": 0.0}
        
        return {
            "avg_hp_diff": np.mean(diffs),
            "std_hp_diff": np.std(diffs),
            "win_hp_diff": np.mean([d for d, g in zip(diffs, games) if g.won]) if any(g.won for g in games) else 0,
            "loss_hp_diff": np.mean([d for d, g in zip(diffs, games) if not g.won]) if any(not g.won for g in games) else 0,
        }
    
    def update(self, game: GameRecord):
        self.differentials.append(game.my_final_hp - game.opp_final_hp)
    
    def reset(self):
        self.differentials = []

