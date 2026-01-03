"""Enhanced self-play training with population-based training features.

This module extends the basic self-play system with:
- Hall of Fame for best historical agents
- League system with divisions
- Diversity mechanisms to prevent convergence
- Intrinsic rewards for exploration
"""

import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
import json
import copy
import heapq

import numpy as np
import torch
import torch.nn as nn
from loguru import logger

from .self_play import AgentRecord, EloSystem, AgentPopulation

# Import Team and TeamEvolver for team evolution
try:
    from src.ml.team_builder.team import Team
    from src.ml.team_builder.evolutionary import TeamEvolver, EvolutionConfig
    HAS_TEAM_BUILDER = True
except ImportError:
    HAS_TEAM_BUILDER = False
    Team = None
    TeamEvolver = None
    EvolutionConfig = None


class TeamPoolManager:
    """Manages a pool of teams that evolve alongside agents.
    
    Teams are selected for battles, updated based on results,
    and periodically evolved using genetic algorithms.
    """
    
    def __init__(
        self,
        pool_size: int = 20,
        evolution_interval: int = 5,
        save_dir: Optional[Path] = None,
    ):
        """Initialize team pool.
        
        Args:
            pool_size: Number of teams in the pool
            evolution_interval: Evolve teams every N iterations
            save_dir: Directory to save team data
        """
        self.pool_size = pool_size
        self.evolution_interval = evolution_interval
        self.save_dir = save_dir
        self.teams: List[Team] = []
        self.iteration = 0
        
        # Initialize with random teams
        if HAS_TEAM_BUILDER:
            self._initialize_pool()
        else:
            logger.warning("Team builder not available, team evolution disabled")
    
    def _initialize_pool(self):
        """Initialize pool with random valid teams."""
        logger.info(f"Initializing team pool with {self.pool_size} random teams")
        for i in range(self.pool_size):
            team = Team.random()
            team.name = f"Team_{i:03d}"
            self.teams.append(team)
        logger.info(f"Created {len(self.teams)} initial teams")
    
    def get_team_for_agent(self, agent: AgentRecord) -> Optional[Team]:
        """Get a team for an agent to use in battle.
        
        Selection is based on team ELO with some exploration.
        
        Args:
            agent: Agent requesting a team
            
        Returns:
            Selected team or None if no teams available
        """
        if not self.teams:
            return None
        
        # If agent already has a team, use it with some probability
        if agent.team is not None and random.random() < 0.7:
            return agent.team
        
        # Otherwise, select from pool (weighted by ELO)
        # Higher ELO teams are more likely to be selected
        total_elo = sum(t.elo for t in self.teams)
        if total_elo <= 0:
            return random.choice(self.teams)
        
        weights = [t.elo / total_elo for t in self.teams]
        selected = random.choices(self.teams, weights=weights, k=1)[0]
        
        # Assign to agent for tracking
        agent.team = selected
        return selected
    
    def update_after_battle(self, team: Team, won: bool):
        """Update team statistics after a battle.
        
        Args:
            team: Team that participated
            won: Whether the team won
        """
        if team is None:
            return
        
        team.games_played += 1
        if won:
            team.wins += 1
            # ELO boost for winning
            team.elo += 16
        else:
            team.losses += 1
            # ELO penalty for losing (smaller)
            team.elo = max(1000, team.elo - 12)
    
    def evolve_teams(self):
        """Evolve teams using genetic algorithms.
        
        Replaces worst-performing teams with offspring of best teams.
        """
        if not HAS_TEAM_BUILDER or len(self.teams) < 4:
            return
        
        logger.info("Evolving team pool...")
        
        # Sort by win rate (descending)
        sorted_teams = sorted(
            self.teams, 
            key=lambda t: (t.win_rate, t.elo), 
            reverse=True
        )
        
        # Keep top 50% as parents
        num_parents = len(sorted_teams) // 2
        parents = sorted_teams[:num_parents]
        
        # Generate offspring to replace worst teams
        new_teams = list(parents)  # Keep parents
        
        from src.ml.team_builder.evolutionary import TeamEvolver, EvolutionConfig
        
        # Create temporary evolver for mutation/crossover
        evolver_config = EvolutionConfig(
            population_size=self.pool_size - num_parents,
            mutation_rate=0.3,
            crossover_rate=0.5,
        )
        evolver = TeamEvolver(config=evolver_config)
        evolver.population = parents
        
        while len(new_teams) < self.pool_size:
            # Select two parents
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Crossover
            child = evolver.crossover(parent1, parent2)
            
            # Mutation
            child = evolver.mutate(child)
            child.name = f"Team_g{self.iteration}_{len(new_teams)}"
            child.elo = 1500.0  # Reset ELO for new teams
            child.games_played = 0
            child.wins = 0
            child.losses = 0
            
            new_teams.append(child)
        
        self.teams = new_teams
        self.iteration += 1
        
        logger.info(
            f"Team evolution complete: "
            f"{num_parents} parents kept, "
            f"{self.pool_size - num_parents} new teams created"
        )
    
    def get_best_team(self) -> Optional[Team]:
        """Get the best-performing team."""
        if not self.teams:
            return None
        return max(self.teams, key=lambda t: (t.win_rate, t.elo))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        if not self.teams:
            return {"pool_size": 0}
        
        return {
            "pool_size": len(self.teams),
            "avg_elo": sum(t.elo for t in self.teams) / len(self.teams),
            "max_elo": max(t.elo for t in self.teams),
            "avg_win_rate": sum(t.win_rate for t in self.teams) / len(self.teams),
            "total_games": sum(t.games_played for t in self.teams),
        }
    
    def save(self, path: Optional[Path] = None):
        """Save team pool to disk."""
        save_path = path or (self.save_dir / "team_pool.json" if self.save_dir else None)
        if save_path is None:
            return
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "teams": [t.to_dict() for t in self.teams],
            "iteration": self.iteration,
            "stats": self.get_stats(),
        }
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved team pool to {save_path}")


@dataclass
class HallOfFameEntry:
    """Entry in the Hall of Fame."""
    agent_id: str
    model_path: str
    elo: float
    peak_elo: float
    inducted_at: str
    achievements: List[str] = field(default_factory=list)


class HallOfFame:
    """Hall of Fame for best historical agents.
    
    Preserves the best agents from training history to:
    - Maintain diversity in opponent selection
    - Provide benchmarks for evaluation
    - Prevent catastrophic forgetting
    """
    
    def __init__(
        self,
        max_size: int = 10,
        save_dir: Optional[Path] = None,
    ):
        """Initialize Hall of Fame.
        
        Args:
            max_size: Maximum entries to keep
            save_dir: Directory to save HoF data
        """
        self.max_size = max_size
        self.save_dir = save_dir
        self.entries: List[HallOfFameEntry] = []
        
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)
            self._load()
    
    def consider(self, agent: AgentRecord) -> bool:
        """Consider an agent for Hall of Fame.
        
        Args:
            agent: Agent to consider
            
        Returns:
            True if agent was inducted
        """
        # Check if already in HoF
        for entry in self.entries:
            if entry.agent_id == agent.agent_id:
                # Update peak ELO if improved
                if agent.elo > entry.peak_elo:
                    entry.peak_elo = agent.elo
                return False
        
        # Check if qualifies
        min_elo = min(e.elo for e in self.entries) if self.entries else 0
        
        if len(self.entries) < self.max_size or agent.elo > min_elo:
            # Create entry
            entry = HallOfFameEntry(
                agent_id=agent.agent_id,
                model_path=agent.model_path,
                elo=agent.elo,
                peak_elo=agent.elo,
                inducted_at=datetime.utcnow().isoformat(),
                achievements=self._get_achievements(agent),
            )
            
            self.entries.append(entry)
            
            # Remove lowest if over capacity
            if len(self.entries) > self.max_size:
                self.entries.sort(key=lambda e: e.elo, reverse=True)
                self.entries = self.entries[:self.max_size]
            
            self._save()
            logger.info(f"Agent {agent.agent_id} inducted into Hall of Fame (ELO: {agent.elo:.0f})")
            return True
        
        return False
    
    def _get_achievements(self, agent: AgentRecord) -> List[str]:
        """Get achievements for an agent."""
        achievements = []
        
        if agent.win_rate >= 0.9:
            achievements.append("Dominant (90%+ win rate)")
        elif agent.win_rate >= 0.75:
            achievements.append("Strong (75%+ win rate)")
        
        if agent.elo >= 1800:
            achievements.append("Master (1800+ ELO)")
        elif agent.elo >= 1600:
            achievements.append("Expert (1600+ ELO)")
        
        if agent.games_played >= 1000:
            achievements.append("Veteran (1000+ games)")
        
        return achievements
    
    def get_random_entry(self) -> Optional[HallOfFameEntry]:
        """Get a random Hall of Fame entry."""
        if not self.entries:
            return None
        return random.choice(self.entries)
    
    def get_strongest_entries(self, n: int = 3) -> List[HallOfFameEntry]:
        """Get strongest entries."""
        sorted_entries = sorted(self.entries, key=lambda e: e.elo, reverse=True)
        return sorted_entries[:n]
    
    def _save(self):
        """Save to file."""
        if self.save_dir is None:
            return
        
        path = self.save_dir / "hall_of_fame.json"
        data = [
            {
                "agent_id": e.agent_id,
                "model_path": e.model_path,
                "elo": e.elo,
                "peak_elo": e.peak_elo,
                "inducted_at": e.inducted_at,
                "achievements": e.achievements,
            }
            for e in self.entries
        ]
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load(self):
        """Load from file."""
        if self.save_dir is None:
            return
        
        path = self.save_dir / "hall_of_fame.json"
        if not path.exists():
            return
        
        with open(path) as f:
            data = json.load(f)
        
        self.entries = [
            HallOfFameEntry(**d) for d in data
        ]


class Division:
    """A division in the league system."""
    
    def __init__(
        self,
        name: str,
        min_elo: float,
        max_elo: float,
    ):
        """Initialize division.
        
        Args:
            name: Division name
            min_elo: Minimum ELO for this division
            max_elo: Maximum ELO for this division
        """
        self.name = name
        self.min_elo = min_elo
        self.max_elo = max_elo
        self.agents: List[str] = []  # Agent IDs in this division
    
    def contains_elo(self, elo: float) -> bool:
        """Check if ELO belongs to this division."""
        return self.min_elo <= elo < self.max_elo


class League:
    """League system with divisions for structured training.
    
    Agents are organized into divisions based on ELO.
    Matches are primarily within division with occasional
    cross-division matches.
    """
    
    DEFAULT_DIVISIONS = [
        Division("Bronze", 0, 1400),
        Division("Silver", 1400, 1600),
        Division("Gold", 1600, 1800),
        Division("Platinum", 1800, 2000),
        Division("Diamond", 2000, float("inf")),
    ]
    
    def __init__(
        self,
        divisions: Optional[List[Division]] = None,
        cross_division_rate: float = 0.1,
    ):
        """Initialize league.
        
        Args:
            divisions: List of divisions (default: Bronze-Diamond)
            cross_division_rate: Rate of cross-division matches
        """
        self.divisions = divisions or self.DEFAULT_DIVISIONS.copy()
        self.cross_division_rate = cross_division_rate
    
    def get_division(self, elo: float) -> Division:
        """Get division for an ELO rating."""
        for division in self.divisions:
            if division.contains_elo(elo):
                return division
        return self.divisions[-1]  # Highest division
    
    def update_divisions(self, agents: List[AgentRecord]):
        """Update division assignments for all agents."""
        # Clear current assignments
        for division in self.divisions:
            division.agents.clear()
        
        # Assign agents to divisions
        for agent in agents:
            division = self.get_division(agent.elo)
            division.agents.append(agent.agent_id)
    
    def get_opponent(
        self,
        agent: AgentRecord,
        population: List[AgentRecord],
    ) -> Optional[AgentRecord]:
        """Get opponent for an agent based on league rules.
        
        Args:
            agent: Agent seeking opponent
            population: All available agents
            
        Returns:
            Selected opponent or None
        """
        if not population:
            return None
        
        # Decide if cross-division match
        if random.random() < self.cross_division_rate:
            # Random opponent from any division
            candidates = [a for a in population if a.agent_id != agent.agent_id]
        else:
            # Same division opponent
            division = self.get_division(agent.elo)
            candidates = [
                a for a in population
                if a.agent_id != agent.agent_id
                and a.agent_id in division.agents
            ]
        
        if not candidates:
            # Fall back to any opponent
            candidates = [a for a in population if a.agent_id != agent.agent_id]
        
        if not candidates:
            return None
        
        return random.choice(candidates)
    
    def get_standings(self) -> Dict[str, List[Tuple[str, float]]]:
        """Get standings for each division.
        
        Returns:
            Dict mapping division name to list of (agent_id, elo) tuples
        """
        standings = {}
        for division in self.divisions:
            standings[division.name] = []
            # Would need full population to get ELOs
        return standings


class DiversityTracker:
    """Track and maintain diversity in the population.
    
    Prevents all agents from converging to the same strategy
    by rewarding behavioral diversity.
    """
    
    def __init__(
        self,
        action_dim: int = 144,
        history_len: int = 1000,
    ):
        """Initialize diversity tracker.
        
        Args:
            action_dim: Number of possible actions
            history_len: Actions to track per agent
        """
        self.action_dim = action_dim
        self.history_len = history_len
        
        # Action distributions per agent
        self.distributions: Dict[str, np.ndarray] = {}
    
    def record_action(self, agent_id: str, action: int):
        """Record an action for an agent."""
        if agent_id not in self.distributions:
            self.distributions[agent_id] = np.zeros(self.action_dim)
        
        self.distributions[agent_id][action] += 1
    
    def get_diversity_score(self, agent_id: str) -> float:
        """Get diversity score for an agent.
        
        Higher score means more diverse action selection.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Diversity score (entropy of action distribution)
        """
        if agent_id not in self.distributions:
            return 0.0
        
        dist = self.distributions[agent_id]
        total = dist.sum()
        
        if total == 0:
            return 0.0
        
        probs = dist / total
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        
        # Normalize by max entropy
        max_entropy = np.log(self.action_dim)
        return entropy / max_entropy
    
    def get_population_diversity(self) -> float:
        """Get overall population diversity.
        
        Measures how different agents are from each other.
        
        Returns:
            Population diversity score
        """
        if len(self.distributions) < 2:
            return 1.0
        
        # Compute pairwise KL divergences
        agents = list(self.distributions.keys())
        total_kl = 0.0
        n_pairs = 0
        
        for i, agent_a in enumerate(agents):
            for agent_b in agents[i+1:]:
                dist_a = self.distributions[agent_a]
                dist_b = self.distributions[agent_b]
                
                # Normalize
                p = dist_a / (dist_a.sum() + 1e-10)
                q = dist_b / (dist_b.sum() + 1e-10)
                
                # Symmetric KL divergence
                kl = 0.5 * (
                    np.sum(p * np.log((p + 1e-10) / (q + 1e-10))) +
                    np.sum(q * np.log((q + 1e-10) / (p + 1e-10)))
                )
                
                total_kl += kl
                n_pairs += 1
        
        if n_pairs == 0:
            return 1.0
        
        # Higher KL means more diverse
        avg_kl = total_kl / n_pairs
        return min(1.0, avg_kl / 10.0)  # Normalize to 0-1


class EnhancedSelfPlayTrainer:
    """Enhanced self-play trainer with advanced features.
    
    Features:
    - Hall of Fame for historical best agents
    - League system with divisions
    - Diversity tracking and rewards
    - Prioritized experience sampling
    - Real Showdown battles (optional)
    """
    
    def __init__(
        self,
        population_size: int = 10,
        hall_of_fame_size: int = 5,
        save_dir: Optional[Path] = None,
        use_league: bool = True,
        diversity_bonus: float = 0.1,
        use_real_battles: bool = False,
        battle_format: str = "gen9vgc2024regg",
        battle_timeout: float = 60.0,
        max_real_battles_per_iter: int = 5,
        evolve_teams: bool = False,
        team_pool_size: int = 20,
        team_evolution_interval: int = 5,
    ):
        """Initialize enhanced trainer.
        
        Args:
            population_size: Number of agents in population
            hall_of_fame_size: Hall of Fame capacity
            save_dir: Directory for saving data
            use_league: Whether to use league system
            diversity_bonus: Bonus reward for diverse play
            use_real_battles: Whether to use actual Showdown battles
            battle_format: Pokemon Showdown battle format
            battle_timeout: Timeout in seconds for each real battle (default: 60s)
            max_real_battles_per_iter: Max real battles per iteration before falling back (default: 5)
            evolve_teams: Whether to use evolving teams during battles
            team_pool_size: Number of teams in the evolution pool
            team_evolution_interval: Evolve teams every N iterations
        """
        self.save_dir = save_dir or Path("data/models/enhanced_self_play")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Population
        self.population = AgentPopulation(
            population_size=population_size,
            save_dir=self.save_dir / "population",
        )
        
        # Hall of Fame
        self.hall_of_fame = HallOfFame(
            max_size=hall_of_fame_size,
            save_dir=self.save_dir / "hof",
        )
        
        # League
        self.league = League() if use_league else None
        
        # Diversity tracking
        self.diversity_tracker = DiversityTracker()
        self.diversity_bonus = diversity_bonus
        
        # ELO system
        self.elo_system = EloSystem()
        
        # Team evolution
        self.evolve_teams = evolve_teams
        self.team_evolution_interval = team_evolution_interval
        if evolve_teams and HAS_TEAM_BUILDER:
            self.team_pool = TeamPoolManager(
                pool_size=team_pool_size,
                evolution_interval=team_evolution_interval,
                save_dir=self.save_dir / "teams",
            )
            logger.info(f"Team evolution enabled with pool of {team_pool_size} teams")
        else:
            self.team_pool = None
            if evolve_teams:
                logger.warning("Team evolution requested but team builder not available")
        
        # Real battle configuration
        self.use_real_battles = use_real_battles
        self.battle_format = battle_format
        self.battle_timeout = battle_timeout
        self.max_real_battles_per_iter = max_real_battles_per_iter
        self._model_cache: Dict[str, Any] = {}  # Cache loaded models
        self._real_battles_this_iter = 0  # Counter for real battles in current iteration
        
        # Statistics
        self.iteration = 0
        self.total_games = 0
    
    def get_opponent(self, agent: AgentRecord) -> Optional[AgentRecord]:
        """Select opponent for an agent.
        
        Uses probability distribution:
        - 60% current population (via league if enabled)
        - 30% recent population
        - 10% Hall of Fame
        
        Args:
            agent: Agent seeking opponent
            
        Returns:
            Selected opponent
        """
        roll = random.random()
        
        if roll < 0.1 and self.hall_of_fame.entries:
            # Hall of Fame opponent
            hof_entry = self.hall_of_fame.get_random_entry()
            if hof_entry:
                # Create temporary agent record
                return AgentRecord(
                    agent_id=hof_entry.agent_id,
                    model_path=hof_entry.model_path,
                    elo=hof_entry.elo,
                )
        
        # Population opponent
        if self.league:
            return self.league.get_opponent(agent, self.population.agents)
        else:
            return self.population.get_opponent(agent)
    
    def record_match(
        self,
        agent1: AgentRecord,
        agent2: AgentRecord,
        winner_id: Optional[str],
        actions: Optional[Dict[str, List[int]]] = None,
    ):
        """Record a match result.
        
        Args:
            agent1: First agent
            agent2: Second agent
            winner_id: ID of winner (None for draw)
            actions: Actions taken by each agent
        """
        # Update ELOs
        self.population.record_match(agent1, agent2, winner_id)
        
        # Record actions for diversity
        if actions:
            for agent_id, agent_actions in actions.items():
                for action in agent_actions:
                    self.diversity_tracker.record_action(agent_id, action)
        
        # Consider for Hall of Fame
        for agent in [agent1, agent2]:
            self.hall_of_fame.consider(agent)
        
        # Update league divisions
        if self.league:
            self.league.update_divisions(self.population.agents)
        
        self.total_games += 1
    
    def get_diversity_bonus(self, agent_id: str) -> float:
        """Get diversity bonus for an agent.
        
        Args:
            agent_id: Agent ID
            
        Returns:
            Bonus reward
        """
        score = self.diversity_tracker.get_diversity_score(agent_id)
        return self.diversity_bonus * score
    
    def _load_model(self, agent: AgentRecord):
        """Load a model for an agent.
        
        Caches loaded models for efficiency during self-play.
        
        Args:
            agent: Agent to load model for
            
        Returns:
            Loaded model or None
        """
        if agent.agent_id in self._model_cache:
            return self._model_cache[agent.agent_id]
        
        model_path = Path(agent.model_path)
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None
        
        try:
            # Try loading as SB3 model
            if str(model_path).endswith('.zip'):
                try:
                    # Try MaskablePPO first
                    from sb3_contrib import MaskablePPO
                    model = MaskablePPO.load(str(model_path))
                except Exception:
                    # Fall back to regular PPO
                    try:
                        from stable_baselines3 import PPO
                        model = PPO.load(str(model_path))
                    except Exception as e2:
                        logger.error(f"Failed to load as PPO: {e2}")
                        return None
            else:
                # Load as PyTorch state dict
                model = torch.load(str(model_path), map_location='cpu')
            
            self._model_cache[agent.agent_id] = model
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_path}: {e}")
            return None
    
    def _run_match(
        self,
        agent1: AgentRecord,
        agent2: AgentRecord,
    ) -> Optional[str]:
        """Run a match between two agents.
        
        Uses real Showdown battles if enabled and under the limit, otherwise simulates.
        If team evolution is enabled, assigns teams from the pool.
        
        Args:
            agent1: First agent
            agent2: Second agent
            
        Returns:
            Winner agent_id or None for draw
        """
        # Get teams from pool if team evolution is enabled
        team1 = None
        team2 = None
        if self.team_pool is not None:
            team1 = self.team_pool.get_team_for_agent(agent1)
            team2 = self.team_pool.get_team_for_agent(agent2)
        
        # Check if we should use real battles
        use_real = (
            self.use_real_battles 
            and self._real_battles_this_iter < self.max_real_battles_per_iter
        )
        
        winner_id = None
        
        if use_real:
            # Load models
            model1 = self._load_model(agent1)
            model2 = self._load_model(agent2)
            
            if model1 is not None and model2 is not None:
                # Import the shared battle function
                from .self_play import run_showdown_battle
                
                logger.info(
                    f"Running real battle {self._real_battles_this_iter + 1}/"
                    f"{self.max_real_battles_per_iter} (timeout: {self.battle_timeout}s)"
                )
                
                result = run_showdown_battle(
                    model1=model1,
                    model2=model2,
                    agent1_id=agent1.agent_id,
                    agent2_id=agent2.agent_id,
                    battle_format=self.battle_format,
                    timeout_seconds=self.battle_timeout,
                    team1=team1,
                    team2=team2,
                )
                
                self._real_battles_this_iter += 1
                
                if result is not None:
                    logger.info(f"Real battle complete: winner = {result}")
                    winner_id = result
                else:
                    logger.warning("Real battle failed/timed out, falling back to simulation")
            else:
                logger.warning("Could not load models, using simulation")
        
        # Simulated match (fallback or default)
        if winner_id is None:
            winner_id = random.choice([agent1.agent_id, agent2.agent_id, None])
        
        # Update team stats if using team evolution
        if self.team_pool is not None:
            if team1 is not None:
                self.team_pool.update_after_battle(team1, won=(winner_id == agent1.agent_id))
            if team2 is not None:
                self.team_pool.update_after_battle(team2, won=(winner_id == agent2.agent_id))
        
        return winner_id
    
    def run_iteration(
        self,
        train_fn: Optional[Callable] = None,
        matches_per_iteration: int = 20,
    ) -> Dict[str, Any]:
        """Run one training iteration.
        
        Args:
            train_fn: Function to train agents
            matches_per_iteration: Matches to play
            
        Returns:
            Iteration statistics
        """
        logger.info(f"Starting iteration {self.iteration}")
        
        # Reset real battle counter for this iteration
        self._real_battles_this_iter = 0
        
        stats = {
            "iteration": self.iteration,
            "matches_played": 0,
            "real_battles": 0,
            "simulated_battles": 0,
            "population_diversity": self.diversity_tracker.get_population_diversity(),
        }
        
        # Play matches
        real_before = self._real_battles_this_iter
        for _ in range(matches_per_iteration):
            if not self.population.agents:
                break
            
            # Select matchup
            agent1 = random.choice(self.population.agents)
            agent2 = self.get_opponent(agent1)
            
            if agent2 is None:
                continue
            
            # Run match (real or simulated based on configuration)
            winner_id = self._run_match(agent1, agent2)
            
            self.record_match(agent1, agent2, winner_id)
            stats["matches_played"] += 1
        
        # Count real vs simulated
        stats["real_battles"] = self._real_battles_this_iter
        stats["simulated_battles"] = stats["matches_played"] - stats["real_battles"]
        
        # Get population stats
        pop_stats = self.population.get_stats()
        stats.update({
            "population_size": pop_stats["population_size"],
            "max_elo": pop_stats["max_elo"],
            "mean_elo": pop_stats["mean_elo"],
            "hof_size": len(self.hall_of_fame.entries),
        })
        
        # Team evolution (if enabled)
        if self.team_pool is not None:
            team_stats = self.team_pool.get_stats()
            stats["team_pool_size"] = team_stats["pool_size"]
            stats["team_avg_elo"] = team_stats.get("avg_elo", 0)
            stats["team_max_elo"] = team_stats.get("max_elo", 0)
            
            # Evolve teams periodically
            if self.iteration > 0 and self.iteration % self.team_evolution_interval == 0:
                self.team_pool.evolve_teams()
                stats["team_evolution"] = True
            else:
                stats["team_evolution"] = False
        
        logger.info(
            f"Iteration {self.iteration}: "
            f"Matches={stats['matches_played']}, "
            f"MaxELO={stats['max_elo']:.0f}, "
            f"Diversity={stats['population_diversity']:.2f}"
            + (f", TeamMaxELO={stats.get('team_max_elo', 0):.0f}" if self.team_pool else "")
        )
        
        self.iteration += 1
        return stats
    
    def get_best_agent(self) -> Optional[AgentRecord]:
        """Get the current best agent."""
        return self.population.get_best_agent()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        return {
            "iterations": self.iteration,
            "total_games": self.total_games,
            "population": self.population.get_stats(),
            "hall_of_fame": [e.agent_id for e in self.hall_of_fame.entries],
            "diversity": self.diversity_tracker.get_population_diversity(),
        }

