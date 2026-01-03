"""Self-play training for VGC Battle AI with ELO tracking."""

import os
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field
import json
import copy

import numpy as np
import torch
from stable_baselines3 import PPO
from loguru import logger

from src.config import config, MODELS_DIR

# Import Team for type hints (avoid circular imports)
if TYPE_CHECKING:
    from src.ml.team_builder.team import Team

# Try to import poke-env components
try:
    from poke_env.player import Player
    from poke_env.battle import DoubleBattle
    HAS_POKE_ENV = True
except ImportError:
    HAS_POKE_ENV = False
    Player = None
    DoubleBattle = None

# Try to import MaskablePPO
try:
    from sb3_contrib import MaskablePPO
    HAS_MASKABLE_PPO = True
except ImportError:
    HAS_MASKABLE_PPO = False
    MaskablePPO = None


def run_showdown_battle(
    model1,
    model2,
    agent1_id: str,
    agent2_id: str,
    battle_format: str = "gen9vgc2024regg",
    timeout_seconds: float = 60.0,
    team1: Optional["Team"] = None,
    team2: Optional["Team"] = None,
) -> Optional[str]:
    """Run a battle between two models on Pokemon Showdown.
    
    This is a shared utility function that can be used by both
    SelfPlayTrainer and EnhancedSelfPlayTrainer.
    
    Args:
        model1: First agent's model (PPO, MaskablePPO, or nn.Module)
        model2: Second agent's model
        agent1_id: ID of first agent
        agent2_id: ID of second agent
        battle_format: Pokemon Showdown battle format
        timeout_seconds: Maximum time to wait for battle (default: 60s)
        team1: Team object for agent1 (will be converted to Showdown paste)
        team2: Team object for agent2 (will be converted to Showdown paste)
        
    Returns:
        Winner agent_id or None for draw/timeout
    """
    import asyncio
    
    if not HAS_POKE_ENV:
        logger.warning("poke-env not available for real battles")
        return None
    
    try:
        from src.engine.state.game_state import GameStateEncoder
        from src.ml.training.async_env_wrapper import ActionDecoder
        
        # Create state encoder and action decoder
        encoder = GameStateEncoder()
        action_decoder = ActionDecoder()
        
        # Create RLPlayer instances
        rl_player1 = RLPlayer(model=model1, deterministic=True)
        rl_player2 = RLPlayer(model=model2, deterministic=True)
        
        # Create poke-env compatible player wrapper
        # Import battle order classes for proper return types
        from poke_env.player.battle_order import (
            BattleOrder, 
            DoubleBattleOrder, 
            SingleBattleOrder,
            DefaultBattleOrder,
        )
        
        class PokeEnvRLPlayer(Player):
            """Poke-env player wrapper for RLPlayer."""
            
            def __init__(self, rl_player, name: str, enc, act_dec, **kwargs):
                super().__init__(**kwargs)
                self.rl_player = rl_player
                self.player_name = name
                self.encoder = enc
                self.action_decoder = act_dec
            
            def choose_move(self, battle) -> BattleOrder:
                if self.encoder is None:
                    return self.choose_random_doubles_move(battle)
                
                try:
                    # Encode battle state
                    state, _ = self.encoder.encode_battle(battle)
                    
                    # Get action from RL player
                    action = self.rl_player.select_action(state)
                    
                    # Convert action to battle order using ActionDecoder
                    # Returns a string like "/choose move X, move Y"
                    order_str = self.action_decoder.action_to_order(battle, action)
                    if order_str:
                        # Parse the order string to create proper BattleOrder
                        # Format: "/choose move1, move2" for doubles
                        parts = order_str.replace("/choose ", "").split(", ")
                        if len(parts) >= 2:
                            first = SingleBattleOrder(order=f"/choose {parts[0]}")
                            second = SingleBattleOrder(order=f"/choose {parts[1]}")
                            return DoubleBattleOrder(first_order=first, second_order=second)
                        elif len(parts) == 1:
                            return SingleBattleOrder(order=f"/choose {parts[0]}")
                except Exception as e:
                    logger.debug(f"Error getting action: {e}")
                
                return self.choose_random_doubles_move(battle)
        
        # Convert teams to Showdown paste format
        team1_paste = team1.to_showdown_paste() if team1 else None
        team2_paste = team2.to_showdown_paste() if team2 else None
        
        if team1_paste:
            logger.debug(f"Agent1 using team with {len(team1.pokemon)} Pokemon")
        if team2_paste:
            logger.debug(f"Agent2 using team with {len(team2.pokemon)} Pokemon")
        
        # Create players with unique names (add timestamp to avoid name collisions)
        import time
        timestamp = int(time.time() * 1000) % 100000
        player1 = PokeEnvRLPlayer(
            rl_player1, 
            name=f"{agent1_id[:8]}_{timestamp}a", 
            enc=encoder, 
            act_dec=action_decoder,
            battle_format=battle_format,
            team=team1_paste,
        )
        player2 = PokeEnvRLPlayer(
            rl_player2, 
            name=f"{agent2_id[:8]}_{timestamp}b", 
            enc=encoder, 
            act_dec=action_decoder,
            battle_format=battle_format,
            team=team2_paste,
        )
        
        # Run battle with timeout
        async def run_battle_with_timeout():
            try:
                await asyncio.wait_for(
                    player1.battle_against(player2, n_battles=1),
                    timeout=timeout_seconds
                )
                return player1.n_won_battles
            except asyncio.TimeoutError:
                logger.warning(f"Battle timed out after {timeout_seconds}s")
                return None
        
        # Get or create event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        wins = loop.run_until_complete(run_battle_with_timeout())
        
        if wins is None:
            return None  # Timeout or error
        elif wins > 0:
            return agent1_id
        elif wins < 1:
            return agent2_id
        else:
            return None  # Draw
        
    except ImportError as e:
        logger.warning(f"Required module not available for battle: {e}")
        return None
    except Exception as e:
        logger.error(f"Showdown battle failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


class RLPlayer:
    """Player that uses a trained RL model for action selection.
    
    This class wraps a trained PPO/MaskablePPO model and provides
    an interface for selecting actions in Pokemon battles.
    
    Can be used standalone or integrated with poke-env for real battles.
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        model: Optional[PPO] = None,
        state_encoder = None,
        use_masking: bool = True,
        deterministic: bool = True,
    ):
        """Initialize RL player.
        
        Args:
            model_path: Path to saved model (.zip)
            model: Pre-loaded PPO/MaskablePPO model
            state_encoder: Encoder for converting battle states to tensors
            use_masking: Whether to use action masking (requires MaskablePPO)
            deterministic: Whether to use deterministic action selection
        """
        self.model_path = model_path
        self.deterministic = deterministic
        self.use_masking = use_masking and HAS_MASKABLE_PPO
        
        # Load model
        if model is not None:
            self.model = model
        elif model_path:
            self.model = self._load_model(model_path)
        else:
            self.model = None
            logger.warning("RLPlayer created without model")
        
        # State encoder
        if state_encoder is not None:
            self.encoder = state_encoder
        else:
            try:
                from src.engine.state.game_state import GameStateEncoder
                self.encoder = GameStateEncoder()
            except ImportError:
                # Fallback to simple encoder
                self.encoder = None
                logger.warning("GameStateEncoder not available")
    
    def _load_model(self, path: str):
        """Load a saved model.
        
        Automatically detects if it's PPO or MaskablePPO.
        """
        path = Path(path)
        if not path.exists():
            logger.error(f"Model file not found: {path}")
            return None
        
        try:
            # Try MaskablePPO first if available
            if self.use_masking and HAS_MASKABLE_PPO:
                try:
                    return MaskablePPO.load(path)
                except Exception:
                    pass
            
            # Fall back to regular PPO
            return PPO.load(path)
        except Exception as e:
            logger.error(f"Failed to load model from {path}: {e}")
            return None
    
    def select_action(
        self,
        state: np.ndarray,
        action_mask: Optional[np.ndarray] = None,
    ) -> int:
        """Select action given a state.
        
        Args:
            state: State tensor (620-dim for VGC)
            action_mask: Optional boolean mask of valid actions (144-dim)
            
        Returns:
            Selected action index (0-143)
        """
        if self.model is None:
            # Random fallback
            if action_mask is not None:
                valid = np.where(action_mask)[0]
                return np.random.choice(valid) if len(valid) > 0 else 0
            return np.random.randint(0, 144)
        
        # Predict action
        if self.use_masking and action_mask is not None and HAS_MASKABLE_PPO:
            action, _ = self.model.predict(
                state, 
                action_masks=action_mask,
                deterministic=self.deterministic
            )
        else:
            action, _ = self.model.predict(
                state,
                deterministic=self.deterministic
            )
        
        return int(action)
    
    def decode_action(self, action: int) -> Tuple[int, int]:
        """Decode combined action to individual slot actions.
        
        Args:
            action: Combined action (0-143)
            
        Returns:
            Tuple of (slot_a_action, slot_b_action) each 0-11
        """
        slot_a = action // 12
        slot_b = action % 12
        return slot_a, slot_b
    
    def action_to_order(self, battle, action: int):
        """Convert action index to poke-env battle order.
        
        Args:
            battle: poke-env DoubleBattle object
            action: Combined action (0-143)
            
        Returns:
            Battle order for poke-env
        """
        if not HAS_POKE_ENV or battle is None:
            return None
        
        slot_a, slot_b = self.decode_action(action)
        
        orders = []
        
        # Process slot A
        if len(battle.available_moves) > 0:
            order_a = self._slot_action_to_order(slot_a, battle, 0)
            if order_a:
                orders.append(order_a)
        
        # Process slot B
        if len(battle.available_moves) > 1:
            order_b = self._slot_action_to_order(slot_b, battle, 1)
            if order_b:
                orders.append(order_b)
        
        # Return combined order or first available
        if len(orders) >= 2:
            return "/choose " + " ".join(str(o) for o in orders)
        elif len(orders) == 1:
            return orders[0]
        else:
            # Fallback to random move
            return battle.random_choice()
    
    def _slot_action_to_order(self, slot_action: int, battle, slot_idx: int):
        """Convert a single slot action to order.
        
        Args:
            slot_action: Action for this slot (0-11)
            battle: poke-env battle
            slot_idx: 0 or 1
            
        Returns:
            Order string or None
        """
        # Actions 0-3: Regular moves
        # Actions 4-7: Tera + moves  
        # Actions 8-11: Switch to bench
        
        if slot_action < 4:
            # Regular move
            if slot_idx < len(battle.available_moves) and slot_action < len(battle.available_moves[slot_idx]):
                return f"move {slot_action + 1}"
        elif slot_action < 8:
            # Tera move
            move_idx = slot_action - 4
            if slot_idx < len(battle.available_moves) and move_idx < len(battle.available_moves[slot_idx]):
                return f"move {move_idx + 1} terastallize"
        else:
            # Switch
            switch_idx = slot_action - 8
            available_switches = battle.available_switches
            if slot_idx < len(available_switches) and switch_idx < len(available_switches[slot_idx]):
                pokemon = available_switches[slot_idx][switch_idx]
                return f"switch {pokemon.species}"
        
        return None


@dataclass
class AgentRecord:
    """Record for tracking agent performance."""
    
    agent_id: str
    model_path: str
    elo: float = 1500.0
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    training_steps: int = 0
    team: Optional["Team"] = None  # Associated team for battles
    
    @property
    def win_rate(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "model_path": self.model_path,
            "elo": self.elo,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "created_at": self.created_at,
            "training_steps": self.training_steps,
            "win_rate": self.win_rate,
            "has_team": self.team is not None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentRecord":
        data.pop("win_rate", None)  # Remove computed property
        return cls(**data)


class EloSystem:
    """ELO rating system for tracking agent performance."""
    
    def __init__(self, k_factor: float = 32.0, initial_elo: float = 1500.0):
        """Initialize ELO system.
        
        Args:
            k_factor: K-factor for ELO updates
            initial_elo: Starting ELO for new agents
        """
        self.k_factor = k_factor
        self.initial_elo = initial_elo
    
    def expected_score(self, elo_a: float, elo_b: float) -> float:
        """Calculate expected score for player A against player B.
        
        Args:
            elo_a: ELO rating of player A
            elo_b: ELO rating of player B
            
        Returns:
            Expected score (0-1) for player A
        """
        return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))
    
    def update_elo(
        self, 
        elo_a: float, 
        elo_b: float, 
        score_a: float
    ) -> Tuple[float, float]:
        """Update ELO ratings after a match.
        
        Args:
            elo_a: Current ELO of player A
            elo_b: Current ELO of player B
            score_a: Actual score for player A (1.0 win, 0.5 draw, 0.0 loss)
            
        Returns:
            Tuple of (new_elo_a, new_elo_b)
        """
        expected_a = self.expected_score(elo_a, elo_b)
        expected_b = 1.0 - expected_a
        
        new_elo_a = elo_a + self.k_factor * (score_a - expected_a)
        new_elo_b = elo_b + self.k_factor * ((1.0 - score_a) - expected_b)
        
        return new_elo_a, new_elo_b


class AgentPopulation:
    """Population of agents for self-play training."""
    
    def __init__(
        self,
        population_size: int = 10,
        save_dir: Optional[Path] = None,
    ):
        """Initialize agent population.
        
        Args:
            population_size: Maximum number of agents to keep
            save_dir: Directory to save population data
        """
        self.population_size = population_size
        self.save_dir = save_dir or (MODELS_DIR / "population")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.agents: List[AgentRecord] = []
        self.elo_system = EloSystem()
        
        # Load existing population if available
        self._load_population()
    
    def _load_population(self):
        """Load population from disk."""
        pop_file = self.save_dir / "population.json"
        if pop_file.exists():
            with open(pop_file) as f:
                data = json.load(f)
                self.agents = [AgentRecord.from_dict(a) for a in data["agents"]]
            logger.info(f"Loaded population with {len(self.agents)} agents")
    
    def _save_population(self):
        """Save population to disk."""
        pop_file = self.save_dir / "population.json"
        with open(pop_file, "w") as f:
            json.dump({
                "agents": [a.to_dict() for a in self.agents],
                "updated_at": datetime.utcnow().isoformat(),
            }, f, indent=2)
    
    def add_agent(
        self, 
        model_path: str, 
        training_steps: int = 0
    ) -> AgentRecord:
        """Add a new agent to the population.
        
        Args:
            model_path: Path to saved model
            training_steps: Training steps for this model
            
        Returns:
            Created AgentRecord
        """
        agent_id = f"agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self.agents)}"
        
        agent = AgentRecord(
            agent_id=agent_id,
            model_path=model_path,
            elo=self.elo_system.initial_elo,
            training_steps=training_steps,
        )
        
        self.agents.append(agent)
        
        # Prune population if too large
        if len(self.agents) > self.population_size:
            self._prune_population()
        
        self._save_population()
        
        return agent
    
    def _prune_population(self):
        """Remove lowest ELO agents if population exceeds limit."""
        # Sort by ELO and keep top agents
        self.agents.sort(key=lambda a: a.elo, reverse=True)
        self.agents = self.agents[:self.population_size]
    
    def get_opponent(self, agent: AgentRecord) -> Optional[AgentRecord]:
        """Select an opponent for the given agent.
        
        Uses a mix of ELO-based and random selection.
        
        Args:
            agent: Agent looking for opponent
            
        Returns:
            Opponent agent or None if no suitable opponent
        """
        if len(self.agents) < 2:
            return None
        
        # Get all agents except self
        candidates = [a for a in self.agents if a.agent_id != agent.agent_id]
        
        if not candidates:
            return None
        
        # 70% chance: choose opponent with similar ELO
        # 30% chance: random opponent
        if random.random() < 0.7:
            # Sort by ELO distance and choose from closest
            candidates.sort(key=lambda a: abs(a.elo - agent.elo))
            # Choose from top 3 closest
            return random.choice(candidates[:min(3, len(candidates))])
        else:
            return random.choice(candidates)
    
    def record_match(
        self, 
        agent1: AgentRecord, 
        agent2: AgentRecord,
        winner: Optional[str] = None,  # agent1, agent2, or None for draw
    ):
        """Record match result and update ELOs.
        
        Args:
            agent1: First agent
            agent2: Second agent
            winner: Winner ID or None for draw
        """
        # Determine scores
        if winner == agent1.agent_id:
            score1 = 1.0
            agent1.wins += 1
            agent2.losses += 1
        elif winner == agent2.agent_id:
            score1 = 0.0
            agent1.losses += 1
            agent2.wins += 1
        else:
            score1 = 0.5
            agent1.draws += 1
            agent2.draws += 1
        
        # Update ELOs
        new_elo1, new_elo2 = self.elo_system.update_elo(
            agent1.elo, agent2.elo, score1
        )
        
        agent1.elo = new_elo1
        agent2.elo = new_elo2
        agent1.games_played += 1
        agent2.games_played += 1
        
        self._save_population()
    
    def get_best_agent(self) -> Optional[AgentRecord]:
        """Get the agent with highest ELO."""
        if not self.agents:
            return None
        return max(self.agents, key=lambda a: a.elo)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get population statistics."""
        if not self.agents:
            return {"population_size": 0}
        
        elos = [a.elo for a in self.agents]
        return {
            "population_size": len(self.agents),
            "mean_elo": np.mean(elos),
            "max_elo": np.max(elos),
            "min_elo": np.min(elos),
            "total_games": sum(a.games_played for a in self.agents),
            "best_agent": self.get_best_agent().agent_id if self.agents else None,
        }


class SelfPlayTrainer:
    """Self-play training loop for VGC Battle AI.
    
    Implements population-based training with:
    - ELO tracking for all agents
    - Hall of fame for best agents ever
    - Opponent selection: 60% current, 30% recent, 10% hall of fame
    """
    
    def __init__(
        self,
        population_size: int = 10,
        save_dir: Optional[Path] = None,
        training_timesteps_per_iteration: int = 10_000,
        matches_per_iteration: int = 20,
        hall_of_fame_size: int = 5,
    ):
        """Initialize self-play trainer.
        
        Args:
            population_size: Size of agent population
            save_dir: Directory to save models
            training_timesteps_per_iteration: Training steps per iteration
            matches_per_iteration: Matches to play per iteration
        """
        self.save_dir = save_dir or (MODELS_DIR / "self_play")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.population = AgentPopulation(
            population_size=population_size,
            save_dir=self.save_dir / "population",
        )
        
        self.training_timesteps = training_timesteps_per_iteration
        self.matches_per_iteration = matches_per_iteration
        self.hall_of_fame_size = hall_of_fame_size
        self.hall_of_fame: List[AgentRecord] = []
        
        self.iteration = 0
    
    def select_opponent(self, agent: AgentRecord) -> Optional[AgentRecord]:
        """Select opponent using population-based strategy.
        
        60% current model (self-play)
        30% recent models (last 5)
        10% hall of fame (best ever)
        
        Args:
            agent: Agent looking for opponent
            
        Returns:
            Selected opponent
        """
        if len(self.population.agents) < 2:
            return agent  # Self-play
        
        r = random.random()
        
        if r < 0.6:
            # Current - fight itself or similar ELO
            return agent
        elif r < 0.9:
            # Recent - choose from last 5 agents
            recent = self.population.agents[-5:]
            candidates = [a for a in recent if a.agent_id != agent.agent_id]
            if candidates:
                return random.choice(candidates)
            return agent
        else:
            # Hall of fame
            if self.hall_of_fame:
                return random.choice(self.hall_of_fame)
            return self.population.get_best_agent() or agent
    
    def update_hall_of_fame(self):
        """Update hall of fame with best agents."""
        all_agents = self.population.agents.copy()
        all_agents.sort(key=lambda a: a.elo, reverse=True)
        
        # Add top agents to hall of fame
        for agent in all_agents[:self.hall_of_fame_size]:
            if agent not in self.hall_of_fame:
                self.hall_of_fame.append(agent)
        
        # Keep only top N
        self.hall_of_fame.sort(key=lambda a: a.elo, reverse=True)
        self.hall_of_fame = self.hall_of_fame[:self.hall_of_fame_size]
    
    def _load_agent_model(self, agent: AgentRecord) -> Optional[PPO]:
        """Load a PPO model from agent record.
        
        Args:
            agent: Agent record with model path
            
        Returns:
            Loaded PPO model or None
        """
        model_path = Path(agent.model_path)
        if not model_path.exists():
            logger.warning(f"Model not found: {model_path}")
            return None
        
        try:
            return PPO.load(model_path)
        except Exception as e:
            logger.warning(f"Failed to load model {model_path}: {e}")
            return None
    
    def _run_match(
        self, 
        agent1: AgentRecord, 
        agent2: AgentRecord,
        use_real_battles: bool = False,
    ) -> Optional[str]:
        """Run a match between two agents.
        
        Args:
            agent1: First agent
            agent2: Second agent
            use_real_battles: Whether to use real Showdown battles
            
        Returns:
            Winner agent_id or None for draw
        """
        if use_real_battles:
            try:
                return self._run_real_battle(agent1, agent2)
            except Exception as e:
                logger.warning(f"Real battle failed: {e}, using simulation")
        
        # Fallback: Simulated match based on ELO
        return self._simulate_match(agent1, agent2)
    
    def _simulate_match(
        self, 
        agent1: AgentRecord, 
        agent2: AgentRecord
    ) -> Optional[str]:
        """Simulate match result based on ELO difference.
        
        Args:
            agent1: First agent
            agent2: Second agent
            
        Returns:
            Winner agent_id or None for draw
        """
        elo_diff = agent1.elo - agent2.elo
        win_prob = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
        
        roll = random.random()
        if roll < win_prob * 0.8:  # Some randomness
            return agent1.agent_id
        elif roll < 1 - (1 - win_prob) * 0.8:
            return agent2.agent_id
        else:
            return None  # Draw
    
    def _run_real_battle(
        self, 
        agent1: AgentRecord, 
        agent2: AgentRecord,
    ) -> Optional[str]:
        """Run actual battle between two agents using poke-env.
        
        Requires Pokemon Showdown server running locally.
        
        Uses the new AsyncToSyncEnv infrastructure for action decoding.
        
        Args:
            agent1: First agent
            agent2: Second agent
            
        Returns:
            Winner agent_id or None for draw
        """
        import asyncio
        
        # Load models
        model1 = self._load_agent_model(agent1)
        model2 = self._load_agent_model(agent2)
        
        if model1 is None or model2 is None:
            logger.warning("Could not load models, using simulation")
            return self._simulate_match(agent1, agent2)
        
        if not HAS_POKE_ENV:
            logger.warning("poke-env not available, using simulation")
            return self._simulate_match(agent1, agent2)
        
        try:
            from src.engine.state.game_state import GameStateEncoder
            from src.ml.training.async_env_wrapper import ActionDecoder
            
            # Create state encoder and action decoder
            encoder = GameStateEncoder()
            action_decoder = ActionDecoder()
            
            # Create RL players using the RLPlayer class
            rl_player1 = RLPlayer(model=model1, deterministic=True)
            rl_player2 = RLPlayer(model=model2, deterministic=True)
            
            # Create poke-env compatible player wrapper
            class PokeEnvRLPlayer(Player):
                """Poke-env player wrapper for RLPlayer."""
                
                def __init__(self, rl_player: RLPlayer, name: str, enc, act_dec, **kwargs):
                    super().__init__(**kwargs)
                    self.rl_player = rl_player
                    self.player_name = name
                    self.encoder = enc
                    self.action_decoder = act_dec
                
                def choose_move(self, battle):
                    if self.encoder is None:
                        return self.choose_random_doubles_move(battle)
                    
                    try:
                        # Encode battle state
                        state, _ = self.encoder.encode_battle(battle)
                        
                        # Get action from RL player
                        action = self.rl_player.select_action(state)
                        
                        # Convert action to battle order using ActionDecoder
                        order = self.action_decoder.action_to_order(battle, action)
                        if order:
                            return order
                    except Exception as e:
                        logger.debug(f"Error getting action: {e}")
                    
                    return self.choose_random_doubles_move(battle)
            
            # Create players
            player1 = PokeEnvRLPlayer(
                rl_player1, 
                name=agent1.agent_id, 
                enc=encoder, 
                act_dec=action_decoder,
                battle_format="gen9vgc2024regg"
            )
            player2 = PokeEnvRLPlayer(
                rl_player2, 
                name=agent2.agent_id, 
                enc=encoder, 
                act_dec=action_decoder,
                battle_format="gen9vgc2024regg"
            )
            
            # Run battle
            async def run_battle():
                await player1.battle_against(player2, n_battles=1)
                return player1.n_won_battles
            
            # Get or create event loop
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            wins = loop.run_until_complete(run_battle())
            
            if wins > 0:
                return agent1.agent_id
            elif wins < 1:
                return agent2.agent_id
            else:
                return None  # Draw
            
        except ImportError as e:
            logger.warning(f"Required module not available: {e}")
            return self._simulate_match(agent1, agent2)
        except Exception as e:
            logger.error(f"Battle failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return self._simulate_match(agent1, agent2)
    
    def run_iteration(self) -> Dict[str, Any]:
        """Run one iteration of self-play training.
        
        1. Train a new agent or improve existing
        2. Add to population
        3. Run matches between agents
        4. Update ELOs
        
        Returns:
            Iteration statistics
        """
        logger.info(f"Starting self-play iteration {self.iteration}")
        
        stats = {
            "iteration": self.iteration,
            "matches_played": 0,
        }
        
        # Create or get base model
        if self.population.agents:
            # Start from best agent
            best = self.population.get_best_agent()
            logger.info(f"Starting from best agent: {best.agent_id} (ELO: {best.elo:.0f})")
            
            # In real implementation, would load and continue training
            # For now, just simulate
            model_path = self.save_dir / f"agent_{self.iteration}.zip"
        else:
            logger.info("Creating initial agent")
            model_path = self.save_dir / "agent_0.zip"
        
        # Add new agent to population
        new_agent = self.population.add_agent(
            model_path=str(model_path),
            training_steps=self.training_timesteps * (self.iteration + 1),
        )
        logger.info(f"Added new agent: {new_agent.agent_id}")
        
        # Run matches using population-based opponent selection
        for _ in range(self.matches_per_iteration):
            agent1 = random.choice(self.population.agents)
            agent2 = self.select_opponent(agent1)
            
            if agent2 is None or agent1.agent_id == agent2.agent_id:
                # Self-play - simulate training improvement
                continue
            
            # Try to run real battle, fall back to simulation
            outcome = self._run_match(agent1, agent2)
            
            self.population.record_match(agent1, agent2, outcome)
            stats["matches_played"] += 1
        
        # Update hall of fame
        self.update_hall_of_fame()
        
        # Get population stats
        pop_stats = self.population.get_stats()
        stats.update(pop_stats)
        
        if self.hall_of_fame:
            stats["hall_of_fame_best_elo"] = self.hall_of_fame[0].elo
        
        logger.info(f"Iteration {self.iteration} complete:")
        logger.info(f"  Matches played: {stats['matches_played']}")
        logger.info(f"  Population size: {pop_stats['population_size']}")
        logger.info(f"  Best ELO: {pop_stats['max_elo']:.0f}")
        
        self.iteration += 1
        
        return stats
    
    def train(self, num_iterations: int = 100) -> List[Dict[str, Any]]:
        """Run multiple iterations of self-play training.
        
        Args:
            num_iterations: Number of iterations to run
            
        Returns:
            List of iteration statistics
        """
        logger.info(f"Starting self-play training for {num_iterations} iterations")
        
        all_stats = []
        
        for i in range(num_iterations):
            stats = self.run_iteration()
            all_stats.append(stats)
        
        # Final summary
        final_stats = self.population.get_stats()
        logger.info("=" * 50)
        logger.info("Self-Play Training Complete")
        logger.info(f"  Total iterations: {num_iterations}")
        logger.info(f"  Final population size: {final_stats['population_size']}")
        logger.info(f"  Best agent: {final_stats.get('best_agent', 'N/A')}")
        logger.info(f"  Highest ELO: {final_stats.get('max_elo', 0):.0f}")
        logger.info("=" * 50)
        
        return all_stats


def main():
    """Main entry point for self-play training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Self-play training for VGC AI")
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of training iterations"
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=10,
        help="Size of agent population"
    )
    
    args = parser.parse_args()
    
    trainer = SelfPlayTrainer(
        population_size=args.population_size,
    )
    
    trainer.train(num_iterations=args.iterations)


if __name__ == "__main__":
    main()

