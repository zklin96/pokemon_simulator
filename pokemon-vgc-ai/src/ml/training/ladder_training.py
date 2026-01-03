"""Live Showdown Ladder Training.

Train RL agents by playing on the live Pokemon Showdown ladder.
"""

import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime
import json
import numpy as np
from loguru import logger

try:
    from poke_env.player import Player
    from poke_env.environment import DoubleBattle
    POKE_ENV_AVAILABLE = True
except ImportError:
    POKE_ENV_AVAILABLE = False
    logger.warning("poke-env not available for ladder training")


@dataclass
class LadderSession:
    """Tracks a ladder training session."""
    session_id: str
    start_time: str
    format_id: str
    games_played: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    elo: int = 1000
    peak_elo: int = 1000
    transitions: List[Dict] = field(default_factory=list)
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate."""
        if self.games_played == 0:
            return 0.0
        return self.wins / self.games_played
    
    def record_game(self, won: bool, elo_change: int = 0) -> None:
        """Record a game result."""
        self.games_played += 1
        if won:
            self.wins += 1
        else:
            self.losses += 1
        
        self.elo += elo_change
        self.peak_elo = max(self.peak_elo, self.elo)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "start_time": self.start_time,
            "format_id": self.format_id,
            "games_played": self.games_played,
            "wins": self.wins,
            "losses": self.losses,
            "draws": self.draws,
            "elo": self.elo,
            "peak_elo": self.peak_elo,
            "win_rate": self.win_rate,
        }


@dataclass
class LadderConfig:
    """Configuration for ladder training."""
    username: str
    password: str = ""
    server_url: str = "sim.smogon.com:8000"
    format_id: str = "gen9vgc2024regg"
    max_games: int = 100
    min_games_per_checkpoint: int = 10
    learning_rate: float = 3e-4
    batch_size: int = 32
    output_dir: Path = Path("data/models/ladder")
    use_team_file: Optional[Path] = None
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)


class LadderTrainer:
    """Trains an RL agent on the live Showdown ladder.
    
    This provides real competitive experience against human players
    at various skill levels.
    """
    
    def __init__(
        self,
        config: LadderConfig,
        model_path: Optional[Path] = None,
    ):
        if not POKE_ENV_AVAILABLE:
            raise ImportError("poke-env is required for ladder training")
        
        self.config = config
        self.model_path = model_path
        self.model = None
        
        # Load model if provided
        if model_path and model_path.exists():
            self._load_model(model_path)
        
        self.session: Optional[LadderSession] = None
        self._replay_buffer: List[Dict] = []
    
    def _load_model(self, path: Path) -> None:
        """Load a trained model."""
        try:
            from stable_baselines3 import PPO
            from sb3_contrib import MaskablePPO
            
            # Try MaskablePPO first
            try:
                self.model = MaskablePPO.load(path)
                logger.info(f"Loaded MaskablePPO from {path}")
            except Exception:
                self.model = PPO.load(path)
                logger.info(f"Loaded PPO from {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model = None
    
    def _save_checkpoint(self, games_played: int) -> Path:
        """Save a model checkpoint."""
        if self.model is None:
            logger.warning("No model to save")
            return None
        
        checkpoint_path = self.config.output_dir / f"ladder_checkpoint_{games_played}.zip"
        self.model.save(checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    
    def _update_model(self) -> None:
        """Update model from collected experience."""
        if not self._replay_buffer:
            logger.warning("No experience to learn from")
            return
        
        # Convert replay buffer to training format
        # This is a simplified version - real implementation would
        # properly format and batch the data
        logger.info(f"Updating model with {len(self._replay_buffer)} transitions")
        
        # Clear buffer after update
        self._replay_buffer = []
    
    def _get_team(self) -> str:
        """Get team for ladder matches."""
        if self.config.use_team_file and self.config.use_team_file.exists():
            with open(self.config.use_team_file) as f:
                return f.read()
        
        # Generate random team
        from src.ml.team_builder.team import Team
        team = Team.random()
        return team.to_showdown_paste()
    
    async def _run_ladder_game(self) -> Tuple[bool, List[Dict]]:
        """Run a single ladder game.
        
        Returns:
            Tuple of (won, transitions)
        """
        # This is a placeholder implementation
        # Real implementation would:
        # 1. Create a LadderPlayer that uses self.model
        # 2. Connect to Showdown
        # 3. Accept ladder challenges
        # 4. Play the game
        # 5. Collect transitions
        
        logger.info("Running ladder game (simulated)")
        
        # Simulate game duration
        await asyncio.sleep(1)
        
        # Simulated result
        won = np.random.random() > 0.5
        transitions = []  # Would be filled with actual game data
        
        return won, transitions
    
    async def run_ladder_session(
        self,
        max_games: Optional[int] = None,
        callback: Optional[callable] = None,
    ) -> LadderSession:
        """Run a ladder training session.
        
        Args:
            max_games: Maximum games to play
            callback: Optional callback after each game
            
        Returns:
            LadderSession with results
        """
        max_games = max_games or self.config.max_games
        
        self.session = LadderSession(
            session_id=f"ladder_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_time=datetime.now().isoformat(),
            format_id=self.config.format_id,
        )
        
        logger.info(f"Starting ladder session: {self.session.session_id}")
        logger.info(f"Format: {self.config.format_id}, Max games: {max_games}")
        
        games_since_checkpoint = 0
        
        for game_num in range(max_games):
            try:
                won, transitions = await self._run_ladder_game()
                
                # Estimate ELO change (simplified)
                elo_change = 20 if won else -20
                self.session.record_game(won, elo_change)
                
                # Store transitions
                self._replay_buffer.extend(transitions)
                
                games_since_checkpoint += 1
                
                logger.info(
                    f"Game {game_num + 1}/{max_games} | "
                    f"{'Won' if won else 'Lost'} | "
                    f"ELO: {self.session.elo} | "
                    f"Win rate: {self.session.win_rate:.1%}"
                )
                
                # Checkpoint and update
                if games_since_checkpoint >= self.config.min_games_per_checkpoint:
                    self._update_model()
                    self._save_checkpoint(self.session.games_played)
                    games_since_checkpoint = 0
                
                # Callback
                if callback:
                    callback(self.session)
                
            except Exception as e:
                logger.error(f"Error in game {game_num + 1}: {e}")
                continue
        
        # Final checkpoint
        if games_since_checkpoint > 0:
            self._update_model()
            self._save_checkpoint(self.session.games_played)
        
        # Save session results
        session_file = self.config.output_dir / f"{self.session.session_id}.json"
        with open(session_file, 'w') as f:
            json.dump(self.session.to_dict(), f, indent=2)
        
        logger.info(f"Session complete: {self.session.games_played} games, {self.session.win_rate:.1%} win rate")
        
        return self.session


def run_ladder_training(
    username: str,
    model_path: Optional[Path] = None,
    max_games: int = 50,
    format_id: str = "gen9vgc2024regg",
) -> LadderSession:
    """Convenience function to run ladder training.
    
    Args:
        username: Showdown username
        model_path: Path to trained model
        max_games: Maximum games to play
        format_id: Battle format
        
    Returns:
        LadderSession with results
    """
    config = LadderConfig(
        username=username,
        format_id=format_id,
        max_games=max_games,
    )
    
    trainer = LadderTrainer(config, model_path)
    return asyncio.run(trainer.run_ladder_session())

