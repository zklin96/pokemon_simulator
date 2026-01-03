"""Tournament mode for evaluating AI agents.

Implements Best-of-3 VGC tournament matches between agents.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from loguru import logger
import random
import json
from pathlib import Path
from datetime import datetime


class MatchResult(Enum):
    """Result of a single game."""
    WIN = "win"
    LOSS = "loss"
    DRAW = "draw"


@dataclass
class GameRecord:
    """Record of a single game in a match."""
    game_number: int
    winner: str  # player id
    turns: int
    my_remaining: int
    opp_remaining: int
    my_tera_used: bool
    opp_tera_used: bool
    duration_seconds: float
    log: List[str] = field(default_factory=list)


@dataclass
class MatchRecord:
    """Record of a Best-of-N match."""
    match_id: str
    player_a: str
    player_b: str
    best_of: int
    games: List[GameRecord] = field(default_factory=list)
    winner: Optional[str] = None
    player_a_wins: int = 0
    player_b_wins: int = 0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def is_complete(self) -> bool:
        """Check if the match is complete."""
        wins_needed = (self.best_of // 2) + 1
        return self.player_a_wins >= wins_needed or self.player_b_wins >= wins_needed
    
    def record_game(self, game: GameRecord) -> None:
        """Record a game result."""
        self.games.append(game)
        if game.winner == self.player_a:
            self.player_a_wins += 1
        elif game.winner == self.player_b:
            self.player_b_wins += 1
        
        if self.is_complete():
            if self.player_a_wins > self.player_b_wins:
                self.winner = self.player_a
            else:
                self.winner = self.player_b
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "match_id": self.match_id,
            "player_a": self.player_a,
            "player_b": self.player_b,
            "best_of": self.best_of,
            "player_a_wins": self.player_a_wins,
            "player_b_wins": self.player_b_wins,
            "winner": self.winner,
            "timestamp": self.timestamp,
            "games": [
                {
                    "game_number": g.game_number,
                    "winner": g.winner,
                    "turns": g.turns,
                    "my_remaining": g.my_remaining,
                    "opp_remaining": g.opp_remaining,
                    "duration_seconds": g.duration_seconds,
                }
                for g in self.games
            ]
        }


@dataclass
class TournamentBracket:
    """Single elimination tournament bracket."""
    name: str
    participants: List[str]
    current_round: int = 1
    matches: List[MatchRecord] = field(default_factory=list)
    bracket: List[List[Tuple[str, str]]] = field(default_factory=list)
    champion: Optional[str] = None
    
    def __post_init__(self):
        """Initialize bracket."""
        self._generate_bracket()
    
    def _generate_bracket(self) -> None:
        """Generate initial bracket pairings."""
        # Pad to power of 2 if needed
        n = len(self.participants)
        next_pow2 = 1
        while next_pow2 < n:
            next_pow2 *= 2
        
        # Add byes
        padded = self.participants + ["BYE"] * (next_pow2 - n)
        
        # Shuffle
        random.shuffle(padded)
        
        # Create first round pairings
        first_round = []
        for i in range(0, len(padded), 2):
            first_round.append((padded[i], padded[i + 1]))
        
        self.bracket = [first_round]
    
    def get_current_matches(self) -> List[Tuple[str, str]]:
        """Get matches for current round."""
        if self.current_round > len(self.bracket):
            return []
        return self.bracket[self.current_round - 1]
    
    def advance_winner(self, match: MatchRecord) -> None:
        """Advance winner to next round."""
        self.matches.append(match)
        
        # Check if round is complete
        current_matches = self.get_current_matches()
        completed = [m for m in self.matches if any(
            (m.player_a, m.player_b) == pair or (m.player_b, m.player_a) == pair
            for pair in current_matches
        )]
        
        if len(completed) >= len(current_matches):
            # Generate next round
            winners = [m.winner for m in completed if m.winner]
            
            if len(winners) == 1:
                self.champion = winners[0]
                logger.info(f"Tournament champion: {self.champion}")
            else:
                # Pair winners for next round
                next_round = []
                for i in range(0, len(winners), 2):
                    if i + 1 < len(winners):
                        next_round.append((winners[i], winners[i + 1]))
                    else:
                        next_round.append((winners[i], "BYE"))
                
                self.bracket.append(next_round)
                self.current_round += 1
                logger.info(f"Advancing to round {self.current_round}")


class TournamentRunner:
    """Runs tournament matches between agents."""
    
    def __init__(
        self,
        format_id: str = "gen9vgc2024regg",
        best_of: int = 3,
        output_dir: Optional[Path] = None,
    ):
        self.format_id = format_id
        self.best_of = best_of
        self.output_dir = output_dir or Path("data/tournaments")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def run_game(
        self,
        player_a,  # Agent instance
        player_b,  # Agent instance
        game_number: int,
    ) -> GameRecord:
        """Run a single game between two agents.
        
        Args:
            player_a: First agent
            player_b: Second agent
            game_number: Game number in the series
            
        Returns:
            GameRecord with results
        """
        import time
        start_time = time.time()
        
        # Simulated game (replace with actual poke-env battle)
        # This is a placeholder - real implementation would use
        # run_showdown_battle or similar
        
        logger.info(f"Running game {game_number}: {player_a} vs {player_b}")
        
        # Simulate game result (placeholder)
        # In real implementation, this would run the actual battle
        await asyncio.sleep(0.1)  # Placeholder for actual battle
        
        winner = random.choice([str(player_a), str(player_b)])
        turns = random.randint(8, 25)
        
        duration = time.time() - start_time
        
        return GameRecord(
            game_number=game_number,
            winner=winner,
            turns=turns,
            my_remaining=random.randint(1, 4),
            opp_remaining=0,
            my_tera_used=random.choice([True, False]),
            opp_tera_used=random.choice([True, False]),
            duration_seconds=duration,
            log=[]
        )
    
    async def run_match(
        self,
        player_a: str,
        player_b: str,
        match_id: Optional[str] = None,
    ) -> MatchRecord:
        """Run a Best-of-N match between two agents.
        
        Args:
            player_a: First agent name
            player_b: Second agent name
            match_id: Optional match identifier
            
        Returns:
            MatchRecord with results
        """
        match_id = match_id or f"{player_a}_vs_{player_b}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        match = MatchRecord(
            match_id=match_id,
            player_a=player_a,
            player_b=player_b,
            best_of=self.best_of,
        )
        
        game_number = 1
        while not match.is_complete():
            game = await self.run_game(player_a, player_b, game_number)
            match.record_game(game)
            
            logger.info(
                f"Game {game_number}: {game.winner} wins | "
                f"Score: {match.player_a_wins}-{match.player_b_wins}"
            )
            game_number += 1
        
        logger.info(f"Match complete: {match.winner} wins {match.player_a_wins}-{match.player_b_wins}")
        
        # Save match record
        match_file = self.output_dir / f"{match_id}.json"
        with open(match_file, 'w') as f:
            json.dump(match.to_dict(), f, indent=2)
        
        return match
    
    async def run_tournament(
        self,
        participants: List[str],
        tournament_name: str = "VGC Tournament",
    ) -> TournamentBracket:
        """Run a single elimination tournament.
        
        Args:
            participants: List of agent names
            tournament_name: Name of the tournament
            
        Returns:
            TournamentBracket with results
        """
        bracket = TournamentBracket(name=tournament_name, participants=participants)
        
        logger.info(f"Starting tournament: {tournament_name}")
        logger.info(f"Participants: {participants}")
        
        while not bracket.champion:
            current_matches = bracket.get_current_matches()
            logger.info(f"Round {bracket.current_round}: {len(current_matches)} matches")
            
            for player_a, player_b in current_matches:
                if player_b == "BYE":
                    # Auto-advance on bye
                    match = MatchRecord(
                        match_id=f"{player_a}_bye",
                        player_a=player_a,
                        player_b="BYE",
                        best_of=1,
                        winner=player_a,
                        player_a_wins=1,
                    )
                else:
                    match = await self.run_match(player_a, player_b)
                
                bracket.advance_winner(match)
        
        # Save tournament results
        tournament_file = self.output_dir / f"{tournament_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        results = {
            "name": bracket.name,
            "champion": bracket.champion,
            "participants": bracket.participants,
            "rounds": len(bracket.bracket),
            "matches": [m.to_dict() for m in bracket.matches],
        }
        with open(tournament_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Tournament complete! Champion: {bracket.champion}")
        
        return bracket


def run_tournament_sync(
    participants: List[str],
    tournament_name: str = "VGC Tournament",
    best_of: int = 3,
    format_id: str = "gen9vgc2024regg",
) -> TournamentBracket:
    """Synchronous wrapper for running a tournament.
    
    Args:
        participants: List of agent names
        tournament_name: Name of the tournament
        best_of: Best-of-N for each match
        format_id: Pokemon Showdown format
        
    Returns:
        TournamentBracket with results
    """
    runner = TournamentRunner(format_id=format_id, best_of=best_of)
    return asyncio.run(runner.run_tournament(participants, tournament_name))

