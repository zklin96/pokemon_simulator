"""Integration tests for Pokemon Showdown connection.

These tests require a running Pokemon Showdown server on localhost:8000.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def is_showdown_running() -> bool:
    """Check if Pokemon Showdown server is accessible."""
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        return result == 0
    except Exception:
        return False


# Skip all tests if Showdown is not running
pytestmark = pytest.mark.skipif(
    not is_showdown_running(),
    reason="Pokemon Showdown server not running on localhost:8000"
)


class TestShowdownConnection:
    """Test basic connection to Pokemon Showdown."""
    
    def test_server_is_accessible(self):
        """Test that the server responds on port 8000."""
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex(('localhost', 8000))
        sock.close()
        assert result == 0, "Could not connect to Pokemon Showdown on port 8000"
    
    def test_http_response(self):
        """Test that HTTP endpoint returns valid HTML."""
        import urllib.request
        
        with urllib.request.urlopen('http://localhost:8000', timeout=5) as response:
            html = response.read().decode('utf-8')
            assert '<!DOCTYPE html>' in html
            assert 'pokemon' in html.lower() or 'showdown' in html.lower()


class TestPokeEnvConnection:
    """Test poke-env connection to Showdown."""
    
    @pytest.mark.asyncio
    async def test_player_instantiation(self):
        """Test that poke-env players can be created."""
        from poke_env.player import RandomPlayer
        
        player = RandomPlayer(
            battle_format="gen9vgc2024regg",
            max_concurrent_battles=1,
        )
        
        assert player is not None
        # poke-env uses _format internally
        assert player._format == "gen9vgc2024regg" or hasattr(player, 'format')
    
    @pytest.mark.asyncio
    async def test_custom_player_creation(self):
        """Test creating custom VGC player."""
        from src.engine.showdown.player import RandomVGCPlayer, HeuristicVGCPlayer
        
        random_player = RandomVGCPlayer(
            battle_format="gen9vgc2024regg",
            max_concurrent_battles=1,
        )
        
        heuristic_player = HeuristicVGCPlayer(
            battle_format="gen9vgc2024regg",
            max_concurrent_battles=1,
        )
        
        assert random_player is not None
        assert heuristic_player is not None


class TestShowdownBattle:
    """Test actual battles on Showdown server."""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # Battle should complete within 60 seconds
    async def test_local_battle_between_random_players(self):
        """Test that two random players can battle each other."""
        from poke_env.player import RandomPlayer
        
        # Create two players with unique names
        player1 = RandomPlayer(
            battle_format="gen9vgc2024regg",
            max_concurrent_battles=1,
        )
        
        player2 = RandomPlayer(
            battle_format="gen9vgc2024regg",
            max_concurrent_battles=1,
        )
        
        # Run a single battle
        await player1.battle_against(player2, n_battles=1)
        
        # Verify battle completed
        assert player1.n_finished_battles == 1
        assert player2.n_finished_battles == 1
        
        # One player should have won
        total_wins = player1.n_won_battles + player2.n_won_battles
        assert total_wins == 1, "Expected exactly one winner"
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(120)
    async def test_multiple_battles(self):
        """Test running multiple battles in sequence."""
        from poke_env.player import RandomPlayer
        
        player1 = RandomPlayer(
            battle_format="gen9vgc2024regg",
            max_concurrent_battles=1,
        )
        
        player2 = RandomPlayer(
            battle_format="gen9vgc2024regg",
            max_concurrent_battles=1,
        )
        
        # Run 3 battles
        await player1.battle_against(player2, n_battles=3)
        
        assert player1.n_finished_battles == 3
        assert player2.n_finished_battles == 3
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_heuristic_vs_random(self):
        """Test heuristic player against random player."""
        from poke_env.player import RandomPlayer
        from src.engine.showdown.player import HeuristicVGCPlayer
        
        heuristic = HeuristicVGCPlayer(
            battle_format="gen9vgc2024regg",
            max_concurrent_battles=1,
        )
        
        random_player = RandomPlayer(
            battle_format="gen9vgc2024regg",
            max_concurrent_battles=1,
        )
        
        # Run battles
        await heuristic.battle_against(random_player, n_battles=3)
        
        # Heuristic should generally win more often
        # But we just check that battles complete
        assert heuristic.n_finished_battles == 3


class TestBattleStateEncoding:
    """Test that battle states can be encoded during real battles."""
    
    @pytest.mark.asyncio
    @pytest.mark.timeout(60)
    async def test_state_encoding_during_battle(self):
        """Test that GameStateEncoder works during a real battle."""
        from poke_env.player import Player
        from src.engine.state.game_state import GameStateEncoder
        
        encoder = GameStateEncoder()
        states_captured = []
        
        class StateCapturingPlayer(Player):
            def choose_move(self, battle):
                try:
                    state_vec, structured = encoder.encode_battle(battle)
                    states_captured.append({
                        'shape': state_vec.shape,
                        'turn': battle.turn,
                    })
                except Exception as e:
                    print(f"Encoding error: {e}")
                
                return self.choose_random_doubles_move(battle)
        
        player1 = StateCapturingPlayer(
            battle_format="gen9vgc2024regg",
            max_concurrent_battles=1,
        )
        
        from poke_env.player import RandomPlayer
        player2 = RandomPlayer(
            battle_format="gen9vgc2024regg",
            max_concurrent_battles=1,
        )
        
        await player1.battle_against(player2, n_battles=1)
        
        # Should have captured some states
        assert len(states_captured) > 0, "No states were captured during battle"
        
        # All states should have correct shape
        for state_info in states_captured:
            assert state_info['shape'] == (620,), f"Wrong state shape: {state_info['shape']}"

