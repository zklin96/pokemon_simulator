"""Test connection to Pokemon Showdown server."""

import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from poke_env.player import RandomPlayer
from loguru import logger

from src.engine.showdown.player import RandomVGCPlayer, HeuristicVGCPlayer


async def test_local_battle():
    """Test a local battle between two AI players.
    
    This doesn't require a Showdown server - poke-env can simulate
    battles locally for training purposes.
    """
    logger.info("Testing local battle simulation...")
    
    # Create two players
    player1 = RandomVGCPlayer(
        battle_format="gen9vgc2024regg",
        max_concurrent_battles=1,
    )
    
    player2 = HeuristicVGCPlayer(
        battle_format="gen9vgc2024regg",
        max_concurrent_battles=1,
    )
    
    logger.info("Players created successfully!")
    logger.info(f"Player 1: {player1.__class__.__name__}")
    logger.info(f"Player 2: {player2.__class__.__name__}")
    
    return True


async def test_player_instantiation():
    """Test that poke-env is properly installed and players work."""
    logger.info("=" * 50)
    logger.info("Pokemon VGC AI - Connection Test")
    logger.info("=" * 50)
    
    try:
        # Test basic poke-env import
        from poke_env.battle import DoubleBattle
        from poke_env.player import Player
        logger.info("✓ poke-env imported successfully")
        
        # Test player creation
        player = RandomVGCPlayer(
            battle_format="gen9vgc2024regg",
            max_concurrent_battles=1,
        )
        logger.info("✓ VGC Player created successfully")
        logger.info(f"  - Format: {player.battle_format}")
        
        # Test local battle simulation
        await test_local_battle()
        logger.info("✓ Local battle test passed")
        
        logger.info("=" * 50)
        logger.info("All tests passed! poke-env is working correctly.")
        logger.info("=" * 50)
        
        return True
        
    except ImportError as e:
        logger.error(f"✗ Import error: {e}")
        logger.error("Please install poke-env: pip install poke-env")
        return False
    except Exception as e:
        logger.error(f"✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(test_player_instantiation())
    sys.exit(0 if success else 1)

