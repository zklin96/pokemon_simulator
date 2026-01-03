"""Scraper for Pokemon Showdown battle replays."""

import re
import json
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from urllib.parse import urljoin

import aiohttp
import requests
from bs4 import BeautifulSoup
from loguru import logger

from src.config import config, RAW_DATA_DIR


class ShowdownReplayScraper:
    """Scraper for Pokemon Showdown battle replays."""
    
    REPLAY_URL = "https://replay.pokemonshowdown.com"
    SEARCH_URL = "https://replay.pokemonshowdown.com/search.json"
    
    def __init__(self, format_name: str = "gen9vgc2024regg"):
        """Initialize the scraper.
        
        Args:
            format_name: The battle format to scrape replays for
        """
        self.format_name = format_name
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Pokemon-VGC-AI-Research/1.0"
        })
    
    def search_replays(
        self, 
        username: Optional[str] = None,
        page: int = 1
    ) -> List[Dict[str, Any]]:
        """Search for replays by format and optionally username.
        
        Args:
            username: Optional username to filter by
            page: Page number for pagination
            
        Returns:
            List of replay metadata
        """
        params = {
            "format": self.format_name,
            "page": page,
        }
        
        if username:
            params["user"] = username
        
        try:
            response = self.session.get(self.SEARCH_URL, params=params)
            response.raise_for_status()
            
            data = response.json()
            return data if isinstance(data, list) else []
            
        except Exception as e:
            logger.error(f"Error searching replays: {e}")
            return []
    
    def get_replay(self, replay_id: str) -> Optional[Dict[str, Any]]:
        """Get a single replay by ID.
        
        Args:
            replay_id: Replay ID (e.g., "gen9vgc2024regg-1234567890")
            
        Returns:
            Replay data dictionary or None
        """
        url = f"{self.REPLAY_URL}/{replay_id}.json"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Error fetching replay {replay_id}: {e}")
            return None
    
    def parse_replay_log(self, log: str) -> Dict[str, Any]:
        """Parse a replay log into structured data.
        
        Args:
            log: Raw replay log string
            
        Returns:
            Parsed replay data
        """
        parsed = {
            "players": {},
            "teams": {1: [], 2: []},
            "turns": [],
            "winner": None,
            "total_turns": 0,
        }
        
        current_turn = 0
        turn_events = []
        
        lines = log.strip().split("\n")
        
        for line in lines:
            if not line.startswith("|"):
                continue
            
            parts = line.split("|")
            if len(parts) < 2:
                continue
            
            event_type = parts[1]
            
            # Player info
            if event_type == "player":
                if len(parts) >= 4:
                    player_id = parts[2]  # p1 or p2
                    player_name = parts[3]
                    player_num = 1 if player_id == "p1" else 2
                    parsed["players"][player_num] = player_name
            
            # Team preview / Pokemon
            elif event_type == "poke":
                if len(parts) >= 4:
                    player_id = parts[2]
                    pokemon_info = parts[3]
                    player_num = 1 if player_id.startswith("p1") else 2
                    
                    # Parse Pokemon name and form
                    pokemon_name = pokemon_info.split(",")[0].strip()
                    parsed["teams"][player_num].append(pokemon_name)
            
            # Turn marker
            elif event_type == "turn":
                if turn_events:
                    parsed["turns"].append({
                        "turn": current_turn,
                        "events": turn_events
                    })
                
                current_turn = int(parts[2]) if len(parts) > 2 else current_turn + 1
                turn_events = []
            
            # Move used
            elif event_type == "move":
                if len(parts) >= 4:
                    pokemon = parts[2]
                    move = parts[3]
                    target = parts[4] if len(parts) > 4 else None
                    turn_events.append({
                        "type": "move",
                        "pokemon": pokemon,
                        "move": move,
                        "target": target,
                    })
            
            # Damage
            elif event_type == "-damage":
                if len(parts) >= 4:
                    pokemon = parts[2]
                    hp = parts[3]
                    turn_events.append({
                        "type": "damage",
                        "pokemon": pokemon,
                        "hp": hp,
                    })
            
            # Faint
            elif event_type == "faint":
                if len(parts) >= 3:
                    pokemon = parts[2]
                    turn_events.append({
                        "type": "faint",
                        "pokemon": pokemon,
                    })
            
            # Switch
            elif event_type in ("switch", "drag"):
                if len(parts) >= 4:
                    pokemon_slot = parts[2]
                    pokemon_info = parts[3]
                    turn_events.append({
                        "type": "switch",
                        "slot": pokemon_slot,
                        "pokemon": pokemon_info.split(",")[0],
                    })
            
            # Winner
            elif event_type == "win":
                if len(parts) >= 3:
                    parsed["winner"] = parts[2]
            
            # Terastallize
            elif event_type == "-terastallize":
                if len(parts) >= 4:
                    pokemon = parts[2]
                    tera_type = parts[3]
                    turn_events.append({
                        "type": "terastallize",
                        "pokemon": pokemon,
                        "tera_type": tera_type,
                    })
        
        # Add final turn
        if turn_events:
            parsed["turns"].append({
                "turn": current_turn,
                "events": turn_events
            })
        
        parsed["total_turns"] = len(parsed["turns"])
        
        return parsed
    
    def scrape_replays(
        self, 
        count: int = 100,
        parse: bool = True
    ) -> List[Dict[str, Any]]:
        """Scrape multiple replays.
        
        Args:
            count: Number of replays to scrape
            parse: Whether to parse replay logs
            
        Returns:
            List of replay data
        """
        all_replays = []
        page = 1
        
        while len(all_replays) < count:
            logger.info(f"Fetching replay page {page}...")
            
            results = self.search_replays(page=page)
            if not results:
                break
            
            for result in results:
                if len(all_replays) >= count:
                    break
                
                replay_id = result.get("id")
                if not replay_id:
                    continue
                
                logger.debug(f"Fetching replay {replay_id}")
                replay = self.get_replay(replay_id)
                
                if replay:
                    if parse and "log" in replay:
                        replay["parsed"] = self.parse_replay_log(replay["log"])
                    
                    replay["scraped_at"] = datetime.utcnow().isoformat()
                    all_replays.append(replay)
            
            page += 1
        
        return all_replays
    
    def scrape_and_save(
        self, 
        count: int = 100,
        output_file: Optional[Path] = None
    ) -> List[Dict[str, Any]]:
        """Scrape replays and save to disk.
        
        Args:
            count: Number of replays to scrape
            output_file: Optional output file path
            
        Returns:
            List of scraped replays
        """
        output_dir = RAW_DATA_DIR / "replays"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"{self.format_name}_{timestamp}.json"
        
        replays = self.scrape_replays(count=count)
        
        with open(output_file, "w") as f:
            json.dump(replays, f, indent=2)
        
        logger.info(f"Saved {len(replays)} replays to {output_file}")
        
        return replays


def main():
    """Example usage of the scraper."""
    logger.info("Starting Showdown replay scraper...")
    
    scraper = ShowdownReplayScraper(format_name="gen9vgc2024regg")
    
    # Search for replays
    results = scraper.search_replays()
    logger.info(f"Found {len(results)} replays")
    
    if results:
        # Show first few
        for replay in results[:3]:
            logger.info(f"  - {replay.get('id')}: {replay.get('p1')} vs {replay.get('p2')}")
        
        # Get full replay for first result
        replay_id = results[0].get("id")
        if replay_id:
            replay = scraper.get_replay(replay_id)
            if replay:
                parsed = scraper.parse_replay_log(replay.get("log", ""))
                logger.info(f"Parsed replay: {parsed['total_turns']} turns, winner: {parsed['winner']}")


if __name__ == "__main__":
    main()

