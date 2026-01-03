"""Scraper for Smogon usage statistics."""

import re
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from loguru import logger

from src.config import config, RAW_DATA_DIR


class SmogonStatsScraper:
    """Scraper for Pokemon usage statistics from Smogon."""
    
    BASE_URL = "https://www.smogon.com/stats"
    
    def __init__(self, format_name: str = "gen9vgc2024regg"):
        """Initialize the scraper.
        
        Args:
            format_name: The battle format to scrape stats for
        """
        self.format_name = format_name
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Pokemon-VGC-AI-Research/1.0"
        })
    
    def get_available_months(self) -> List[str]:
        """Get list of available months with stats.
        
        Returns:
            List of month strings (e.g., ["2024-06", "2024-05", ...])
        """
        try:
            response = self.session.get(self.BASE_URL)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, "html.parser")
            months = []
            
            # Parse directory listing
            for link in soup.find_all("a"):
                href = link.get("href", "")
                # Match YYYY-MM pattern
                if re.match(r"^\d{4}-\d{2}/$", href):
                    months.append(href.rstrip("/"))
            
            return sorted(months, reverse=True)
        except Exception as e:
            logger.error(f"Error fetching available months: {e}")
            return []
    
    def get_usage_stats(self, month: str) -> Optional[Dict[str, Any]]:
        """Get usage statistics for a specific month.
        
        Args:
            month: Month string (e.g., "2024-06")
            
        Returns:
            Dictionary with usage stats or None on error
        """
        # Try different rating cutoffs (1500, 1630, 1760, etc.)
        for rating in [1760, 1630, 1500, 0]:
            url = f"{self.BASE_URL}/{month}/{self.format_name}-{rating}.txt"
            
            try:
                response = self.session.get(url)
                if response.status_code == 200:
                    return self._parse_usage_stats(response.text, month, rating)
            except Exception as e:
                logger.debug(f"Error fetching {url}: {e}")
                continue
        
        logger.warning(f"No usage stats found for {month}")
        return None
    
    def _parse_usage_stats(
        self, 
        content: str, 
        month: str, 
        rating: int
    ) -> Dict[str, Any]:
        """Parse usage statistics text file.
        
        Args:
            content: Raw text content from Smogon
            month: Month string
            rating: Rating cutoff
            
        Returns:
            Parsed statistics dictionary
        """
        lines = content.strip().split("\n")
        stats = {
            "month": month,
            "rating_cutoff": rating,
            "format": self.format_name,
            "pokemon": [],
            "scraped_at": datetime.utcnow().isoformat(),
        }
        
        # Parse header info
        for i, line in enumerate(lines):
            if line.startswith(" Total battles:"):
                try:
                    stats["total_battles"] = int(line.split(":")[1].strip())
                except (ValueError, IndexError):
                    pass
            
            # Start of Pokemon data (after header separator)
            if line.strip().startswith("+"):
                break
        
        # Parse Pokemon entries
        # Format: | rank | Pokemon | usage % | raw | raw% |
        pokemon_pattern = re.compile(
            r"\|\s*(\d+)\s*\|\s*(\w+(?:\s*\w+)*)\s*\|\s*([\d.]+)%?\s*\|"
        )
        
        for line in lines:
            match = pokemon_pattern.match(line)
            if match:
                rank, name, usage = match.groups()
                stats["pokemon"].append({
                    "rank": int(rank),
                    "name": name.strip(),
                    "usage_percent": float(usage),
                })
        
        return stats
    
    def get_moveset_stats(self, month: str, pokemon_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed moveset statistics for a Pokemon.
        
        Args:
            month: Month string
            pokemon_name: Pokemon name
            
        Returns:
            Dictionary with moveset stats or None
        """
        # Normalize Pokemon name for URL
        pokemon_id = pokemon_name.lower().replace(" ", "").replace("-", "")
        
        for rating in [1760, 1630, 1500, 0]:
            url = f"{self.BASE_URL}/{month}/moveset/{self.format_name}-{rating}/{pokemon_id}.txt"
            
            try:
                response = self.session.get(url)
                if response.status_code == 200:
                    return self._parse_moveset_stats(response.text, pokemon_name, month)
            except Exception as e:
                logger.debug(f"Error fetching moveset for {pokemon_name}: {e}")
                continue
        
        return None
    
    def _parse_moveset_stats(
        self, 
        content: str, 
        pokemon_name: str, 
        month: str
    ) -> Dict[str, Any]:
        """Parse moveset statistics.
        
        Args:
            content: Raw text content
            pokemon_name: Pokemon name
            month: Month string
            
        Returns:
            Parsed moveset dictionary
        """
        stats = {
            "pokemon": pokemon_name,
            "month": month,
            "format": self.format_name,
            "abilities": {},
            "items": {},
            "spreads": {},
            "moves": {},
            "teammates": {},
            "checks_counters": {},
            "tera_types": {},
        }
        
        current_section = None
        
        lines = content.strip().split("\n")
        for line in lines:
            line = line.strip()
            
            # Detect section headers
            if line == "Abilities":
                current_section = "abilities"
                continue
            elif line == "Items":
                current_section = "items"
                continue
            elif line == "Spreads":
                current_section = "spreads"
                continue
            elif line == "Moves":
                current_section = "moves"
                continue
            elif line == "Teammates":
                current_section = "teammates"
                continue
            elif line in ("Checks and Counters", "Counters"):
                current_section = "checks_counters"
                continue
            elif line == "Tera Types":
                current_section = "tera_types"
                continue
            
            # Parse data lines
            if current_section and "%" in line:
                # Format: "Item Name 45.678%"
                match = re.match(r"(.+?)\s+([\d.]+)%", line)
                if match:
                    name, percent = match.groups()
                    stats[current_section][name.strip()] = float(percent)
        
        return stats
    
    def scrape_and_save(
        self, 
        months: Optional[List[str]] = None,
        top_n_pokemon: int = 50
    ) -> List[Dict[str, Any]]:
        """Scrape stats and save to disk.
        
        Args:
            months: List of months to scrape (default: latest)
            top_n_pokemon: Number of top Pokemon to get detailed stats for
            
        Returns:
            List of scraped stats
        """
        if months is None:
            available = self.get_available_months()
            months = available[:1] if available else []
        
        all_stats = []
        output_dir = RAW_DATA_DIR / "smogon"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for month in months:
            logger.info(f"Scraping stats for {month}...")
            
            usage = self.get_usage_stats(month)
            if not usage:
                continue
            
            # Get moveset stats for top Pokemon
            for pokemon_entry in usage["pokemon"][:top_n_pokemon]:
                pokemon_name = pokemon_entry["name"]
                logger.debug(f"  Getting moveset for {pokemon_name}")
                
                moveset = self.get_moveset_stats(month, pokemon_name)
                if moveset:
                    pokemon_entry["moveset"] = moveset
            
            all_stats.append(usage)
            
            # Save to file
            output_file = output_dir / f"{self.format_name}_{month}.json"
            with open(output_file, "w") as f:
                json.dump(usage, f, indent=2)
            logger.info(f"Saved stats to {output_file}")
        
        return all_stats


def main():
    """Example usage of the scraper."""
    logger.info("Starting Smogon stats scraper...")
    
    scraper = SmogonStatsScraper(format_name="gen9vgc2024regg")
    
    # Get available months
    months = scraper.get_available_months()
    logger.info(f"Available months: {months[:5]}...")
    
    if months:
        # Scrape latest month
        stats = scraper.scrape_and_save(months=[months[0]], top_n_pokemon=20)
        
        if stats:
            logger.info(f"Scraped {len(stats[0]['pokemon'])} Pokemon")
            # Show top 5
            for entry in stats[0]["pokemon"][:5]:
                logger.info(f"  {entry['rank']}. {entry['name']}: {entry['usage_percent']}%")


if __name__ == "__main__":
    main()

