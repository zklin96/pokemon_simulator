"""Web scrapers for Pokemon battle data."""

from .smogon_scraper import SmogonStatsScraper
from .showdown_scraper import ShowdownReplayScraper

__all__ = ["SmogonStatsScraper", "ShowdownReplayScraper"]

