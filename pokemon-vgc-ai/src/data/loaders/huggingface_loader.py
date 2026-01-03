"""HuggingFace dataset loader for Pokemon Showdown replays."""

import os
from pathlib import Path
from typing import Optional, List, Dict, Any, Iterator, Callable
from dataclasses import dataclass
import json

from datasets import load_dataset, Dataset, DatasetDict
from loguru import logger
import pandas as pd

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


@dataclass
class ReplayData:
    """Container for a single replay."""
    
    replay_id: str
    format: str
    player1: str
    player2: str
    winner: Optional[str]
    rating: Optional[int]
    log: str
    upload_time: Optional[str]
    
    def is_vgc(self) -> bool:
        """Check if this is a VGC format replay."""
        format_lower = self.format.lower()
        return "vgc" in format_lower or "battlestadium" in format_lower
    
    def is_high_level(self, min_rating: int = 1500) -> bool:
        """Check if this is a high-level battle."""
        return self.rating is not None and self.rating >= min_rating


class HuggingFaceReplayLoader:
    """Loader for Pokemon Showdown replays from HuggingFace datasets."""
    
    # Available datasets
    DATASETS = {
        "metamon": "jakegrigsby/metamon-raw-replays",
        "holidayougi": "HolidayOugi/pokemon-showdown-replays",
    }
    
    # VGC format patterns
    VGC_PATTERNS = [
        "vgc",
        "battlestadium",
        "bss",
    ]
    
    def __init__(
        self,
        dataset_name: str = "metamon",
        cache_dir: Optional[Path] = None,
    ):
        """Initialize the loader.
        
        Args:
            dataset_name: Name of dataset to load ("metamon" or "holidayougi")
            cache_dir: Directory to cache downloaded data
        """
        self.dataset_name = dataset_name
        self.cache_dir = cache_dir or (RAW_DATA_DIR / "huggingface")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.dataset_id = self.DATASETS.get(dataset_name)
        if not self.dataset_id:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(self.DATASETS.keys())}")
        
        self._dataset: Optional[Dataset] = None
    
    def load_dataset(
        self,
        split: str = "train",
        streaming: bool = True,
    ) -> Dataset:
        """Load the full dataset from HuggingFace.
        
        Args:
            split: Dataset split to load
            streaming: If True, use streaming mode (lower memory)
            
        Returns:
            HuggingFace Dataset object
        """
        logger.info(f"Loading dataset {self.dataset_id} (streaming={streaming})...")
        
        self._dataset = load_dataset(
            self.dataset_id,
            split=split,
            cache_dir=str(self.cache_dir),
            streaming=streaming,
        )
        
        if not streaming and hasattr(self._dataset, "__len__"):
            logger.info(f"Loaded dataset with {len(self._dataset)} entries")
        else:
            logger.info("Loaded dataset in streaming mode")
        
        return self._dataset
    
    def filter_vgc(
        self,
        dataset: Optional[Dataset] = None,
        generation: Optional[int] = None,
        min_rating: Optional[int] = None,
    ) -> Dataset:
        """Filter dataset for VGC format battles.
        
        Args:
            dataset: Dataset to filter (uses loaded dataset if None)
            generation: Filter by generation (e.g., 9 for Scarlet/Violet)
            min_rating: Minimum rating filter
            
        Returns:
            Filtered Dataset
        """
        ds = dataset or self._dataset
        if ds is None:
            raise ValueError("No dataset loaded. Call load_dataset() first.")
        
        def is_vgc_format(example: Dict[str, Any]) -> bool:
            format_str = example.get("format", "").lower()
            
            # Check VGC patterns
            is_vgc = any(pattern in format_str for pattern in self.VGC_PATTERNS)
            
            # Check generation if specified
            if generation is not None and is_vgc:
                gen_str = f"gen{generation}"
                is_vgc = gen_str in format_str
            
            # Check rating if specified
            if min_rating is not None and is_vgc:
                rating = example.get("rating")
                if rating is not None:
                    is_vgc = rating >= min_rating
            
            return is_vgc
        
        logger.info("Filtering for VGC format replays...")
        filtered = ds.filter(is_vgc_format)
        
        count = len(filtered) if hasattr(filtered, "__len__") else "unknown"
        logger.info(f"Filtered to {count} VGC replays")
        
        return filtered
    
    def get_formats_summary(
        self,
        dataset: Optional[Dataset] = None,
        sample_size: int = 10000,
    ) -> Dict[str, int]:
        """Get summary of formats in the dataset.
        
        Args:
            dataset: Dataset to analyze
            sample_size: Number of entries to sample
            
        Returns:
            Dictionary of format -> count
        """
        ds = dataset or self._dataset
        if ds is None:
            raise ValueError("No dataset loaded.")
        
        # Sample dataset
        if hasattr(ds, "__len__") and len(ds) > sample_size:
            ds = ds.shuffle(seed=42).select(range(sample_size))
        
        formats = {}
        for example in ds:
            fmt = example.get("format", "unknown")
            formats[fmt] = formats.get(fmt, 0) + 1
        
        # Sort by count
        formats = dict(sorted(formats.items(), key=lambda x: x[1], reverse=True))
        
        return formats
    
    def iterate_replays(
        self,
        dataset: Optional[Dataset] = None,
        batch_size: int = 100,
    ) -> Iterator[List[ReplayData]]:
        """Iterate over replays in batches.
        
        Args:
            dataset: Dataset to iterate
            batch_size: Number of replays per batch
            
        Yields:
            Batches of ReplayData objects
        """
        ds = dataset or self._dataset
        if ds is None:
            raise ValueError("No dataset loaded.")
        
        batch = []
        for example in ds:
            replay = ReplayData(
                replay_id=example.get("id", ""),
                format=example.get("format", ""),
                player1=example.get("p1", example.get("player1", "")),
                player2=example.get("p2", example.get("player2", "")),
                winner=example.get("winner"),
                rating=example.get("rating"),
                log=example.get("log", ""),
                upload_time=example.get("uploadtime"),
            )
            batch.append(replay)
            
            if len(batch) >= batch_size:
                yield batch
                batch = []
        
        if batch:
            yield batch
    
    def save_filtered_dataset(
        self,
        dataset: Dataset,
        output_name: str,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """Save filtered dataset to disk.
        
        Args:
            dataset: Dataset to save
            output_name: Name for the output file
            output_dir: Directory to save to
            
        Returns:
            Path to saved file
        """
        output_dir = output_dir or PROCESSED_DATA_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"{output_name}.parquet"
        
        logger.info(f"Saving dataset to {output_path}...")
        dataset.to_parquet(str(output_path))
        
        logger.info(f"Saved {len(dataset)} entries")
        
        return output_path
    
    def load_filtered_dataset(
        self,
        input_name: str,
        input_dir: Optional[Path] = None,
    ) -> Dataset:
        """Load a previously saved filtered dataset.
        
        Args:
            input_name: Name of the saved file
            input_dir: Directory to load from
            
        Returns:
            Loaded Dataset
        """
        input_dir = input_dir or PROCESSED_DATA_DIR
        input_path = input_dir / f"{input_name}.parquet"
        
        if not input_path.exists():
            raise FileNotFoundError(f"Dataset not found: {input_path}")
        
        logger.info(f"Loading dataset from {input_path}...")
        dataset = Dataset.from_parquet(str(input_path))
        
        logger.info(f"Loaded {len(dataset)} entries")
        
        return dataset


def download_vgc_replays(
    dataset_name: str = "metamon",
    generation: int = 9,
    min_rating: Optional[int] = 1500,
    max_replays: Optional[int] = None,
    output_name: Optional[str] = None,
) -> Path:
    """Download and filter VGC replays from HuggingFace.
    
    Args:
        dataset_name: Dataset to download from
        generation: Pokemon generation to filter for
        min_rating: Minimum rating filter
        max_replays: Maximum number of replays to keep
        output_name: Name for saved file
        
    Returns:
        Path to saved dataset
    """
    loader = HuggingFaceReplayLoader(dataset_name=dataset_name)
    
    # Load dataset
    logger.info(f"Downloading {dataset_name} dataset...")
    dataset = loader.load_dataset()
    
    # Show format summary
    logger.info("Analyzing formats...")
    formats = loader.get_formats_summary(sample_size=50000)
    vgc_formats = {k: v for k, v in formats.items() if "vgc" in k.lower()}
    logger.info(f"VGC formats found: {vgc_formats}")
    
    # Filter for VGC
    filtered = loader.filter_vgc(
        generation=generation,
        min_rating=min_rating,
    )
    
    # Limit size if specified
    if max_replays is not None and len(filtered) > max_replays:
        logger.info(f"Limiting to {max_replays} replays...")
        filtered = filtered.shuffle(seed=42).select(range(max_replays))
    
    # Save
    output_name = output_name or f"vgc_gen{generation}_replays"
    output_path = loader.save_filtered_dataset(filtered, output_name)
    
    return output_path


def main():
    """Download VGC replays from HuggingFace."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download VGC replays from HuggingFace")
    parser.add_argument(
        "--dataset",
        type=str,
        default="metamon",
        choices=["metamon", "holidayougi"],
        help="Dataset to download"
    )
    parser.add_argument(
        "--generation",
        type=int,
        default=9,
        help="Pokemon generation to filter"
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=1500,
        help="Minimum rating filter"
    )
    parser.add_argument(
        "--max-replays",
        type=int,
        default=50000,
        help="Maximum replays to download"
    )
    parser.add_argument(
        "--sample-only",
        action="store_true",
        help="Only download a small sample for testing"
    )
    
    args = parser.parse_args()
    
    if args.sample_only:
        args.max_replays = 1000
    
    output_path = download_vgc_replays(
        dataset_name=args.dataset,
        generation=args.generation,
        min_rating=args.min_rating,
        max_replays=args.max_replays,
    )
    
    logger.info(f"Downloaded to: {output_path}")


if __name__ == "__main__":
    main()

