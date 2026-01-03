"""Parquet-based storage for trajectories.

This module provides efficient storage for trajectory data using Parquet,
achieving ~10x compression over JSON while enabling columnar access.

Features:
- Efficient columnar storage with snappy compression
- Streaming read/write for large datasets
- Partition by battle for incremental processing
- Metadata preservation
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator, Tuple, Union
from dataclasses import dataclass, asdict
import json
from loguru import logger
from tqdm import tqdm


@dataclass
class StorageConfig:
    """Configuration for Parquet storage."""
    compression: str = "snappy"  # snappy, gzip, lz4, zstd
    row_group_size: int = 10000
    use_dictionary: bool = True
    partition_by: Optional[str] = None  # e.g., "battle_id"


class TrajectoryStorage:
    """Efficient storage for trajectory data using Parquet.
    
    The storage format optimizes for:
    - Fast sequential reads for training
    - Columnar access for analytics
    - Compression (10x vs JSON)
    - Incremental appends
    
    Schema:
    - battle_id: string
    - player: string (p1/p2)
    - turn: int32
    - state: fixed_size_list[float32, 620]
    - action: int32
    - reward: float32
    - done: bool
    - won: bool
    - metadata: string (JSON)
    """
    
    # State dimension
    STATE_DIM = 620
    
    def __init__(
        self,
        storage_dir: Path,
        config: Optional[StorageConfig] = None,
    ):
        """Initialize storage.
        
        Args:
            storage_dir: Directory for storing Parquet files
            config: Storage configuration
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.config = config or StorageConfig()
        
        # Define schema
        self.schema = pa.schema([
            ("battle_id", pa.string()),
            ("player", pa.string()),
            ("turn", pa.int32()),
            ("state", pa.list_(pa.float32(), self.STATE_DIM)),
            ("action", pa.int32()),
            ("reward", pa.float32()),
            ("done", pa.bool_()),
            ("won", pa.bool_()),
            ("metadata", pa.string()),
        ])
    
    def save_trajectories(
        self,
        trajectories: List[Dict[str, Any]],
        filename: str = "trajectories.parquet",
    ) -> Path:
        """Save trajectories to Parquet file.
        
        Args:
            trajectories: List of trajectory dictionaries
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        # Flatten trajectories to rows
        rows = []
        for traj in tqdm(trajectories, desc="Preparing data"):
            battle_id = traj.get("battle_id", "unknown")
            player = traj.get("player", "p1")
            won = traj.get("winner", "") == player
            
            for i, trans in enumerate(traj.get("transitions", [])):
                state = trans.get("state", [0.0] * self.STATE_DIM)
                # Ensure state is correct length
                if len(state) < self.STATE_DIM:
                    state = state + [0.0] * (self.STATE_DIM - len(state))
                elif len(state) > self.STATE_DIM:
                    state = state[:self.STATE_DIM]
                
                rows.append({
                    "battle_id": battle_id,
                    "player": player,
                    "turn": i,
                    "state": [float(x) for x in state],
                    "action": int(trans.get("action", 0)),
                    "reward": float(trans.get("reward", 0.0)),
                    "done": bool(trans.get("done", False)),
                    "won": won,
                    "metadata": json.dumps(trans.get("metadata", {})),
                })
        
        # Convert to PyArrow table
        table = pa.Table.from_pylist(rows, schema=self.schema)
        
        # Write to Parquet
        output_path = self.storage_dir / filename
        pq.write_table(
            table,
            output_path,
            compression=self.config.compression,
            row_group_size=self.config.row_group_size,
            use_dictionary=self.config.use_dictionary,
        )
        
        # Log stats
        file_size = output_path.stat().st_size / 1024 / 1024
        logger.info(
            f"Saved {len(rows)} transitions from {len(trajectories)} trajectories "
            f"to {output_path} ({file_size:.2f} MB)"
        )
        
        return output_path
    
    def load_trajectories(
        self,
        filename: str = "trajectories.parquet",
        columns: Optional[List[str]] = None,
        filters: Optional[List[Tuple]] = None,
    ) -> pd.DataFrame:
        """Load trajectories from Parquet file.
        
        Args:
            filename: Input filename
            columns: Specific columns to load (None = all)
            filters: Row filters e.g., [("won", "=", True)]
            
        Returns:
            DataFrame with trajectory data
        """
        path = self.storage_dir / filename
        
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file not found: {path}")
        
        table = pq.read_table(
            path,
            columns=columns,
            filters=filters,
        )
        
        return table.to_pandas()
    
    def load_for_training(
        self,
        filename: str = "trajectories.parquet",
        batch_size: int = 1000,
        shuffle: bool = True,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Load data in batches for training.
        
        Args:
            filename: Input filename
            batch_size: Number of rows per batch
            shuffle: Whether to shuffle data
            
        Yields:
            Tuples of (states, actions, rewards, dones)
        """
        path = self.storage_dir / filename
        
        # Read in batches
        parquet_file = pq.ParquetFile(path)
        
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=["state", "action", "reward", "done"],
        ):
            df = batch.to_pandas()
            
            if shuffle:
                df = df.sample(frac=1.0)
            
            # Extract arrays
            states = np.stack(df["state"].values)
            actions = df["action"].values.astype(np.int64)
            rewards = df["reward"].values.astype(np.float32)
            dones = df["done"].values
            
            yield states, actions, rewards, dones
    
    def get_stats(self, filename: str = "trajectories.parquet") -> Dict[str, Any]:
        """Get statistics about stored data.
        
        Args:
            filename: Input filename
            
        Returns:
            Dictionary of statistics
        """
        path = self.storage_dir / filename
        
        if not path.exists():
            return {"error": "File not found"}
        
        parquet_file = pq.ParquetFile(path)
        metadata = parquet_file.metadata
        
        # Read sample for stats
        df = self.load_trajectories(filename)
        
        return {
            "file_size_mb": path.stat().st_size / 1024 / 1024,
            "num_rows": metadata.num_rows,
            "num_row_groups": metadata.num_row_groups,
            "num_columns": metadata.num_columns,
            "schema": str(self.schema),
            "num_battles": df["battle_id"].nunique(),
            "num_transitions": len(df),
            "win_rate": df["won"].mean(),
            "avg_reward": df["reward"].mean(),
            "action_distribution": df["action"].value_counts().head(10).to_dict(),
        }
    
    def append_trajectories(
        self,
        trajectories: List[Dict[str, Any]],
        filename: str = "trajectories.parquet",
    ) -> Path:
        """Append trajectories to existing file.
        
        Args:
            trajectories: New trajectories to append
            filename: Target filename
            
        Returns:
            Path to updated file
        """
        path = self.storage_dir / filename
        
        # Prepare new data
        rows = []
        for traj in trajectories:
            battle_id = traj.get("battle_id", "unknown")
            player = traj.get("player", "p1")
            won = traj.get("winner", "") == player
            
            for i, trans in enumerate(traj.get("transitions", [])):
                state = trans.get("state", [0.0] * self.STATE_DIM)
                if len(state) < self.STATE_DIM:
                    state = state + [0.0] * (self.STATE_DIM - len(state))
                elif len(state) > self.STATE_DIM:
                    state = state[:self.STATE_DIM]
                
                rows.append({
                    "battle_id": battle_id,
                    "player": player,
                    "turn": i,
                    "state": [float(x) for x in state],
                    "action": int(trans.get("action", 0)),
                    "reward": float(trans.get("reward", 0.0)),
                    "done": bool(trans.get("done", False)),
                    "won": won,
                    "metadata": "{}",
                })
        
        new_table = pa.Table.from_pylist(rows, schema=self.schema)
        
        if path.exists():
            # Read existing and concatenate
            existing_table = pq.read_table(path)
            combined_table = pa.concat_tables([existing_table, new_table])
        else:
            combined_table = new_table
        
        # Write back
        pq.write_table(
            combined_table,
            path,
            compression=self.config.compression,
            row_group_size=self.config.row_group_size,
        )
        
        logger.info(f"Appended {len(rows)} transitions to {path}")
        return path


class StreamingTrajectoryReader:
    """Read trajectories from multiple batch Parquet files efficiently.
    
    This class handles reading from batched Parquet files created by
    StreamingBattleProcessor, providing lazy loading for memory efficiency.
    """
    
    STATE_DIM = 620
    
    def __init__(self, batch_dir: Path):
        """Initialize the streaming reader.
        
        Args:
            batch_dir: Directory containing batch Parquet files and metadata.json
        """
        self.batch_dir = Path(batch_dir)
        self.metadata = self._load_metadata()
        self.batch_files = self._discover_batch_files()
        
        logger.info(f"StreamingTrajectoryReader initialized with {len(self.batch_files)} batch files")
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load metadata from metadata.json if it exists."""
        metadata_path = self.batch_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path) as f:
                return json.load(f)
        return {}
    
    def _discover_batch_files(self) -> List[Path]:
        """Discover all batch Parquet files in the directory."""
        # Check metadata for batch files
        if 'batch_files' in self.metadata:
            return [self.batch_dir / f for f in self.metadata['batch_files']]
        
        # Otherwise, discover by pattern
        batch_files = sorted(self.batch_dir.glob('batch_*.parquet'))
        
        # Also include any regular trajectory files
        if not batch_files:
            batch_files = sorted(self.batch_dir.glob('*.parquet'))
        
        return batch_files
    
    def iter_batches(
        self, 
        batch_size: int = 1000,
        columns: Optional[List[str]] = None,
        shuffle_files: bool = False,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
        """Iterate over batches of training data from all files.
        
        Args:
            batch_size: Number of transitions per batch
            columns: Columns to load (default: state, action, reward, done)
            shuffle_files: Whether to shuffle the order of files
            
        Yields:
            Tuples of (states, actions, rewards, dones) as numpy arrays
        """
        if columns is None:
            columns = ["state", "action", "reward", "done"]
        
        files = list(self.batch_files)
        if shuffle_files:
            import random
            random.shuffle(files)
        
        for batch_file in files:
            if not batch_file.exists():
                logger.warning(f"Batch file not found: {batch_file}")
                continue
            
            try:
                parquet_file = pq.ParquetFile(batch_file)
                
                for batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
                    df = batch.to_pandas()
                    
                    if len(df) == 0:
                        continue
                    
                    # Handle state column - could be bytes or list
                    if 'state' in columns:
                        states = self._extract_states(df['state'])
                    else:
                        states = None
                    
                    actions = df['action'].values.astype(np.int64) if 'action' in columns else None
                    rewards = df['reward'].values.astype(np.float32) if 'reward' in columns else None
                    dones = df['done'].values if 'done' in columns else None
                    
                    yield states, actions, rewards, dones
                    
            except Exception as e:
                logger.warning(f"Error reading {batch_file}: {e}")
                continue
    
    def _extract_states(self, state_column: pd.Series) -> np.ndarray:
        """Extract state vectors from column, handling bytes or list format."""
        sample = state_column.iloc[0] if len(state_column) > 0 else None
        
        if sample is None:
            return np.zeros((len(state_column), self.STATE_DIM), dtype=np.float32)
        
        if isinstance(sample, bytes):
            # States stored as bytes
            states = np.array([
                np.frombuffer(s, dtype=np.float32) if isinstance(s, bytes) else np.zeros(self.STATE_DIM)
                for s in state_column
            ])
        elif isinstance(sample, (list, np.ndarray)):
            # States stored as lists
            states = np.stack(state_column.values)
        else:
            logger.warning(f"Unknown state format: {type(sample)}")
            states = np.zeros((len(state_column), self.STATE_DIM), dtype=np.float32)
        
        return states.astype(np.float32)
    
    def load_all(self) -> pd.DataFrame:
        """Load all data from all batch files into a single DataFrame.
        
        Warning: This may use significant memory for large datasets.
        
        Returns:
            Combined DataFrame with all trajectories
        """
        dfs = []
        for batch_file in self.batch_files:
            if batch_file.exists():
                try:
                    df = pq.read_table(batch_file).to_pandas()
                    dfs.append(df)
                except Exception as e:
                    logger.warning(f"Error reading {batch_file}: {e}")
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs, ignore_index=True)
    
    def get_total_transitions(self) -> int:
        """Get total number of transitions across all batch files."""
        if 'stats' in self.metadata and 'total_transitions' in self.metadata['stats']:
            return self.metadata['stats']['total_transitions']
        
        total = 0
        for batch_file in self.batch_files:
            if batch_file.exists():
                try:
                    parquet_file = pq.ParquetFile(batch_file)
                    total += parquet_file.metadata.num_rows
                except Exception:
                    pass
        return total
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the stored data."""
        total_size = sum(f.stat().st_size for f in self.batch_files if f.exists())
        
        return {
            'num_batch_files': len(self.batch_files),
            'total_size_mb': total_size / 1024 / 1024,
            'total_transitions': self.get_total_transitions(),
            'metadata': self.metadata.get('stats', {}),
        }


def save_trajectories_parquet(
    trajectories: List[Dict[str, Any]],
    output_path: Path,
    compression: str = "snappy",
) -> Path:
    """Save trajectories to Parquet file.
    
    Convenience function for one-off saves.
    
    Args:
        trajectories: List of trajectory dictionaries
        output_path: Output path
        compression: Compression algorithm
        
    Returns:
        Path to saved file
    """
    storage = TrajectoryStorage(
        output_path.parent,
        StorageConfig(compression=compression),
    )
    return storage.save_trajectories(trajectories, output_path.name)


def load_trajectories_parquet(
    input_path: Path,
    columns: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load trajectories from Parquet file.
    
    Convenience function for one-off loads.
    
    Args:
        input_path: Input path
        columns: Specific columns to load
        
    Returns:
        DataFrame with trajectory data
    """
    storage = TrajectoryStorage(input_path.parent)
    return storage.load_trajectories(input_path.name, columns=columns)


def convert_json_to_parquet(
    json_path: Path,
    output_path: Optional[Path] = None,
    compression: str = "snappy",
) -> Tuple[Path, Dict[str, Any]]:
    """Convert JSON trajectories file to Parquet.
    
    Args:
        json_path: Path to JSON file
        output_path: Output path (default: same name with .parquet)
        compression: Compression algorithm
        
    Returns:
        Tuple of (output path, statistics)
    """
    if output_path is None:
        output_path = json_path.with_suffix(".parquet")
    
    logger.info(f"Converting {json_path} to Parquet...")
    
    # Load JSON
    with open(json_path) as f:
        trajectories = json.load(f)
    
    json_size = json_path.stat().st_size / 1024 / 1024
    logger.info(f"Loaded {len(trajectories)} trajectories from JSON ({json_size:.2f} MB)")
    
    # Save as Parquet
    storage = TrajectoryStorage(
        output_path.parent,
        StorageConfig(compression=compression),
    )
    storage.save_trajectories(trajectories, output_path.name)
    
    # Get stats
    stats = storage.get_stats(output_path.name)
    stats["json_size_mb"] = json_size
    stats["compression_ratio"] = json_size / stats["file_size_mb"]
    
    logger.info(
        f"Converted to Parquet: {stats['file_size_mb']:.2f} MB "
        f"(compression ratio: {stats['compression_ratio']:.1f}x)"
    )
    
    return output_path, stats


if __name__ == "__main__":
    """Example usage and testing."""
    import tempfile
    
    # Create test data
    test_trajectories = [
        {
            "battle_id": f"battle_{i}",
            "player": "p1",
            "winner": "p1" if i % 2 == 0 else "p2",
            "transitions": [
                {
                    "state": [float(j) for j in range(620)],
                    "action": j % 144,
                    "reward": 0.1 * j,
                    "done": j == 9,
                }
                for j in range(10)
            ],
        }
        for i in range(100)
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Test save/load
        storage = TrajectoryStorage(tmpdir)
        storage.save_trajectories(test_trajectories)
        
        # Load and verify
        df = storage.load_trajectories()
        print(f"Loaded {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        
        # Get stats
        stats = storage.get_stats()
        print(f"Stats: {stats}")
        
        # Test batch loading
        for batch_idx, (states, actions, rewards, dones) in enumerate(
            storage.load_for_training(batch_size=100)
        ):
            print(f"Batch {batch_idx}: states={states.shape}, actions={actions.shape}")
            if batch_idx >= 2:
                break

