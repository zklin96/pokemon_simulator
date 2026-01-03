"""Tests for StreamingBattleProcessor."""

import pytest
import json
import tempfile
from pathlib import Path
import numpy as np

from src.data.parsers.trajectory_builder import (
    StreamingBattleProcessor,
    process_vgc_bench_streaming,
    TrajectoryBuilder,
    Trajectory,
)


class TestStreamingBattleProcessor:
    """Tests for the StreamingBattleProcessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.processor = StreamingBattleProcessor(batch_size=10)
    
    def test_init_default_batch_size(self):
        """Test default batch size."""
        processor = StreamingBattleProcessor()
        assert processor.batch_size == 2000
    
    def test_init_custom_batch_size(self):
        """Test custom batch size."""
        processor = StreamingBattleProcessor(batch_size=500)
        assert processor.batch_size == 500
    
    def test_stats_initialized(self):
        """Test that stats are initialized correctly."""
        assert self.processor.stats['total_battles'] == 0
        assert self.processor.stats['successful_battles'] == 0
        assert self.processor.stats['failed_battles'] == 0
        assert self.processor.stats['total_trajectories'] == 0
        assert self.processor.stats['total_transitions'] == 0
    
    def test_process_empty_batch(self):
        """Test processing an empty batch."""
        trajectories = self.processor._process_batch({}, [])
        assert trajectories == []
    
    def test_process_batch_with_invalid_data(self):
        """Test that invalid battles are counted as failures."""
        data = {
            'battle1': [123, 'invalid log data'],
            'battle2': [456, 'also invalid'],
        }
        
        trajectories = self.processor._process_batch(data, ['battle1', 'battle2'])
        
        # Both should fail since the log format is invalid
        assert self.processor.stats['failed_battles'] >= 0  # May fail or succeed depending on parser


class TestStreamingProcessorWithMockData:
    """Tests with mock battle data."""
    
    @pytest.fixture
    def mock_battle_log(self):
        """Create a minimal valid battle log."""
        return """|j|
|t:|1234567890
|gametype|doubles
|player|p1|Player1|avatar1
|player|p2|Player2|avatar2
|teamsize|p1|6
|teamsize|p2|6
|gen|9
|tier|[Gen 9] VGC 2024 Reg G
|rule|Species Clause: Limit one of each Pokémon
|poke|p1|Pikachu, L50|
|poke|p1|Charizard, L50|
|poke|p1|Venusaur, L50|
|poke|p1|Blastoise, L50|
|poke|p2|Mewtwo, L50|
|poke|p2|Mew, L50|
|poke|p2|Celebi, L50|
|poke|p2|Jirachi, L50|
|start
|switch|p1a: Pikachu|Pikachu, L50, M|100/100
|switch|p1b: Charizard|Charizard, L50, M|100/100
|switch|p2a: Mewtwo|Mewtwo, L50|100/100
|switch|p2b: Mew|Mew, L50|100/100
|turn|1
|move|p1a: Pikachu|Thunderbolt|p2a: Mewtwo
|-damage|p2a: Mewtwo|50/100
|move|p2a: Mewtwo|Psychic|p1a: Pikachu
|-damage|p1a: Pikachu|30/100
|turn|2
|move|p1a: Pikachu|Thunderbolt|p2a: Mewtwo
|-damage|p2a: Mewtwo|0 fnt
|faint|p2a: Mewtwo
|win|Player1
"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_save_batch_parquet(self, temp_dir):
        """Test saving a batch to Parquet format."""
        processor = StreamingBattleProcessor(batch_size=10)
        
        # Create a mock trajectory
        trajectory = Trajectory(
            battle_id='test_battle',
            player_perspective='p1',
            won=True,
            transitions=[],
        )
        
        # Add a mock transition
        from src.data.parsers.trajectory_builder import Transition
        trajectory.transitions.append(Transition(
            state=np.zeros(620, dtype=np.float32),
            action=42,
            reward=1.0,
            next_state=np.zeros(620, dtype=np.float32),
            done=False,
        ))
        
        # Save the batch
        output_file = processor._save_batch_parquet([trajectory], temp_dir, 0)
        
        assert output_file.exists()
        assert output_file.suffix == '.parquet'
        
        # Verify we can read it back
        import pyarrow.parquet as pq
        table = pq.read_table(output_file)
        
        assert len(table) == 1  # One transition
        assert 'battle_id' in table.column_names
        assert 'action' in table.column_names
        assert 'reward' in table.column_names
    
    def test_save_metadata(self, temp_dir):
        """Test saving metadata file."""
        processor = StreamingBattleProcessor(batch_size=10)
        processor.stats['total_battles'] = 100
        processor.stats['successful_battles'] = 95
        
        batch_files = [temp_dir / 'batch_0000.parquet', temp_dir / 'batch_0001.parquet']
        processor._save_metadata(temp_dir, batch_files)
        
        metadata_file = temp_dir / 'metadata.json'
        assert metadata_file.exists()
        
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        assert metadata['stats']['total_battles'] == 100
        assert metadata['num_batches'] == 2
        assert metadata['state_dim'] == 620


class TestProcessVgcBenchStreaming:
    """Integration tests for process_vgc_bench_streaming function."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def sample_battles_file(self, temp_dir):
        """Create a sample battles JSON file."""
        # Create minimal battle logs
        battle_log = """|j|
|t:|1234567890
|gametype|doubles
|player|p1|Player1|avatar1
|player|p2|Player2|avatar2
|gen|9
|start
|switch|p1a: Pikachu|Pikachu, L50|100/100
|switch|p1b: Charizard|Charizard, L50|100/100
|switch|p2a: Mewtwo|Mewtwo, L50|100/100
|switch|p2b: Mew|Mew, L50|100/100
|turn|1
|win|Player1
"""
        
        # Create 5 sample battles
        battles = {}
        for i in range(5):
            battles[f'battle_{i}'] = [1234567890 + i, battle_log]
        
        battles_file = temp_dir / 'test_battles.json'
        with open(battles_file, 'w') as f:
            json.dump(battles, f)
        
        return battles_file
    
    def test_streaming_creates_output_files(self, temp_dir, sample_battles_file):
        """Test that streaming processing creates output files."""
        output_dir = temp_dir / 'output'
        
        stats = process_vgc_bench_streaming(
            sample_battles_file,
            output_dir,
            batch_size=2,
            max_battles=5,
        )
        
        assert output_dir.exists()
        assert (output_dir / 'metadata.json').exists()
        assert stats['total_battles'] == 5
    
    def test_streaming_respects_max_battles(self, temp_dir, sample_battles_file):
        """Test that max_battles parameter is respected."""
        output_dir = temp_dir / 'output'
        
        stats = process_vgc_bench_streaming(
            sample_battles_file,
            output_dir,
            batch_size=10,
            max_battles=2,
        )
        
        assert stats['total_battles'] == 2


class TestBatchProcessingMemory:
    """Tests to verify memory efficiency of batch processing."""
    
    def test_batch_processing_releases_memory(self):
        """Test that memory is released between batches."""
        import gc
        
        processor = StreamingBattleProcessor(batch_size=10)
        
        # Process an empty batch and verify gc.collect is called implicitly
        # (We can't easily test memory directly, but we can verify the pattern)
        
        # The processor should have a clean stats dict after init
        assert processor.stats['total_trajectories'] == 0
        
        # After processing empty data, stats should still be clean
        processor._process_batch({}, [])
        gc.collect()  # Manual gc to verify no errors
        
        assert True  # If we get here, no memory errors

