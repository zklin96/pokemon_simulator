"""Tests for battle history tracking."""

import pytest
from src.data.parsers.trajectory_builder import BattleHistoryTracker


class TestBattleHistoryTracker:
    """Tests for temporal context tracking."""
    
    @pytest.fixture
    def tracker(self):
        return BattleHistoryTracker(history_len=5)
    
    def test_initial_state(self, tracker):
        """Tracker starts empty."""
        assert tracker.action_history == []
        assert tracker.damage_dealt == []
        assert len(tracker.get_context()['action_history']) == 5
    
    def test_records_actions(self, tracker):
        """Actions are recorded correctly."""
        tracker.record_turn(action=10, damage_dealt=0.2, damage_received=0.1)
        tracker.record_turn(action=25, damage_dealt=0.3, damage_received=0.0)
        
        assert tracker.action_history == [10, 25]
        assert tracker.damage_dealt == [0.2, 0.3]
        assert tracker.damage_received == [0.1, 0.0]
    
    def test_overflow_handling(self, tracker):
        """Old entries are removed when exceeding history_len."""
        for i in range(7):  # More than history_len=5
            tracker.record_turn(action=i, damage_dealt=0.1, damage_received=0.1)
        
        assert len(tracker.action_history) == 5
        assert tracker.action_history == [2, 3, 4, 5, 6]  # Only last 5
    
    def test_padding(self, tracker):
        """get_context() pads short histories correctly."""
        tracker.record_turn(action=10, damage_dealt=0.2, damage_received=0.1)
        
        context = tracker.get_context()
        
        # Should be padded to length 5
        assert len(context['action_history']) == 5
        
        # First 4 should be padding (0), last is real data
        assert context['action_history'][:4] == [0, 0, 0, 0]
        assert context['action_history'][4] == 10
    
    def test_history_mask(self, tracker):
        """History mask correctly marks real vs padded data."""
        tracker.record_turn(action=10, damage_dealt=0.2, damage_received=0.1)
        tracker.record_turn(action=20, damage_dealt=0.3, damage_received=0.2)
        
        context = tracker.get_context()
        
        # First 3 are padding (0.0), last 2 are real (1.0)
        assert context['history_mask'] == [0.0, 0.0, 0.0, 1.0, 1.0]
    
    def test_reset(self, tracker):
        """Tracker can be reset for new battle."""
        tracker.record_turn(action=10, damage_dealt=0.2, damage_received=0.1)
        tracker.reset()
        
        assert tracker.action_history == []
        assert tracker.damage_dealt == []
        assert tracker.damage_received == []
    
    def test_switch_tracking(self, tracker):
        """Switch events are tracked."""
        tracker.record_turn(action=10, did_switch=True)
        tracker.record_turn(action=20, did_switch=False)
        tracker.record_turn(action=30, did_switch=True)
        
        assert tracker.switch_events == [True, False, True]
        
        context = tracker.get_context()
        # Padded: [0, 0, 1, 0, 1]
        assert context['switch_history'][-3:] == [1.0, 0.0, 1.0]
    
    def test_outcome_tracking(self, tracker):
        """Outcomes are computed correctly."""
        # Good turn: KO without losing
        tracker.record_turn(action=0, got_ko=True, lost_pokemon=False)
        assert tracker.outcome_history[-1] == 1
        
        # Bad turn: Lost Pokemon without KO
        tracker.record_turn(action=0, got_ko=False, lost_pokemon=True)
        assert tracker.outcome_history[-1] == -1
        
        # Good turn: More damage dealt
        tracker.record_turn(action=0, damage_dealt=0.3, damage_received=0.1)
        assert tracker.outcome_history[-1] == 1
        
        # Bad turn: More damage received
        tracker.record_turn(action=0, damage_dealt=0.1, damage_received=0.3)
        assert tracker.outcome_history[-1] == -1
        
        # Neutral: Equal exchange
        tracker.record_turn(action=0, damage_dealt=0.2, damage_received=0.2)
        # Note: this exceeds history_len, so oldest is removed
        # But the last outcome should be neutral
        assert tracker.outcome_history[-1] == 0
    
    def test_damage_history_shape(self, tracker):
        """Damage history has correct shape [history_len, 2]."""
        tracker.record_turn(action=0, damage_dealt=0.2, damage_received=0.1)
        tracker.record_turn(action=0, damage_dealt=0.3, damage_received=0.0)
        
        context = tracker.get_context()
        damage_history = context['damage_history']
        
        assert len(damage_history) == 5
        assert all(len(entry) == 2 for entry in damage_history)
        
        # Check last two entries have our data
        assert damage_history[-2] == [0.2, 0.1]
        assert damage_history[-1] == [0.3, 0.0]
        
        # First three should be zeros (padding)
        assert damage_history[0] == [0.0, 0.0]
    
    def test_current_turn_count(self, tracker):
        """Current turn is tracked correctly."""
        assert tracker.get_context()['current_turn'] == 0
        
        tracker.record_turn(action=0)
        assert tracker.get_context()['current_turn'] == 1
        
        tracker.record_turn(action=0)
        tracker.record_turn(action=0)
        assert tracker.get_context()['current_turn'] == 3
    
    def test_summary_stats_empty(self, tracker):
        """Summary stats work with empty history."""
        stats = tracker.get_summary_stats()
        
        assert stats['total_damage_dealt'] == 0.0
        assert stats['ko_count'] == 0
        assert stats['switch_count'] == 0
    
    def test_summary_stats(self, tracker):
        """Summary stats aggregate correctly."""
        tracker.record_turn(action=0, damage_dealt=0.2, damage_received=0.1, got_ko=True)
        tracker.record_turn(action=0, damage_dealt=0.3, damage_received=0.0, did_switch=True)
        tracker.record_turn(action=0, damage_dealt=0.0, damage_received=0.4, lost_pokemon=True)
        
        stats = tracker.get_summary_stats()
        
        assert stats['total_damage_dealt'] == pytest.approx(0.5)
        assert stats['total_damage_received'] == pytest.approx(0.5)
        assert stats['ko_count'] == 1
        assert stats['deaths'] == 1
        assert stats['switch_count'] == 1


class TestBattleHistoryTrackerEdgeCases:
    """Edge case tests for history tracker."""
    
    def test_very_short_history(self):
        """Works with history_len=1."""
        tracker = BattleHistoryTracker(history_len=1)
        tracker.record_turn(action=5)
        tracker.record_turn(action=10)
        
        assert tracker.action_history == [10]
        assert tracker.get_context()['action_history'] == [10]
    
    def test_very_long_history(self):
        """Works with large history_len."""
        tracker = BattleHistoryTracker(history_len=100)
        for i in range(50):
            tracker.record_turn(action=i)
        
        context = tracker.get_context()
        assert len(context['action_history']) == 100
        assert context['action_history'][:50] == [0] * 50  # Padding
        assert context['action_history'][50:] == list(range(50))
    
    def test_negative_damage(self):
        """Handles negative values (healing) gracefully."""
        tracker = BattleHistoryTracker(history_len=5)
        tracker.record_turn(action=0, damage_dealt=-0.2, damage_received=-0.1)
        
        assert tracker.damage_dealt[-1] == -0.2  # Stored as-is
        assert tracker.damage_received[-1] == -0.1

