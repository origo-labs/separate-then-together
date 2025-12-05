"""Tests for collaboration strategies."""

import pytest
from separate_then_together.strategies import (
    SeparateStrategy,
    CollaborativeStrategy,
    SeparateTogetherStrategy,
)


def test_separate_strategy_filters_own_history() -> None:
    """Test that SeparateStrategy only shows agent's own history."""
    strategy = SeparateStrategy(separate_turns=10, collab_turns=0)
    
    full_history = [
        {"role": "Agent1", "content": "Idea 1 from Agent1"},
        {"role": "Agent2", "content": "Idea 1 from Agent2"},
        {"role": "Agent1", "content": "Idea 2 from Agent1"},
    ]
    
    # Agent1 should only see their own history
    filtered = strategy.filter_history(full_history, "Agent1")
    assert len(filtered) == 2
    assert all(entry["role"] == "Agent1" for entry in filtered)
    
    # Agent2 should only see their own history
    filtered = strategy.filter_history(full_history, "Agent2")
    assert len(filtered) == 1
    assert filtered[0]["role"] == "Agent2"


def test_separate_strategy_phase_name() -> None:
    """Test that SeparateStrategy returns correct phase name."""
    strategy = SeparateStrategy()
    assert strategy.get_phase_name() == "Separate"


def test_separate_strategy_turn_limit() -> None:
    """Test that SeparateStrategy respects turn limit."""
    strategy = SeparateStrategy(separate_turns=5, collab_turns=0)
    
    for i in range(5):
        assert strategy.should_continue()
        strategy.increment_turn()
    
    assert not strategy.should_continue()


def test_collaborative_strategy_shows_full_history() -> None:
    """Test that CollaborativeStrategy shows full history."""
    strategy = CollaborativeStrategy(separate_turns=0, collab_turns=10)
    
    full_history = [
        {"role": "Agent1", "content": "Idea 1"},
        {"role": "Agent2", "content": "Idea 2"},
        {"role": "Agent1", "content": "Idea 3"},
    ]
    
    # Both agents should see full history
    filtered1 = strategy.filter_history(full_history, "Agent1")
    filtered2 = strategy.filter_history(full_history, "Agent2")
    
    assert filtered1 == full_history
    assert filtered2 == full_history


def test_collaborative_strategy_phase_name() -> None:
    """Test that CollaborativeStrategy returns correct phase name."""
    strategy = CollaborativeStrategy()
    assert strategy.get_phase_name() == "Collaborative"


def test_separate_together_phase_transition() -> None:
    """Test that SeparateTogetherStrategy transitions phases correctly."""
    strategy = SeparateTogetherStrategy(separate_turns=4, collab_turns=4)
    
    full_history = [
        {"role": "Agent1", "content": "Idea 1"},
        {"role": "Agent2", "content": "Idea 2"},
    ]
    
    # First 4 turns should be Separate phase
    for i in range(4):
        assert strategy.get_phase_name() == "Separate"
        
        # Should filter to own history
        filtered = strategy.filter_history(full_history, "Agent1")
        assert all(entry["role"] == "Agent1" for entry in filtered)
        
        strategy.increment_turn()
    
    # Next 4 turns should be Collaborative phase
    for i in range(4):
        assert strategy.get_phase_name() == "Collaborative"
        
        # Should show full history
        filtered = strategy.filter_history(full_history, "Agent1")
        assert filtered == full_history
        
        strategy.increment_turn()
    
    # Should be complete
    assert not strategy.should_continue()


def test_separate_together_transition_detection() -> None:
    """Test transition detection in SeparateTogetherStrategy."""
    strategy = SeparateTogetherStrategy(separate_turns=2, collab_turns=2)
    
    assert not strategy.is_transitioning()
    
    strategy.increment_turn()
    assert not strategy.is_transitioning()
    
    strategy.increment_turn()
    assert strategy.is_transitioning()  # Turn 2 is the transition
    
    strategy.increment_turn()
    assert not strategy.is_transitioning()


def test_strategy_turn_counting() -> None:
    """Test that all strategies count turns correctly."""
    strategies = [
        SeparateStrategy(separate_turns=5, collab_turns=0),
        CollaborativeStrategy(separate_turns=0, collab_turns=5),
        SeparateTogetherStrategy(separate_turns=3, collab_turns=2),
    ]
    
    for strategy in strategies:
        assert strategy.current_turn == 0
        
        strategy.increment_turn()
        assert strategy.current_turn == 1
        
        strategy.increment_turn()
        assert strategy.current_turn == 2
