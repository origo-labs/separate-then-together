"""Integration tests for session engine."""

from pathlib import Path
from unittest.mock import Mock, patch
import pytest
import json

from separate_then_together.agent import LLMAgent
from separate_then_together.config import Config
from separate_then_together.persona import Persona
from separate_then_together.session import SessionEngine
from separate_then_together.strategies import SeparateTogetherStrategy


@pytest.fixture
def config() -> Config:
    """Create a test configuration."""
    return Config(
        openai_api_key="test-key",
        openai_base_url="http://localhost:11434/v1",
        openai_model="test-model",
        separate_turns=4,
        collab_turns=4,
    )


@pytest.fixture
def personas() -> tuple[Persona, Persona]:
    """Create test personas."""
    p1 = Persona("Agent1", "You are agent 1")
    p2 = Persona("Agent2", "You are agent 2")
    return p1, p2


@patch('separate_then_together.agent.OpenAI')
def test_session_runs_correctly(
    mock_openai: Mock,
    personas: tuple[Persona, Persona],
    config: Config
) -> None:
    """Test that session runs and produces results."""
    # Mock OpenAI responses
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test idea"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    p1, p2 = personas
    agent1 = LLMAgent(p1, config)
    agent2 = LLMAgent(p2, config)
    
    strategy = SeparateTogetherStrategy(separate_turns=4, collab_turns=4)
    session = SessionEngine(agent1, agent2, "Test topic", strategy, config)
    
    results = session.run(verbose=False)
    
    # Should have 8 total results (4 separate + 4 collaborative)
    assert len(results) == 8
    
    # Check that results have required fields
    for result in results:
        assert "turn" in result
        assert "role" in result
        assert "phase" in result
        assert "content" in result
        assert "timestamp" in result


@patch('separate_then_together.agent.OpenAI')
def test_session_alternates_agents(
    mock_openai: Mock,
    personas: tuple[Persona, Persona],
    config: Config
) -> None:
    """Test that session alternates between agents."""
    # Mock OpenAI responses
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test idea"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    p1, p2 = personas
    agent1 = LLMAgent(p1, config)
    agent2 = LLMAgent(p2, config)
    
    strategy = SeparateTogetherStrategy(separate_turns=4, collab_turns=0)
    session = SessionEngine(agent1, agent2, "Test topic", strategy, config)
    
    results = session.run(verbose=False)
    
    # Should alternate: Agent1, Agent2, Agent1, Agent2
    assert results[0]["role"] == "Agent1"
    assert results[1]["role"] == "Agent2"
    assert results[2]["role"] == "Agent1"
    assert results[3]["role"] == "Agent2"


@patch('separate_then_together.agent.OpenAI')
def test_session_groups_by_phase(
    mock_openai: Mock,
    personas: tuple[Persona, Persona],
    config: Config
) -> None:
    """Test that results are correctly grouped by phase."""
    # Mock OpenAI responses
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test idea"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    p1, p2 = personas
    agent1 = LLMAgent(p1, config)
    agent2 = LLMAgent(p2, config)
    
    strategy = SeparateTogetherStrategy(separate_turns=2, collab_turns=2)
    session = SessionEngine(agent1, agent2, "Test topic", strategy, config)
    
    session.run(verbose=False)
    by_phase = session.get_results_by_phase()
    
    assert "Separate" in by_phase
    assert "Collaborative" in by_phase
    assert len(by_phase["Separate"]) == 2
    assert len(by_phase["Collaborative"]) == 2


@patch('separate_then_together.agent.OpenAI')
def test_session_export_json(
    mock_openai: Mock,
    personas: tuple[Persona, Persona],
    config: Config,
    tmp_path: Path
) -> None:
    """Test exporting session results to JSON."""
    # Mock OpenAI responses
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test idea"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    p1, p2 = personas
    agent1 = LLMAgent(p1, config)
    agent2 = LLMAgent(p2, config)
    
    strategy = SeparateTogetherStrategy(separate_turns=2, collab_turns=2)
    session = SessionEngine(agent1, agent2, "Test topic", strategy, config)
    
    session.run(verbose=False)
    
    output_file = tmp_path / "results.json"
    session.export_to_json(output_file)
    
    assert output_file.exists()
    
    # Verify JSON structure
    with open(output_file) as f:
        data = json.load(f)
    
    assert data["topic"] == "Test topic"
    assert data["agents"] == ["Agent1", "Agent2"]
    assert "results" in data
    assert "summary" in data


@patch('separate_then_together.agent.OpenAI')
def test_session_export_markdown(
    mock_openai: Mock,
    personas: tuple[Persona, Persona],
    config: Config,
    tmp_path: Path
) -> None:
    """Test exporting session results to Markdown."""
    # Mock OpenAI responses
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test idea"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    p1, p2 = personas
    agent1 = LLMAgent(p1, config)
    agent2 = LLMAgent(p2, config)
    
    strategy = SeparateTogetherStrategy(separate_turns=2, collab_turns=2)
    session = SessionEngine(agent1, agent2, "Test topic", strategy, config)
    
    session.run(verbose=False)
    
    output_file = tmp_path / "results.md"
    session.export_to_markdown(output_file)
    
    assert output_file.exists()
    
    # Verify markdown content
    content = output_file.read_text()
    assert "# Multi-Agent Planning Session" in content
    assert "Test topic" in content
    assert "Agent1" in content
    assert "Agent2" in content
