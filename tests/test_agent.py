"""Tests for LLM agent."""

from unittest.mock import Mock, patch
import pytest

from separate_then_together.agent import LLMAgent
from separate_then_together.config import Config
from separate_then_together.persona import Persona


@pytest.fixture
def config() -> Config:
    """Create a test configuration."""
    return Config(
        openai_api_key="test-key",
        openai_base_url="http://localhost:11434/v1",
        openai_model="test-model",
        temperature=0.7,
        max_tokens=100,
    )


@pytest.fixture
def persona() -> Persona:
    """Create a test persona."""
    return Persona(
        name="Test Agent",
        system_prompt="You are a test agent for unit testing."
    )


def test_agent_initialization(persona: Persona, config: Config) -> None:
    """Test that agent initializes correctly."""
    agent = LLMAgent(persona, config)
    
    assert agent.name == "Test Agent"
    assert agent.persona == persona
    assert agent.config == config


def test_agent_name_property(persona: Persona, config: Config) -> None:
    """Test that agent name property works."""
    agent = LLMAgent(persona, config)
    assert agent.name == persona.name


@patch('separate_then_together.agent.OpenAI')
def test_generate_idea_separate_phase(
    mock_openai: Mock,
    persona: Persona,
    config: Config
) -> None:
    """Test idea generation in Separate phase."""
    # Mock the OpenAI response
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Test idea"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    agent = LLMAgent(persona, config)
    
    history = [
        {"role": "Test Agent", "content": "Previous idea"}
    ]
    
    result = agent.generate_idea(
        topic="Test topic",
        history=history,
        phase="Separate"
    )
    
    assert result == "Test idea"
    
    # Verify the API was called
    mock_client.chat.completions.create.assert_called_once()
    call_args = mock_client.chat.completions.create.call_args
    
    assert call_args.kwargs["model"] == "test-model"
    assert call_args.kwargs["temperature"] == 0.7
    assert call_args.kwargs["max_tokens"] == 100


@patch('separate_then_together.agent.OpenAI')
def test_generate_idea_collaborative_phase(
    mock_openai: Mock,
    persona: Persona,
    config: Config
) -> None:
    """Test idea generation in Collaborative phase."""
    # Mock the OpenAI response
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="Collaborative idea"))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    agent = LLMAgent(persona, config)
    
    history = [
        {"role": "Test Agent", "content": "My idea"},
        {"role": "Other Agent", "content": "Other idea"},
    ]
    
    result = agent.generate_idea(
        topic="Test topic",
        history=history,
        phase="Collaborative"
    )
    
    assert result == "Collaborative idea"


@patch('separate_then_together.agent.OpenAI')
def test_generate_idea_handles_errors(
    mock_openai: Mock,
    persona: Persona,
    config: Config
) -> None:
    """Test that agent handles API errors gracefully."""
    # Mock an API error
    mock_client = Mock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")
    mock_openai.return_value = mock_client
    
    agent = LLMAgent(persona, config)
    
    result = agent.generate_idea(
        topic="Test topic",
        history=[],
        phase="Separate"
    )
    
    assert "Error generating idea" in result
    assert "API Error" in result


@patch('separate_then_together.agent.OpenAI')
def test_generate_idea_handles_none_response(
    mock_openai: Mock,
    persona: Persona,
    config: Config
) -> None:
    """Test that agent handles None response from API."""
    # Mock a None response
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=None))]
    mock_client.chat.completions.create.return_value = mock_response
    mock_openai.return_value = mock_client
    
    agent = LLMAgent(persona, config)
    
    result = agent.generate_idea(
        topic="Test topic",
        history=[],
        phase="Separate"
    )
    
    assert result == "[No response generated]"


def test_agent_string_representations(persona: Persona, config: Config) -> None:
    """Test agent string representations."""
    agent = LLMAgent(persona, config)
    
    assert str(agent) == "LLMAgent(Test Agent)"
    assert "Test Agent" in repr(agent)
    assert "test-model" in repr(agent)
