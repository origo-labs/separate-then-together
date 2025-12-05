"""Configuration management for the multi-agent system."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Configuration for the multi-agent collaboration system.
    
    Attributes:
        openai_api_key: API key for OpenAI-compatible endpoint
        openai_base_url: Base URL for API endpoint (supports Ollama, OpenAI, etc.)
        openai_model: Model name to use for LLM calls
        embedding_model: Sentence transformer model for persona similarity
        separate_turns: Number of turns in the Separate phase
        collab_turns: Number of turns in the Collaborative phase
        temperature: Sampling temperature for LLM
        max_tokens: Maximum tokens for LLM response
    """
    
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", "ollama"))
    openai_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    )
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gemma3:4b"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    separate_turns: int = 5
    collab_turns: int = 10
    temperature: float = 0.7
    max_tokens: int = 2000
    
    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable must be set. "
                "For Ollama, use 'ollama' as a placeholder."
            )
        
        if self.separate_turns < 0 or self.collab_turns < 0:
            raise ValueError("Turn counts must be non-negative")
        
        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables.
        
        Returns:
            Config instance with values from environment
        """
        return cls()
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary.
        
        Returns:
            Dictionary representation of configuration
        """
        return {
            "openai_api_key": "***" if self.openai_api_key else None,
            "openai_base_url": self.openai_base_url,
            "openai_model": self.openai_model,
            "embedding_model": self.embedding_model,
            "separate_turns": self.separate_turns,
            "collab_turns": self.collab_turns,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
