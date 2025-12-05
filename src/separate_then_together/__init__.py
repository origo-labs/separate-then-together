"""
Separate-Then-Together: Persona-based Multi-Agent Collaboration System

A research framework for multi-agent planning using the Separate-Then-Together
collaboration strategy, inspired by human-centered design principles.
"""

from separate_then_together.agent import LLMAgent
from separate_then_together.config import Config
from separate_then_together.persona import Persona, PersonaSelector
from separate_then_together.session import SessionEngine
from separate_then_together.report import ReportGenerator
from separate_then_together.strategies import (
    CollaborationStrategy,
    CollaborativeStrategy,
    SeparateStrategy,
    SeparateTogetherStrategy,
)

__version__ = "0.1.0"

__all__ = [
    "LLMAgent",
    "Config",
    "Persona",
    "PersonaSelector",
    "SessionEngine",
    "ReportGenerator",
    "CollaborationStrategy",
    "CollaborativeStrategy",
    "SeparateStrategy",
    "SeparateTogetherStrategy",
]
