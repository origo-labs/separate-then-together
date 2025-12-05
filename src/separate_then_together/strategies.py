"""Collaboration strategies for multi-agent interaction."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class CollaborationStrategy(ABC):
    """Abstract base class for collaboration strategies.
    
    Strategies control how agents access conversation history,
    implementing different A2A (Agent-to-Agent) dynamics.
    """
    
    def __init__(self, separate_turns: int = 10, collab_turns: int = 20):
        """Initialize the strategy.
        
        Args:
            separate_turns: Number of turns for separate/divergent phase
            collab_turns: Number of turns for collaborative/convergent phase
        """
        self.separate_turns = separate_turns
        self.collab_turns = collab_turns
        self.current_turn = 0
    
    @abstractmethod
    def filter_history(
        self, 
        full_history: List[Dict[str, Any]], 
        current_agent_name: str
    ) -> List[Dict[str, Any]]:
        """Filter conversation history based on strategy.
        
        Args:
            full_history: Complete conversation history
            current_agent_name: Name of the agent requesting history
        
        Returns:
            Filtered history appropriate for the current agent and phase
        """
        pass
    
    @abstractmethod
    def get_phase_name(self) -> str:
        """Get the name of the current collaboration phase.
        
        Returns:
            Phase name (e.g., "Separate", "Collaborative")
        """
        pass
    
    def should_continue(self) -> bool:
        """Check if the collaboration should continue.
        
        Returns:
            True if more turns are needed, False otherwise
        """
        total_turns = self.separate_turns + self.collab_turns
        return self.current_turn < total_turns
    
    def increment_turn(self) -> None:
        """Increment the turn counter."""
        self.current_turn += 1


class SeparateStrategy(CollaborationStrategy):
    """Separate strategy: Agents work in epistemic isolation (divergence).
    
    Each agent only sees their own prior contributions, promoting
    independent ideation and maximum conceptual diversity.
    """
    
    def filter_history(
        self, 
        full_history: List[Dict[str, Any]], 
        current_agent_name: str
    ) -> List[Dict[str, Any]]:
        """Return only the current agent's own history.
        
        Args:
            full_history: Complete conversation history
            current_agent_name: Name of the agent requesting history
        
        Returns:
            History containing only the current agent's contributions
        """
        return [
            entry for entry in full_history 
            if entry.get("role") == current_agent_name
        ]
    
    def get_phase_name(self) -> str:
        """Get phase name.
        
        Returns:
            "Separate"
        """
        return "Separate"
    
    def should_continue(self) -> bool:
        """Check if more separate turns are needed.
        
        Returns:
            True if current_turn < separate_turns
        """
        return self.current_turn < self.separate_turns


class CollaborativeStrategy(CollaborationStrategy):
    """Collaborative strategy: Agents share full history (convergence).
    
    Agents have access to all prior contributions, enabling them to
    reference, critique, and build upon each other's ideas.
    """
    
    def filter_history(
        self, 
        full_history: List[Dict[str, Any]], 
        current_agent_name: str
    ) -> List[Dict[str, Any]]:
        """Return the complete conversation history.
        
        Args:
            full_history: Complete conversation history
            current_agent_name: Name of the agent requesting history (unused)
        
        Returns:
            Unfiltered full history
        """
        return full_history
    
    def get_phase_name(self) -> str:
        """Get phase name.
        
        Returns:
            "Collaborative"
        """
        return "Collaborative"
    
    def should_continue(self) -> bool:
        """Check if more collaborative turns are needed.
        
        Returns:
            True if current_turn < collab_turns
        """
        return self.current_turn < self.collab_turns


class SeparateTogetherStrategy(CollaborationStrategy):
    """Separate-Then-Together: Two-phase hybrid strategy (optimal).
    
    Phase 1 (Separate): Agents work independently for divergence
    Phase 2 (Collaborative): Agents collaborate for synthesis
    
    This strategy empirically produces the highest Novelty and Depth scores.
    """
    
    def __init__(self, separate_turns: int = 10, collab_turns: int = 20):
        """Initialize the two-phase strategy.
        
        Args:
            separate_turns: Number of turns in the Separate phase
            collab_turns: Number of turns in the Collaborative phase
        """
        super().__init__(separate_turns, collab_turns)
        self._in_separate_phase = True
    
    def filter_history(
        self, 
        full_history: List[Dict[str, Any]], 
        current_agent_name: str
    ) -> List[Dict[str, Any]]:
        """Filter history based on current phase.
        
        Args:
            full_history: Complete conversation history
            current_agent_name: Name of the agent requesting history
        
        Returns:
            Filtered history (own only) in Separate phase,
            full history in Collaborative phase
        """
        # Check if we should transition to collaborative phase
        if self.current_turn >= self.separate_turns:
            self._in_separate_phase = False
        
        if self._in_separate_phase:
            # Separate phase: only own history
            return [
                entry for entry in full_history 
                if entry.get("role") == current_agent_name
            ]
        else:
            # Collaborative phase: full history
            return full_history
    
    def get_phase_name(self) -> str:
        """Get the name of the current phase.
        
        Returns:
            "Separate" or "Collaborative" depending on current turn
        """
        if self.current_turn < self.separate_turns:
            return "Separate"
        else:
            return "Collaborative"
    
    def is_transitioning(self) -> bool:
        """Check if this is the transition turn from Separate to Collaborative.
        
        Returns:
            True if this is the first turn of the Collaborative phase
        """
        return self.current_turn == self.separate_turns
