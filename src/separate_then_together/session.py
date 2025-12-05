"""Session engine for orchestrating multi-agent collaboration."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from separate_then_together.agent import LLMAgent
from separate_then_together.config import Config
from separate_then_together.strategies import CollaborationStrategy, SeparateTogetherStrategy


class SessionEngine:
    """Orchestrates multi-agent collaboration sessions.
    
    The session engine manages turn-taking, history filtering based on
    collaboration strategy, and result collection.
    """
    
    def __init__(
        self,
        agent1: LLMAgent,
        agent2: LLMAgent,
        topic: str,
        strategy: CollaborationStrategy,
        config: Config,
    ):
        """Initialize the session engine.
        
        Args:
            agent1: First agent
            agent2: Second agent
            topic: The planning task or brainstorming topic
            strategy: Collaboration strategy to use
            config: System configuration
        """
        self.agents = {agent1.name: agent1, agent2.name: agent2}
        self.agent_names = [agent1.name, agent2.name]
        self.topic = topic
        self.strategy = strategy
        self.config = config
        
        self.full_history: List[Dict[str, Any]] = []
        self.session_start: Optional[datetime] = None
        self.session_end: Optional[datetime] = None
    
    def run(self, verbose: bool = True) -> List[Dict[str, Any]]:
        """Execute the collaboration session.
        
        Args:
            verbose: If True, print progress information
        
        Returns:
            List of all generated ideas/planning steps
        """
        self.session_start = datetime.now()
        
        if verbose:
            self._print_session_header()
        
        # Track phase transitions for Separate-Then-Together
        last_phase = self.strategy.get_phase_name()
        
        while self.strategy.should_continue():
            # Check for phase transition
            current_phase = self.strategy.get_phase_name()
            if current_phase != last_phase and verbose:
                self._print_phase_transition(current_phase)
                last_phase = current_phase
            
            # Determine which agent acts next (alternating)
            agent_index = self.strategy.current_turn % 2
            current_agent_name = self.agent_names[agent_index]
            current_agent = self.agents[current_agent_name]
            
            # Get filtered history based on strategy
            filtered_history = self.strategy.filter_history(
                self.full_history, 
                current_agent_name
            )
            
            # Execute agent turn
            if verbose:
                phase_name = self.strategy.get_phase_name()
                turn_num = self.strategy.current_turn + 1
                print(f"\n[Turn {turn_num}] {current_agent_name} ({phase_name} phase)")
                print("-" * 60)
            
            idea = current_agent.generate_idea(
                self.topic, 
                filtered_history, 
                self.strategy.get_phase_name()
            )
            
            # Record the action
            action = {
                "turn": self.strategy.current_turn + 1,
                "role": current_agent_name,
                "phase": self.strategy.get_phase_name(),
                "content": idea,
                "timestamp": datetime.now().isoformat(),
            }
            self.full_history.append(action)
            
            if verbose:
                print(f"{idea}")
            
            # Increment turn counter
            self.strategy.increment_turn()
        
        self.session_end = datetime.now()
        
        if verbose:
            self._print_session_footer()
        
        return self.full_history
    
    def get_results_by_phase(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get results grouped by collaboration phase.
        
        Returns:
            Dictionary mapping phase names to lists of ideas
        """
        results: Dict[str, List[Dict[str, Any]]] = {}
        
        for entry in self.full_history:
            phase = entry.get("phase", "Unknown")
            if phase not in results:
                results[phase] = []
            results[phase].append(entry)
        
        return results
    
    def export_to_json(self, filepath: Path) -> None:
        """Export session results to JSON file.
        
        Args:
            filepath: Path to save the JSON file
        """
        data = {
            "topic": self.topic,
            "agents": self.agent_names,
            "strategy": self.strategy.__class__.__name__,
            "config": self.config.to_dict(),
            "session_start": self.session_start.isoformat() if self.session_start else None,
            "session_end": self.session_end.isoformat() if self.session_end else None,
            "results": self.full_history,
            "summary": {
                "total_turns": len(self.full_history),
                "by_phase": {
                    phase: len(entries) 
                    for phase, entries in self.get_results_by_phase().items()
                },
                "by_agent": {
                    agent: len([e for e in self.full_history if e["role"] == agent])
                    for agent in self.agent_names
                },
            }
        }
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"\n✓ Results exported to: {filepath}")
    
    def export_to_markdown(self, filepath: Path) -> None:
        """Export session results to Markdown file.
        
        Args:
            filepath: Path to save the Markdown file
        """
        lines = [
            f"# Multi-Agent Planning Session",
            f"\n## Topic\n\n{self.topic}",
            f"\n## Configuration\n",
            f"- **Strategy**: {self.strategy.__class__.__name__}",
            f"- **Agents**: {', '.join(self.agent_names)}",
            f"- **Model**: {self.config.openai_model}",
            f"- **Session Start**: {self.session_start.isoformat() if self.session_start else 'N/A'}",
            f"- **Session End**: {self.session_end.isoformat() if self.session_end else 'N/A'}",
            f"\n## Results\n",
        ]
        
        # Group by phase
        by_phase = self.get_results_by_phase()
        
        for phase, entries in by_phase.items():
            lines.append(f"\n### {phase} Phase ({len(entries)} turns)\n")
            
            for entry in entries:
                turn = entry.get("turn", "?")
                role = entry.get("role", "Unknown")
                content = entry.get("content", "")
                lines.append(f"**Turn {turn} - {role}**\n\n{content}\n")
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            f.write("\n".join(lines))
        
        print(f"✓ Results exported to: {filepath}")
    
    def _print_session_header(self) -> None:
        """Print session start information."""
        print("\n" + "=" * 70)
        print("MULTI-AGENT PLANNING SESSION")
        print("=" * 70)
        print(f"\nTopic: {self.topic}")
        print(f"Strategy: {self.strategy.__class__.__name__}")
        print(f"Agents: {self.agent_names[0]} × {self.agent_names[1]}")
        print(f"Model: {self.config.openai_model}")
        print("\n" + "=" * 70)
    
    def _print_phase_transition(self, new_phase: str) -> None:
        """Print phase transition notification.
        
        Args:
            new_phase: Name of the new phase
        """
        print("\n" + "=" * 70)
        print(f"PHASE TRANSITION → {new_phase.upper()}")
        print("=" * 70)
    
    def _print_session_footer(self) -> None:
        """Print session completion information."""
        duration = (
            (self.session_end - self.session_start).total_seconds()
            if self.session_start and self.session_end
            else 0
        )
        
        print("\n" + "=" * 70)
        print("SESSION COMPLETE")
        print("=" * 70)
        print(f"Total turns: {len(self.full_history)}")
        print(f"Duration: {duration:.1f} seconds")
        
        by_phase = self.get_results_by_phase()
        print(f"\nTurns by phase:")
        for phase, entries in by_phase.items():
            print(f"  {phase}: {len(entries)}")
        
        print("=" * 70 + "\n")
