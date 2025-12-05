"""LLM Agent implementation with OpenAI-compatible API support."""

from typing import List, Dict, Any, Optional
import time

from openai import OpenAI

from separate_then_together.config import Config
from separate_then_together.persona import Persona


class LLMAgent:
    """Represents an LLM-powered agent with a specific persona.
    
    The agent generates ideas and planning steps using an OpenAI-compatible
    API (supports OpenAI, Ollama, OpenRouter, etc.).
    """
    
    def __init__(self, persona: Persona, config: Config):
        """Initialize the LLM agent.
        
        Args:
            persona: Persona defining the agent's expertise and role
            config: Configuration for API access and model parameters
        """
        self.persona = persona
        self.config = config
        
        # Initialize OpenAI client with custom base URL for Ollama support
        self.client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        )
    
    @property
    def name(self) -> str:
        """Get the agent's name.
        
        Returns:
            Persona name
        """
        return self.persona.name
    
    def generate_idea(
        self, 
        topic: str, 
        history: List[Dict[str, Any]], 
        phase: str
    ) -> str:
        """Generate a planning step or idea using the LLM.
        
        The context and instructions vary based on the collaboration phase:
        - Separate: Agent sees only its own history, instructed to work independently
        - Collaborative: Agent sees full history, instructed to build on others' ideas
        
        Args:
            topic: The planning task or brainstorming topic
            history: Filtered conversation history (based on strategy)
            phase: Current collaboration phase ("Separate" or "Collaborative")
        
        Returns:
            Generated idea or planning step as a string
        """
        messages = self._build_messages(topic, history, phase)
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=messages,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            
            content = response.choices[0].message.content
            if content is None:
                return "[No response generated]"
            
            return content.strip()
        
        except Exception as e:
            error_msg = f"[Error generating idea: {str(e)}]"
            print(f"⚠️  {self.name}: {error_msg}")
            return error_msg
    
    def _build_messages(
        self, 
        topic: str, 
        history: List[Dict[str, Any]], 
        phase: str
    ) -> List[Dict[str, str]]:
        """Build the message list for the LLM API call.
        
        Args:
            topic: The planning task
            history: Filtered conversation history
            phase: Current collaboration phase
        
        Returns:
            List of message dictionaries for the API
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.persona.system_prompt}
        ]
        
        if phase == "Separate":
            # Separate phase: emphasize independence
            context_summary = self._format_own_history(history)
            
            user_message = (
                f"TASK: {topic}\n\n"
                f"PHASE: Independent Ideation\n"
                f"INSTRUCTION: Generate ONE detailed, specific planning step or idea. "
                f"Work independently - DO NOT reference or build upon your partner's ideas. "
                f"Focus on your unique domain expertise.\n\n"
            )
            
            if context_summary:
                user_message += f"Your Previous Ideas:\n{context_summary}\n\n"
            
            user_message += "Generate your next planning step:"
            
            messages.append({"role": "user", "content": user_message})
        
        else:  # Collaborative phase
            # Collaborative phase: encourage synthesis
            user_message = (
                f"TASK: {topic}\n\n"
                f"PHASE: Collaborative Discussion\n"
                f"INSTRUCTION: Generate ONE refined or integrated planning step. "
                f"Reference, critique, and build upon the existing ideas from both agents. "
                f"Focus on synthesis and cross-domain integration.\n\n"
            )
            
            # Add conversation history
            if history:
                user_message += "Conversation History:\n"
                user_message += self._format_full_history(history)
                user_message += "\n\n"
            
            user_message += "Generate your next planning step:"
            
            messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _format_own_history(self, history: List[Dict[str, Any]]) -> str:
        """Format the agent's own history for display.
        
        Args:
            history: List of history entries (should only contain own entries)
        
        Returns:
            Formatted string of previous ideas
        """
        if not history:
            return "None yet."
        
        formatted = []
        for i, entry in enumerate(history, 1):
            formatted.append(f"{i}. {entry.get('content', '')}")
        
        return "\n".join(formatted)
    
    def _format_full_history(self, history: List[Dict[str, Any]]) -> str:
        """Format the full conversation history for display.
        
        Args:
            history: List of all history entries
        
        Returns:
            Formatted string of conversation
        """
        if not history:
            return "None yet."
        
        formatted = []
        for entry in history:
            role = entry.get("role", "Unknown")
            content = entry.get("content", "")
            formatted.append(f"[{role}]: {content}")
        
        return "\n\n".join(formatted)
    
    def __str__(self) -> str:
        return f"LLMAgent({self.name})"
    
    def __repr__(self) -> str:
        return f"LLMAgent(persona={self.persona.name}, model={self.config.openai_model})"
