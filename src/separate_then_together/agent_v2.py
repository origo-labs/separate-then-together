"""Enhanced agent with history summarization for better collaborative planning."""

from typing import List, Dict, Any
from openai import OpenAI

from separate_then_together.config import Config
from separate_then_together.persona import Persona


class LLMAgentV2:
    """Enhanced LLM agent with intelligent history management.
    
    This version addresses the context window limitation by:
    1. Summarizing covered topics instead of sending full history
    2. Providing structured topic lists to prevent repetition
    3. Including progress indicators for better planning
    """
    
    def __init__(self, persona: Persona, config: Config):
        self.persona = persona
        self.config = config
        self.client = OpenAI(
            api_key=config.openai_api_key,
            base_url=config.openai_base_url,
        )
    
    @property
    def name(self) -> str:
        return self.persona.name
    
    def generate_idea(
        self, 
        topic: str, 
        history: List[Dict[str, Any]], 
        phase: str,
        current_turn: int = 0,
        total_turns: int = 0
    ) -> str:
        """Generate a planning step with enhanced context management."""
        messages = self._build_messages_v2(topic, history, phase, current_turn, total_turns)
        
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
    
    def _build_messages_v2(
        self, 
        topic: str, 
        history: List[Dict[str, Any]], 
        phase: str,
        current_turn: int,
        total_turns: int
    ) -> List[Dict[str, str]]:
        """Build messages with intelligent history summarization."""
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.persona.system_prompt}
        ]
        
        if phase == "Separate":
            # Separate phase: show own history
            context_summary = self._format_own_history(history)
            
            user_message = (
                f"TASK: {topic}\\n\\n"
                f"PHASE: Independent Ideation (Turn {current_turn + 1}/{total_turns})\\n"
                f"INSTRUCTION: Generate ONE detailed, specific planning step or idea. "
                f"Work independently - DO NOT reference your partner's ideas. "
                f"Focus on your unique domain expertise.\\n\\n"
            )
            
            if context_summary:
                user_message += f"Your Previous Ideas:\\n{context_summary}\\n\\n"
            
            user_message += "Generate your next planning step:"
            messages.append({"role": "user", "content": user_message})
        
        else:  # Collaborative phase
            # Use summarized history instead of full text
            topic_summary = self._summarize_covered_topics(history)
            recent_context = self._format_recent_history(history, max_entries=5)
            
            progress_pct = int((current_turn / total_turns) * 100) if total_turns > 0 else 0
            
            user_message = (
                f"TASK: {topic}\\n\\n"
                f"PHASE: Collaborative Discussion (Turn {current_turn + 1}/{total_turns}, {progress_pct}% complete)\\n\\n"
                f"TOPICS ALREADY COVERED:\\n{topic_summary}\\n\\n"
                f"RECENT DISCUSSION (last 5 turns):\\n{recent_context}\\n\\n"
                f"INSTRUCTION: Generate ONE refined or integrated planning step that:\\n"
                f"  1. Builds upon or integrates ideas from both agents\\n"
                f"  2. Does NOT repeat topics already covered above\\n"
                f"  3. Moves the plan forward toward completion\\n"
                f"  4. Focuses on synthesis and cross-domain integration\\n\\n"
            )
            
            # Add stage-specific guidance
            if progress_pct < 40:
                user_message += "STAGE: Early collaboration - focus on exploring connections between ideas.\\n"
            elif progress_pct < 75:
                user_message += "STAGE: Mid collaboration - focus on integration and identifying dependencies.\\n"
            else:
                user_message += "STAGE: Final collaboration - focus on consolidation and creating a coherent roadmap.\\n"
            
            user_message += "\\nGenerate your next planning step:"
            messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _summarize_covered_topics(self, history: List[Dict[str, Any]]) -> str:
        """Create a concise summary of topics already covered."""
        if not history:
            return "None yet."
        
        topics_by_agent: Dict[str, List[str]] = {}
        
        for entry in history:
            agent = entry.get("role", "Unknown")
            content = entry.get("content", "")
            
            # Extract topic from first line or first sentence
            lines = content.split('\\n')
            topic = ""
            for line in lines:
                line = line.strip()
                if line and not line.startswith('['):  # Skip role prefixes
                    # Look for bold markers or just take first substantial line
                    if '**' in line:
                        # Extract text between ** markers
                        parts = line.split('**')
                        if len(parts) >= 2:
                            topic = parts[1]
                    else:
                        topic = line
                    break
            
            if not topic:
                topic = content[:80] + "..." if len(content) > 80 else content
            
            # Clean up
            topic = topic.replace('*', '').replace('#', '').strip()
            if len(topic) > 100:
                topic = topic[:100] + "..."
            
            if agent not in topics_by_agent:
                topics_by_agent[agent] = []
            topics_by_agent[agent].append(topic)
        
        # Format summary
        summary_lines = []
        for agent, topics in topics_by_agent.items():
            summary_lines.append(f"\\n{agent}:")
            for i, topic in enumerate(topics, 1):
                summary_lines.append(f"  {i}. {topic}")
        
        return "\\n".join(summary_lines)
    
    def _format_recent_history(self, history: List[Dict[str, Any]], max_entries: int = 5) -> str:
        """Format only the most recent history entries."""
        if not history:
            return "None yet."
        
        recent = history[-max_entries:] if len(history) > max_entries else history
        
        formatted = []
        for entry in recent:
            role = entry.get("role", "Unknown")
            content = entry.get("content", "")
            # Truncate long content
            if len(content) > 300:
                content = content[:300] + "... [truncated]"
            formatted.append(f"[{role}]: {content}")
        
        return "\\n\\n".join(formatted)
    
    def _format_own_history(self, history: List[Dict[str, Any]]) -> str:
        """Format the agent's own history for display."""
        if not history:
            return "None yet."
        
        formatted = []
        for i, entry in enumerate(history, 1):
            content = entry.get('content', '')
            # Extract just the topic/title
            first_line = content.split('\\n')[0]
            if len(first_line) > 100:
                first_line = first_line[:100] + "..."
            formatted.append(f"{i}. {first_line}")
        
        return "\\n".join(formatted)
    
    def __str__(self) -> str:
        return f"LLMAgentV2({self.name})"
    
    def __repr__(self) -> str:
        return f"LLMAgentV2(persona={self.persona.name}, model={self.config.openai_model})"
