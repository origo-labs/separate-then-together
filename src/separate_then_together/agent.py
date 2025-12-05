"""LLM Agent implementation with OpenAI-compatible API support and intelligent history management."""

from typing import List, Dict, Any, Optional

from openai import OpenAI

from separate_then_together.config import Config
from separate_then_together.persona import Persona


class LLMAgent:
    """Represents an LLM-powered agent with a specific persona.
    
    The agent generates ideas and planning steps using an OpenAI-compatible
    API (supports OpenAI, Ollama, OpenRouter, etc.).
    
    This version includes intelligent history management to prevent context overflow:
    - Summarizes covered topics instead of sending full history
    - Provides structured topic lists to prevent repetition
    - Includes progress indicators for better planning
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
        
        # Cache for LLM-generated summaries to avoid regenerating
        # Key: "turns_{start}-{end}", Value: summary text
        self._summary_cache: Dict[str, str] = {}
    
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
        phase: str,
        current_turn: int = 0,
        total_turns: int = 0
    ) -> str:
        """Generate a planning step or idea using the LLM.
        
        The context and instructions vary based on the collaboration phase:
        - Separate: Agent sees only its own history, instructed to work independently
        - Collaborative: Agent sees summarized history, instructed to build on others' ideas
        
        Args:
            topic: The planning task or brainstorming topic
            history: Filtered conversation history (based on strategy)
            phase: Current collaboration phase ("Separate" or "Collaborative")
            current_turn: Current turn number (0-indexed)
            total_turns: Total number of turns in the session
        
        Returns:
            Generated idea or planning step as a string
        """
        messages = self._build_messages(topic, history, phase, current_turn, total_turns)
        
        # Log prompts if verbose mode is enabled
        if self.config.verbose_prompts:
            self._log_prompt(messages, current_turn, total_turns, phase)
        
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
        phase: str,
        current_turn: int,
        total_turns: int
    ) -> List[Dict[str, str]]:
        """Build the message list for the LLM API call.
        
        Args:
            topic: The planning task
            history: Filtered conversation history
            phase: Current collaboration phase
            current_turn: Current turn number
            total_turns: Total turns in session
        
        Returns:
            List of message dictionaries for the API
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.persona.system_prompt}
        ]
        
        if phase == "Separate":
            # Separate phase: emphasize independence, show own full history
            user_message = (
                f"<task>{topic}</task>\n\n"
                f"<phase>Independent Ideation</phase>\n"
                f"<progress>\n"
                f"  <turn>{current_turn + 1}/{total_turns}</turn>\n"
                f"</progress>\n\n"
                f"<instruction>\n"
                f"Generate ONE detailed, specific planning step or idea.\n"
                f"Work independently - DO NOT reference or build upon your partner's ideas.\n"
                f"Focus on your unique domain expertise.\n"
                f"</instruction>\n\n"
            )
            
            if history:
                # Show full previous ideas verbatim with XML structure
                user_message += "<previous_ideas>\n"
                for entry in history:
                    turn = entry.get("turn", "?")
                    role = entry.get("role", "Unknown")
                    content = entry.get("content", "")
                    user_message += f'<idea turn="{turn}" agent="{role}">\n{content}\n</idea>\n\n'
                user_message += "</previous_ideas>\n\n"
            
            user_message += "Generate your next planning step:"
            
            messages.append({"role": "user", "content": user_message})
        
        else:  # Collaborative phase
            # Hybrid approach: last 10 messages verbatim + LLM summaries for older chunks
            history_context = self._build_hybrid_history(history, topic)
            
            progress_pct = int((current_turn / total_turns) * 100) if total_turns > 0 else 0
            
            user_message = (
                f"<task>{topic}</task>\n\n"
                f"<phase>Collaborative Discussion</phase>\n"
                f"<progress>\n"
                f"  <turn>{current_turn + 1}/{total_turns}</turn>\n"
                f"  <percentage>{progress_pct}%</percentage>\n"
                f"</progress>\n\n"
                f"{history_context}\n\n"
                f"<instruction>\n"
                f"- Generate ONE refined or integrated planning step that:\n"
                f"  1. Builds upon or integrates ideas from both agents\n"
                f"  2. Advances the plan toward completion\n"
                f"  3. Focuses on synthesis and cross-domain integration\n"
                f"- Refrain from followup questions or requests for clarification.\n"
                f"</instruction>\n\n"
            )
            
            # Add stage-specific guidance based on progress
            if progress_pct < 40:
                guidance = "Early collaboration - focus on exploring connections between ideas."
            elif progress_pct < 75:
                guidance = "Mid collaboration - focus on integration and identifying dependencies."
            else:
                guidance = "Final collaboration - focus on consolidation and creating a coherent roadmap."
            
            user_message += f"<stage_guidance>\n{guidance}\n</stage_guidance>\n\n"
            user_message += "Generate your next planning step:"
            
            messages.append({"role": "user", "content": user_message})
        
        return messages
    
    def _build_hybrid_history(self, history: List[Dict[str, Any]], topic: str) -> str:
        """Build hybrid history using cumulative summarization.
        
        Strategy:
        - Recent messages (N): Shown verbatim, where N is 1 to threshold
        - Older messages: Represented by a SINGLE cumulative summary
        - The summary is updated every 'threshold' turns
        
        Args:
            history: Full conversation history
            topic: The planning task (for context)
        
        Returns:
            XML-formatted history string
        """
        if not history:
            return "<conversation_history>\n  <recent_discussion>\n    (No previous discussion)\n  </recent_discussion>\n</conversation_history>"
        
        threshold = self.config.summary_threshold
        total_turns = len(history)
        
        # Calculate how many recent message to show verbatim
        # We want to show between 1 and threshold messages
        # Example (threshold=5):
        # Turn 1-5: Show 1-5 verbatim (summary covers 0)
        # Turn 6: Show 6 verbatim (summary covers 1-5)
        # ...
        # Turn 10: Show 6-10 verbatim (summary covers 1-5)
        # Turn 11: Show 11 verbatim (summary covers 1-10)
        
        verbatim_count = total_turns % threshold
        if verbatim_count == 0:
            verbatim_count = threshold
            
        summary_end_index = total_turns - verbatim_count
        
        # Case 1: No summary needed (early in conversation)
        if summary_end_index == 0:
            result = "<conversation_history>\n  <recent_discussion>\n"
            for entry in history:
                turn = entry.get("turn", "?")
                role = entry.get("role", "Unknown")
                content = entry.get("content", "")
                result += f'    <message turn="{turn}" agent="{role}">\n      {content}\n    </message>\n\n'
            result += "  </recent_discussion>\n</conversation_history>"
            return result
            
        # Case 2: Summary + Recent
        # Get or generate variable cumulative summary up to summary_end_index
        summary_text = self._get_cumulative_summary(history[:summary_end_index], topic, threshold)
        
        recent_messages = history[summary_end_index:]
        
        result = "<conversation_history>\n"
        result += f"  <earlier_discussion_summary>\n    {summary_text}\n  </earlier_discussion_summary>\n\n"
        result += "  <recent_discussion>\n"
        for entry in recent_messages:
            turn = entry.get("turn", "?")
            role = entry.get("role", "Unknown")
            content = entry.get("content", "")
            result += f'    <message turn="{turn}" agent="{role}">\n      {content}\n    </message>\n\n'
        result += "  </recent_discussion>\n</conversation_history>"
        
        return result

    def _get_cumulative_summary(
        self, 
        history_segment: List[Dict[str, Any]], 
        topic: str,
        threshold: int
    ) -> str:
        """Get or generate a cumulative summary for the given history segment.
        
        Recursive logic:
        - If segment size <= threshold: Simple summary of just those messages
        - If segment size > threshold: Update previous summary with new chunk
        """
        end_turn = len(history_segment)
        cache_key = f"summary_upto_{end_turn}"
        
        # 1. Check cache
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]
            
        # 2. Generate new summary
        # Determine strict base: is there a previous block?
        prev_end_turn = end_turn - threshold
        
        if prev_end_turn <= 0:
            # Base case: First block (e.g., turns 1-5)
            # Summarize from scratch
            summary = self._summarize_chunk_with_llm(history_segment, topic, 1, end_turn)
        else:
            # Recursive case: Update previous summary
            # Get previous summary (this will recurse/use cache)
            prev_summary = self._get_cumulative_summary(history_segment[:prev_end_turn], topic, threshold)
            
            # Get new chunk to add
            new_chunk = history_segment[prev_end_turn:]
            new_chunk_text = self._format_messages_verbatim(new_chunk)
            
            # Generate updated summary
            summary = self._update_summary_with_llm(prev_summary, new_chunk_text, topic, 1, end_turn)
            
        # 3. Cache and return
        self._summary_cache[cache_key] = summary
        return summary
    
    def _update_summary_with_llm(
        self,
        prev_summary: str,
        new_chunk_text: str,
        topic: str,
        start_turn: int,
        end_turn: int
    ) -> str:
        """Update an existing summary with new information."""
        prompt = (
            f"Update the following conversation summary with the latest discussion steps.\n\n"
            f"TASK: {topic}\n\n"
            f"EXISTING SUMMARY (Turns {start_turn}-{end_turn - self.config.summary_threshold}):\n{prev_summary}\n\n"
            f"LATEST DISCUSSION (Turns {end_turn - self.config.summary_threshold + 1}-{end_turn}):\n{new_chunk_text}\n\n"
            f"INSTRUCTION: Create a new, consolidated summary covering turns {start_turn}-{end_turn}. "
            f"Integrate the new information seamlessly. "
            f"Focus on key decisions, plan progressions, and open dependencies. "
            f"Keep it concise (under 400 words)."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes planning discussions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️  Summary update failed: {e}")
            return f"{prev_summary}\n\n[Update failed]\n{new_chunk_text}"
    
    def _summarize_chunk_with_llm(
        self, 
        chunk: List[Dict[str, Any]], 
        topic: str,
        chunk_start: int,
        chunk_end: int
    ) -> str:
        """Use LLM to generate a concise summary of a conversation chunk.
        
        Args:
            chunk: List of conversation entries to summarize
            topic: The planning task
            chunk_start: Starting turn number
            chunk_end: Ending turn number
        
        Returns:
            LLM-generated summary of the chunk
        """
        # Format chunk for summarization
        chunk_text = self._format_messages_verbatim(chunk)
        
        # Create summarization prompt
        summary_prompt = (
            f"Summarize the following {len(chunk)} turns of a multi-agent planning discussion.\n\n"
            f"TASK: {topic}\n\n"
            f"DISCUSSION (turns {chunk_start}-{chunk_end}):\n{chunk_text}\n\n"
            f"Provide a concise summary that captures:\n"
            f"1. Key planning steps or ideas proposed by each agent\n"
            f"2. Important decisions or agreements\n"
            f"3. Any unresolved issues or dependencies\n\n"
            f"Keep the summary under 300 words. Be specific and preserve important details."
        )
        
        try:
            response = self.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes planning discussions."},
                    {"role": "user", "content": summary_prompt}
                ],
                temperature=0.3,  # Lower temperature for consistent summaries
                max_tokens=500,  # Limit summary length
            )
            
            summary = response.choices[0].message.content
            if summary is None:
                return f"[Summary unavailable for turns {chunk_start}-{chunk_end}]"
            
            return summary.strip()
        
        except Exception as e:
            # Fallback: return a simple list of topics if LLM summarization fails
            print(f"⚠️  Summary generation failed for turns {chunk_start}-{chunk_end}: {e}")
            return self._fallback_summary(chunk)
    
    def _format_messages_verbatim(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages exactly as they were, preserving all content.
        
        Args:
            messages: List of conversation entries
        
        Returns:
            Formatted string with full message content
        """
        formatted = []
        for entry in messages:
            turn = entry.get("turn", "?")
            role = entry.get("role", "Unknown")
            content = entry.get("content", "")
            formatted.append(f"Turn {turn} - {role}:\n{content}")
        
        return "\n\n".join(formatted)
    
    def _fallback_summary(self, chunk: List[Dict[str, Any]]) -> str:
        """Create a simple fallback summary if LLM summarization fails.
        
        Args:
            chunk: List of conversation entries
        
        Returns:
            Simple bullet-point summary
        """
        summary_lines = []
        for entry in chunk:
            role = entry.get("role", "Unknown")
            content = entry.get("content", "")
            # Take first 100 chars as topic
            topic = content.split('\n')[0][:100]
            if len(topic) < len(content.split('\n')[0]):
                topic += "..."
            summary_lines.append(f"- {role}: {topic}")
        
        return "\n".join(summary_lines)
    
    def _summarize_covered_topics(self, history: List[Dict[str, Any]]) -> str:
        """Create a concise summary of topics already covered.
        
        Args:
            history: List of all history entries
        
        Returns:
            Formatted string summarizing topics by agent
        """
        if not history:
            return "None yet."
        
        topics_by_agent: Dict[str, List[str]] = {}
        
        for entry in history:
            agent = entry.get("role", "Unknown")
            content = entry.get("content", "")
            
            # Extract topic from first line or first sentence
            lines = content.split('\n')
            topic = ""
            for line in lines:
                line = line.strip()
                if line and not line.startswith('['):  # Skip role prefixes like [Security Engineer]:
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
            summary_lines.append(f"\n{agent}:")
            for i, topic in enumerate(topics, 1):
                summary_lines.append(f"  {i}. {topic}")
        
        return "\n".join(summary_lines)
    
    def _format_recent_history(self, history: List[Dict[str, Any]], max_entries: int = 5) -> str:
        """Format only the most recent history entries.
        
        Args:
            history: List of all history entries
            max_entries: Maximum number of recent entries to include
        
        Returns:
            Formatted string of recent conversation
        """
        if not history:
            return "None yet."
        
        recent = history[-max_entries:] if len(history) > max_entries else history
        
        formatted = []
        for entry in recent:
            role = entry.get("role", "Unknown")
            content = entry.get("content", "")
            # Truncate long content to prevent context overflow
            if len(content) > 300:
                content = content[:300] + "... [truncated]"
            formatted.append(f"[{role}]: {content}")
        
        return "\n\n".join(formatted)
    
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
            content = entry.get('content', '')
            # Extract just the topic/title for brevity
            first_line = content.split('\n')[0]
            if len(first_line) > 100:
                first_line = first_line[:100] + "..."
            formatted.append(f"{i}. {first_line}")
        
        return "\n".join(formatted)
    
    def _log_prompt(
        self, 
        messages: List[Dict[str, str]], 
        current_turn: int, 
        total_turns: int,
        phase: str
    ) -> None:
        """Log the full prompt being sent to the LLM for debugging.
        
        Args:
            messages: The messages being sent to the API
            current_turn: Current turn number
            total_turns: Total turns in session
            phase: Current phase name
        """
        print("\n" + "="*80)
        print(f"PROMPT LOG - {self.name} - Turn {current_turn + 1}/{total_turns} - {phase}")
        print("="*80)
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            
            print(f"\n[{role.upper()}]")
            print("-"*80)
            
            if role == "system":
                # Show first 200 chars of system prompt
                if len(content) > 200:
                    print(content[:200] + f"... ({len(content) - 200} more chars)")
                else:
                    print(content)
            else:
                # Show full user message
                print(content)
            
            # Show token estimate
            token_estimate = len(content) // 4
            print(f"\n[Estimated tokens: ~{token_estimate:,}]")
        
        total_tokens = sum(len(msg.get("content", "")) for msg in messages) // 4
        print("\n" + "="*80)
        print(f"TOTAL ESTIMATED INPUT TOKENS: ~{total_tokens:,}")
        print(f"MAX RESPONSE TOKENS: {self.config.max_tokens:,}")
        print(f"TOTAL TOKENS NEEDED: ~{total_tokens + self.config.max_tokens:,}")
        print("="*80 + "\n")
    
    def __str__(self) -> str:
        return f"LLMAgent({self.name})"
    
    def __repr__(self) -> str:
        return f"LLMAgent(persona={self.persona.name}, model={self.config.openai_model})"
