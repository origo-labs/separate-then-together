"""Report generation module for creating comprehensive design documents."""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

from separate_then_together.agent import LLMAgent
from separate_then_together.config import Config

class ReportGenerator:
    """Generates a structured comprehensive plan from session history."""
    
    def __init__(self, agent: LLMAgent, config: Config):
        """Initialize the report generator.
        
        Args:
            agent: LLMAgent instance to use for generation (reusing client/config)
            config: Configuration settings
        """
        self.agent = agent
        self.config = config
        
    def generate_report(self, topic: str, history: List[Dict[str, Any]], metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate a full comprehensive report.
        
        Args:
            topic: The main topic/goal of the planning session
            history: Full conversation history
            metadata: Optional session metadata (e.g. personas, config settings)
            
        Returns:
            Markdown formatted report
        """
        print(f"\n[ReportGenerator] Generating comprehensive plan for: {topic}")
        
        # 1. Prepare Context
        context_str = self.agent._build_hybrid_history(history, topic)
        
        # 2. Determine Document Structure & Title
        print("[ReportGenerator] Step 1/3: Analyzing content and generating structure...")
        structure = self._generate_structure_and_title(topic, context_str)
        doc_title = structure.get("title", f"Comprehensive Report: {topic}")
        outline = structure.get("sections", [])
        
        if not outline:
            print("⚠️  Could not determine structure, falling back to default.")
            outline = ["Executive Summary", "Key Discussion Points", "Proposed Solutions", "Next Steps"]
            
        print(f"[ReportGenerator] Document Type: {doc_title}")
        print(f"[ReportGenerator] Generated {len(outline)} sections.")
        
        # 3. Add Table of Contents
        print("[ReportGenerator] Step 2/3: Generating Table of Contents...")
        toc_lines = ["\n## Table of Contents"]
        for i, section_title in enumerate(outline):
            # Create a simple anchor link (lowercase, spaces to hyphens, remove special chars)
            anchor = section_title.lower().replace(" ", "-")
            # Remove non-alphanumeric chars (except hyphens) for better compatibility
            anchor = "".join(c for c in anchor if c.isalnum() or c == "-")
            toc_lines.append(f"{i+1}. [{section_title}](#{anchor})")
        
        full_report = f"# {doc_title}\n\n**Topic:** {topic}\n\n" + "\n".join(toc_lines) + "\n\n"
        
        # 4. Generate Sections
        print("[ReportGenerator] Step 3/3: Synthesizing section content...")
        
        for i, section_title in enumerate(outline):
            print(f"  - Writing section {i+1}/{len(outline)}: {section_title}")
            section_content = self._generate_section(section_title, doc_title, topic, context_str)
            
            # Post-process: Remove the section title if the LLM included it despite instructions
            # Check for lines starting with #, ##, ### that match the section title closely
            lines = section_content.split('\n')
            if lines and (section_title.lower() in lines[0].lower() or lines[0].strip().strip("#").strip().lower() == section_title.lower()):
                section_content = "\n".join(lines[1:]).strip()
            
            # Ensure we use the clean section title for the anchor
            full_report += f"## {section_title}\n\n{section_content}\n\n"
            
        # 5. Add Metadata/Appendix
        if metadata:
            print("[ReportGenerator] Step 4/3: Appending session metadata...")
            full_report += "## Appendix: Generation Metadata\n\n"
            
            if "agents" in metadata:
                full_report += "**Participating Personas:**\n"
                for agent in metadata["agents"]:
                    full_report += f"- {agent}\n"
                full_report += "\n"
                
            if "config" in metadata:
                full_report += "**Configuration:**\n"
                # Filter out sensitive keys if needed, for now just dump it nicely
                for key, value in metadata["config"].items():
                    if "key" not in key.lower(): # Basic filter for API keys
                         full_report += f"- **{key}:** {value}\n"
                full_report += "\n"
        
            if "session_start" in metadata:
                full_report += f"**Session Start:** {metadata['session_start']}\n"
            
        return full_report

    def _generate_structure_and_title(self, topic: str, context: str) -> Dict[str, Any]:
        """Determine the best document title and Table of Contents."""
        prompt = (
            f"Review this summary of a multi-agent collaboration session:\n\n"
            f"{context}\n\n"
            f"TASK: Determine the most appropriate type of formal document to generate from this discussion "
            f"(e.g., 'Software Architecture Design', 'Strategic Roadmap', 'Research Summary', 'Project Proposal').\n"
            f"Then, propose a logical Table of Contents (sections) for this document.\n"
            f"Structure the sections by logical themes strategies, or components, NOT chronologically.\n"
            f"Do NOT include generic 'Introduction' or 'Conclusion' sections unless critical.\n\n"
            f"Return ONLY a JSON object with this format:\n"
            f'{{\n  "title": "The Document Title",\n  "sections": ["Section 1", "Section 2", "Section 3"]\n}}'
        )
        
        try:
            response = self.agent.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are an expert documentarian capable of synthesizing complex discussions into professional artifacts."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content.strip()
            
            # Simple fallback regex if JSON parse fails
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                content = json_match.group(0)
                
            return json.loads(content)
            
        except Exception as e:
            print(f"⚠️  Structure generation failed: {e}")
            return {}

    def _generate_section(self, section_title: str, doc_title: str, topic: str, context: str) -> str:
        """Generate content for a specific section."""
        prompt = (
            f"Write the '{section_title}' section of the '{doc_title}' for the topic: {topic}.\n\n"
            f"SOURCE MATERIAL (Conversation History):\n{context}\n\n"
            f"INSTRUCTIONS:\n"
            f"- Synthesize all relevant points discussed about this specific section.\n"
            f"- Adopt a professional tone suitable for a '{doc_title}'.\n"
            f"- Resolve any initial conflicts by presenting the *final* evolved solution.\n"
            f"- Use technical language and standard Markdown formatting.\n"
            f"- Do NOT introduce yourself. Just write the document content.\n"
            f"- Do NOT include the section title as a heading. Start directly with the content.\n"
        )
        
        try:
            response = self.agent.client.chat.completions.create(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are a senior technical writer creating a formal design document."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=1500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️  Section generation failed for {section_title}: {e}")
            return "[Content generation failed]"
