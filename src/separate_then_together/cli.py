"""Command-line interface for the multi-agent system."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from separate_then_together.agent import LLMAgent
from separate_then_together.config import Config
from separate_then_together.persona import Persona, PersonaSelector
from separate_then_together.session import SessionEngine
from separate_then_together.report import ReportGenerator
from separate_then_together.strategies import (
    SeparateStrategy,
    CollaborativeStrategy,
    SeparateTogetherStrategy,
)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser.
    
    Returns:
        Configured ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Persona-based Multi-Agent Collaboration System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default software engineering personas
  separate-then-together --topic "Plan a microservices migration"
  
  # Use specific strategy
  separate-then-together --strategy separate --topic "Design a new API"
  
  # Export results
  separate-then-together --topic "Refactor authentication" --output results.json
  
  # Use custom model
  OPENAI_MODEL=llama3.1 separate-then-together --topic "Plan database migration"
        """
    )
    
    parser.add_argument(
        "--topic",
        type=str,
        required=True,
        help="The planning task or brainstorming topic"
    )
    
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["separate", "collaborative", "separate-together"],
        default="separate-together",
        help="Collaboration strategy (default: separate-together)"
    )
    
    parser.add_argument(
        "--separate-turns",
        type=int,
        default=10,
        help="Number of turns in Separate phase (default: 10)"
    )
    
    parser.add_argument(
        "--collab-turns",
        type=int,
        default=20,
        help="Number of turns in Collaborative phase (default: 20)"
    )
    
    parser.add_argument(
        "--summary-threshold",
        type=int,
        default=5,
        help="Number of turns before summarizing history (default: 5)"
    )
    
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (JSON or Markdown)"
    )
    
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate a comprehensive design document after the session"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Override OPENAI_MODEL environment variable"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        help="Override OPENAI_BASE_URL environment variable"
    )
    
    parser.add_argument(
        "--verbose-prompts",
        action="store_true",
        help="Log full prompts sent to LLM (for debugging)"
    )
    
    return parser


def get_default_personas() -> list[Persona]:
    """Get default software engineering personas.
    
    Returns:
        List of Persona objects
    """
    return [
        Persona(
            name="System Architect",
            system_prompt=(
                "You are a seasoned System Architect focused on high-level design, "
                "modularity, performance scaling, and dependency management. Your goal "
                "is to ensure the new framework foundation is structurally sound and "
                "future-proof. You think in terms of components, interfaces, and system "
                "boundaries."
            )
        ),
        Persona(
            name="Security Engineer",
            system_prompt=(
                "You are a dedicated Security Engineer focused on risk assessment, "
                "vulnerability mitigation, and compliance with data privacy standards "
                "(e.g., GDPR, SOC2). Your priority is securing all refactored components. "
                "You think in terms of threat models, attack surfaces, and defense in depth."
            )
        ),
        Persona(
            name="Frontend Designer",
            system_prompt=(
                "You are a meticulous Frontend Designer focusing on user interaction flow, "
                "UX research findings, and component library migration. Your goal is "
                "to ensure visual consistency and accessibility in the new framework. "
                "You think in terms of user journeys, design systems, and accessibility standards."
            )
        ),
    ]


def main() -> int:
    """Main entry point for the CLI.
    
    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = create_parser()
    args = parser.parse_args()
    
    try:
        # Create configuration
        config = Config.from_env()
        
        # Override with CLI arguments if provided
        if args.model:
            config.openai_model = args.model
        if args.base_url:
            config.openai_base_url = args.base_url
        
        config.separate_turns = args.separate_turns
        config.collab_turns = args.collab_turns
        config.summary_threshold = args.summary_threshold
        config.verbose_prompts = args.verbose_prompts
        
        # Get personas and select dissimilar pair
        personas = get_default_personas()
        selector = PersonaSelector(personas, config.embedding_model)
        
        persona1, persona2 = selector.select_dissimilar_pair(verbose=not args.quiet)
        
        # Create agents
        agent1 = LLMAgent(persona1, config)
        agent2 = LLMAgent(persona2, config)
        
        # Create strategy
        if args.strategy == "separate":
            strategy = SeparateStrategy(
                separate_turns=args.separate_turns,
                collab_turns=0
            )
        elif args.strategy == "collaborative":
            strategy = CollaborativeStrategy(
                separate_turns=0,
                collab_turns=args.collab_turns
            )
        else:  # separate-together
            strategy = SeparateTogetherStrategy(
                separate_turns=args.separate_turns,
                collab_turns=args.collab_turns
            )
        
        # Create and run session
        session = SessionEngine(agent1, agent2, args.topic, strategy, config)
        session.run(verbose=not args.quiet)
        
        # Export results if requested
        if args.output:
            if args.output.suffix == ".json":
                session.export_to_json(args.output)
            elif args.output.suffix in [".md", ".markdown"]:
                session.export_to_markdown(args.output)
            else:
                # Default to JSON
                session.export_to_json(args.output.with_suffix(".json"))
        
        # Generate comprehensive report if requested
        if args.generate_report:
            print(f"\nGeneratin comprehensive design document...")
            # Reuse agent1 as the reporter
            generator = ReportGenerator(agent1, config)
            topic = args.topic
            history = session.full_history
            
            report_content = generator.generate_report(topic, history)
            
            output_path = Path("DESIGN_DOCUMENT.md")
            if args.output:
                 # If user specified output dir, verify and use it
                 # If output is file (e.g. output/results.json), use parent dir
                 if args.output.suffix:
                     output_dir = args.output.parent
                 else:
                     output_dir = args.output
                     
                 output_dir.mkdir(parents=True, exist_ok=True)
                 output_path = output_dir / "DESIGN_DOCUMENT.md"
            
            with open(output_path, "w") as f:
                f.write(report_content)
                
            print(f"\n✓ Comprehensive plan saved to: {output_path}")
        
        return 0
    
    except KeyboardInterrupt:
        print("\n\nSession interrupted by user.")
        return 1
    
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        if not args.quiet:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
