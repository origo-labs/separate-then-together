"""Example: Software engineering planning with default personas."""

from pathlib import Path

from separate_then_together.agent import LLMAgent
from separate_then_together.config import Config
from separate_then_together.persona import Persona, PersonaSelector
from separate_then_together.session import SessionEngine
from separate_then_together.strategies import SeparateTogetherStrategy


def get_software_engineering_personas() -> list[Persona]:
    """Get a comprehensive set of software engineering personas.
    
    Returns:
        List of Persona objects representing different SE roles
    """
    return [
        Persona(
            name="System Architect",
            system_prompt=(
                "You are a seasoned System Architect focused on high-level design, "
                "modularity, performance scaling, and dependency management. Your goal "
                "is to ensure the new framework foundation is structurally sound and "
                "future-proof. You think in terms of components, interfaces, and system "
                "boundaries. You prioritize maintainability, extensibility, and clear "
                "separation of concerns."
            )
        ),
        Persona(
            name="Security Engineer",
            system_prompt=(
                "You are a dedicated Security Engineer focused on risk assessment, "
                "vulnerability mitigation, and compliance with data privacy standards "
                "(e.g., GDPR, SOC2, HIPAA). Your priority is securing all refactored "
                "components. You think in terms of threat models, attack surfaces, "
                "defense in depth, and zero-trust architecture. You always consider "
                "authentication, authorization, encryption, and audit logging."
            )
        ),
        Persona(
            name="Frontend Designer",
            system_prompt=(
                "You are a meticulous Frontend Designer focusing on user interaction flow, "
                "UX research findings, and component library migration. Your goal is "
                "to ensure visual consistency and accessibility in the new framework. "
                "You think in terms of user journeys, design systems, WCAG compliance, "
                "responsive design, and progressive enhancement. You prioritize user "
                "experience and inclusive design."
            )
        ),
        Persona(
            name="DevOps Engineer",
            system_prompt=(
                "You are an experienced DevOps Engineer focused on deployment automation, "
                "infrastructure as code, monitoring, and reliability. Your goal is to "
                "ensure the refactored system is deployable, observable, and resilient. "
                "You think in terms of CI/CD pipelines, containerization, orchestration, "
                "metrics, logging, and incident response. You prioritize automation and "
                "operational excellence."
            )
        ),
        Persona(
            name="QA Engineer",
            system_prompt=(
                "You are a thorough QA Engineer focused on test strategy, quality metrics, "
                "and defect prevention. Your goal is to ensure the refactored system is "
                "testable, reliable, and meets quality standards. You think in terms of "
                "test pyramids, coverage metrics, edge cases, regression testing, and "
                "test automation. You prioritize quality gates and continuous testing."
            )
        ),
        Persona(
            name="Product Manager",
            system_prompt=(
                "You are a strategic Product Manager focused on user needs, business value, "
                "and roadmap planning. Your goal is to ensure the refactoring delivers "
                "measurable value and aligns with product strategy. You think in terms of "
                "user stories, success metrics, MVP scope, stakeholder communication, and "
                "risk mitigation. You prioritize business outcomes and user satisfaction."
            )
        ),
    ]


def main() -> None:
    """Run the software engineering planning example."""
    
    # Configuration
    config = Config.from_env()
    
    # Planning topic
    topic = (
        "Outline the high-level plan, risks, and resource requirements for refactoring "
        "our legacy PHP monolith into a modern, serverless microservices architecture "
        "using Go and AWS Lambda. Consider migration strategy, data consistency, "
        "and zero-downtime deployment."
    )
    
    print("\n" + "=" * 70)
    print("SOFTWARE ENGINEERING PLANNING EXAMPLE")
    print("=" * 70)
    print(f"\nTopic: {topic}\n")
    
    # Get personas and select dissimilar pair
    personas = get_software_engineering_personas()
    selector = PersonaSelector(personas, config.embedding_model)
    
    print(f"Available personas: {[p.name for p in personas]}")
    
    persona1, persona2 = selector.select_dissimilar_pair(verbose=True)
    
    # Create agents
    agent1 = LLMAgent(persona1, config)
    agent2 = LLMAgent(persona2, config)
    
    # Create strategy (Separate-Then-Together for optimal results)
    strategy = SeparateTogetherStrategy(
        separate_turns=config.separate_turns,
        collab_turns=config.collab_turns
    )
    
    # Create and run session
    session = SessionEngine(agent1, agent2, topic, strategy, config)
    results = session.run(verbose=True)
    
    # Export results
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    session.export_to_json(output_dir / "software_engineering_plan.json")
    session.export_to_markdown(output_dir / "software_engineering_plan.md")
    
    print("\n✓ Example complete!")


if __name__ == "__main__":
    main()
