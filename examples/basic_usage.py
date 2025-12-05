"""Basic usage example for researchers."""

from separate_then_together import (
    Config,
    LLMAgent,
    Persona,
    PersonaSelector,
    SessionEngine,
    SeparateTogetherStrategy,
)


def main() -> None:
    """Demonstrate basic API usage."""
    
    # 1. Create configuration
    config = Config.from_env()
    
    # 2. Define personas
    personas = [
        Persona(
            name="Creative Thinker",
            system_prompt=(
                "You are a creative thinker who generates novel, unconventional ideas. "
                "You prioritize innovation and thinking outside the box."
            )
        ),
        Persona(
            name="Analytical Thinker",
            system_prompt=(
                "You are an analytical thinker who focuses on logical reasoning, "
                "data-driven decisions, and systematic problem-solving."
            )
        ),
    ]
    
    # 3. Select dissimilar pair (optional - you can also choose manually)
    selector = PersonaSelector(personas, config.embedding_model)
    persona1, persona2 = selector.select_dissimilar_pair(verbose=True)
    
    # 4. Create agents
    agent1 = LLMAgent(persona1, config)
    agent2 = LLMAgent(persona2, config)
    
    # 5. Define your planning task
    topic = "How can we improve team collaboration in a remote-first company?"
    
    # 6. Create collaboration strategy
    strategy = SeparateTogetherStrategy(
        separate_turns=6,   # 3 turns per agent in divergent phase
        collab_turns=10     # 5 turns per agent in convergent phase
    )
    
    # 7. Create and run session
    session = SessionEngine(agent1, agent2, topic, strategy, config)
    results = session.run(verbose=True)
    
    # 8. Access results
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    
    by_phase = session.get_results_by_phase()
    for phase, entries in by_phase.items():
        print(f"\n{phase} Phase: {len(entries)} ideas")
    
    # 9. Export results (optional)
    # session.export_to_json(Path("results.json"))
    # session.export_to_markdown(Path("results.md"))


if __name__ == "__main__":
    main()
