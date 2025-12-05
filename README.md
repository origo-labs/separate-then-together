# Separate-Then-Together

A professional, research-oriented multi-agent collaboration system based on the **Separate-Then-Together** framework for persona-based planning and brainstorming.

## Overview

This system implements the empirically-validated **Separate-Then-Together** collaboration strategy, which combines:

1. **Separate Phase (Divergence)**: Agents work in epistemic isolation, seeing only their own contributions, to maximize conceptual diversity
2. **Collaborative Phase (Convergence)**: Agents share full conversation history to synthesize, critique, and integrate ideas

This two-phase approach produces the highest **Novelty** and **Depth** scores compared to purely separate or purely collaborative strategies.

## Key Features

- 🎭 **Persona Selection**: Automatic selection of dissimilar personas using embedding-based cosine similarity
- 🔄 **Multiple Strategies**: Separate, Collaborative, and Separate-Then-Together modes
- 🤖 **OpenAI-Compatible**: Works with OpenAI, Ollama, OpenRouter, and other compatible APIs
- 📊 **Rich Output**: Export results to JSON or Markdown
- 🧪 **Research-Ready**: Modular architecture for experimentation
- ✅ **Type-Safe**: Full type hints and Pydantic validation

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd separate-then-together

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

## Quick Start

### Using Ollama (Local Models)

```bash
# Start Ollama and pull a model
ollama pull gemma3:4b

# Set environment variables
export OPENAI_API_KEY=ollama
export OPENAI_BASE_URL=http://localhost:11434/v1
export OPENAI_MODEL=gemma3:4b

# Run the CLI
separate-then-together --topic "Plan a microservices migration"
```

### Using OpenAI

```bash
# Set your OpenAI API key
export OPENAI_API_KEY=sk-...
export OPENAI_MODEL=gpt-4o

# Run the CLI
separate-then-together --topic "Design a new authentication system"
```

## Usage

### Command-Line Interface

```bash
# Basic usage with default personas
separate-then-together --topic "Your planning task here"

# Use a specific strategy
separate-then-together --strategy separate --topic "Your task"
separate-then-together --strategy collaborative --topic "Your task"
separate-then-together --strategy separate-together --topic "Your task"

# Customize turn counts
separate-then-together \
  --topic "Your task" \
  --separate-turns 10 \
  --collab-turns 20

# Export results
separate-then-together \
  --topic "Your task" \
  --output results.json

# Quiet mode (no progress output)
separate-then-together --topic "Your task" --quiet
```

### Python API

```python
from separate_then_together import (
    Config,
    LLMAgent,
    Persona,
    PersonaSelector,
    SessionEngine,
    SeparateTogetherStrategy,
)

# 1. Create configuration
config = Config.from_env()

# 2. Define personas
personas = [
    Persona(
        name="System Architect",
        system_prompt="You are a system architect focused on..."
    ),
    Persona(
        name="Security Engineer",
        system_prompt="You are a security engineer focused on..."
    ),
]

# 3. Select dissimilar pair
selector = PersonaSelector(personas, config.embedding_model)
persona1, persona2 = selector.select_dissimilar_pair()

# 4. Create agents
agent1 = LLMAgent(persona1, config)
agent2 = LLMAgent(persona2, config)

# 5. Create strategy
strategy = SeparateTogetherStrategy(
    separate_turns=10,
    collab_turns=20
)

# 6. Run session
session = SessionEngine(
    agent1, agent2,
    topic="Your planning task",
    strategy=strategy,
    config=config
)
results = session.run()

# 7. Export results
session.export_to_json("results.json")
session.export_to_markdown("results.md")
```

## Configuration

The system uses environment variables for configuration:

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | API key for OpenAI-compatible endpoint | `ollama` |
| `OPENAI_BASE_URL` | Base URL for API endpoint | `http://localhost:11434/v1` |
| `OPENAI_MODEL` | Model name | `gemma2:2b` |
| `EMBEDDING_MODEL` | Sentence transformer model for persona similarity | `all-MiniLM-L6-v2` |

## Examples

See the `examples/` directory for complete examples:

- `basic_usage.py`: Simple API usage demonstration
- `software_engineering.py`: Comprehensive software planning example with 6 personas

Run an example:

```bash
python examples/software_engineering.py
```

## Research Applications

This framework is designed for research in multi-agent planning and collaboration:

### Persona Diversity Experiments

```python
# Compare similar vs dissimilar persona pairs
selector = PersonaSelector(personas)

# Most dissimilar (maximum diversity)
p1, p2 = selector.select_dissimilar_pair()

# Most similar (minimum diversity)
p1, p2 = selector.select_similar_pair()
```

### Strategy Comparison

```python
# Test different collaboration modes
strategies = [
    SeparateStrategy(separate_turns=30),
    CollaborativeStrategy(collab_turns=30),
    SeparateTogetherStrategy(separate_turns=15, collab_turns=15),
]

for strategy in strategies:
    session = SessionEngine(agent1, agent2, topic, strategy, config)
    results = session.run()
    # Analyze results...
```

### Custom Personas

```python
# Define domain-specific personas
personas = [
    Persona("ML Engineer", "You specialize in machine learning..."),
    Persona("Data Scientist", "You focus on statistical analysis..."),
    Persona("MLOps Engineer", "You ensure ML systems are production-ready..."),
]
```

## Architecture

```
src/separate_then_together/
├── __init__.py          # Package exports
├── config.py            # Configuration management
├── persona.py           # Persona selection with embeddings
├── agent.py             # LLM agent implementation
├── strategies.py        # Collaboration strategies
├── session.py           # Session orchestration
└── cli.py               # Command-line interface

tests/
├── test_persona.py      # Persona selection tests
├── test_strategies.py   # Strategy tests
├── test_agent.py        # Agent tests
└── test_session.py      # Integration tests

examples/
├── basic_usage.py       # Simple example
└── software_engineering.py  # Comprehensive example
```

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=separate_then_together

# Run specific test file
pytest tests/test_persona.py
```

## Development

```bash
# Install development dependencies
uv sync --all-extras

# Format code
black src/ tests/

# Type checking
mypy src/

# Linting
ruff check src/ tests/
```

## Research Background

This implementation is based on research into persona-based multi-agent collaboration for brainstorming and planning tasks. Key findings:

- **Dissimilar personas** (low cosine similarity) produce higher novelty and depth
- **Separate-Then-Together** strategy outperforms pure separate or collaborative modes
- **Epistemic isolation** in the divergence phase is critical for diversity
- **Cross-domain synthesis** in the convergence phase improves implementation-readiness

See `docs/SUMMARY.md` and `docs/conceptual.md` for detailed research background.

## License

[Add your license here]

## Citation

If you use this framework in your research, please cite:

```bibtex
[Add citation information]
```

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For issues, questions, or contributions, please open an issue on GitHub.
