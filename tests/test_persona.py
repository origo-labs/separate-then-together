"""Tests for persona selection and similarity calculations."""

import pytest
from separate_then_together.persona import Persona, PersonaSelector


def test_persona_creation() -> None:
    """Test creating a Persona."""
    persona = Persona(
        name="Test Persona",
        system_prompt="You are a test persona."
    )
    
    assert persona.name == "Test Persona"
    assert persona.system_prompt == "You are a test persona."


def test_persona_selector_requires_multiple_personas() -> None:
    """Test that PersonaSelector requires at least 2 personas."""
    persona = Persona("Single", "Single persona")
    
    with pytest.raises(ValueError, match="At least 2 personas"):
        PersonaSelector([persona])


def test_persona_selector_embeddings() -> None:
    """Test that embeddings are computed correctly."""
    personas = [
        Persona("P1", "This is about software architecture and system design."),
        Persona("P2", "This is about security and vulnerability assessment."),
    ]
    
    selector = PersonaSelector(personas)
    embeddings = selector.embeddings
    
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0  # Has embedding dimensions


def test_similarity_matrix() -> None:
    """Test similarity matrix calculation."""
    personas = [
        Persona("Similar1", "Software development and programming"),
        Persona("Similar2", "Software engineering and coding"),
        Persona("Different", "Cooking and culinary arts"),
    ]
    
    selector = PersonaSelector(personas)
    similarities = selector.calculate_similarity_matrix()
    
    # Should have 3 pairs
    assert len(similarities) == 3
    
    # Similar personas should have higher similarity
    similar_pair = "Similar1 × Similar2"
    different_pair1 = "Similar1 × Different"
    
    assert similar_pair in similarities
    assert different_pair1 in similarities
    assert similarities[similar_pair] > similarities[different_pair1]


def test_select_dissimilar_pair() -> None:
    """Test selecting the most dissimilar pair."""
    personas = [
        Persona("Tech", "Software engineering and programming"),
        Persona("Art", "Painting and visual arts"),
        Persona("Science", "Physics and mathematics"),
    ]
    
    selector = PersonaSelector(personas)
    p1, p2 = selector.select_dissimilar_pair(verbose=False)
    
    assert p1 in personas
    assert p2 in personas
    assert p1 != p2


def test_select_similar_pair() -> None:
    """Test selecting the most similar pair."""
    personas = [
        Persona("Dev1", "Python programming"),
        Persona("Dev2", "Python development"),
        Persona("Chef", "Cooking recipes"),
    ]
    
    selector = PersonaSelector(personas)
    p1, p2 = selector.select_similar_pair(verbose=False)
    
    assert p1 in personas
    assert p2 in personas
    assert p1 != p2
    
    # Should select the two dev personas
    assert {p1.name, p2.name} == {"Dev1", "Dev2"}


def test_get_persona_by_name() -> None:
    """Test retrieving persona by name."""
    personas = [
        Persona("Alice", "Prompt A"),
        Persona("Bob", "Prompt B"),
    ]
    
    selector = PersonaSelector(personas)
    
    alice = selector.get_persona_by_name("Alice")
    assert alice is not None
    assert alice.name == "Alice"
    
    charlie = selector.get_persona_by_name("Charlie")
    assert charlie is None
