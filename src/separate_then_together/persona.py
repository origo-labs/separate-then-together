"""Persona selection and management using embedding-based similarity."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Persona:
    """Represents an agent persona with a name and system prompt.

    Attributes:
        name: Unique identifier for the persona
        system_prompt: Detailed description of the persona's expertise and role
    """

    name: str
    system_prompt: str

    def __str__(self) -> str:
        return f"Persona({self.name})"

    def __repr__(self) -> str:
        return f"Persona(name='{self.name}', prompt_length={len(self.system_prompt)})"


class PersonaSelector:
    """Selects heterogeneous persona pairs based on cosine similarity of embeddings.

    This class implements the persona selection methodology from the research paper,
    using embedding similarity to quantify semantic distance between personas.
    """

    def __init__(self, personas: List[Persona], embedding_model: str = "all-MiniLM-L6-v2"):
        """Initialize the persona selector.

        Args:
            personas: List of Persona objects to select from
            embedding_model: Name of the sentence-transformers model to use
        """
        if len(personas) < 2:
            raise ValueError("At least 2 personas are required for selection")

        self.personas = personas
        self.embedding_model_name = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self._embeddings: Optional[np.ndarray] = None

    @property
    def embeddings(self) -> np.ndarray:
        """Get or compute embeddings for all personas.

        Returns:
            Numpy array of embeddings, shape (n_personas, embedding_dim)
        """
        if self._embeddings is None:
            prompts = [p.system_prompt for p in self.personas]
            self._embeddings = self.model.encode(prompts, convert_to_tensor=False)
        return self._embeddings

    def calculate_similarity_matrix(self) -> Dict[str, float]:
        """Calculate cosine similarity between all persona pairs.

        Returns:
            Dictionary mapping persona pair names to similarity scores.
            Similarity of 1.0 = highly similar, 0.0 = orthogonal/dissimilar.
        """
        similarity_matrix = cosine_similarity(self.embeddings)

        results = {}
        for i in range(len(self.personas)):
            for j in range(i + 1, len(self.personas)):
                pair_name = f"{self.personas[i].name} × {self.personas[j].name}"
                similarity = float(similarity_matrix[i, j])
                results[pair_name] = similarity

        return results

    def select_dissimilar_pair(self, verbose: bool = True) -> Tuple[Persona, Persona]:
        """Find the pair with the lowest semantic similarity (maximum divergence).

        Args:
            verbose: If True, print selection details

        Returns:
            Tuple of (persona1, persona2) with minimum similarity
        """
        similarities = self.calculate_similarity_matrix()

        # Find pair with minimum similarity
        min_pair_name = min(similarities, key=similarities.get)  # type: ignore
        min_similarity = similarities[min_pair_name]

        # Extract persona names
        p1_name, p2_name = min_pair_name.split(" × ")
        p1 = next(p for p in self.personas if p.name == p1_name)
        p2 = next(p for p in self.personas if p.name == p2_name)

        if verbose:
            print(f"\n{'=' * 60}")
            print("PERSONA SELECTION")
            print(f"{'=' * 60}")
            print("\nAll Pair Similarities:")
            for pair, sim in sorted(similarities.items(), key=lambda x: x[1]):
                print(f"  {pair}: {sim:.3f}")
            print(f"\n✓ Selected Dissimilar Pair: {min_pair_name}")
            print(f"  Cosine Similarity: {min_similarity:.3f}")
            print("  Rationale: Maximum semantic divergence promotes")
            print("             conceptual diversity and cross-domain synthesis.")
            print(f"{'=' * 60}\n")

        return p1, p2

    def select_similar_pair(self, verbose: bool = True) -> Tuple[Persona, Persona]:
        """Find the pair with the highest semantic similarity.

        Args:
            verbose: If True, print selection details

        Returns:
            Tuple of (persona1, persona2) with maximum similarity
        """
        similarities = self.calculate_similarity_matrix()

        # Find pair with maximum similarity
        max_pair_name = max(similarities, key=similarities.get)  # type: ignore
        max_similarity = similarities[max_pair_name]

        # Extract persona names
        p1_name, p2_name = max_pair_name.split(" × ")
        p1 = next(p for p in self.personas if p.name == p1_name)
        p2 = next(p for p in self.personas if p.name == p2_name)

        if verbose:
            print(f"\n{'=' * 60}")
            print("PERSONA SELECTION")
            print(f"{'=' * 60}")
            print(f"\nSelected Similar Pair: {max_pair_name}")
            print(f"Cosine Similarity: {max_similarity:.3f}")
            print(f"{'=' * 60}\n")

        return p1, p2

    def get_persona_by_name(self, name: str) -> Optional[Persona]:
        """Get a persona by name.

        Args:
            name: Name of the persona to retrieve

        Returns:
            Persona object if found, None otherwise
        """
        return next((p for p in self.personas if p.name == name), None)
