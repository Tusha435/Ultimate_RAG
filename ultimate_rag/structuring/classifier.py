"""Content classification for chunks and documents."""

import re
from typing import Optional, Any
from loguru import logger

from ultimate_rag.models.chunk import Chunk, ChunkType
from ultimate_rag.config import StructuringConfig


class ContentClassifier:
    """Classify content type, domain, and difficulty."""

    def __init__(self, config: Optional[StructuringConfig] = None):
        self.config = config or StructuringConfig()
        self._classifier_model = None

        self.domain_keywords = {
            "physics": [
                "force", "mass", "acceleration", "velocity", "momentum",
                "energy", "work", "power", "wave", "field", "potential",
                "electric", "magnetic", "quantum", "relativity", "particle",
                "thermodynamics", "entropy", "optics", "mechanics"
            ],
            "mathematics": [
                "theorem", "proof", "lemma", "corollary", "axiom",
                "function", "derivative", "integral", "limit", "series",
                "matrix", "vector", "topology", "algebra", "geometry",
                "calculus", "equation", "polynomial", "differential"
            ],
            "chemistry": [
                "molecule", "atom", "bond", "reaction", "compound",
                "element", "orbital", "electron", "ion", "acid", "base",
                "oxidation", "reduction", "equilibrium", "solution",
                "concentration", "mole", "catalyst", "polymer"
            ],
            "data_science": [
                "model", "algorithm", "dataset", "feature", "prediction",
                "regression", "classification", "clustering", "neural",
                "training", "validation", "accuracy", "precision", "recall",
                "optimization", "gradient", "loss", "batch", "epoch"
            ],
            "statistics": [
                "probability", "distribution", "variance", "deviation",
                "mean", "median", "mode", "hypothesis", "confidence",
                "correlation", "regression", "sample", "population",
                "significance", "bayesian", "likelihood", "estimation"
            ]
        }

        self.difficulty_indicators = {
            "introductory": [
                "basic", "introduction", "fundamental", "simple",
                "beginner", "elementary", "first", "overview"
            ],
            "intermediate": [
                "application", "example", "practice", "typical",
                "standard", "common", "general"
            ],
            "advanced": [
                "advanced", "complex", "rigorous", "theoretical",
                "proof", "derivation", "generalization", "abstract"
            ]
        }

    def classify_chunk(self, chunk: Chunk) -> dict[str, Any]:
        """Classify a chunk's type, domain, and difficulty."""
        content = chunk.content.lower()

        chunk_type = self._classify_type(content, chunk)
        domain = self._classify_domain(content)
        difficulty = self._classify_difficulty(content)
        concepts = self._extract_concepts(content, domain)

        return {
            "chunk_type": chunk_type,
            "domain": domain,
            "difficulty": difficulty,
            "concepts": concepts
        }

    def _classify_type(self, content: str, chunk: Chunk) -> ChunkType:
        """Classify the structural type of content."""
        type_patterns = [
            (ChunkType.DEFINITION, r'\b(defin(e|ition|ed)|is\s+called|we\s+define|denote)\b'),
            (ChunkType.THEOREM, r'\b(theorem|lemma|corollary|proposition)\b'),
            (ChunkType.PROOF, r'\b(proof|prove|qed|∎|we\s+show\s+that)\b'),
            (ChunkType.EXAMPLE, r'\b(example|consider|suppose|let\s+us)\b'),
            (ChunkType.EXERCISE, r'\b(exercise|problem|question|find|calculate|show\s+that)\b'),
        ]

        for chunk_type, pattern in type_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return chunk_type

        formula_indicators = ['=', '∫', '∑', '∂', '∇', '→']
        if any(ind in chunk.content for ind in formula_indicators):
            return ChunkType.FORMULA_BLOCK

        return chunk.chunk_type

    def _classify_domain(self, content: str) -> str:
        """Classify the domain of content."""
        domain_scores = {}

        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for kw in keywords if kw in content)
            if score > 0:
                domain_scores[domain] = score

        if not domain_scores:
            return "general"

        return max(domain_scores, key=domain_scores.get)

    def _classify_difficulty(self, content: str) -> str:
        """Classify difficulty level."""
        diff_scores = {}

        for level, indicators in self.difficulty_indicators.items():
            score = sum(1 for ind in indicators if ind in content)
            diff_scores[level] = score

        formula_count = content.count('=') + content.count('∫')
        if formula_count > 5:
            diff_scores["advanced"] = diff_scores.get("advanced", 0) + 2

        if re.search(r'\\begin\{proof\}|∎|qed', content, re.IGNORECASE):
            diff_scores["advanced"] = diff_scores.get("advanced", 0) + 3

        if not any(diff_scores.values()):
            return "intermediate"

        return max(diff_scores, key=diff_scores.get)

    def _extract_concepts(self, content: str, domain: str) -> list[str]:
        """Extract key concepts from content."""
        concepts = []

        pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\'s)?)\s+(?:law|theorem|equation|principle|rule|formula|constant)'
        matches = re.findall(pattern, content, re.IGNORECASE)
        concepts.extend(matches)

        domain_keywords = self.domain_keywords.get(domain, [])
        for keyword in domain_keywords:
            if keyword in content:
                concepts.append(keyword)

        physics_concepts = [
            "Newton", "Maxwell", "Einstein", "Schrödinger", "Heisenberg",
            "Boltzmann", "Planck", "Faraday", "Gauss", "Coulomb"
        ]
        for concept in physics_concepts:
            if concept.lower() in content:
                concepts.append(concept)

        return list(set(concepts))[:10]

    def classify_batch(self, chunks: list[Chunk]) -> list[dict[str, Any]]:
        """Classify multiple chunks."""
        return [self.classify_chunk(chunk) for chunk in chunks]

    def enrich_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Enrich chunks with classification metadata."""
        for chunk in chunks:
            classification = self.classify_chunk(chunk)

            chunk.metadata.domain = classification["domain"]
            chunk.metadata.difficulty_level = classification["difficulty"]
            chunk.metadata.concepts = classification["concepts"]

            if classification["chunk_type"] != chunk.chunk_type:
                chunk.chunk_type = classification["chunk_type"]

        return chunks

    def find_related_chunks(
        self,
        chunk: Chunk,
        all_chunks: list[Chunk],
        threshold: float = 0.5
    ) -> list[tuple[Chunk, float]]:
        """Find chunks related to a given chunk."""
        chunk_class = self.classify_chunk(chunk)
        related = []

        for other in all_chunks:
            if other.id == chunk.id:
                continue

            other_class = self.classify_chunk(other)

            score = 0.0

            if chunk_class["domain"] == other_class["domain"]:
                score += 0.3

            common_concepts = set(chunk_class["concepts"]) & set(other_class["concepts"])
            if common_concepts:
                score += 0.2 * len(common_concepts)

            if chunk.metadata.page_numbers and other.metadata.page_numbers:
                page_diff = abs(
                    chunk.metadata.page_numbers[0] - other.metadata.page_numbers[0]
                )
                if page_diff <= 2:
                    score += 0.2

            if chunk.metadata.section_hierarchy and other.metadata.section_hierarchy:
                common_sections = (
                    set(chunk.metadata.section_hierarchy) &
                    set(other.metadata.section_hierarchy)
                )
                if common_sections:
                    score += 0.15 * len(common_sections)

            if score >= threshold:
                related.append((other, min(score, 1.0)))

        return sorted(related, key=lambda x: x[1], reverse=True)
