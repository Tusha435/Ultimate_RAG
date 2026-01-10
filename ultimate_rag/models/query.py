"""Query and result models for retrieval system."""

from typing import Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class QueryType(str, Enum):
    """Types of queries."""

    TEXT = "text"
    FORMULA = "formula"
    DIAGRAM = "diagram"
    MIXED = "mixed"
    KEYWORD = "keyword"


class QueryIntent(str, Enum):
    """Detected intent of a query."""

    DEFINITION = "definition"
    EXPLANATION = "explanation"
    DERIVATION = "derivation"
    EXAMPLE = "example"
    COMPARISON = "comparison"
    FORMULA_LOOKUP = "formula_lookup"
    PROOF = "proof"
    APPLICATION = "application"
    VISUALIZATION = "visualization"
    CALCULATION = "calculation"
    GENERAL = "general"


class Query(BaseModel):
    """Represents a user query."""

    id: str = Field(default_factory=lambda: f"Q_{datetime.now().strftime('%Y%m%d%H%M%S%f')}")
    raw_text: str
    processed_text: Optional[str] = None
    query_type: QueryType = QueryType.TEXT
    intent: QueryIntent = QueryIntent.GENERAL

    keywords: list[str] = Field(default_factory=list)
    entities: list[str] = Field(default_factory=list)
    formulas: list[str] = Field(default_factory=list)

    embedding: Optional[list[float]] = None
    filters: dict[str, Any] = Field(default_factory=dict)

    domain_hint: Optional[str] = None
    context: Optional[str] = None
    conversation_history: list[str] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def has_formula_query(self) -> bool:
        return len(self.formulas) > 0 or self.query_type == QueryType.FORMULA

    def with_embedding(self, embedding: list[float]) -> "Query":
        """Create a copy with embedding attached."""
        return self.model_copy(update={"embedding": embedding})


class SearchResult(BaseModel):
    """Single search result from any index."""

    id: str
    source_type: str
    content: str
    score: float
    rank: int = 0

    chunk_id: Optional[str] = None
    formula_id: Optional[str] = None
    diagram_id: Optional[str] = None
    table_id: Optional[str] = None

    page_number: Optional[int] = None
    section: Optional[str] = None
    highlight: Optional[str] = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    def __lt__(self, other: "SearchResult") -> bool:
        return self.score > other.score


class RetrievalResult(BaseModel):
    """Combined retrieval result with fusion scores."""

    query: Query
    results: list[SearchResult] = Field(default_factory=list)
    total_results: int = 0

    text_results: list[SearchResult] = Field(default_factory=list)
    formula_results: list[SearchResult] = Field(default_factory=list)
    diagram_results: list[SearchResult] = Field(default_factory=list)
    keyword_results: list[SearchResult] = Field(default_factory=list)

    fusion_scores: dict[str, float] = Field(default_factory=dict)
    retrieval_time_ms: float = 0.0

    context_chunks: list[Any] = Field(default_factory=list)
    expanded_context: Optional[str] = None

    confidence: float = 0.0
    explanation: Optional[str] = None

    def get_top_k(self, k: int = 10) -> list[SearchResult]:
        """Get top k results by score."""
        return sorted(self.results, reverse=True)[:k]

    def get_unique_pages(self) -> list[int]:
        """Get unique page numbers from results."""
        pages = set()
        for result in self.results:
            if result.page_number:
                pages.add(result.page_number)
        return sorted(pages)

    def get_sources_summary(self) -> str:
        """Get a summary of sources."""
        pages = self.get_unique_pages()
        sources = []
        for result in self.results[:5]:
            source = f"- {result.source_type}"
            if result.page_number:
                source += f", Page {result.page_number}"
            if result.section:
                source += f", {result.section}"
            source += f" (score: {result.score:.2f})"
            sources.append(source)
        return "\n".join(sources)


class QueryResult(BaseModel):
    """Final query result with generated response."""

    query: Query
    retrieval: RetrievalResult
    answer: str
    answer_type: str = "generated"

    sources: list[dict[str, Any]] = Field(default_factory=list)
    formulas_used: list[str] = Field(default_factory=list)
    diagrams_used: list[str] = Field(default_factory=list)

    confidence_breakdown: dict[str, float] = Field(default_factory=dict)
    processing_time_ms: float = 0.0
    model_used: Optional[str] = None
    tokens_used: Optional[int] = None

    def to_display_dict(self) -> dict:
        """Convert to dictionary for display."""
        return {
            "question": self.query.raw_text,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence_breakdown,
            "formulas": self.formulas_used,
            "diagrams": self.diagrams_used
        }
