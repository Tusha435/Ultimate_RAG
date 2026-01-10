"""Assemble context from retrieval results."""

from typing import Optional, Any
from loguru import logger

from ultimate_rag.models.query import Query, SearchResult, RetrievalResult
from ultimate_rag.models.chunk import Chunk, ChunkStore
from ultimate_rag.config import RetrievalConfig


class ContextAssembler:
    """Assemble context from retrieval results."""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()

    def assemble(
        self,
        query: Query,
        results: list[SearchResult],
        chunk_store: dict[str, dict],
        formula_store: Optional[dict[str, dict]] = None,
        diagram_store: Optional[dict[str, dict]] = None,
        top_k: Optional[int] = None
    ) -> RetrievalResult:
        """Assemble complete retrieval result with context."""
        top_k = top_k or self.config.default_top_k
        top_results = results[:top_k]

        retrieval_result = RetrievalResult(
            query=query,
            results=top_results,
            total_results=len(results)
        )

        for result in top_results:
            if result.source_type == "text" or result.source_type == "keyword":
                retrieval_result.text_results.append(result)
            elif result.source_type == "formula":
                retrieval_result.formula_results.append(result)
            elif result.source_type == "diagram":
                retrieval_result.diagram_results.append(result)

        self._populate_content(top_results, chunk_store, formula_store, diagram_store)

        if self.config.enable_context_expansion:
            expanded = self._expand_context(top_results, chunk_store)
            retrieval_result.context_chunks = expanded

        retrieval_result.expanded_context = self._build_context_string(
            top_results, chunk_store, formula_store
        )

        retrieval_result.confidence = self._compute_confidence(top_results)

        return retrieval_result

    def _populate_content(
        self,
        results: list[SearchResult],
        chunk_store: dict[str, dict],
        formula_store: Optional[dict[str, dict]],
        diagram_store: Optional[dict[str, dict]]
    ):
        """Populate result content from stores."""
        for result in results:
            if result.chunk_id and result.chunk_id in chunk_store:
                chunk_data = chunk_store[result.chunk_id]
                result.content = chunk_data.get("content", "")
                result.page_number = chunk_data.get("metadata", {}).get("page_numbers", [None])[0]
                result.section = chunk_data.get("metadata", {}).get("heading")

            elif result.formula_id and formula_store and result.formula_id in formula_store:
                formula_data = formula_store[result.formula_id]
                result.content = formula_data.get("latex", "")
                result.page_number = formula_data.get("source_page")

            elif result.diagram_id and diagram_store:
                for diag in diagram_store if isinstance(diagram_store, list) else diagram_store.values():
                    if diag.get("id") == result.diagram_id:
                        result.content = diag.get("caption", "")
                        result.page_number = diag.get("metadata", {}).get("page")
                        break

    def _expand_context(
        self,
        results: list[SearchResult],
        chunk_store: dict[str, dict]
    ) -> list[dict]:
        """Expand context with surrounding chunks."""
        expanded = []
        seen_chunks = set()

        chunk_sequence = self._build_chunk_sequence(chunk_store)

        for result in results:
            if not result.chunk_id or result.chunk_id in seen_chunks:
                continue

            chunk_data = chunk_store.get(result.chunk_id)
            if not chunk_data:
                continue

            seen_chunks.add(result.chunk_id)
            expanded.append(chunk_data)

            try:
                idx = chunk_sequence.index(result.chunk_id)
            except ValueError:
                continue

            window = self.config.context_window_size

            for i in range(max(0, idx - window), idx):
                neighbor_id = chunk_sequence[i]
                if neighbor_id not in seen_chunks and neighbor_id in chunk_store:
                    seen_chunks.add(neighbor_id)
                    expanded.append(chunk_store[neighbor_id])

            for i in range(idx + 1, min(len(chunk_sequence), idx + window + 1)):
                neighbor_id = chunk_sequence[i]
                if neighbor_id not in seen_chunks and neighbor_id in chunk_store:
                    seen_chunks.add(neighbor_id)
                    expanded.append(chunk_store[neighbor_id])

        return expanded

    def _build_chunk_sequence(self, chunk_store: dict[str, dict]) -> list[str]:
        """Build ordered sequence of chunks."""
        chunks = list(chunk_store.items())

        def sort_key(item):
            chunk_id, data = item
            pages = data.get("metadata", {}).get("page_numbers", [0])
            page = pages[0] if pages else 0
            return (page, chunk_id)

        chunks.sort(key=sort_key)
        return [chunk_id for chunk_id, _ in chunks]

    def _build_context_string(
        self,
        results: list[SearchResult],
        chunk_store: dict[str, dict],
        formula_store: Optional[dict[str, dict]]
    ) -> str:
        """Build formatted context string for LLM."""
        context_parts = []

        text_results = [r for r in results if r.chunk_id]
        for i, result in enumerate(text_results[:5]):
            chunk_data = chunk_store.get(result.chunk_id, {})
            content = chunk_data.get("content", result.content)
            page = result.page_number or "?"
            section = result.section or "Unknown section"

            context_parts.append(
                f"[Source {i+1} - Page {page}, {section}]\n{content}\n"
            )

        formula_results = [r for r in results if r.formula_id]
        if formula_results and formula_store:
            context_parts.append("\n[Relevant Formulas]")
            for result in formula_results[:3]:
                formula_data = formula_store.get(result.formula_id, {})
                latex = formula_data.get("latex", result.content)
                plain = formula_data.get("plain_text", "")
                context_parts.append(f"  ${latex}$")
                if plain:
                    context_parts.append(f"  ({plain})")

        return "\n".join(context_parts)

    def _compute_confidence(self, results: list[SearchResult]) -> float:
        """Compute confidence score for retrieval."""
        if not results:
            return 0.0

        top_score = results[0].score if results else 0.0

        score_variance = 0.0
        if len(results) > 1:
            scores = [r.score for r in results]
            mean_score = sum(scores) / len(scores)
            score_variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)

        score_gap = (results[0].score - results[-1].score) if len(results) > 1 else 0

        confidence = min(1.0, (
            0.5 * top_score +
            0.3 * (1.0 / (1.0 + score_variance)) +
            0.2 * min(1.0, score_gap)
        ))

        return confidence

    def get_sources_summary(self, retrieval_result: RetrievalResult) -> list[dict]:
        """Get summary of sources for citation."""
        sources = []

        for result in retrieval_result.results[:10]:
            source = {
                "id": result.id,
                "type": result.source_type,
                "score": result.score,
                "page": result.page_number,
                "section": result.section
            }

            if result.content:
                source["preview"] = result.content[:200] + "..." if len(result.content) > 200 else result.content

            sources.append(source)

        return sources
