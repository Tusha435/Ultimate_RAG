"""Search strategies for different modalities."""

from typing import Optional, Any
from loguru import logger

from ultimate_rag.models.query import Query, QueryType, SearchResult
from ultimate_rag.indexing.index_manager import IndexManager
from ultimate_rag.config import RetrievalConfig


class MultiModalSearcher:
    """Search across all modalities based on query type."""

    def __init__(
        self,
        index_manager: IndexManager,
        config: Optional[RetrievalConfig] = None
    ):
        self.index_manager = index_manager
        self.config = config or RetrievalConfig()

    def search(
        self,
        query: Query,
        top_k: Optional[int] = None
    ) -> dict[str, list[SearchResult]]:
        """Execute multi-modal search based on query type."""
        top_k = top_k or self.config.default_top_k

        results = {
            "text": [],
            "formula": [],
            "diagram": [],
            "keyword": []
        }

        results["text"] = self._search_text(query, top_k)

        results["keyword"] = self._search_keywords(query, top_k)

        if query.query_type in [QueryType.FORMULA, QueryType.MIXED] or query.formulas:
            results["formula"] = self._search_formulas(query, top_k)

        if query.query_type == QueryType.DIAGRAM or "diagram" in query.raw_text.lower():
            results["diagram"] = self._search_diagrams(query, top_k)

        return results

    def _search_text(self, query: Query, top_k: int) -> list[SearchResult]:
        """Semantic text search."""
        query_text = query.processed_text or query.raw_text

        if query.embedding:
            import numpy as np
            embedding = np.array(query.embedding)
            raw_results = self.index_manager.embedding_index.search_by_embedding(
                embedding, top_k
            )
        else:
            raw_results = self.index_manager.embedding_index.search(query_text, top_k)

        results = []
        for rank, (chunk_id, score) in enumerate(raw_results):
            results.append(SearchResult(
                id=f"text_{chunk_id}",
                source_type="text",
                content="",
                score=score,
                rank=rank,
                chunk_id=chunk_id
            ))

        return results

    def _search_keywords(self, query: Query, top_k: int) -> list[SearchResult]:
        """Keyword-based search."""
        if not self.index_manager.index_config.enable_keyword_index:
            return []

        search_text = " ".join(query.keywords) if query.keywords else query.raw_text
        raw_results = self.index_manager.keyword_index.search(search_text, top_k)

        results = []
        for rank, (chunk_id, score) in enumerate(raw_results):
            results.append(SearchResult(
                id=f"keyword_{chunk_id}",
                source_type="keyword",
                content="",
                score=score,
                rank=rank,
                chunk_id=chunk_id
            ))

        return results

    def _search_formulas(self, query: Query, top_k: int) -> list[SearchResult]:
        """Formula search using symbolic matching."""
        if not self.index_manager.index_config.enable_formula_index:
            return []

        results = []

        if query.formulas:
            for formula_latex in query.formulas:
                raw_results = self.index_manager.formula_index.search_by_latex(
                    formula_latex, top_k
                )
                for rank, (formula_id, score) in enumerate(raw_results):
                    results.append(SearchResult(
                        id=f"formula_{formula_id}",
                        source_type="formula",
                        content="",
                        score=score,
                        rank=rank,
                        formula_id=formula_id
                    ))

        from ultimate_rag.utils.math_utils import extract_variables, extract_operators
        query_text = query.raw_text

        variables = extract_variables(query_text)
        operators = extract_operators(query_text)

        if variables or operators:
            raw_results = self.index_manager.formula_index.search_symbolic(
                variables=variables if variables else None,
                operators=operators if operators else None,
                domain=query.domain_hint,
                top_k=top_k
            )
            for rank, (formula_id, score) in enumerate(raw_results):
                results.append(SearchResult(
                    id=f"formula_sym_{formula_id}",
                    source_type="formula",
                    content="",
                    score=score * 0.8,
                    rank=rank,
                    formula_id=formula_id
                ))

        seen = set()
        unique_results = []
        for r in results:
            if r.formula_id not in seen:
                seen.add(r.formula_id)
                unique_results.append(r)

        return unique_results[:top_k]

    def _search_diagrams(self, query: Query, top_k: int) -> list[SearchResult]:
        """Diagram search using visual and text features."""
        if not self.index_manager.index_config.enable_diagram_index:
            return []

        results = []

        raw_results = self.index_manager.diagram_index.search_by_text(
            query.raw_text, top_k
        )

        for rank, (diagram_id, score) in enumerate(raw_results):
            metadata = self.index_manager.diagram_index.get_metadata(diagram_id)
            results.append(SearchResult(
                id=f"diagram_{diagram_id}",
                source_type="diagram",
                content=metadata.get("caption", "") if metadata else "",
                score=score,
                rank=rank,
                diagram_id=diagram_id,
                page_number=metadata.get("page") if metadata else None
            ))

        return results


class DomainSpecificSearcher:
    """Domain-specific search strategies."""

    def __init__(
        self,
        index_manager: IndexManager,
        config: Optional[RetrievalConfig] = None
    ):
        self.index_manager = index_manager
        self.config = config or RetrievalConfig()

    def search_physics(self, query: Query, top_k: int = 10) -> list[SearchResult]:
        """Physics-specific search prioritizing formulas and concepts."""
        results = []

        formula_results = self.index_manager.formula_index.search_by_domain(
            "physics", top_k * 2
        )

        query_text = query.raw_text.lower()
        for formula_id, _ in formula_results:
            formula_data = self.index_manager.formula_index.get_formula(formula_id)
            if formula_data:
                plain_text = formula_data.get("plain_text", "").lower()
                if any(kw in plain_text for kw in query.keywords):
                    results.append(SearchResult(
                        id=f"physics_{formula_id}",
                        source_type="formula",
                        content=formula_data.get("latex", ""),
                        score=1.0,
                        formula_id=formula_id
                    ))

        return results[:top_k]

    def search_mathematics(self, query: Query, top_k: int = 10) -> list[SearchResult]:
        """Math-specific search with theorem/proof awareness."""
        results = []

        text_results = self.index_manager.embedding_index.search(
            query.raw_text, top_k * 2
        )

        for chunk_id, score in text_results:
            results.append(SearchResult(
                id=f"math_{chunk_id}",
                source_type="text",
                content="",
                score=score,
                chunk_id=chunk_id
            ))

        return results[:top_k]

    def search_chemistry(self, query: Query, top_k: int = 10) -> list[SearchResult]:
        """Chemistry-specific search with reaction awareness."""
        results = []

        formula_results = self.index_manager.formula_index.search_by_domain(
            "chemistry", top_k
        )

        for formula_id, score in formula_results:
            formula_data = self.index_manager.formula_index.get_formula(formula_id)
            results.append(SearchResult(
                id=f"chem_{formula_id}",
                source_type="formula",
                content=formula_data.get("latex", "") if formula_data else "",
                score=score,
                formula_id=formula_id
            ))

        text_results = self.index_manager.embedding_index.search(
            query.raw_text, top_k
        )

        for chunk_id, score in text_results:
            results.append(SearchResult(
                id=f"chem_text_{chunk_id}",
                source_type="text",
                content="",
                score=score * 0.9,
                chunk_id=chunk_id
            ))

        return results[:top_k]

    def search_data_science(self, query: Query, top_k: int = 10) -> list[SearchResult]:
        """Data science search with algorithm awareness."""
        results = []

        keyword_results = self.index_manager.keyword_index.search(
            query.raw_text, top_k * 2
        )

        for chunk_id, score in keyword_results:
            results.append(SearchResult(
                id=f"ds_{chunk_id}",
                source_type="keyword",
                content="",
                score=score,
                chunk_id=chunk_id
            ))

        text_results = self.index_manager.embedding_index.search(
            query.raw_text, top_k
        )

        for chunk_id, score in text_results:
            results.append(SearchResult(
                id=f"ds_text_{chunk_id}",
                source_type="text",
                content="",
                score=score * 0.9,
                chunk_id=chunk_id
            ))

        seen = set()
        unique = []
        for r in results:
            key = r.chunk_id or r.formula_id
            if key not in seen:
                seen.add(key)
                unique.append(r)

        return unique[:top_k]
