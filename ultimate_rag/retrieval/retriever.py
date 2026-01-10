"""Main retriever combining all retrieval components."""

import time
from typing import Optional, Any
from pathlib import Path
from loguru import logger

from ultimate_rag.models.query import Query, RetrievalResult
from ultimate_rag.retrieval.query_processor import QueryProcessor
from ultimate_rag.retrieval.searchers import MultiModalSearcher, DomainSpecificSearcher
from ultimate_rag.retrieval.fusion import ResultFusion, ReRanker
from ultimate_rag.retrieval.context_assembler import ContextAssembler
from ultimate_rag.indexing.index_manager import IndexManager
from ultimate_rag.config import Config, RetrievalConfig
from ultimate_rag.utils.io import load_json


class Retriever:
    """Main retriever orchestrating the retrieval pipeline."""

    def __init__(
        self,
        index_manager: IndexManager,
        knowledge_base: Optional[dict[str, Any]] = None,
        config: Optional[Config] = None
    ):
        self.config = config or Config()
        self.retrieval_config = self.config.retrieval

        self.index_manager = index_manager
        self.query_processor = QueryProcessor(self.retrieval_config)
        self.searcher = MultiModalSearcher(index_manager, self.retrieval_config)
        self.domain_searcher = DomainSpecificSearcher(index_manager, self.retrieval_config)
        self.fusion = ResultFusion(self.retrieval_config)
        self.reranker = ReRanker(self.retrieval_config)
        self.context_assembler = ContextAssembler(self.retrieval_config)

        self.chunk_store: dict[str, dict] = {}
        self.formula_store: dict[str, dict] = {}
        self.diagram_store: list[dict] = []

        if knowledge_base:
            self.load_knowledge_base(knowledge_base)

    def load_knowledge_base(self, knowledge_base: dict[str, Any]):
        """Load knowledge base data for context assembly."""
        self.chunk_store = knowledge_base.get("chunks", {})
        self.formula_store = knowledge_base.get("formulas", {})
        self.diagram_store = knowledge_base.get("diagrams", [])
        logger.info(
            f"Loaded knowledge base: {len(self.chunk_store)} chunks, "
            f"{len(self.formula_store)} formulas, {len(self.diagram_store)} diagrams"
        )

    def load_from_file(self, knowledge_path: Path):
        """Load knowledge base from file."""
        knowledge_base = load_json(Path(knowledge_path))
        self.load_knowledge_base(knowledge_base)

    def retrieve(
        self,
        query_text: str,
        top_k: Optional[int] = None,
        context: Optional[str] = None
    ) -> RetrievalResult:
        """Execute full retrieval pipeline."""
        start_time = time.time()
        top_k = top_k or self.retrieval_config.default_top_k

        query = self.query_processor.process(query_text, context)

        query = self.query_processor.add_embedding(query, self.index_manager.embedding_index)

        raw_results = self.searcher.search(query, top_k * 2)

        if query.domain_hint:
            domain_results = self._get_domain_results(query, top_k)
            if domain_results:
                raw_results["domain"] = domain_results

        fused_results = self.fusion.fuse(raw_results, query)

        if self.retrieval_config.rerank_results and self.chunk_store:
            chunk_texts = {cid: data.get("content", "") for cid, data in self.chunk_store.items()}
            fused_results = self.reranker.rerank(query, fused_results, chunk_texts, top_k)

        retrieval_result = self.context_assembler.assemble(
            query=query,
            results=fused_results,
            chunk_store=self.chunk_store,
            formula_store=self.formula_store,
            diagram_store=self.diagram_store,
            top_k=top_k
        )

        retrieval_result.retrieval_time_ms = (time.time() - start_time) * 1000

        return retrieval_result

    def _get_domain_results(self, query: Query, top_k: int):
        """Get domain-specific search results."""
        domain = query.domain_hint

        if domain == "physics":
            return self.domain_searcher.search_physics(query, top_k)
        elif domain == "mathematics":
            return self.domain_searcher.search_mathematics(query, top_k)
        elif domain == "chemistry":
            return self.domain_searcher.search_chemistry(query, top_k)
        elif domain == "data_science":
            return self.domain_searcher.search_data_science(query, top_k)

        return []

    def retrieve_formula(
        self,
        formula_latex: str,
        top_k: int = 5
    ) -> list[tuple[str, float, dict]]:
        """Retrieve similar formulas."""
        results = self.index_manager.formula_index.search_by_latex(formula_latex, top_k)

        enriched = []
        for formula_id, score in results:
            formula_data = self.formula_store.get(formula_id, {})
            enriched.append((formula_id, score, formula_data))

        return enriched

    def retrieve_by_variable(
        self,
        variable: str,
        top_k: int = 10
    ) -> list[tuple[str, float, dict]]:
        """Find formulas containing a specific variable."""
        results = self.index_manager.formula_index.search_by_variable(variable, top_k)

        enriched = []
        for formula_id, score in results:
            formula_data = self.formula_store.get(formula_id, {})
            enriched.append((formula_id, score, formula_data))

        return enriched

    def retrieve_diagram(
        self,
        query_text: str,
        top_k: int = 5
    ) -> list[tuple[str, float, dict]]:
        """Retrieve relevant diagrams."""
        results = self.index_manager.diagram_index.search_by_text(query_text, top_k)

        enriched = []
        for diagram_id, score in results:
            diagram_data = next(
                (d for d in self.diagram_store if d.get("id") == diagram_id),
                {}
            )
            enriched.append((diagram_id, score, diagram_data))

        return enriched

    def get_chunk_context(
        self,
        chunk_id: str,
        window_size: int = 2
    ) -> list[dict]:
        """Get chunk with surrounding context."""
        if chunk_id not in self.chunk_store:
            return []

        chunk_data = self.chunk_store[chunk_id]
        context_chunks = [chunk_data]

        prev_id = chunk_data.get("previous_id")
        next_id = chunk_data.get("next_id")

        for _ in range(window_size):
            if prev_id and prev_id in self.chunk_store:
                context_chunks.insert(0, self.chunk_store[prev_id])
                prev_id = self.chunk_store[prev_id].get("previous_id")

        for _ in range(window_size):
            if next_id and next_id in self.chunk_store:
                context_chunks.append(self.chunk_store[next_id])
                next_id = self.chunk_store[next_id].get("next_id")

        return context_chunks

    def explain_retrieval(self, retrieval_result: RetrievalResult) -> str:
        """Generate explanation of retrieval process."""
        lines = [
            f"Query: {retrieval_result.query.raw_text}",
            f"Query Type: {retrieval_result.query.query_type.value}",
            f"Intent: {retrieval_result.query.intent.value}",
            f"Domain: {retrieval_result.query.domain_hint or 'general'}",
            f"",
            f"Retrieved {retrieval_result.total_results} total results",
            f"  - Text results: {len(retrieval_result.text_results)}",
            f"  - Formula results: {len(retrieval_result.formula_results)}",
            f"  - Diagram results: {len(retrieval_result.diagram_results)}",
            f"",
            f"Confidence: {retrieval_result.confidence:.2f}",
            f"Retrieval time: {retrieval_result.retrieval_time_ms:.0f}ms",
            f"",
            "Top sources:"
        ]

        for i, result in enumerate(retrieval_result.results[:5]):
            source_info = f"  {i+1}. [{result.source_type}]"
            if result.page_number:
                source_info += f" Page {result.page_number}"
            if result.section:
                source_info += f" - {result.section}"
            source_info += f" (score: {result.score:.3f})"
            lines.append(source_info)

        return "\n".join(lines)
