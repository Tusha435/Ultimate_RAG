"""Result fusion strategies for combining multi-modal search results."""

from typing import Optional
from collections import defaultdict
from loguru import logger

from ultimate_rag.models.query import Query, SearchResult
from ultimate_rag.config import RetrievalConfig


class ResultFusion:
    """Fuse results from multiple search modalities."""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()

    def fuse(
        self,
        results: dict[str, list[SearchResult]],
        query: Query
    ) -> list[SearchResult]:
        """Fuse results using configured strategy."""
        if self.config.fusion_method == "rrf":
            return self._reciprocal_rank_fusion(results)
        elif self.config.fusion_method == "weighted":
            return self._weighted_fusion(results, query)
        elif self.config.fusion_method == "convex":
            return self._convex_combination(results, query)
        else:
            return self._reciprocal_rank_fusion(results)

    def _reciprocal_rank_fusion(
        self,
        results: dict[str, list[SearchResult]],
        k: Optional[int] = None
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion (RRF) algorithm."""
        k = k or self.config.rrf_k

        doc_scores: dict[str, float] = defaultdict(float)
        doc_results: dict[str, SearchResult] = {}

        for source_type, source_results in results.items():
            for rank, result in enumerate(source_results):
                doc_key = result.chunk_id or result.formula_id or result.diagram_id or result.id

                rrf_score = 1.0 / (k + rank + 1)
                doc_scores[doc_key] += rrf_score

                if doc_key not in doc_results or result.score > doc_results[doc_key].score:
                    doc_results[doc_key] = result

        fused_results = []
        for doc_key, rrf_score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            result = doc_results[doc_key]
            fused_result = result.model_copy(update={
                "score": rrf_score,
                "metadata": {**result.metadata, "fusion_method": "rrf"}
            })
            fused_results.append(fused_result)

        for rank, result in enumerate(fused_results):
            result.rank = rank

        return fused_results

    def _weighted_fusion(
        self,
        results: dict[str, list[SearchResult]],
        query: Query
    ) -> list[SearchResult]:
        """Weighted score fusion based on query type."""
        weights = self._get_weights_for_query(query)

        doc_scores: dict[str, float] = defaultdict(float)
        doc_results: dict[str, SearchResult] = {}

        for source_type, source_results in results.items():
            weight = weights.get(source_type, 0.1)

            if not source_results:
                continue

            max_score = max(r.score for r in source_results) if source_results else 1.0
            max_score = max_score if max_score > 0 else 1.0

            for result in source_results:
                doc_key = result.chunk_id or result.formula_id or result.diagram_id or result.id
                normalized_score = result.score / max_score
                doc_scores[doc_key] += weight * normalized_score

                if doc_key not in doc_results or result.score > doc_results[doc_key].score:
                    doc_results[doc_key] = result

        fused_results = []
        for doc_key, weighted_score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            result = doc_results[doc_key]
            fused_result = result.model_copy(update={
                "score": weighted_score,
                "metadata": {**result.metadata, "fusion_method": "weighted"}
            })
            fused_results.append(fused_result)

        for rank, result in enumerate(fused_results):
            result.rank = rank

        return fused_results

    def _convex_combination(
        self,
        results: dict[str, list[SearchResult]],
        query: Query
    ) -> list[SearchResult]:
        """Convex combination of normalized scores."""
        weights = self._get_weights_for_query(query)

        total_weight = sum(
            weights.get(source, 0.1)
            for source in results.keys()
            if results[source]
        )

        if total_weight == 0:
            return []

        normalized_weights = {
            source: weights.get(source, 0.1) / total_weight
            for source in results.keys()
        }

        doc_scores: dict[str, float] = defaultdict(float)
        doc_results: dict[str, SearchResult] = {}

        for source_type, source_results in results.items():
            if not source_results:
                continue

            scores = [r.score for r in source_results]
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score if max_score > min_score else 1.0

            weight = normalized_weights.get(source_type, 0.1)

            for result in source_results:
                doc_key = result.chunk_id or result.formula_id or result.diagram_id or result.id
                normalized_score = (result.score - min_score) / score_range
                doc_scores[doc_key] += weight * normalized_score

                if doc_key not in doc_results:
                    doc_results[doc_key] = result

        fused_results = []
        for doc_key, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            result = doc_results[doc_key]
            fused_result = result.model_copy(update={
                "score": score,
                "metadata": {**result.metadata, "fusion_method": "convex"}
            })
            fused_results.append(fused_result)

        for rank, result in enumerate(fused_results):
            result.rank = rank

        return fused_results

    def _get_weights_for_query(self, query: Query) -> dict[str, float]:
        """Get fusion weights based on query characteristics."""
        from ultimate_rag.models.query import QueryType, QueryIntent

        base_weights = {
            "text": self.config.text_weight,
            "formula": self.config.formula_weight,
            "diagram": self.config.diagram_weight,
            "keyword": self.config.keyword_weight
        }

        if query.query_type == QueryType.FORMULA:
            base_weights["formula"] *= 2.0
            base_weights["text"] *= 0.7

        elif query.query_type == QueryType.DIAGRAM:
            base_weights["diagram"] *= 2.0
            base_weights["text"] *= 0.7

        elif query.query_type == QueryType.KEYWORD:
            base_weights["keyword"] *= 1.5
            base_weights["text"] *= 0.8

        if query.intent == QueryIntent.FORMULA_LOOKUP:
            base_weights["formula"] *= 1.5

        elif query.intent == QueryIntent.VISUALIZATION:
            base_weights["diagram"] *= 1.5

        elif query.intent in [QueryIntent.DEFINITION, QueryIntent.EXPLANATION]:
            base_weights["text"] *= 1.3

        total = sum(base_weights.values())
        return {k: v / total for k, v in base_weights.items()}


class ReRanker:
    """Re-rank results using cross-encoder or other methods."""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self._model = None

    def _load_model(self):
        """Load cross-encoder model."""
        if self._model is None and self.config.rerank_model:
            try:
                from sentence_transformers import CrossEncoder
                self._model = CrossEncoder(self.config.rerank_model)
                logger.info(f"Loaded reranker: {self.config.rerank_model}")
            except ImportError:
                logger.warning("sentence-transformers not available for reranking")
            except Exception as e:
                logger.warning(f"Failed to load reranker: {e}")

    def rerank(
        self,
        query: Query,
        results: list[SearchResult],
        chunks: dict[str, str],
        top_k: Optional[int] = None
    ) -> list[SearchResult]:
        """Re-rank results using cross-encoder."""
        if not self.config.rerank_results or not results:
            return results

        self._load_model()

        if not self._model:
            return results

        top_k = top_k or len(results)
        query_text = query.processed_text or query.raw_text

        pairs = []
        valid_results = []

        for result in results[:min(len(results), 100)]:
            chunk_id = result.chunk_id
            if chunk_id and chunk_id in chunks:
                pairs.append([query_text, chunks[chunk_id]])
                valid_results.append(result)

        if not pairs:
            return results

        try:
            scores = self._model.predict(pairs)

            for i, score in enumerate(scores):
                valid_results[i] = valid_results[i].model_copy(update={
                    "score": float(score),
                    "metadata": {**valid_results[i].metadata, "reranked": True}
                })

            valid_results.sort(key=lambda x: x.score, reverse=True)

            for rank, result in enumerate(valid_results):
                result.rank = rank

            return valid_results[:top_k]

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results
