"""Keyword index for exact and fuzzy matching."""

import re
import json
from pathlib import Path
from typing import Optional, Any
from collections import defaultdict
from loguru import logger

from ultimate_rag.models.chunk import Chunk
from ultimate_rag.config import IndexingConfig


class KeywordIndex:
    """Inverted index for keyword-based search."""

    def __init__(self, config: Optional[IndexingConfig] = None):
        self.config = config or IndexingConfig()

        self._inverted_index: dict[str, set[str]] = defaultdict(set)
        self._chunk_keywords: dict[str, set[str]] = {}
        self._document_freq: dict[str, int] = defaultdict(int)
        self._total_chunks = 0

        self._stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'that', 'which', 'who', 'whom', 'this', 'these', 'those',
            'it', 'its', 'as', 'if', 'when', 'than', 'so', 'no', 'not'
        }

    def add_chunk(self, chunk: Chunk):
        """Add a chunk to the keyword index."""
        keywords = self._extract_keywords(chunk.content)

        if chunk.metadata.keywords:
            keywords.update(chunk.metadata.keywords)

        self._chunk_keywords[chunk.id] = keywords

        for keyword in keywords:
            if chunk.id not in self._inverted_index[keyword]:
                self._inverted_index[keyword].add(chunk.id)
                self._document_freq[keyword] += 1

        self._total_chunks += 1

    def add_chunks(self, chunks: list[Chunk]):
        """Add multiple chunks to the index."""
        for chunk in chunks:
            self.add_chunk(chunk)

    def _extract_keywords(self, text: str) -> set[str]:
        """Extract keywords from text."""
        text = text.lower()
        text = re.sub(r'[^\w\s-]', ' ', text)

        words = text.split()

        keywords = set()
        for word in words:
            word = word.strip('-')
            if len(word) > 2 and word not in self._stopwords:
                keywords.add(word)

        for i in range(len(words) - 1):
            if words[i] not in self._stopwords and words[i+1] not in self._stopwords:
                bigram = f"{words[i]}_{words[i+1]}"
                keywords.add(bigram)

        return keywords

    def search(
        self,
        query: str,
        top_k: int = 20,
        use_bm25: bool = True
    ) -> list[tuple[str, float]]:
        """Search for chunks matching query keywords."""
        query_keywords = self._extract_keywords(query)

        if not query_keywords:
            return []

        chunk_scores: dict[str, float] = defaultdict(float)

        for keyword in query_keywords:
            matching_chunks = self._inverted_index.get(keyword, set())

            for chunk_id in matching_chunks:
                if use_bm25:
                    score = self._bm25_score(keyword, chunk_id)
                else:
                    score = 1.0

                chunk_scores[chunk_id] += score

        results = [(cid, score) for cid, score in chunk_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def search_exact(
        self,
        phrase: str,
        top_k: int = 20
    ) -> list[tuple[str, float]]:
        """Search for exact phrase match."""
        phrase_lower = phrase.lower()
        results = []

        for chunk_id, keywords in self._chunk_keywords.items():
            phrase_words = phrase_lower.split()
            if all(word in keywords or any(word in kw for kw in keywords) for word in phrase_words):
                results.append((chunk_id, 1.0))

        return results[:top_k]

    def search_fuzzy(
        self,
        query: str,
        top_k: int = 20,
        threshold: float = 0.7
    ) -> list[tuple[str, float]]:
        """Fuzzy keyword search with edit distance."""
        query_keywords = self._extract_keywords(query)
        chunk_scores: dict[str, float] = defaultdict(float)

        for query_kw in query_keywords:
            for index_kw, chunk_ids in self._inverted_index.items():
                similarity = self._string_similarity(query_kw, index_kw)
                if similarity >= threshold:
                    for chunk_id in chunk_ids:
                        chunk_scores[chunk_id] += similarity

        results = [(cid, score) for cid, score in chunk_scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:top_k]

    def _bm25_score(
        self,
        keyword: str,
        chunk_id: str,
        k1: float = 1.5,
        b: float = 0.75
    ) -> float:
        """Calculate BM25 score for keyword-chunk pair."""
        import math

        df = self._document_freq.get(keyword, 0)
        if df == 0:
            return 0.0

        idf = math.log((self._total_chunks - df + 0.5) / (df + 0.5) + 1)

        chunk_keywords = self._chunk_keywords.get(chunk_id, set())
        tf = 1 if keyword in chunk_keywords else 0

        avg_dl = sum(len(kws) for kws in self._chunk_keywords.values()) / max(self._total_chunks, 1)
        dl = len(chunk_keywords)

        score = idf * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avg_dl))

        return score

    def _string_similarity(self, s1: str, s2: str) -> float:
        """Calculate string similarity using Levenshtein ratio."""
        if s1 == s2:
            return 1.0

        if s1 in s2 or s2 in s1:
            return 0.8

        len1, len2 = len(s1), len(s2)
        if abs(len1 - len2) > max(len1, len2) * 0.5:
            return 0.0

        matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]

        for i in range(len1 + 1):
            matrix[i][0] = i
        for j in range(len2 + 1):
            matrix[0][j] = j

        for i in range(1, len1 + 1):
            for j in range(1, len2 + 1):
                cost = 0 if s1[i-1] == s2[j-1] else 1
                matrix[i][j] = min(
                    matrix[i-1][j] + 1,
                    matrix[i][j-1] + 1,
                    matrix[i-1][j-1] + cost
                )

        distance = matrix[len1][len2]
        max_len = max(len1, len2)

        return 1.0 - (distance / max_len)

    def get_keywords(self, chunk_id: str) -> set[str]:
        """Get keywords for a chunk."""
        return self._chunk_keywords.get(chunk_id, set())

    def get_chunks_with_keyword(self, keyword: str) -> set[str]:
        """Get all chunks containing a keyword."""
        return self._inverted_index.get(keyword.lower(), set())

    def save(self, path: Path):
        """Save index to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "inverted_index": {k: list(v) for k, v in self._inverted_index.items()},
            "chunk_keywords": {k: list(v) for k, v in self._chunk_keywords.items()},
            "document_freq": dict(self._document_freq),
            "total_chunks": self._total_chunks
        }

        with open(path, 'w') as f:
            json.dump(data, f)

        logger.info(f"Saved keyword index to {path}")

    def load(self, path: Path):
        """Load index from disk."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Keyword index not found: {path}")
            return

        with open(path, 'r') as f:
            data = json.load(f)

        self._inverted_index = defaultdict(set, {k: set(v) for k, v in data["inverted_index"].items()})
        self._chunk_keywords = {k: set(v) for k, v in data["chunk_keywords"].items()}
        self._document_freq = defaultdict(int, data["document_freq"])
        self._total_chunks = data["total_chunks"]

        logger.info(f"Loaded keyword index from {path}")

    @property
    def size(self) -> int:
        """Get number of indexed chunks."""
        return self._total_chunks

    @property
    def vocabulary_size(self) -> int:
        """Get number of unique keywords."""
        return len(self._inverted_index)
