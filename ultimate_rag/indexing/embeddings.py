"""Text embedding index using FAISS."""

import time
from pathlib import Path
from typing import Optional, Any
import numpy as np
from loguru import logger

from ultimate_rag.models.chunk import Chunk
from ultimate_rag.config import IndexingConfig


class EmbeddingIndex:
    """Manage text embeddings and FAISS index."""

    def __init__(self, config: Optional[IndexingConfig] = None):
        self.config = config or IndexingConfig()
        self._model = None
        self._index = None
        self._id_map: dict[int, str] = {}
        self._reverse_map: dict[str, int] = {}
        self._initialized = False

    def initialize(self):
        """Load embedding model and initialize index."""
        if self._initialized:
            return

        try:
            from sentence_transformers import SentenceTransformer

            device = "cuda" if self.config.use_gpu else "cpu"
            self._model = SentenceTransformer(
                self.config.embedding_model,
                device=device
            )
            logger.info(f"Loaded embedding model: {self.config.embedding_model}")

        except ImportError:
            logger.error("sentence-transformers not available")
            raise

        self._create_empty_index()
        self._initialized = True

    def _create_empty_index(self):
        """Create empty FAISS index."""
        try:
            import faiss

            dimension = self.config.embedding_dimension

            if self.config.faiss_index_type == "flat":
                self._index = faiss.IndexFlatIP(dimension)

            elif self.config.faiss_index_type == "ivf":
                quantizer = faiss.IndexFlatIP(dimension)
                self._index = faiss.IndexIVFFlat(
                    quantizer, dimension,
                    self.config.faiss_nlist,
                    faiss.METRIC_INNER_PRODUCT
                )

            elif self.config.faiss_index_type == "hnsw":
                self._index = faiss.IndexHNSWFlat(dimension, 32)

            else:
                self._index = faiss.IndexFlatIP(dimension)

            logger.info(f"Created FAISS index: {self.config.faiss_index_type}")

        except ImportError:
            logger.error("FAISS not available")
            raise

    def embed_text(self, text: str) -> np.ndarray:
        """Generate embedding for single text."""
        self.initialize()
        embedding = self._model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        return embedding

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Generate embeddings for batch of texts."""
        self.initialize()
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=self.config.batch_embedding_size,
            show_progress_bar=True
        )
        return embeddings

    def add_chunks(self, chunks: list[Chunk]):
        """Add chunks to the index."""
        self.initialize()

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embed_batch(texts)

        start_idx = len(self._id_map)
        for i, chunk in enumerate(chunks):
            idx = start_idx + i
            self._id_map[idx] = chunk.id
            self._reverse_map[chunk.id] = idx
            chunk.text_embedding = embeddings[i].tolist()

        if self.config.faiss_index_type == "ivf" and not self._index.is_trained:
            if len(embeddings) >= self.config.faiss_nlist:
                self._index.train(embeddings.astype('float32'))
            else:
                quantizer = self._index.quantizer
                self._index = quantizer
                logger.warning("Not enough vectors for IVF, using flat index")

        self._index.add(embeddings.astype('float32'))
        logger.info(f"Added {len(chunks)} chunks to index. Total: {self._index.ntotal}")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_ids: Optional[list[str]] = None
    ) -> list[tuple[str, float]]:
        """Search for similar chunks."""
        self.initialize()

        if self._index.ntotal == 0:
            return []

        query_embedding = self.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')

        search_k = top_k * 3 if filter_ids else top_k
        scores, indices = self._index.search(query_embedding, min(search_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            chunk_id = self._id_map.get(idx)
            if chunk_id is None:
                continue

            if filter_ids and chunk_id not in filter_ids:
                continue

            results.append((chunk_id, float(score)))

            if len(results) >= top_k:
                break

        return results

    def search_by_embedding(
        self,
        embedding: np.ndarray,
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Search using pre-computed embedding."""
        if self._index is None or self._index.ntotal == 0:
            return []

        embedding = embedding.reshape(1, -1).astype('float32')
        scores, indices = self._index.search(embedding, min(top_k, self._index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk_id = self._id_map.get(idx)
            if chunk_id:
                results.append((chunk_id, float(score)))

        return results

    def get_embedding(self, chunk_id: str) -> Optional[np.ndarray]:
        """Get embedding for a chunk by ID."""
        if chunk_id not in self._reverse_map:
            return None

        idx = self._reverse_map[chunk_id]
        return self._index.reconstruct(idx)

    def save(self, path: Path):
        """Save index to disk."""
        import faiss
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self._index, str(path.with_suffix('.faiss')))

        metadata = {
            "id_map": self._id_map,
            "reverse_map": self._reverse_map,
            "config": self.config.model_dump()
        }
        with open(path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(metadata, f)

        logger.info(f"Saved embedding index to {path}")

    def load(self, path: Path):
        """Load index from disk."""
        import faiss
        import pickle

        path = Path(path)

        self._index = faiss.read_index(str(path.with_suffix('.faiss')))

        with open(path.with_suffix('.pkl'), 'rb') as f:
            metadata = pickle.load(f)

        self._id_map = metadata["id_map"]
        self._reverse_map = metadata["reverse_map"]

        self._initialized = True
        logger.info(f"Loaded embedding index from {path}, {self._index.ntotal} vectors")

    def clear(self):
        """Clear the index."""
        self._create_empty_index()
        self._id_map = {}
        self._reverse_map = {}

    @property
    def size(self) -> int:
        """Get number of indexed items."""
        return self._index.ntotal if self._index else 0
