"""Unified index manager for all indices."""

from pathlib import Path
from typing import Optional, Any
from loguru import logger

from ultimate_rag.models.chunk import Chunk, ChunkStore
from ultimate_rag.models.formula import Formula
from ultimate_rag.models.document import ImageBlock
from ultimate_rag.indexing.embeddings import EmbeddingIndex
from ultimate_rag.indexing.formula_index import FormulaIndex
from ultimate_rag.indexing.diagram_index import DiagramIndex
from ultimate_rag.indexing.keyword_index import KeywordIndex
from ultimate_rag.config import Config, IndexingConfig
from ultimate_rag.utils.io import load_json


class IndexManager:
    """Manage all indices for the RAG system."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.index_config = self.config.indexing

        self.embedding_index = EmbeddingIndex(self.index_config)
        self.formula_index = FormulaIndex(self.index_config)
        self.diagram_index = DiagramIndex(self.index_config)
        self.keyword_index = KeywordIndex(self.index_config)

        self._initialized = False

    def initialize(self):
        """Initialize all indices."""
        if self._initialized:
            return

        self.embedding_index.initialize()
        self.formula_index.initialize()
        self.diagram_index.initialize()

        self._initialized = True
        logger.info("All indices initialized")

    def index_knowledge_base(
        self,
        knowledge_base: dict[str, Any],
        indices_dir: Optional[Path] = None
    ):
        """Index a complete knowledge base."""
        self.initialize()
        indices_dir = indices_dir or self.config.indices_dir
        indices_dir.mkdir(parents=True, exist_ok=True)

        doc_id = knowledge_base.get("document_id", "doc")

        chunks_data = knowledge_base.get("chunks", {})
        chunks = self._reconstruct_chunks(chunks_data)

        if chunks:
            logger.info(f"Indexing {len(chunks)} chunks...")
            self.embedding_index.add_chunks(chunks)

            if self.index_config.enable_keyword_index:
                self.keyword_index.add_chunks(chunks)

        formulas_data = knowledge_base.get("formulas", {})
        if formulas_data and self.index_config.enable_formula_index:
            formulas = self._reconstruct_formulas(formulas_data)
            logger.info(f"Indexing {len(formulas)} formulas...")
            self.formula_index.add_formulas(formulas)

        diagrams_data = knowledge_base.get("diagrams", [])
        if diagrams_data and self.index_config.enable_diagram_index:
            diagrams = self._reconstruct_diagrams(diagrams_data)
            logger.info(f"Indexing {len(diagrams)} diagrams...")
            self.diagram_index.add_diagrams(diagrams)

        self.save(indices_dir, doc_id)
        logger.info(f"Indexing complete. Saved to {indices_dir}")

    def _reconstruct_chunks(self, chunks_data: dict) -> list[Chunk]:
        """Reconstruct Chunk objects from dictionary data."""
        from ultimate_rag.models.chunk import Chunk, ChunkType, ChunkMetadata

        chunks = []
        for chunk_id, data in chunks_data.items():
            metadata = ChunkMetadata(**data.get("metadata", {}))

            chunk = Chunk(
                id=data.get("id", chunk_id),
                content=data.get("content", ""),
                chunk_type=ChunkType(data.get("chunk_type", "paragraph")),
                metadata=metadata,
                formula_ids=data.get("formula_ids", []),
                table_ids=data.get("table_ids", []),
                diagram_ids=data.get("diagram_ids", []),
                parent_id=data.get("parent_id"),
                children_ids=data.get("children_ids", []),
                previous_id=data.get("previous_id"),
                next_id=data.get("next_id"),
                related_ids=data.get("related_ids", [])
            )
            chunks.append(chunk)

        return chunks

    def _reconstruct_formulas(self, formulas_data: dict) -> list[Formula]:
        """Reconstruct Formula objects from dictionary data."""
        from ultimate_rag.models.formula import Formula, FormulaType, SymbolicForm

        formulas = []
        for formula_id, data in formulas_data.items():
            symbolic = None
            if data.get("symbolic"):
                symbolic = SymbolicForm(**data["symbolic"])

            formula = Formula(
                id=data.get("id", formula_id),
                latex=data.get("latex", ""),
                formula_type=data.get("formula_type", "unknown"),
                symbolic=symbolic,
                plain_text=data.get("plain_text"),
                description=data.get("description"),
                source_page=data.get("source_page"),
                source_chunk_id=data.get("source_chunk_id"),
                domain=data.get("domain"),
                dependencies=data.get("dependencies", []),
                tags=data.get("tags", [])
            )
            formulas.append(formula)

        return formulas

    def _reconstruct_diagrams(self, diagrams_data: list) -> list[ImageBlock]:
        """Reconstruct ImageBlock objects from dictionary data."""
        from ultimate_rag.models.document import ImageBlock, BoundingBox, BlockType

        diagrams = []
        for data in diagrams_data:
            bbox = None
            if data.get("bbox"):
                bbox = BoundingBox(**data["bbox"])

            diagram = ImageBlock(
                id=data.get("id", ""),
                bbox=bbox,
                block_type=BlockType(data.get("block_type", "diagram")),
                diagram_type=data.get("diagram_type"),
                caption=data.get("caption"),
                metadata=data.get("metadata", {})
            )
            diagrams.append(diagram)

        return diagrams

    def search(
        self,
        query: str,
        top_k: int = 10,
        search_types: Optional[list[str]] = None
    ) -> dict[str, list[tuple[str, float]]]:
        """Search across all indices."""
        self.initialize()

        search_types = search_types or ["embedding", "keyword", "formula"]
        results = {}

        if "embedding" in search_types:
            results["embedding"] = self.embedding_index.search(query, top_k)

        if "keyword" in search_types and self.index_config.enable_keyword_index:
            results["keyword"] = self.keyword_index.search(query, top_k)

        if "formula" in search_types and self.index_config.enable_formula_index:
            if self._looks_like_formula(query):
                results["formula"] = self.formula_index.search_by_latex(query, top_k)

        if "diagram" in search_types and self.index_config.enable_diagram_index:
            results["diagram"] = self.diagram_index.search_by_text(query, top_k)

        return results

    def _looks_like_formula(self, query: str) -> bool:
        """Check if query looks like a formula."""
        formula_indicators = ['=', '\\', '^', '_', '∫', '∑', '∂', '→']
        return any(ind in query for ind in formula_indicators)

    def save(self, indices_dir: Path, prefix: str = ""):
        """Save all indices to disk."""
        indices_dir = Path(indices_dir)
        indices_dir.mkdir(parents=True, exist_ok=True)

        prefix = f"{prefix}_" if prefix else ""

        self.embedding_index.save(indices_dir / f"{prefix}embeddings")
        self.formula_index.save(indices_dir / f"{prefix}formulas.db")
        self.diagram_index.save(indices_dir / f"{prefix}diagrams")
        self.keyword_index.save(indices_dir / f"{prefix}keywords.json")

        logger.info(f"All indices saved to {indices_dir}")

    def load(self, indices_dir: Path, prefix: str = ""):
        """Load all indices from disk."""
        indices_dir = Path(indices_dir)
        prefix = f"{prefix}_" if prefix else ""

        emb_path = indices_dir / f"{prefix}embeddings"
        if emb_path.with_suffix('.faiss').exists():
            self.embedding_index.load(emb_path)

        formula_path = indices_dir / f"{prefix}formulas.db"
        if formula_path.exists():
            self.formula_index.load(formula_path)

        diagram_path = indices_dir / f"{prefix}diagrams"
        if diagram_path.with_suffix('.pkl').exists():
            self.diagram_index.load(diagram_path)

        keyword_path = indices_dir / f"{prefix}keywords.json"
        if keyword_path.exists():
            self.keyword_index.load(keyword_path)

        self._initialized = True
        logger.info(f"All indices loaded from {indices_dir}")

    def get_statistics(self) -> dict[str, Any]:
        """Get statistics for all indices."""
        return {
            "embedding_index_size": self.embedding_index.size,
            "formula_index_size": self.formula_index.size,
            "diagram_index_size": self.diagram_index.size,
            "keyword_index_size": self.keyword_index.size,
            "keyword_vocabulary_size": self.keyword_index.vocabulary_size
        }

    def index_from_file(
        self,
        knowledge_path: Path,
        indices_dir: Optional[Path] = None
    ):
        """Index knowledge base from file."""
        knowledge_base = load_json(Path(knowledge_path))
        self.index_knowledge_base(knowledge_base, indices_dir)
