"""Build structured knowledge from extracted content."""

from typing import Optional, Any
from pathlib import Path
from loguru import logger

from ultimate_rag.models.document import Document
from ultimate_rag.models.chunk import Chunk, ChunkStore, ChunkType
from ultimate_rag.models.formula import Formula, FormulaCollection
from ultimate_rag.structuring.chunker import HierarchicalChunker, SemanticChunker, Chunker
from ultimate_rag.structuring.classifier import ContentClassifier
from ultimate_rag.config import Config, StructuringConfig
from ultimate_rag.utils.io import save_json, load_json


class KnowledgeBuilder:
    """Build structured knowledge objects from documents."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.struct_config = self.config.structuring

        if self.struct_config.chunk_strategy == "hierarchical":
            self.chunker = HierarchicalChunker(self.struct_config)
        elif self.struct_config.chunk_strategy == "semantic":
            self.chunker = SemanticChunker(self.struct_config)
        else:
            self.chunker = Chunker(self.struct_config)

        self.classifier = ContentClassifier(self.struct_config)

    def build_knowledge_base(
        self,
        extracted_data: dict[str, Any],
        output_dir: Optional[Path] = None
    ) -> dict[str, Any]:
        """Build complete knowledge base from extracted data."""
        output_dir = output_dir or self.config.processed_dir / "knowledge"
        output_dir.mkdir(parents=True, exist_ok=True)

        doc_data = extracted_data.get("document", {})
        document = self._reconstruct_document(doc_data)

        chunk_store = self.chunker.chunk_document(document)
        logger.info(f"Created {len(chunk_store.chunks)} chunks")

        chunks = list(chunk_store.chunks.values())
        self.classifier.enrich_chunks(chunks)

        formulas = self._process_formulas(
            extracted_data.get("formulas", []),
            chunks
        )
        logger.info(f"Processed {len(formulas.formulas)} formulas")

        self._link_formulas_to_chunks(formulas, chunks)

        self._build_relationships(chunks)

        knowledge_base = {
            "document_id": extracted_data.get("document_id"),
            "chunks": {cid: c.model_dump() for cid, c in chunk_store.chunks.items()},
            "hierarchy": chunk_store.hierarchy,
            "sequence": chunk_store.sequence,
            "formulas": {fid: f.model_dump() for fid, f in formulas.formulas.items()},
            "tables": extracted_data.get("tables", []),
            "diagrams": extracted_data.get("diagrams", []),
            "statistics": self._compute_statistics(chunk_store, formulas)
        }

        kb_path = output_dir / f"{extracted_data.get('document_id', 'doc')}_knowledge.json"
        save_json(knowledge_base, kb_path)
        logger.info(f"Knowledge base saved to {kb_path}")

        return knowledge_base

    def _reconstruct_document(self, doc_data: dict) -> Document:
        """Reconstruct Document object from dictionary."""
        from ultimate_rag.models.document import Document, Page, TextBlock, ImageBlock, TableBlock, BoundingBox, BlockType

        pages = []
        for page_data in doc_data.get("pages", []):
            text_blocks = []
            for tb_data in page_data.get("text_blocks", []):
                bbox = None
                if tb_data.get("bbox"):
                    bbox = BoundingBox(**tb_data["bbox"])

                text_blocks.append(TextBlock(
                    id=tb_data.get("id", ""),
                    content=tb_data.get("content", ""),
                    block_type=BlockType(tb_data.get("block_type", "text")),
                    bbox=bbox,
                    font_name=tb_data.get("font_name"),
                    font_size=tb_data.get("font_size"),
                    is_bold=tb_data.get("is_bold", False),
                    is_italic=tb_data.get("is_italic", False)
                ))

            image_blocks = []
            for ib_data in page_data.get("image_blocks", []):
                bbox = BoundingBox(**ib_data["bbox"]) if ib_data.get("bbox") else None
                image_blocks.append(ImageBlock(
                    id=ib_data.get("id", ""),
                    bbox=bbox,
                    block_type=BlockType(ib_data.get("block_type", "image"))
                ))

            table_blocks = []
            for tab_data in page_data.get("table_blocks", []):
                table_blocks.append(TableBlock(
                    id=tab_data.get("id", ""),
                    headers=tab_data.get("headers", []),
                    rows=tab_data.get("rows", [])
                ))

            pages.append(Page(
                page_number=page_data.get("page_number", 0),
                width=page_data.get("width", 0),
                height=page_data.get("height", 0),
                text_blocks=text_blocks,
                image_blocks=image_blocks,
                table_blocks=table_blocks,
                raw_text=page_data.get("raw_text")
            ))

        return Document(
            id=doc_data.get("id", ""),
            filename=doc_data.get("filename", ""),
            filepath=doc_data.get("filepath"),
            title=doc_data.get("title"),
            pages=pages,
            total_pages=len(pages)
        )

    def _process_formulas(
        self,
        formula_data: list[dict],
        chunks: list[Chunk]
    ) -> FormulaCollection:
        """Process and organize formulas."""
        collection = FormulaCollection()

        for f_data in formula_data:
            formula = Formula(
                id=f_data.get("id", ""),
                latex=f_data.get("latex", ""),
                formula_type=f_data.get("formula_type", "unknown"),
                plain_text=f_data.get("plain_text"),
                description=f_data.get("description"),
                source_page=f_data.get("source_page"),
                domain=f_data.get("domain"),
                tags=f_data.get("tags", [])
            )

            if f_data.get("symbolic"):
                from ultimate_rag.models.formula import SymbolicForm
                formula.symbolic = SymbolicForm(**f_data["symbolic"])

            collection.add(formula)

        self._find_formula_dependencies(collection)

        return collection

    def _find_formula_dependencies(self, collection: FormulaCollection):
        """Detect dependencies between formulas."""
        formula_list = list(collection.formulas.values())

        for i, formula in enumerate(formula_list):
            if not formula.symbolic:
                continue

            for j, other in enumerate(formula_list):
                if i == j or not other.symbolic:
                    continue

                other_vars = set(other.symbolic.variables)
                formula_vars = set(formula.symbolic.variables)

                if other_vars & formula_vars and len(other_vars) < len(formula_vars):
                    if other.id not in formula.dependencies:
                        formula.dependencies.append(other.id)

    def _link_formulas_to_chunks(
        self,
        formulas: FormulaCollection,
        chunks: list[Chunk]
    ):
        """Link formulas to their containing chunks."""
        for chunk in chunks:
            chunk_formulas = []

            for formula in formulas.formulas.values():
                if formula.source_page in chunk.metadata.page_numbers:
                    if (formula.latex in chunk.content or
                        any(var in chunk.content for var in (formula.symbolic.variables if formula.symbolic else []))):
                        chunk_formulas.append(formula.id)
                        formula.source_chunk_id = chunk.id

            chunk.formula_ids = chunk_formulas

    def _build_relationships(self, chunks: list[Chunk]):
        """Build relationships between chunks."""
        for i, chunk in enumerate(chunks):
            if chunk.chunk_type in [ChunkType.THEOREM, ChunkType.DEFINITION]:
                for j, other in enumerate(chunks):
                    if i == j:
                        continue

                    if other.chunk_type == ChunkType.PROOF:
                        if (other.metadata.page_numbers and chunk.metadata.page_numbers and
                            abs(other.metadata.page_numbers[0] - chunk.metadata.page_numbers[0]) <= 1):
                            if chunk.id not in other.related_ids:
                                other.related_ids.append(chunk.id)
                            if other.id not in chunk.related_ids:
                                chunk.related_ids.append(other.id)

                    if other.chunk_type == ChunkType.EXAMPLE:
                        common_concepts = (
                            set(chunk.metadata.concepts) &
                            set(other.metadata.concepts)
                        )
                        if common_concepts:
                            if other.id not in chunk.related_ids:
                                chunk.related_ids.append(other.id)

    def _compute_statistics(
        self,
        chunk_store: ChunkStore,
        formulas: FormulaCollection
    ) -> dict[str, Any]:
        """Compute knowledge base statistics."""
        chunks = list(chunk_store.chunks.values())

        chunk_types = {}
        for chunk in chunks:
            t = chunk.chunk_type.value
            chunk_types[t] = chunk_types.get(t, 0) + 1

        domains = {}
        for chunk in chunks:
            d = chunk.metadata.domain or "general"
            domains[d] = domains.get(d, 0) + 1

        formula_domains = {}
        for formula in formulas.formulas.values():
            d = formula.domain or "general"
            formula_domains[d] = formula_domains.get(d, 0) + 1

        return {
            "total_chunks": len(chunks),
            "chunk_types": chunk_types,
            "domains": domains,
            "total_formulas": len(formulas.formulas),
            "formula_domains": formula_domains,
            "avg_chunk_length": sum(len(c.content) for c in chunks) / len(chunks) if chunks else 0,
            "chunks_with_formulas": sum(1 for c in chunks if c.formula_ids)
        }

    def build_from_file(
        self,
        extracted_path: Path,
        output_dir: Optional[Path] = None
    ) -> dict[str, Any]:
        """Build knowledge base from extracted data file."""
        extracted_data = load_json(Path(extracted_path))
        return self.build_knowledge_base(extracted_data, output_dir)
