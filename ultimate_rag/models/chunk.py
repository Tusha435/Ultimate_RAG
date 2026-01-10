"""Chunk models for semantic document units."""

from typing import Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib


class ChunkType(str, Enum):
    """Types of content chunks."""

    CHAPTER = "chapter"
    SECTION = "section"
    SUBSECTION = "subsection"
    CONCEPT = "concept"
    DEFINITION = "definition"
    THEOREM = "theorem"
    PROOF = "proof"
    EXAMPLE = "example"
    EXERCISE = "exercise"
    FORMULA_BLOCK = "formula_block"
    TABLE_BLOCK = "table_block"
    DIAGRAM_BLOCK = "diagram_block"
    PARAGRAPH = "paragraph"
    LIST = "list"
    MIXED = "mixed"


class ChunkMetadata(BaseModel):
    """Metadata associated with a chunk."""

    document_id: str
    page_numbers: list[int] = Field(default_factory=list)
    section_hierarchy: list[str] = Field(default_factory=list)
    heading: Optional[str] = None
    domain: Optional[str] = None
    difficulty_level: Optional[str] = None
    keywords: list[str] = Field(default_factory=list)
    concepts: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    is_definition: bool = False
    is_theorem: bool = False
    is_example: bool = False
    custom: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """Atomic unit of retrievable content."""

    id: str
    content: str
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    metadata: ChunkMetadata

    text_embedding: Optional[list[float]] = None
    formula_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    diagram_ids: list[str] = Field(default_factory=list)

    parent_id: Optional[str] = None
    children_ids: list[str] = Field(default_factory=list)
    previous_id: Optional[str] = None
    next_id: Optional[str] = None
    related_ids: list[str] = Field(default_factory=list)

    token_count: Optional[int] = None
    char_count: Optional[int] = None
    word_count: Optional[int] = None

    def model_post_init(self, __context):
        """Calculate counts if not provided."""
        if self.char_count is None:
            self.char_count = len(self.content)
        if self.word_count is None:
            self.word_count = len(self.content.split())

    @property
    def content_hash(self) -> str:
        """Get hash of content for deduplication."""
        return hashlib.md5(self.content.encode()).hexdigest()[:12]

    @property
    def has_formulas(self) -> bool:
        return len(self.formula_ids) > 0

    @property
    def has_diagrams(self) -> bool:
        return len(self.diagram_ids) > 0

    @property
    def has_tables(self) -> bool:
        return len(self.table_ids) > 0

    @property
    def is_multimodal(self) -> bool:
        return self.has_formulas or self.has_diagrams or self.has_tables

    def get_display_text(self, include_formulas: bool = True) -> str:
        """Get text for display with optional formula inclusion."""
        text = self.content
        if include_formulas and self.formula_ids:
            text += f"\n[Contains {len(self.formula_ids)} formula(s)]"
        if self.diagram_ids:
            text += f"\n[Contains {len(self.diagram_ids)} diagram(s)]"
        if self.table_ids:
            text += f"\n[Contains {len(self.table_ids)} table(s)]"
        return text

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()

    @classmethod
    def create(cls, content: str, document_id: str, **kwargs) -> "Chunk":
        """Factory method to create a chunk."""
        chunk_id = kwargs.pop("id", f"C_{hashlib.md5(content.encode()).hexdigest()[:8]}")
        metadata = ChunkMetadata(document_id=document_id, **kwargs.pop("metadata", {}))
        return cls(id=chunk_id, content=content, metadata=metadata, **kwargs)


class HierarchicalChunk(BaseModel):
    """Chunk with full hierarchical context."""

    chunk: Chunk
    parent: Optional["HierarchicalChunk"] = None
    children: list["HierarchicalChunk"] = Field(default_factory=list)

    @property
    def depth(self) -> int:
        """Get depth in hierarchy."""
        depth = 0
        current = self.parent
        while current:
            depth += 1
            current = current.parent
        return depth

    def get_full_context(self) -> str:
        """Get content with parent context."""
        context_parts = []
        current = self
        while current:
            context_parts.insert(0, current.chunk.content)
            current = current.parent
        return "\n\n".join(context_parts)

    def get_ancestors(self) -> list["HierarchicalChunk"]:
        """Get all ancestor chunks."""
        ancestors = []
        current = self.parent
        while current:
            ancestors.append(current)
            current = current.parent
        return ancestors

    def get_descendants(self) -> list["HierarchicalChunk"]:
        """Get all descendant chunks."""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_descendants())
        return descendants


class ChunkStore(BaseModel):
    """Storage for chunks with relationship tracking."""

    chunks: dict[str, Chunk] = Field(default_factory=dict)
    hierarchy: dict[str, list[str]] = Field(default_factory=dict)
    sequence: list[str] = Field(default_factory=list)

    def add(self, chunk: Chunk):
        """Add a chunk to the store."""
        self.chunks[chunk.id] = chunk
        self.sequence.append(chunk.id)

        if chunk.parent_id:
            if chunk.parent_id not in self.hierarchy:
                self.hierarchy[chunk.parent_id] = []
            self.hierarchy[chunk.parent_id].append(chunk.id)

    def get(self, chunk_id: str) -> Optional[Chunk]:
        """Get chunk by ID."""
        return self.chunks.get(chunk_id)

    def get_children(self, chunk_id: str) -> list[Chunk]:
        """Get child chunks."""
        child_ids = self.hierarchy.get(chunk_id, [])
        return [self.chunks[cid] for cid in child_ids if cid in self.chunks]

    def get_context_window(self, chunk_id: str, window_size: int = 2) -> list[Chunk]:
        """Get surrounding chunks for context."""
        if chunk_id not in self.chunks:
            return []

        try:
            idx = self.sequence.index(chunk_id)
        except ValueError:
            return [self.chunks[chunk_id]]

        start = max(0, idx - window_size)
        end = min(len(self.sequence), idx + window_size + 1)

        return [self.chunks[cid] for cid in self.sequence[start:end] if cid in self.chunks]

    def find_by_type(self, chunk_type: ChunkType) -> list[Chunk]:
        """Find chunks by type."""
        return [c for c in self.chunks.values() if c.chunk_type == chunk_type]

    def find_by_page(self, page_number: int) -> list[Chunk]:
        """Find chunks from a specific page."""
        return [
            c for c in self.chunks.values()
            if page_number in c.metadata.page_numbers
        ]
