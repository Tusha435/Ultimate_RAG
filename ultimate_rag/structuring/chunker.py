"""Hierarchical and semantic chunking strategies."""

import re
from typing import Optional, Any
from loguru import logger

from ultimate_rag.models.document import Document, Page, TextBlock, BlockType
from ultimate_rag.models.chunk import Chunk, ChunkType, ChunkMetadata, ChunkStore
from ultimate_rag.config import StructuringConfig


class Chunker:
    """Base chunker with fixed-size strategy."""

    def __init__(self, config: Optional[StructuringConfig] = None):
        self.config = config or StructuringConfig()

    def chunk_document(self, document: Document) -> ChunkStore:
        """Chunk a document into retrievable units."""
        store = ChunkStore()

        for page in document.pages:
            chunks = self.chunk_page(page, document.id)
            for chunk in chunks:
                store.add(chunk)

        self._link_chunks(store)
        return store

    def chunk_page(self, page: Page, document_id: str) -> list[Chunk]:
        """Chunk a single page."""
        chunks = []
        current_text = []
        current_start_block = 0

        for i, block in enumerate(page.text_blocks):
            current_text.append(block.content)
            combined = "\n".join(current_text)

            if len(combined) >= self.config.max_chunk_size:
                chunk = self._create_chunk(
                    "\n".join(current_text[:-1]) if len(current_text) > 1 else combined,
                    document_id,
                    page.page_number,
                    len(chunks)
                )
                chunks.append(chunk)

                if self.config.chunk_overlap > 0 and current_text:
                    overlap_text = current_text[-1]
                    current_text = [overlap_text, block.content] if block.content != overlap_text else [block.content]
                else:
                    current_text = [block.content]
                current_start_block = i

        if current_text:
            combined = "\n".join(current_text)
            if len(combined) >= self.config.min_chunk_size:
                chunk = self._create_chunk(
                    combined,
                    document_id,
                    page.page_number,
                    len(chunks)
                )
                chunks.append(chunk)

        return chunks

    def _create_chunk(
        self,
        content: str,
        document_id: str,
        page_number: int,
        chunk_idx: int
    ) -> Chunk:
        """Create a chunk with metadata."""
        chunk_id = f"C_{document_id}_{page_number}_{chunk_idx}"

        metadata = ChunkMetadata(
            document_id=document_id,
            page_numbers=[page_number]
        )

        return Chunk(
            id=chunk_id,
            content=content,
            chunk_type=ChunkType.PARAGRAPH,
            metadata=metadata
        )

    def _link_chunks(self, store: ChunkStore):
        """Link chunks with previous/next relationships."""
        for i, chunk_id in enumerate(store.sequence):
            chunk = store.chunks[chunk_id]
            if i > 0:
                chunk.previous_id = store.sequence[i - 1]
            if i < len(store.sequence) - 1:
                chunk.next_id = store.sequence[i + 1]


class HierarchicalChunker(Chunker):
    """Chunk with hierarchical structure preservation."""

    def __init__(self, config: Optional[StructuringConfig] = None):
        super().__init__(config)
        self.heading_patterns = [re.compile(p) for p in self.config.heading_patterns]

    def chunk_document(self, document: Document) -> ChunkStore:
        """Chunk document preserving hierarchy."""
        store = ChunkStore()

        hierarchy_stack = []
        current_content = []
        current_page_numbers = []

        for page in document.pages:
            for block in page.text_blocks:
                heading_level = self._detect_heading_level(block)

                if heading_level is not None:
                    if current_content:
                        chunk = self._create_hierarchical_chunk(
                            current_content,
                            document.id,
                            current_page_numbers,
                            hierarchy_stack,
                            store
                        )
                        if chunk:
                            store.add(chunk)

                    while hierarchy_stack and hierarchy_stack[-1][0] >= heading_level:
                        hierarchy_stack.pop()

                    hierarchy_stack.append((heading_level, block.content.strip()))

                    heading_chunk = self._create_heading_chunk(
                        block,
                        document.id,
                        page.page_number,
                        heading_level,
                        hierarchy_stack,
                        store
                    )
                    store.add(heading_chunk)

                    current_content = []
                    current_page_numbers = [page.page_number]

                else:
                    current_content.append(block)
                    if page.page_number not in current_page_numbers:
                        current_page_numbers.append(page.page_number)

                    combined_len = sum(len(b.content) for b in current_content)
                    if combined_len >= self.config.max_chunk_size:
                        chunk = self._create_hierarchical_chunk(
                            current_content,
                            document.id,
                            current_page_numbers,
                            hierarchy_stack,
                            store
                        )
                        if chunk:
                            store.add(chunk)
                        current_content = []
                        current_page_numbers = [page.page_number]

        if current_content:
            chunk = self._create_hierarchical_chunk(
                current_content,
                document.id,
                current_page_numbers,
                hierarchy_stack,
                store
            )
            if chunk:
                store.add(chunk)

        self._link_chunks(store)
        return store

    def _detect_heading_level(self, block: TextBlock) -> Optional[int]:
        """Detect if block is a heading and its level."""
        if block.block_type == BlockType.HEADING:
            if block.font_size and block.font_size > 18:
                return 1
            elif block.font_size and block.font_size > 14:
                return 2
            else:
                return 3

        content = block.content.strip()

        chapter_match = re.match(r'^Chapter\s+(\d+)', content, re.IGNORECASE)
        if chapter_match:
            return 1

        section_match = re.match(r'^(\d+)\.(\d+)?(\.(\d+))?', content)
        if section_match:
            if section_match.group(4):
                return 3
            elif section_match.group(2):
                return 2
            else:
                return 1

        if block.is_bold and len(content) < 100:
            return 2

        return None

    def _create_heading_chunk(
        self,
        block: TextBlock,
        document_id: str,
        page_number: int,
        level: int,
        hierarchy_stack: list,
        store: ChunkStore
    ) -> Chunk:
        """Create a chunk for a heading."""
        chunk_type = {
            1: ChunkType.CHAPTER,
            2: ChunkType.SECTION,
            3: ChunkType.SUBSECTION
        }.get(level, ChunkType.SECTION)

        chunk_id = f"H_{document_id}_{page_number}_{len(store.chunks)}"

        metadata = ChunkMetadata(
            document_id=document_id,
            page_numbers=[page_number],
            heading=block.content.strip(),
            section_hierarchy=[h[1] for h in hierarchy_stack]
        )

        parent_id = None
        for i in range(len(hierarchy_stack) - 2, -1, -1):
            parent_heading = hierarchy_stack[i][1]
            for cid, chunk in store.chunks.items():
                if chunk.metadata.heading == parent_heading:
                    parent_id = cid
                    break
            if parent_id:
                break

        return Chunk(
            id=chunk_id,
            content=block.content.strip(),
            chunk_type=chunk_type,
            metadata=metadata,
            parent_id=parent_id
        )

    def _create_hierarchical_chunk(
        self,
        blocks: list[TextBlock],
        document_id: str,
        page_numbers: list[int],
        hierarchy_stack: list,
        store: ChunkStore
    ) -> Optional[Chunk]:
        """Create a content chunk with hierarchical context."""
        content = "\n".join(b.content for b in blocks)

        if len(content.strip()) < self.config.min_chunk_size:
            return None

        chunk_type = self._classify_content(blocks)
        chunk_id = f"C_{document_id}_{page_numbers[0]}_{len(store.chunks)}"

        metadata = ChunkMetadata(
            document_id=document_id,
            page_numbers=page_numbers,
            section_hierarchy=[h[1] for h in hierarchy_stack],
            heading=hierarchy_stack[-1][1] if hierarchy_stack else None
        )

        self._enrich_metadata(metadata, blocks, content)

        parent_id = None
        for cid in reversed(store.sequence):
            chunk = store.chunks[cid]
            if chunk.chunk_type in [ChunkType.CHAPTER, ChunkType.SECTION, ChunkType.SUBSECTION]:
                parent_id = cid
                break

        return Chunk(
            id=chunk_id,
            content=content,
            chunk_type=chunk_type,
            metadata=metadata,
            parent_id=parent_id
        )

    def _classify_content(self, blocks: list[TextBlock]) -> ChunkType:
        """Classify content type from blocks."""
        has_formula = any(b.block_type == BlockType.FORMULA for b in blocks)

        content = " ".join(b.content for b in blocks).lower()

        if re.search(r'\bdef(ine|inition)\b', content):
            return ChunkType.DEFINITION
        if re.search(r'\btheorem\b|\blemma\b|\bcorollary\b', content):
            return ChunkType.THEOREM
        if re.search(r'\bproof\b', content):
            return ChunkType.PROOF
        if re.search(r'\bexample\b', content):
            return ChunkType.EXAMPLE
        if re.search(r'\bexercise\b|\bproblem\b', content):
            return ChunkType.EXERCISE

        if has_formula:
            return ChunkType.FORMULA_BLOCK

        return ChunkType.PARAGRAPH

    def _enrich_metadata(
        self,
        metadata: ChunkMetadata,
        blocks: list[TextBlock],
        content: str
    ):
        """Enrich metadata with extracted information."""
        keywords = self._extract_keywords(content)
        metadata.keywords = keywords

        content_lower = content.lower()
        metadata.is_definition = bool(re.search(r'\bdef(ine|inition)\b', content_lower))
        metadata.is_theorem = bool(re.search(r'\btheorem\b|\blemma\b', content_lower))
        metadata.is_example = bool(re.search(r'\bexample\b', content_lower))

    def _extract_keywords(self, content: str) -> list[str]:
        """Extract important keywords from content."""
        important_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            r'\b([a-z]+(?:tion|ment|ness|ity))\b',
        ]

        keywords = set()
        for pattern in important_patterns:
            matches = re.findall(pattern, content)
            keywords.update(matches)

        stopwords = {'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had'}
        keywords = [k for k in keywords if k.lower() not in stopwords and len(k) > 3]

        return keywords[:20]


class SemanticChunker(Chunker):
    """Chunk based on semantic boundaries."""

    def __init__(self, config: Optional[StructuringConfig] = None):
        super().__init__(config)
        self._embedding_model = None

    def _load_embedding_model(self):
        """Load embedding model for semantic similarity."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            except ImportError:
                logger.warning("sentence-transformers not available")

    def chunk_document(self, document: Document) -> ChunkStore:
        """Chunk based on semantic similarity."""
        self._load_embedding_model()
        store = ChunkStore()

        if not self._embedding_model:
            return super().chunk_document(document)

        sentences = []
        sentence_meta = []

        for page in document.pages:
            for block in page.text_blocks:
                block_sentences = self._split_sentences(block.content)
                for sent in block_sentences:
                    if sent.strip():
                        sentences.append(sent)
                        sentence_meta.append({
                            "page": page.page_number,
                            "block_type": block.block_type
                        })

        if not sentences:
            return store

        embeddings = self._embedding_model.encode(sentences)

        boundaries = self._find_semantic_boundaries(embeddings)

        current_sentences = []
        current_pages = set()
        chunk_idx = 0

        for i, (sent, meta) in enumerate(zip(sentences, sentence_meta)):
            current_sentences.append(sent)
            current_pages.add(meta["page"])

            if i in boundaries or len(" ".join(current_sentences)) >= self.config.max_chunk_size:
                content = " ".join(current_sentences)
                if len(content) >= self.config.min_chunk_size:
                    chunk = Chunk(
                        id=f"SC_{document.id}_{chunk_idx}",
                        content=content,
                        chunk_type=ChunkType.PARAGRAPH,
                        metadata=ChunkMetadata(
                            document_id=document.id,
                            page_numbers=sorted(current_pages)
                        )
                    )
                    store.add(chunk)
                    chunk_idx += 1

                current_sentences = []
                current_pages = set()

        if current_sentences:
            content = " ".join(current_sentences)
            if len(content) >= self.config.min_chunk_size:
                chunk = Chunk(
                    id=f"SC_{document.id}_{chunk_idx}",
                    content=content,
                    chunk_type=ChunkType.PARAGRAPH,
                    metadata=ChunkMetadata(
                        document_id=document.id,
                        page_numbers=sorted(current_pages)
                    )
                )
                store.add(chunk)

        self._link_chunks(store)
        return store

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentence_pattern = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]

    def _find_semantic_boundaries(
        self,
        embeddings: Any,
        threshold: float = 0.3
    ) -> set[int]:
        """Find boundaries where semantic similarity drops."""
        import numpy as np
        from numpy.linalg import norm

        boundaries = set()

        for i in range(1, len(embeddings)):
            sim = np.dot(embeddings[i-1], embeddings[i]) / (
                norm(embeddings[i-1]) * norm(embeddings[i])
            )

            if sim < threshold:
                boundaries.add(i)

        return boundaries
