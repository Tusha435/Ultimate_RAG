"""PDF extraction with layout preservation."""

import time
from pathlib import Path
from typing import Optional, Any
from loguru import logger

from ultimate_rag.extraction.base import BaseExtractor, ExtractionResult
from ultimate_rag.models.document import (
    Document, Page, TextBlock, ImageBlock, BoundingBox, BlockType
)
from ultimate_rag.config import ExtractionConfig


class PDFExtractor(BaseExtractor):
    """Extract structured content from PDF files."""

    def __init__(self, config: Optional[ExtractionConfig] = None):
        super().__init__(config)
        self.config = config or ExtractionConfig()
        self._fitz = None
        self._pdfplumber = None

    def _setup(self):
        """Import PDF libraries."""
        try:
            import fitz
            self._fitz = fitz
        except ImportError:
            logger.warning("PyMuPDF not available, falling back to pdfplumber only")

        try:
            import pdfplumber
            self._pdfplumber = pdfplumber
        except ImportError:
            logger.warning("pdfplumber not available")

    def extract(self, source: Path, **kwargs) -> ExtractionResult:
        """Extract content from PDF file."""
        self.initialize()
        start_time = time.time()

        source = Path(source)
        if not source.exists():
            return ExtractionResult(
                success=False,
                error=f"File not found: {source}"
            )

        try:
            document = self._extract_with_pymupdf(source)

            if self._pdfplumber and self.config.table_extraction_method == "pdfplumber":
                self._enhance_with_pdfplumber(document, source)

            processing_time = (time.time() - start_time) * 1000

            return ExtractionResult(
                success=True,
                data=document,
                processing_time_ms=processing_time,
                metadata={
                    "pages": document.total_pages,
                    "text_blocks": sum(len(p.text_blocks) for p in document.pages),
                    "images": sum(len(p.image_blocks) for p in document.pages)
                }
            )

        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            return ExtractionResult(
                success=False,
                error=str(e)
            )

    def _extract_with_pymupdf(self, source: Path) -> Document:
        """Extract using PyMuPDF (fitz)."""
        if not self._fitz:
            raise ImportError("PyMuPDF required for extraction")

        doc = self._fitz.open(str(source))

        document = Document(
            id=source.stem,
            filename=source.name,
            filepath=str(source),
            title=doc.metadata.get("title", ""),
            author=doc.metadata.get("author", ""),
            total_pages=len(doc)
        )

        for page_num in range(len(doc)):
            page = doc[page_num]
            extracted_page = self._extract_page(page, page_num + 1)
            document.pages.append(extracted_page)

        doc.close()
        return document

    def _extract_page(self, page: Any, page_number: int) -> Page:
        """Extract content from a single page."""
        rect = page.rect

        extracted_page = Page(
            page_number=page_number,
            width=rect.width,
            height=rect.height,
            raw_text=page.get_text("text")
        )

        blocks = page.get_text("dict", flags=self._fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        block_idx = 0

        for block in blocks:
            if block["type"] == 0:
                text_block = self._process_text_block(block, page_number, block_idx)
                if text_block and text_block.content.strip():
                    extracted_page.text_blocks.append(text_block)
                    block_idx += 1

            elif block["type"] == 1 and self.config.extract_images:
                image_block = self._process_image_block(block, page, page_number, block_idx)
                if image_block:
                    extracted_page.image_blocks.append(image_block)
                    block_idx += 1

        return extracted_page

    def _process_text_block(
        self,
        block: dict,
        page_number: int,
        block_idx: int
    ) -> Optional[TextBlock]:
        """Process a text block from PDF."""
        bbox = BoundingBox(
            x0=block["bbox"][0],
            y0=block["bbox"][1],
            x1=block["bbox"][2],
            y1=block["bbox"][3]
        )

        text_parts = []
        font_sizes = []
        is_bold = False
        is_italic = False
        font_name = None

        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "")
                if text.strip():
                    text_parts.append(text)
                    font_sizes.append(span.get("size", 12))
                    font_name = font_name or span.get("font", "")

                    flags = span.get("flags", 0)
                    if flags & 2 ** 4:
                        is_bold = True
                    if flags & 2 ** 1:
                        is_italic = True

        content = " ".join(text_parts)
        if not content.strip():
            return None

        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12

        block_type = self._classify_block_type(content, avg_font_size, is_bold)

        return TextBlock(
            id=f"TB_{page_number}_{block_idx}",
            content=content,
            block_type=block_type,
            bbox=bbox,
            font_name=font_name,
            font_size=avg_font_size,
            is_bold=is_bold,
            is_italic=is_italic
        )

    def _process_image_block(
        self,
        block: dict,
        page: Any,
        page_number: int,
        block_idx: int
    ) -> Optional[ImageBlock]:
        """Process an image block from PDF."""
        try:
            bbox = BoundingBox(
                x0=block["bbox"][0],
                y0=block["bbox"][1],
                x1=block["bbox"][2],
                y1=block["bbox"][3]
            )

            if bbox.area < 100:
                return None

            image_bytes = None
            if "image" in block:
                xref = block.get("xref", 0)
                if xref:
                    base_image = page.parent.extract_image(xref)
                    if base_image:
                        image_bytes = base_image.get("image")

            return ImageBlock(
                id=f"IB_{page_number}_{block_idx}",
                bbox=bbox,
                image_bytes=image_bytes,
                block_type=BlockType.IMAGE
            )

        except Exception as e:
            logger.debug(f"Image extraction failed: {e}")
            return None

    def _classify_block_type(
        self,
        content: str,
        font_size: float,
        is_bold: bool
    ) -> BlockType:
        """Classify block type based on content and style."""
        content_lower = content.lower().strip()

        if font_size > 16 or (font_size > 14 and is_bold):
            return BlockType.HEADING

        heading_patterns = [
            r'^chapter\s+\d+',
            r'^section\s+\d+',
            r'^\d+\.\d+',
            r'^[a-z]\)',
            r'^\(\d+\)',
        ]

        import re
        for pattern in heading_patterns:
            if re.match(pattern, content_lower):
                if font_size > 12 or is_bold:
                    return BlockType.HEADING

        if content_lower.startswith(('•', '-', '*', '–')):
            return BlockType.LIST_ITEM
        if re.match(r'^\d+\.?\s', content_lower):
            return BlockType.LIST_ITEM

        if re.match(r'^(figure|fig\.?|table|tab\.?)\s*\d+', content_lower):
            return BlockType.CAPTION

        formula_indicators = ['=', '∫', '∑', '∏', '∂', '∇', 'lim', '→']
        if any(ind in content for ind in formula_indicators) and len(content) < 200:
            return BlockType.FORMULA

        return BlockType.PARAGRAPH

    def _enhance_with_pdfplumber(self, document: Document, source: Path):
        """Enhance extraction with pdfplumber for tables."""
        try:
            with self._pdfplumber.open(str(source)) as pdf:
                for i, page in enumerate(pdf.pages):
                    if i < len(document.pages):
                        tables = page.extract_tables()
                        for table in tables:
                            if table and len(table) > 0:
                                from ultimate_rag.models.document import TableBlock
                                headers = table[0] if table else []
                                rows = table[1:] if len(table) > 1 else []

                                table_block = TableBlock(
                                    id=f"TAB_{i+1}_{len(document.pages[i].table_blocks)}",
                                    headers=[str(h) if h else "" for h in headers],
                                    rows=[[str(c) if c else "" for c in row] for row in rows]
                                )
                                document.pages[i].table_blocks.append(table_block)

        except Exception as e:
            logger.warning(f"pdfplumber enhancement failed: {e}")

    def extract_page_images(
        self,
        source: Path,
        output_dir: Path,
        dpi: int = 150
    ) -> list[Path]:
        """Extract pages as images for OCR or diagram detection."""
        try:
            from pdf2image import convert_from_path

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            images = convert_from_path(str(source), dpi=dpi)
            image_paths = []

            for i, image in enumerate(images):
                image_path = output_dir / f"page_{i+1:04d}.png"
                image.save(str(image_path), "PNG")
                image_paths.append(image_path)

            return image_paths

        except ImportError:
            logger.error("pdf2image not available")
            return []
        except Exception as e:
            logger.error(f"Page image extraction failed: {e}")
            return []
