"""Document models for representing PDF structure."""

from typing import Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib


class BoundingBox(BaseModel):
    """Represents a bounding box on a page."""

    x0: float
    y0: float
    x1: float
    y1: float
    page_width: Optional[float] = None
    page_height: Optional[float] = None

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2, (self.y0 + self.y1) / 2)

    def overlaps(self, other: "BoundingBox") -> bool:
        """Check if this bbox overlaps with another."""
        return not (
            self.x1 < other.x0 or self.x0 > other.x1 or
            self.y1 < other.y0 or self.y0 > other.y1
        )

    def iou(self, other: "BoundingBox") -> float:
        """Calculate Intersection over Union with another bbox."""
        x_left = max(self.x0, other.x0)
        y_top = max(self.y0, other.y0)
        x_right = min(self.x1, other.x1)
        y_bottom = min(self.y1, other.y1)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


class BlockType(str, Enum):
    """Types of content blocks."""

    TEXT = "text"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    FORMULA = "formula"
    TABLE = "table"
    IMAGE = "image"
    DIAGRAM = "diagram"
    CAPTION = "caption"
    FOOTER = "footer"
    HEADER = "header"


class TextBlock(BaseModel):
    """Represents a text block extracted from PDF."""

    id: str
    content: str
    block_type: BlockType = BlockType.TEXT
    bbox: Optional[BoundingBox] = None
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    is_bold: bool = False
    is_italic: bool = False
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def is_heading(self) -> bool:
        return self.block_type == BlockType.HEADING or (
            self.font_size and self.font_size > 14
        )


class ImageBlock(BaseModel):
    """Represents an image/diagram extracted from PDF."""

    id: str
    image_path: Optional[str] = None
    image_bytes: Optional[bytes] = None
    bbox: BoundingBox
    block_type: BlockType = BlockType.IMAGE
    caption: Optional[str] = None
    ocr_text: Optional[str] = None
    detected_formulas: list[str] = Field(default_factory=list)
    diagram_type: Optional[str] = None
    visual_features: Optional[list[float]] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class TableBlock(BaseModel):
    """Represents a table extracted from PDF."""

    id: str
    bbox: Optional[BoundingBox] = None
    block_type: BlockType = BlockType.TABLE
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    caption: Optional[str] = None
    markdown: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_markdown(self) -> str:
        """Convert table to markdown format."""
        if self.markdown:
            return self.markdown

        if not self.headers and not self.rows:
            return ""

        lines = []
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("| " + " | ".join(["---"] * len(self.headers)) + " |")

        for row in self.rows:
            lines.append("| " + " | ".join(str(cell) for cell in row) + " |")

        return "\n".join(lines)


class Page(BaseModel):
    """Represents a single page from a document."""

    page_number: int
    width: float
    height: float
    text_blocks: list[TextBlock] = Field(default_factory=list)
    image_blocks: list[ImageBlock] = Field(default_factory=list)
    table_blocks: list[TableBlock] = Field(default_factory=list)
    raw_text: Optional[str] = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def all_blocks(self) -> list:
        """Get all blocks sorted by vertical position."""
        blocks = self.text_blocks + self.image_blocks + self.table_blocks
        return sorted(blocks, key=lambda b: (b.bbox.y0 if b.bbox else 0, b.bbox.x0 if b.bbox else 0))

    @property
    def full_text(self) -> str:
        """Get concatenated text content."""
        texts = []
        for block in self.text_blocks:
            texts.append(block.content)
        return "\n".join(texts)


class Document(BaseModel):
    """Represents a complete document."""

    id: str
    filename: str
    filepath: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    subject: Optional[str] = None
    pages: list[Page] = Field(default_factory=list)
    total_pages: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
    processing_status: str = "pending"
    content_hash: Optional[str] = None

    def model_post_init(self, __context):
        """Initialize derived fields."""
        if not self.total_pages:
            self.total_pages = len(self.pages)

    def compute_hash(self) -> str:
        """Compute content hash for change detection."""
        content = f"{self.filename}:{self.total_pages}"
        for page in self.pages:
            content += page.full_text
        self.content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self.content_hash

    def get_page(self, page_number: int) -> Optional[Page]:
        """Get page by number (1-indexed)."""
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    @property
    def full_text(self) -> str:
        """Get all text from document."""
        return "\n\n".join(page.full_text for page in self.pages)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return self.model_dump(exclude={"image_bytes"})
