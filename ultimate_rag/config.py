"""Configuration management for Ultimate RAG system."""

from pathlib import Path
from typing import Optional, Literal
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()


class ExtractionConfig(BaseModel):
    """Configuration for extraction layer."""

    use_ocr: bool = True
    ocr_language: str = "eng"
    formula_backend: Literal["pix2tex", "mathpix", "tesseract"] = "pix2tex"
    mathpix_app_id: Optional[str] = Field(default_factory=lambda: os.getenv("MATHPIX_APP_ID"))
    mathpix_app_key: Optional[str] = Field(default_factory=lambda: os.getenv("MATHPIX_APP_KEY"))
    diagram_detection_model: str = "yolov8n"
    diagram_confidence_threshold: float = 0.5
    table_extraction_method: Literal["pdfplumber", "camelot", "tabula"] = "pdfplumber"
    extract_images: bool = True
    image_dpi: int = 150
    batch_size: int = 10


class StructuringConfig(BaseModel):
    """Configuration for structuring layer."""

    chunk_strategy: Literal["hierarchical", "semantic", "fixed"] = "hierarchical"
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    chunk_overlap: int = 100
    preserve_formulas: bool = True
    preserve_tables: bool = True
    heading_patterns: list[str] = Field(default=[
        r"^Chapter\s+\d+",
        r"^Section\s+\d+",
        r"^\d+\.\d+",
        r"^[A-Z][A-Za-z\s]+:$"
    ])


class IndexingConfig(BaseModel):
    """Configuration for indexing layer."""

    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    use_gpu: bool = False
    faiss_index_type: Literal["flat", "ivf", "hnsw"] = "flat"
    faiss_nlist: int = 100
    formula_index_backend: Literal["sqlite", "duckdb"] = "sqlite"
    enable_keyword_index: bool = True
    enable_formula_index: bool = True
    enable_diagram_index: bool = True
    batch_embedding_size: int = 32


class RetrievalConfig(BaseModel):
    """Configuration for retrieval layer."""

    default_top_k: int = 10
    fusion_method: Literal["rrf", "weighted", "convex"] = "rrf"
    rrf_k: int = 60
    text_weight: float = 0.4
    formula_weight: float = 0.3
    diagram_weight: float = 0.2
    keyword_weight: float = 0.1
    enable_context_expansion: bool = True
    context_window_size: int = 2
    rerank_results: bool = True
    rerank_model: Optional[str] = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class GenerationConfig(BaseModel):
    """Configuration for generation layer."""

    llm_provider: Literal["openai", "anthropic", "ollama", "none"] = "none"
    llm_model: str = "gpt-4"
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    anthropic_api_key: Optional[str] = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    max_context_tokens: int = 4000
    max_response_tokens: int = 1000
    temperature: float = 0.1
    include_sources: bool = True
    generate_knowledge_book: bool = True


class Config(BaseModel):
    """Main configuration for Ultimate RAG system."""

    project_name: str = "ultimate_rag"
    data_dir: Path = Path("data")
    raw_dir: Path = Field(default=None)
    processed_dir: Path = Field(default=None)
    indices_dir: Path = Field(default=None)
    cache_dir: Path = Field(default=None)

    extraction: ExtractionConfig = Field(default_factory=ExtractionConfig)
    structuring: StructuringConfig = Field(default_factory=StructuringConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    enable_checkpointing: bool = True
    parallel_workers: int = 4

    def model_post_init(self, __context):
        """Initialize derived paths after model creation."""
        if self.raw_dir is None:
            self.raw_dir = self.data_dir / "raw"
        if self.processed_dir is None:
            self.processed_dir = self.data_dir / "processed"
        if self.indices_dir is None:
            self.indices_dir = self.data_dir / "indices"
        if self.cache_dir is None:
            self.cache_dir = self.data_dir / "cache"

    def ensure_directories(self):
        """Create all necessary directories."""
        for dir_path in [self.raw_dir, self.processed_dir, self.indices_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        (self.processed_dir / "pages").mkdir(exist_ok=True)
        (self.processed_dir / "knowledge").mkdir(exist_ok=True)

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        """Load configuration from JSON file."""
        import json
        with open(path) as f:
            return cls(**json.load(f))

    def save(self, path: Path):
        """Save configuration to JSON file."""
        import json
        with open(path, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, default=str)
