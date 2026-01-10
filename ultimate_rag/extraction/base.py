"""Base extractor interface."""

from abc import ABC, abstractmethod
from typing import Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from loguru import logger


class ExtractionResult(BaseModel):
    """Result of an extraction operation."""

    success: bool = True
    data: Any = None
    error: Optional[str] = None
    warnings: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    processing_time_ms: float = 0.0

    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)
        logger.warning(warning)


class BaseExtractor(ABC):
    """Base class for all extractors."""

    def __init__(self, config: Optional[Any] = None):
        self.config = config
        self._initialized = False

    def initialize(self):
        """Initialize extractor resources."""
        if not self._initialized:
            self._setup()
            self._initialized = True

    def _setup(self):
        """Override to setup specific resources."""
        pass

    @abstractmethod
    def extract(self, source: Any, **kwargs) -> ExtractionResult:
        """Extract content from source."""
        pass

    def extract_batch(
        self,
        sources: list[Any],
        **kwargs
    ) -> list[ExtractionResult]:
        """Extract from multiple sources."""
        self.initialize()
        results = []
        for source in sources:
            result = self.extract(source, **kwargs)
            results.append(result)
        return results

    def validate_source(self, source: Any) -> bool:
        """Validate that source is processable."""
        return True

    def cleanup(self):
        """Cleanup resources."""
        pass

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
