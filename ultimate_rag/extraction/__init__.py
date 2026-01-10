"""Extraction layer for Ultimate RAG system."""

from ultimate_rag.extraction.base import BaseExtractor, ExtractionResult
from ultimate_rag.extraction.pdf_extractor import PDFExtractor
from ultimate_rag.extraction.formula_extractor import FormulaExtractor
from ultimate_rag.extraction.diagram_extractor import DiagramExtractor
from ultimate_rag.extraction.table_extractor import TableExtractor
from ultimate_rag.extraction.ocr import OCRProcessor
from ultimate_rag.extraction.document_processor import DocumentProcessor

__all__ = [
    "BaseExtractor", "ExtractionResult",
    "PDFExtractor", "FormulaExtractor", "DiagramExtractor", "TableExtractor",
    "OCRProcessor", "DocumentProcessor"
]
