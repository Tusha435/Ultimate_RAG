"""Main document processor orchestrating all extractors."""

import time
from pathlib import Path
from typing import Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

from ultimate_rag.extraction.base import ExtractionResult
from ultimate_rag.extraction.pdf_extractor import PDFExtractor
from ultimate_rag.extraction.formula_extractor import FormulaExtractor
from ultimate_rag.extraction.diagram_extractor import DiagramExtractor
from ultimate_rag.extraction.table_extractor import TableExtractor
from ultimate_rag.extraction.ocr import OCRProcessor
from ultimate_rag.models.document import Document
from ultimate_rag.models.formula import Formula
from ultimate_rag.config import Config, ExtractionConfig
from ultimate_rag.utils.io import save_json, load_json, save_checkpoint, load_checkpoint


class DocumentProcessor:
    """Orchestrates document extraction pipeline."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.extraction_config = self.config.extraction

        self.pdf_extractor = PDFExtractor(self.extraction_config)
        self.formula_extractor = FormulaExtractor(self.extraction_config)
        self.diagram_extractor = DiagramExtractor(self.extraction_config)
        self.table_extractor = TableExtractor(self.extraction_config)
        self.ocr_processor = OCRProcessor(self.extraction_config)

    def process_document(
        self,
        pdf_path: Path,
        output_dir: Optional[Path] = None,
        use_cache: bool = True
    ) -> dict[str, Any]:
        """Process a complete document through extraction pipeline."""
        pdf_path = Path(pdf_path)
        output_dir = output_dir or self.config.processed_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        doc_id = pdf_path.stem
        cache_path = output_dir / f"{doc_id}_extracted.json"

        if use_cache and cache_path.exists():
            logger.info(f"Loading cached extraction: {cache_path}")
            return load_json(cache_path)

        start_time = time.time()
        logger.info(f"Processing document: {pdf_path}")

        result = {
            "document_id": doc_id,
            "source_path": str(pdf_path),
            "document": None,
            "formulas": [],
            "diagrams": [],
            "tables": [],
            "processing_stats": {}
        }

        pdf_result = self.pdf_extractor.extract(pdf_path)
        if not pdf_result.success:
            logger.error(f"PDF extraction failed: {pdf_result.error}")
            result["error"] = pdf_result.error
            return result

        document = pdf_result.data
        result["document"] = document.model_dump()
        result["processing_stats"]["pdf_extraction_ms"] = pdf_result.processing_time_ms

        logger.info(f"Extracted {document.total_pages} pages, "
                   f"{sum(len(p.text_blocks) for p in document.pages)} text blocks")

        formula_start = time.time()
        formulas = self.formula_extractor.extract_from_document(
            document,
            extract_from_images=self.extraction_config.extract_images
        )
        result["formulas"] = [f.model_dump() for f in formulas]
        result["processing_stats"]["formula_extraction_ms"] = (time.time() - formula_start) * 1000
        logger.info(f"Extracted {len(formulas)} formulas")

        table_start = time.time()
        table_result = self.table_extractor.extract(pdf_path)
        if table_result.success:
            result["tables"] = [t.model_dump() for t in table_result.data]
        result["processing_stats"]["table_extraction_ms"] = (time.time() - table_start) * 1000
        logger.info(f"Extracted {len(result['tables'])} tables")

        if self.extraction_config.extract_images:
            diagram_start = time.time()
            diagrams = []
            for page in document.pages:
                page_diagrams = self.diagram_extractor._extract_from_page(page)
                diagrams.extend(page_diagrams)

            result["diagrams"] = [d.model_dump() for d in diagrams]
            result["processing_stats"]["diagram_extraction_ms"] = (time.time() - diagram_start) * 1000
            logger.info(f"Detected {len(diagrams)} diagrams")

        total_time = (time.time() - start_time) * 1000
        result["processing_stats"]["total_ms"] = total_time

        save_json(result, cache_path)
        logger.info(f"Processing complete in {total_time:.0f}ms, saved to {cache_path}")

        return result

    def process_batch(
        self,
        pdf_paths: list[Path],
        output_dir: Optional[Path] = None,
        max_workers: int = 4,
        progress_callback: Optional[callable] = None
    ) -> list[dict[str, Any]]:
        """Process multiple documents in parallel."""
        output_dir = output_dir or self.config.processed_dir
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self.process_document, path, output_dir
                ): path
                for path in pdf_paths
            }

            for future in as_completed(futures):
                path = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                    if progress_callback:
                        progress_callback(path, result)
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
                    results.append({
                        "document_id": path.stem,
                        "error": str(e)
                    })

        return results

    def process_pages_incrementally(
        self,
        pdf_path: Path,
        output_dir: Optional[Path] = None,
        batch_size: int = 10,
        progress_callback: Optional[callable] = None
    ) -> dict[str, Any]:
        """Process large documents incrementally with checkpointing."""
        pdf_path = Path(pdf_path)
        output_dir = output_dir or self.config.processed_dir
        doc_id = pdf_path.stem

        checkpoint = load_checkpoint(output_dir, doc_id, "extraction")
        if checkpoint:
            processed_pages = checkpoint.get("processed_pages", set())
            result = checkpoint.get("result", self._init_result(doc_id, pdf_path))
        else:
            processed_pages = set()
            result = self._init_result(doc_id, pdf_path)

        try:
            import fitz
            doc = fitz.open(str(pdf_path))
            total_pages = len(doc)
        except Exception as e:
            logger.error(f"Failed to open PDF: {e}")
            return {"error": str(e)}

        for batch_start in range(0, total_pages, batch_size):
            batch_end = min(batch_start + batch_size, total_pages)

            for page_num in range(batch_start, batch_end):
                if page_num in processed_pages:
                    continue

                try:
                    page = doc[page_num]
                    self._process_single_page(page, page_num + 1, result)
                    processed_pages.add(page_num)

                except Exception as e:
                    logger.error(f"Failed to process page {page_num + 1}: {e}")

            save_checkpoint(
                {"processed_pages": processed_pages, "result": result},
                output_dir, doc_id, "extraction"
            )

            if progress_callback:
                progress_callback(batch_end, total_pages)

        doc.close()

        cache_path = output_dir / f"{doc_id}_extracted.json"
        save_json(result, cache_path)

        return result

    def _init_result(self, doc_id: str, pdf_path: Path) -> dict:
        """Initialize result dictionary."""
        return {
            "document_id": doc_id,
            "source_path": str(pdf_path),
            "document": {
                "id": doc_id,
                "filename": pdf_path.name,
                "pages": []
            },
            "formulas": [],
            "diagrams": [],
            "tables": [],
            "processing_stats": {}
        }

    def _process_single_page(
        self,
        fitz_page: Any,
        page_number: int,
        result: dict
    ):
        """Process a single page and update result."""
        page_data = self.pdf_extractor._extract_page(fitz_page, page_number)

        result["document"]["pages"].append(page_data.model_dump())

        for block in page_data.text_blocks:
            formula_result = self.formula_extractor.extract(block.content)
            if formula_result.success and formula_result.data:
                for formula in formula_result.data:
                    formula.source_page = page_number
                    result["formulas"].append(formula.model_dump())

        for img_block in page_data.image_blocks:
            if img_block.image_bytes:
                img_formulas = self.formula_extractor._extract_from_image_bytes(
                    img_block.image_bytes
                )
                for formula in img_formulas:
                    formula.source_page = page_number
                    result["formulas"].append(formula.model_dump())

                diagrams = self.diagram_extractor._extract_from_bytes(
                    img_block.image_bytes
                )
                for diagram in diagrams:
                    diagram.metadata["page"] = page_number
                    result["diagrams"].append(diagram.model_dump())
