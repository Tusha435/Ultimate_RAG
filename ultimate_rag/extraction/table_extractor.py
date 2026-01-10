"""Table extraction from documents."""

import time
from pathlib import Path
from typing import Optional, Any
from loguru import logger

from ultimate_rag.extraction.base import BaseExtractor, ExtractionResult
from ultimate_rag.models.document import TableBlock, BoundingBox
from ultimate_rag.config import ExtractionConfig


class TableExtractor(BaseExtractor):
    """Extract tables from PDF documents."""

    def __init__(self, config: Optional[ExtractionConfig] = None):
        super().__init__(config)
        self.config = config or ExtractionConfig()

    def _setup(self):
        """Initialize table extraction dependencies."""
        pass

    def extract(self, source: Any, **kwargs) -> ExtractionResult:
        """Extract tables from PDF or page."""
        self.initialize()
        start_time = time.time()

        try:
            if isinstance(source, (str, Path)):
                tables = self._extract_from_pdf(Path(source))
            elif hasattr(source, 'page_number'):
                tables = self._extract_from_page(source, kwargs.get('pdf_path'))
            else:
                return ExtractionResult(
                    success=False,
                    error=f"Unsupported source type: {type(source)}"
                )

            processing_time = (time.time() - start_time) * 1000

            return ExtractionResult(
                success=True,
                data=tables,
                processing_time_ms=processing_time,
                metadata={"table_count": len(tables)}
            )

        except Exception as e:
            logger.error(f"Table extraction failed: {e}")
            return ExtractionResult(success=False, error=str(e))

    def _extract_from_pdf(self, pdf_path: Path) -> list[TableBlock]:
        """Extract all tables from PDF."""
        tables = []

        if self.config.table_extraction_method == "pdfplumber":
            tables = self._extract_with_pdfplumber(pdf_path)
        elif self.config.table_extraction_method == "camelot":
            tables = self._extract_with_camelot(pdf_path)
        else:
            tables = self._extract_with_pdfplumber(pdf_path)

        return tables

    def _extract_with_pdfplumber(self, pdf_path: Path) -> list[TableBlock]:
        """Extract tables using pdfplumber."""
        try:
            import pdfplumber

            tables = []

            with pdfplumber.open(str(pdf_path)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    page_tables = page.extract_tables()

                    for idx, table in enumerate(page_tables):
                        if not table or len(table) == 0:
                            continue

                        headers = []
                        rows = []

                        first_row = table[0]
                        if self._is_header_row(first_row):
                            headers = [str(cell) if cell else "" for cell in first_row]
                            rows = [[str(cell) if cell else "" for cell in row] for row in table[1:]]
                        else:
                            rows = [[str(cell) if cell else "" for cell in row] for row in table]

                        table_block = TableBlock(
                            id=f"TAB_{page_num}_{idx}",
                            headers=headers,
                            rows=rows,
                            metadata={
                                "page": page_num,
                                "extraction_method": "pdfplumber"
                            }
                        )
                        tables.append(table_block)

            return tables

        except ImportError:
            logger.error("pdfplumber not available")
            return []
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return []

    def _extract_with_camelot(self, pdf_path: Path) -> list[TableBlock]:
        """Extract tables using camelot."""
        try:
            import camelot

            tables = []
            camelot_tables = camelot.read_pdf(str(pdf_path), pages='all')

            for idx, table in enumerate(camelot_tables):
                df = table.df

                headers = list(df.iloc[0]) if len(df) > 0 else []
                rows = df.iloc[1:].values.tolist() if len(df) > 1 else []

                table_block = TableBlock(
                    id=f"TAB_{table.page}_{idx}",
                    headers=[str(h) for h in headers],
                    rows=[[str(cell) for cell in row] for row in rows],
                    metadata={
                        "page": table.page,
                        "accuracy": table.accuracy,
                        "extraction_method": "camelot"
                    }
                )
                tables.append(table_block)

            return tables

        except ImportError:
            logger.error("camelot not available")
            return []
        except Exception as e:
            logger.error(f"camelot extraction failed: {e}")
            return []

    def _extract_from_page(self, page: Any, pdf_path: Optional[Path]) -> list[TableBlock]:
        """Extract tables from a specific page."""
        if pdf_path:
            try:
                import pdfplumber

                with pdfplumber.open(str(pdf_path)) as pdf:
                    if page.page_number <= len(pdf.pages):
                        pdf_page = pdf.pages[page.page_number - 1]
                        page_tables = pdf_page.extract_tables()

                        tables = []
                        for idx, table in enumerate(page_tables):
                            if not table:
                                continue

                            headers = []
                            rows = []

                            if self._is_header_row(table[0]):
                                headers = [str(c) if c else "" for c in table[0]]
                                rows = [[str(c) if c else "" for c in r] for r in table[1:]]
                            else:
                                rows = [[str(c) if c else "" for c in r] for r in table]

                            tables.append(TableBlock(
                                id=f"TAB_{page.page_number}_{idx}",
                                headers=headers,
                                rows=rows
                            ))

                        return tables

            except Exception as e:
                logger.error(f"Page table extraction failed: {e}")

        return list(page.table_blocks) if hasattr(page, 'table_blocks') else []

    def _is_header_row(self, row: list) -> bool:
        """Determine if row is likely a header."""
        if not row:
            return False

        non_empty = [cell for cell in row if cell and str(cell).strip()]
        if len(non_empty) < len(row) * 0.5:
            return False

        numeric_count = sum(1 for cell in non_empty if self._is_numeric(str(cell)))
        if numeric_count > len(non_empty) * 0.5:
            return False

        return True

    def _is_numeric(self, text: str) -> bool:
        """Check if text is numeric."""
        text = text.strip().replace(',', '').replace('%', '').replace('$', '')
        try:
            float(text)
            return True
        except ValueError:
            return False

    def table_to_markdown(self, table: TableBlock) -> str:
        """Convert table to markdown format."""
        return table.to_markdown()

    def table_to_dict(self, table: TableBlock) -> list[dict]:
        """Convert table to list of dictionaries."""
        if not table.headers:
            return [{"row": row} for row in table.rows]

        return [
            {h: row[i] if i < len(row) else "" for i, h in enumerate(table.headers)}
            for row in table.rows
        ]

    def table_to_dataframe(self, table: TableBlock) -> Any:
        """Convert table to pandas DataFrame."""
        try:
            import pandas as pd

            if table.headers:
                return pd.DataFrame(table.rows, columns=table.headers)
            return pd.DataFrame(table.rows)

        except ImportError:
            logger.error("pandas not available")
            return None
