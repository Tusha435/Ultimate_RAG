"""Main pipeline orchestrator for Ultimate RAG system."""

import time
from pathlib import Path
from typing import Optional, Any, Union
from loguru import logger

from ultimate_rag.config import Config
from ultimate_rag.extraction.document_processor import DocumentProcessor
from ultimate_rag.structuring.knowledge_builder import KnowledgeBuilder
from ultimate_rag.indexing.index_manager import IndexManager
from ultimate_rag.retrieval.retriever import Retriever
from ultimate_rag.generation.response_builder import ResponseBuilder
from ultimate_rag.generation.knowledge_book import KnowledgeBookGenerator
from ultimate_rag.models.query import QueryResult, RetrievalResult
from ultimate_rag.utils.io import save_json, load_json


class UltimateRAG:
    """Main orchestrator for the Ultimate RAG system."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize the Ultimate RAG system."""
        self.config = config or Config()
        self.config.ensure_directories()

        self.document_processor = DocumentProcessor(self.config)
        self.knowledge_builder = KnowledgeBuilder(self.config)
        self.index_manager = IndexManager(self.config)
        self.retriever: Optional[Retriever] = None
        self.response_builder = ResponseBuilder(self.config)
        self.knowledge_book_gen = KnowledgeBookGenerator(self.config)

        self._knowledge_base: Optional[dict] = None
        self._is_ready = False

        logger.info("Ultimate RAG system initialized")

    def process_document(
        self,
        pdf_path: Union[str, Path],
        incremental: bool = False,
        batch_size: int = 10
    ) -> dict[str, Any]:
        """Process a PDF document through the full pipeline."""
        pdf_path = Path(pdf_path)
        start_time = time.time()

        logger.info(f"Starting document processing: {pdf_path}")

        if incremental:
            extracted = self.document_processor.process_pages_incrementally(
                pdf_path,
                batch_size=batch_size,
                progress_callback=lambda done, total: logger.info(f"Progress: {done}/{total} pages")
            )
        else:
            extracted = self.document_processor.process_document(pdf_path)

        if extracted.get("error"):
            logger.error(f"Extraction failed: {extracted['error']}")
            return extracted

        logger.info("Building knowledge base...")
        knowledge_base = self.knowledge_builder.build_knowledge_base(extracted)

        logger.info("Indexing knowledge base...")
        self.index_manager.index_knowledge_base(
            knowledge_base,
            self.config.indices_dir
        )

        self._knowledge_base = knowledge_base
        self._setup_retriever()
        self._is_ready = True

        total_time = time.time() - start_time
        logger.info(f"Document processing complete in {total_time:.1f}s")

        stats = knowledge_base.get("statistics", {})
        return {
            "document_id": knowledge_base.get("document_id"),
            "processing_time_s": total_time,
            "chunks": stats.get("total_chunks", 0),
            "formulas": stats.get("total_formulas", 0),
            "success": True
        }

    def load_knowledge_base(self, knowledge_path: Union[str, Path]):
        """Load an existing knowledge base."""
        knowledge_path = Path(knowledge_path)
        logger.info(f"Loading knowledge base from {knowledge_path}")

        self._knowledge_base = load_json(knowledge_path)

        doc_id = self._knowledge_base.get("document_id", "")
        self.index_manager.load(self.config.indices_dir, doc_id)

        self._setup_retriever()
        self._is_ready = True

        logger.info("Knowledge base loaded successfully")

    def _setup_retriever(self):
        """Set up the retriever with current knowledge base."""
        self.retriever = Retriever(
            index_manager=self.index_manager,
            knowledge_base=self._knowledge_base,
            config=self.config
        )

    def query(
        self,
        question: str,
        top_k: int = 10,
        use_llm: bool = True
    ) -> QueryResult:
        """Query the knowledge base."""
        if not self._is_ready or not self.retriever:
            raise RuntimeError("System not ready. Process or load a document first.")

        start_time = time.time()

        retrieval_result = self.retriever.retrieve(question, top_k)

        query_result = self.response_builder.build_response(
            retrieval_result,
            use_llm=use_llm
        )

        query_result.processing_time_ms = (time.time() - start_time) * 1000

        return query_result

    def search(
        self,
        query_text: str,
        top_k: int = 10
    ) -> RetrievalResult:
        """Search without generating a response."""
        if not self._is_ready or not self.retriever:
            raise RuntimeError("System not ready. Process or load a document first.")

        return self.retriever.retrieve(query_text, top_k)

    def search_formulas(
        self,
        query: str,
        top_k: int = 10
    ) -> list[dict]:
        """Search for formulas."""
        if not self._is_ready or not self.retriever:
            raise RuntimeError("System not ready.")

        if '=' in query or '\\' in query:
            results = self.retriever.retrieve_formula(query, top_k)
        else:
            results = self.retriever.retrieve_by_variable(query, top_k)

        return [
            {"id": fid, "score": score, "data": data}
            for fid, score, data in results
        ]

    def generate_knowledge_book(
        self,
        output_dir: Optional[Union[str, Path]] = None
    ) -> dict[str, Path]:
        """Generate a knowledge book from the processed document."""
        if not self._knowledge_base:
            raise RuntimeError("No knowledge base available.")

        output_dir = Path(output_dir) if output_dir else self.config.data_dir / "output"

        return self.knowledge_book_gen.generate(
            self._knowledge_base,
            output_dir
        )

    def explain_retrieval(self, result: RetrievalResult) -> str:
        """Get explanation of retrieval process."""
        if not self.retriever:
            return "Retriever not initialized"
        return self.retriever.explain_retrieval(result)

    def get_statistics(self) -> dict[str, Any]:
        """Get system statistics."""
        stats = {
            "is_ready": self._is_ready,
            "indices": self.index_manager.get_statistics() if self._is_ready else {}
        }

        if self._knowledge_base:
            stats["knowledge_base"] = self._knowledge_base.get("statistics", {})
            stats["document_id"] = self._knowledge_base.get("document_id")

        return stats

    def interactive_session(self):
        """Start an interactive query session."""
        if not self._is_ready:
            print("System not ready. Please process or load a document first.")
            return

        print("\n" + "=" * 60)
        print("Ultimate RAG Interactive Session")
        print("=" * 60)
        print("Commands: /quit, /stats, /explain, /formulas <query>")
        print("=" * 60 + "\n")

        last_result = None

        while True:
            try:
                query = input("\nYou: ").strip()

                if not query:
                    continue

                if query.lower() in ['/quit', '/exit', '/q']:
                    print("Goodbye!")
                    break

                if query.lower() == '/stats':
                    stats = self.get_statistics()
                    print(f"\nStatistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue

                if query.lower() == '/explain' and last_result:
                    print(f"\n{self.explain_retrieval(last_result.retrieval)}")
                    continue

                if query.lower().startswith('/formulas '):
                    formula_query = query[10:].strip()
                    results = self.search_formulas(formula_query)
                    print(f"\nFound {len(results)} formulas:")
                    for r in results[:5]:
                        print(f"  - {r['data'].get('latex', r['id'])} (score: {r['score']:.2f})")
                    continue

                result = self.query(query)
                last_result = result

                print(f"\n{self.response_builder.format_for_display(result)}")

            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                print(f"\nError: {e}")

    @property
    def is_ready(self) -> bool:
        """Check if system is ready for queries."""
        return self._is_ready


def create_rag_system(
    pdf_path: Optional[Union[str, Path]] = None,
    knowledge_path: Optional[Union[str, Path]] = None,
    config: Optional[Config] = None
) -> UltimateRAG:
    """Factory function to create and initialize RAG system."""
    rag = UltimateRAG(config)

    if pdf_path:
        rag.process_document(pdf_path)
    elif knowledge_path:
        rag.load_knowledge_base(knowledge_path)

    return rag
