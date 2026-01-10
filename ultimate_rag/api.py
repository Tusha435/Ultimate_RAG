"""FastAPI REST API for Ultimate RAG system."""

from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import tempfile
import shutil
from loguru import logger

from ultimate_rag import UltimateRAG, Config

app = FastAPI(
    title="Ultimate RAG API",
    description="Multi-modal Retrieval-Augmented Generation for Technical Documents",
    version="1.0.0"
)

rag_instances: dict[str, UltimateRAG] = {}


class QueryRequest(BaseModel):
    """Query request model."""
    question: str
    document_id: str
    top_k: int = Field(default=10, ge=1, le=100)
    use_llm: bool = True


class QueryResponse(BaseModel):
    """Query response model."""
    question: str
    answer: str
    sources: list[dict]
    confidence: float
    processing_time_ms: float
    formulas_used: list[str]


class SearchRequest(BaseModel):
    """Search request model."""
    query: str
    document_id: str
    top_k: int = Field(default=10, ge=1, le=100)
    search_types: Optional[list[str]] = None


class SearchResult(BaseModel):
    """Search result item."""
    id: str
    source_type: str
    content: str
    score: float
    page: Optional[int]
    section: Optional[str]


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    results: list[SearchResult]
    total_results: int
    retrieval_time_ms: float


class ProcessingStatus(BaseModel):
    """Document processing status."""
    document_id: str
    status: str
    progress: Optional[float] = None
    chunks: Optional[int] = None
    formulas: Optional[int] = None
    error: Optional[str] = None


processing_status: dict[str, ProcessingStatus] = {}


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Ultimate RAG API",
        "version": "1.0.0",
        "endpoints": [
            "/process", "/query", "/search", "/formulas",
            "/statistics", "/knowledge-book"
        ]
    }


@app.post("/process")
async def process_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    incremental: bool = False
):
    """Upload and process a PDF document."""
    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files are supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    doc_id = Path(file.filename).stem

    processing_status[doc_id] = ProcessingStatus(
        document_id=doc_id,
        status="processing"
    )

    background_tasks.add_task(
        _process_document_task,
        doc_id,
        tmp_path,
        incremental
    )

    return {
        "document_id": doc_id,
        "status": "processing",
        "message": "Document processing started"
    }


async def _process_document_task(
    doc_id: str,
    pdf_path: Path,
    incremental: bool
):
    """Background task for document processing."""
    try:
        config = Config()
        rag = UltimateRAG(config)

        result = rag.process_document(pdf_path, incremental=incremental)

        if result.get("success"):
            rag_instances[doc_id] = rag
            processing_status[doc_id] = ProcessingStatus(
                document_id=doc_id,
                status="completed",
                chunks=result.get("chunks"),
                formulas=result.get("formulas")
            )
        else:
            processing_status[doc_id] = ProcessingStatus(
                document_id=doc_id,
                status="failed",
                error=result.get("error")
            )

    except Exception as e:
        logger.error(f"Processing failed: {e}")
        processing_status[doc_id] = ProcessingStatus(
            document_id=doc_id,
            status="failed",
            error=str(e)
        )
    finally:
        if pdf_path.exists():
            pdf_path.unlink()


@app.get("/status/{document_id}")
async def get_processing_status(document_id: str):
    """Get document processing status."""
    if document_id not in processing_status:
        raise HTTPException(404, f"Document {document_id} not found")

    return processing_status[document_id]


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Query a processed document."""
    if request.document_id not in rag_instances:
        raise HTTPException(404, f"Document {request.document_id} not loaded")

    rag = rag_instances[request.document_id]

    try:
        result = rag.query(
            request.question,
            top_k=request.top_k,
            use_llm=request.use_llm
        )

        return QueryResponse(
            question=request.question,
            answer=result.answer,
            sources=result.sources,
            confidence=result.confidence_breakdown.get("retrieval", 0),
            processing_time_ms=result.processing_time_ms,
            formulas_used=result.formulas_used
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Search a processed document."""
    if request.document_id not in rag_instances:
        raise HTTPException(404, f"Document {request.document_id} not loaded")

    rag = rag_instances[request.document_id]

    try:
        retrieval = rag.search(request.query, top_k=request.top_k)

        results = [
            SearchResult(
                id=r.id,
                source_type=r.source_type,
                content=r.content[:500] if r.content else "",
                score=r.score,
                page=r.page_number,
                section=r.section
            )
            for r in retrieval.results
        ]

        return SearchResponse(
            query=request.query,
            results=results,
            total_results=retrieval.total_results,
            retrieval_time_ms=retrieval.retrieval_time_ms
        )

    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/formulas/{document_id}")
async def search_formulas(
    document_id: str,
    query: str,
    top_k: int = 10
):
    """Search for formulas in a document."""
    if document_id not in rag_instances:
        raise HTTPException(404, f"Document {document_id} not loaded")

    rag = rag_instances[document_id]

    try:
        results = rag.search_formulas(query, top_k)

        return {
            "query": query,
            "results": [
                {
                    "id": r["id"],
                    "latex": r["data"].get("latex", ""),
                    "plain_text": r["data"].get("plain_text", ""),
                    "score": r["score"],
                    "page": r["data"].get("source_page"),
                    "domain": r["data"].get("domain")
                }
                for r in results
            ]
        }

    except Exception as e:
        logger.error(f"Formula search failed: {e}")
        raise HTTPException(500, str(e))


@app.get("/statistics/{document_id}")
async def get_statistics(document_id: str):
    """Get statistics for a processed document."""
    if document_id not in rag_instances:
        raise HTTPException(404, f"Document {document_id} not loaded")

    rag = rag_instances[document_id]
    return rag.get_statistics()


@app.post("/knowledge-book/{document_id}")
async def generate_knowledge_book(
    document_id: str,
    output_dir: Optional[str] = None
):
    """Generate knowledge book for a document."""
    if document_id not in rag_instances:
        raise HTTPException(404, f"Document {document_id} not loaded")

    rag = rag_instances[document_id]

    try:
        files = rag.generate_knowledge_book(output_dir)

        return {
            "document_id": document_id,
            "generated_files": {k: str(v) for k, v in files.items()}
        }

    except Exception as e:
        logger.error(f"Knowledge book generation failed: {e}")
        raise HTTPException(500, str(e))


@app.post("/load/{document_id}")
async def load_knowledge_base(
    document_id: str,
    knowledge_path: str
):
    """Load an existing knowledge base."""
    try:
        config = Config()
        rag = UltimateRAG(config)
        rag.load_knowledge_base(knowledge_path)

        rag_instances[document_id] = rag

        return {
            "document_id": document_id,
            "status": "loaded",
            "statistics": rag.get_statistics()
        }

    except Exception as e:
        logger.error(f"Load failed: {e}")
        raise HTTPException(500, str(e))


@app.delete("/unload/{document_id}")
async def unload_document(document_id: str):
    """Unload a document from memory."""
    if document_id in rag_instances:
        del rag_instances[document_id]

    if document_id in processing_status:
        del processing_status[document_id]

    return {"document_id": document_id, "status": "unloaded"}


@app.get("/documents")
async def list_documents():
    """List all loaded documents."""
    return {
        "documents": [
            {
                "document_id": doc_id,
                "is_ready": rag.is_ready,
                "statistics": rag.get_statistics() if rag.is_ready else None
            }
            for doc_id, rag in rag_instances.items()
        ]
    }


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
