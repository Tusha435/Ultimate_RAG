#!/usr/bin/env python3
"""Advanced usage examples for Ultimate RAG system."""

from pathlib import Path
from ultimate_rag import UltimateRAG, Config
from ultimate_rag.retrieval.query_processor import QueryProcessor
from ultimate_rag.indexing.index_manager import IndexManager


def custom_configuration():
    """Example of custom configuration."""
    config = Config()

    # Extraction settings
    config.extraction.formula_backend = "pix2tex"
    config.extraction.use_ocr = True
    config.extraction.extract_images = True

    # Chunking settings
    config.structuring.chunk_strategy = "hierarchical"
    config.structuring.max_chunk_size = 1500
    config.structuring.chunk_overlap = 100

    # Indexing settings
    config.indexing.embedding_model = "all-MiniLM-L6-v2"
    config.indexing.enable_formula_index = True
    config.indexing.enable_diagram_index = True

    # Retrieval settings
    config.retrieval.fusion_method = "rrf"
    config.retrieval.default_top_k = 15
    config.retrieval.enable_context_expansion = True

    # Generation settings
    config.generation.llm_provider = "ollama"  # or "openai", "anthropic"
    config.generation.llm_model = "llama2"
    config.generation.include_sources = True

    return config


def domain_specific_queries(rag: UltimateRAG):
    """Examples of domain-specific queries."""

    # Physics queries
    physics_queries = [
        "What is Newton's second law of motion?",
        "Explain the concept of electromagnetic induction",
        "Find the formula for kinetic energy",
        "Show me diagrams related to electric fields"
    ]

    # Math queries
    math_queries = [
        "What is the derivative of sin(x)?",
        "Explain the fundamental theorem of calculus",
        "Find formulas involving integrals",
        "What is a Taylor series?"
    ]

    # Chemistry queries
    chemistry_queries = [
        "What is oxidation?",
        "Explain chemical equilibrium",
        "Find formulas for reaction rates",
        "What is Le Chatelier's principle?"
    ]

    # Data Science queries
    ds_queries = [
        "What is gradient descent?",
        "Explain backpropagation",
        "Find formulas for loss functions",
        "What is cross-validation?"
    ]

    print("\n--- Domain-Specific Query Examples ---\n")

    for domain, queries in [
        ("Physics", physics_queries),
        ("Mathematics", math_queries),
        ("Chemistry", chemistry_queries),
        ("Data Science", ds_queries)
    ]:
        print(f"\n{domain} Queries:")
        for q in queries[:2]:  # Just show first 2
            print(f"  - {q}")


def formula_search_examples(rag: UltimateRAG):
    """Examples of formula-specific searches."""
    if not rag.is_ready:
        print("RAG system not ready")
        return

    print("\n--- Formula Search Examples ---\n")

    # Search by variable
    print("Formulas containing variable 'E':")
    results = rag.search_formulas("E", top_k=5)
    for r in results:
        print(f"  - {r['data'].get('latex', r['id'])} (domain: {r['data'].get('domain')})")

    # Search by LaTeX pattern
    print("\nFormulas similar to F=ma:")
    results = rag.search_formulas("F = ma", top_k=5)
    for r in results:
        print(f"  - {r['data'].get('latex', r['id'])}")


def batch_processing_example():
    """Example of processing multiple documents."""
    from ultimate_rag.extraction.document_processor import DocumentProcessor

    config = Config()
    processor = DocumentProcessor(config)

    pdf_paths = [
        Path("physics_textbook.pdf"),
        Path("math_reference.pdf"),
        Path("chemistry_handbook.pdf")
    ]

    existing_paths = [p for p in pdf_paths if p.exists()]

    if existing_paths:
        results = processor.process_batch(
            existing_paths,
            max_workers=2,
            progress_callback=lambda path, result: print(f"Processed: {path}")
        )

        for result in results:
            print(f"Document: {result.get('document_id')}")
            print(f"  Status: {'Success' if not result.get('error') else 'Failed'}")
    else:
        print("No PDF files found for batch processing demo")


def query_analysis_example():
    """Example of query processing and analysis."""
    from ultimate_rag.models.query import QueryType, QueryIntent

    processor = QueryProcessor()

    example_queries = [
        "What is the definition of momentum?",
        "E = mc^2",
        "Explain the difference between velocity and acceleration",
        "Show me the diagram of a magnetic field",
        "Derive the formula for centripetal acceleration",
        "Find all formulas with the nabla operator"
    ]

    print("\n--- Query Analysis ---\n")

    for query_text in example_queries:
        query = processor.process(query_text)

        print(f"Query: {query_text}")
        print(f"  Type: {query.query_type.value}")
        print(f"  Intent: {query.intent.value}")
        print(f"  Domain: {query.domain_hint or 'general'}")
        print(f"  Keywords: {', '.join(query.keywords[:5])}")
        if query.formulas:
            print(f"  Formulas detected: {query.formulas}")
        print()


def interactive_demo():
    """Run an interactive demonstration."""
    config = Config()
    config.generation.llm_provider = "none"

    print("\n" + "=" * 60)
    print("Ultimate RAG - Interactive Demo")
    print("=" * 60)

    print("\nThis demo shows the system capabilities without a PDF.")
    print("To use with your documents, run:")
    print("  ultimate-rag process your_document.pdf")
    print("  ultimate-rag interactive -k data/processed/knowledge/your_document_knowledge.json")

    print("\n--- Query Analysis Demo ---")
    query_analysis_example()

    print("\n--- Domain Query Examples ---")
    rag = UltimateRAG(config)
    domain_specific_queries(rag)


def main():
    """Run advanced examples."""
    print("Ultimate RAG - Advanced Usage Examples")
    print("=" * 50)

    # Show custom configuration
    print("\n1. Custom Configuration:")
    config = custom_configuration()
    print(f"   Embedding model: {config.indexing.embedding_model}")
    print(f"   Chunk strategy: {config.structuring.chunk_strategy}")
    print(f"   LLM provider: {config.generation.llm_provider}")

    # Run interactive demo
    interactive_demo()


if __name__ == "__main__":
    main()
