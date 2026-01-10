#!/usr/bin/env python3
"""Basic usage example for Ultimate RAG system."""

from pathlib import Path
from ultimate_rag import UltimateRAG, Config


def main():
    """Demonstrate basic usage of Ultimate RAG."""

    # Initialize with default configuration
    config = Config()
    config.generation.llm_provider = "none"  # Use structured responses

    # Create RAG instance
    rag = UltimateRAG(config)

    # Process a PDF document
    pdf_path = Path("your_document.pdf")  # Replace with actual path

    if pdf_path.exists():
        print(f"Processing {pdf_path}...")
        result = rag.process_document(pdf_path)

        if result.get("success"):
            print(f"Processed successfully!")
            print(f"  Chunks: {result['chunks']}")
            print(f"  Formulas: {result['formulas']}")

            # Query the document
            question = "What is the main concept discussed?"
            response = rag.query(question, use_llm=False)

            print(f"\nQuestion: {question}")
            print(f"Answer: {response.answer}")
            print(f"\nSources:")
            for source in response.sources[:3]:
                print(f"  - Page {source.get('page')}: {source.get('preview', '')[:100]}")

            # Search for formulas
            print("\nSearching for formulas with 'E'...")
            formulas = rag.search_formulas("E", top_k=5)
            for f in formulas:
                print(f"  - {f['data'].get('latex', f['id'])}")

            # Generate knowledge book
            print("\nGenerating knowledge book...")
            files = rag.generate_knowledge_book(Path("output"))
            for file_type, path in files.items():
                print(f"  - {file_type}: {path}")

        else:
            print(f"Processing failed: {result.get('error')}")

    else:
        print("Demo mode - no PDF provided")
        print("Usage: python basic_usage.py")
        print("Make sure to set 'pdf_path' to your document")


if __name__ == "__main__":
    main()
