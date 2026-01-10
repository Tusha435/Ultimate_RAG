"""Command-line interface for Ultimate RAG system."""

import typer
from pathlib import Path
from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel
from rich.markdown import Markdown

app = typer.Typer(
    name="ultimate-rag",
    help="Ultimate RAG: Multi-modal Retrieval-Augmented Generation System",
    add_completion=False
)

console = Console()


@app.command()
def process(
    pdf_path: Path = typer.Argument(..., help="Path to PDF file to process"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    incremental: bool = typer.Option(False, "--incremental", "-i", help="Use incremental processing"),
    batch_size: int = typer.Option(10, "--batch-size", "-b", help="Batch size for incremental processing"),
    config_path: Optional[Path] = typer.Option(None, "--config", "-c", help="Config file path")
):
    """Process a PDF document through the RAG pipeline."""
    from ultimate_rag import UltimateRAG, Config

    console.print(Panel.fit(f"[bold blue]Processing:[/] {pdf_path}"))

    config = Config.from_file(config_path) if config_path else Config()
    if output_dir:
        config.data_dir = output_dir

    rag = UltimateRAG(config)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing document...", total=None)

        result = rag.process_document(
            pdf_path,
            incremental=incremental,
            batch_size=batch_size
        )

        progress.update(task, completed=True)

    if result.get("success"):
        table = Table(title="Processing Complete")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Document ID", result.get("document_id", "N/A"))
        table.add_row("Processing Time", f"{result.get('processing_time_s', 0):.1f}s")
        table.add_row("Chunks", str(result.get("chunks", 0)))
        table.add_row("Formulas", str(result.get("formulas", 0)))

        console.print(table)
    else:
        console.print(f"[red]Error:[/] {result.get('error', 'Unknown error')}")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    knowledge_path: Optional[Path] = typer.Option(None, "--knowledge", "-k", help="Knowledge base path"),
    indices_dir: Optional[Path] = typer.Option(None, "--indices", "-i", help="Indices directory"),
    top_k: int = typer.Option(10, "--top-k", "-n", help="Number of results"),
    no_llm: bool = typer.Option(False, "--no-llm", help="Disable LLM generation"),
    explain: bool = typer.Option(False, "--explain", "-e", help="Show retrieval explanation")
):
    """Query the knowledge base."""
    from ultimate_rag import UltimateRAG, Config
    from ultimate_rag.utils.io import load_json

    if not knowledge_path:
        console.print("[red]Error:[/] Please provide --knowledge path")
        raise typer.Exit(1)

    config = Config()
    if indices_dir:
        config.indices_dir = indices_dir

    rag = UltimateRAG(config)
    rag.load_knowledge_base(knowledge_path)

    with console.status("Searching..."):
        result = rag.query(question, top_k=top_k, use_llm=not no_llm)

    console.print(Panel(Markdown(result.answer), title="Answer"))

    if result.sources:
        table = Table(title="Sources")
        table.add_column("#", style="dim")
        table.add_column("Type")
        table.add_column("Page")
        table.add_column("Score")

        for i, source in enumerate(result.sources[:5]):
            table.add_row(
                str(i + 1),
                source.get("type", ""),
                str(source.get("page", "")),
                f"{source.get('score', 0):.3f}"
            )

        console.print(table)

    if explain:
        console.print(Panel(
            rag.explain_retrieval(result.retrieval),
            title="Retrieval Explanation"
        ))


@app.command()
def search(
    query_text: str = typer.Argument(..., help="Search query"),
    knowledge_path: Path = typer.Option(..., "--knowledge", "-k", help="Knowledge base path"),
    search_type: str = typer.Option("all", "--type", "-t", help="Search type: all, formula, diagram"),
    top_k: int = typer.Option(10, "--top-k", "-n", help="Number of results")
):
    """Search the knowledge base without generating a response."""
    from ultimate_rag import UltimateRAG, Config

    config = Config()
    rag = UltimateRAG(config)
    rag.load_knowledge_base(knowledge_path)

    with console.status("Searching..."):
        if search_type == "formula":
            results = rag.search_formulas(query_text, top_k)
            table = Table(title=f"Formula Results for '{query_text}'")
            table.add_column("ID")
            table.add_column("LaTeX")
            table.add_column("Score")

            for r in results:
                table.add_row(
                    r["id"],
                    r["data"].get("latex", "")[:50],
                    f"{r['score']:.3f}"
                )
        else:
            retrieval = rag.search(query_text, top_k)
            table = Table(title=f"Search Results for '{query_text}'")
            table.add_column("#")
            table.add_column("Type")
            table.add_column("Content Preview")
            table.add_column("Score")

            for i, result in enumerate(retrieval.results[:top_k]):
                preview = result.content[:60] + "..." if len(result.content) > 60 else result.content
                table.add_row(
                    str(i + 1),
                    result.source_type,
                    preview,
                    f"{result.score:.3f}"
                )

    console.print(table)


@app.command()
def generate_book(
    knowledge_path: Path = typer.Option(..., "--knowledge", "-k", help="Knowledge base path"),
    output_dir: Path = typer.Option(Path("output"), "--output", "-o", help="Output directory")
):
    """Generate knowledge book from processed document."""
    from ultimate_rag import UltimateRAG, Config

    config = Config()
    rag = UltimateRAG(config)
    rag.load_knowledge_base(knowledge_path)

    with console.status("Generating knowledge book..."):
        files = rag.generate_knowledge_book(output_dir)

    table = Table(title="Generated Files")
    table.add_column("Type")
    table.add_column("Path")

    for file_type, path in files.items():
        table.add_row(file_type, str(path))

    console.print(table)


@app.command()
def interactive(
    knowledge_path: Path = typer.Option(..., "--knowledge", "-k", help="Knowledge base path")
):
    """Start interactive query session."""
    from ultimate_rag import UltimateRAG, Config

    config = Config()
    rag = UltimateRAG(config)
    rag.load_knowledge_base(knowledge_path)

    rag.interactive_session()


@app.command()
def stats(
    knowledge_path: Path = typer.Option(..., "--knowledge", "-k", help="Knowledge base path")
):
    """Show knowledge base statistics."""
    from ultimate_rag.utils.io import load_json

    kb = load_json(knowledge_path)
    stats = kb.get("statistics", {})

    table = Table(title="Knowledge Base Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Document ID", kb.get("document_id", "N/A"))
    table.add_row("Total Chunks", str(stats.get("total_chunks", 0)))
    table.add_row("Total Formulas", str(stats.get("total_formulas", 0)))
    table.add_row("Avg Chunk Length", f"{stats.get('avg_chunk_length', 0):.0f}")

    console.print(table)

    if stats.get("chunk_types"):
        types_table = Table(title="Chunk Types")
        types_table.add_column("Type")
        types_table.add_column("Count")

        for chunk_type, count in stats["chunk_types"].items():
            types_table.add_row(chunk_type, str(count))

        console.print(types_table)

    if stats.get("domains"):
        domains_table = Table(title="Domains")
        domains_table.add_column("Domain")
        domains_table.add_column("Count")

        for domain, count in stats["domains"].items():
            domains_table.add_row(domain, str(count))

        console.print(domains_table)


def main():
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
