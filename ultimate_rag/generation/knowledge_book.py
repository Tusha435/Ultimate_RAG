"""Generate knowledge book from processed documents."""

from pathlib import Path
from typing import Optional, Any
from datetime import datetime
from loguru import logger

from ultimate_rag.config import Config, GenerationConfig
from ultimate_rag.generation.llm_interface import LLMInterface
from ultimate_rag.utils.io import save_json


class KnowledgeBookGenerator:
    """Generate comprehensive knowledge book from processed documents."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.gen_config = self.config.generation

        self.llm = None
        if self.gen_config.llm_provider != "none":
            self.llm = LLMInterface(self.gen_config)

    def generate(
        self,
        knowledge_base: dict[str, Any],
        output_dir: Path,
        include_summaries: bool = True
    ) -> dict[str, Path]:
        """Generate complete knowledge book."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        doc_id = knowledge_base.get("document_id", "document")
        generated_files = {}

        toc_path = output_dir / f"{doc_id}_table_of_contents.md"
        self._generate_toc(knowledge_base, toc_path)
        generated_files["toc"] = toc_path

        formula_path = output_dir / f"{doc_id}_formula_handbook.md"
        self._generate_formula_handbook(knowledge_base, formula_path)
        generated_files["formulas"] = formula_path

        concepts_path = output_dir / f"{doc_id}_concept_index.md"
        self._generate_concept_index(knowledge_base, concepts_path)
        generated_files["concepts"] = concepts_path

        if include_summaries and self.llm:
            summary_path = output_dir / f"{doc_id}_chapter_summaries.md"
            self._generate_chapter_summaries(knowledge_base, summary_path)
            generated_files["summaries"] = summary_path

        index_path = output_dir / f"{doc_id}_searchable_index.json"
        self._generate_searchable_index(knowledge_base, index_path)
        generated_files["index"] = index_path

        stats = knowledge_base.get("statistics", {})
        meta_path = output_dir / f"{doc_id}_metadata.md"
        self._generate_metadata(knowledge_base, stats, meta_path)
        generated_files["metadata"] = meta_path

        logger.info(f"Knowledge book generated: {len(generated_files)} files")
        return generated_files

    def _generate_toc(self, knowledge_base: dict, output_path: Path):
        """Generate table of contents."""
        chunks = knowledge_base.get("chunks", {})

        lines = [
            "# Table of Contents",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        hierarchy = {}
        for chunk_id, chunk_data in chunks.items():
            chunk_type = chunk_data.get("chunk_type", "")
            if chunk_type in ["chapter", "section", "subsection"]:
                heading = chunk_data.get("metadata", {}).get("heading", "")
                pages = chunk_data.get("metadata", {}).get("page_numbers", [])
                page = pages[0] if pages else "?"

                if heading:
                    level = {"chapter": 1, "section": 2, "subsection": 3}.get(chunk_type, 2)
                    indent = "  " * (level - 1)
                    lines.append(f"{indent}- **{heading}** (Page {page})")

        with open(output_path, 'w') as f:
            f.write("\n".join(lines))

    def _generate_formula_handbook(self, knowledge_base: dict, output_path: Path):
        """Generate formula handbook."""
        formulas = knowledge_base.get("formulas", {})

        lines = [
            "# Formula Handbook",
            "",
            f"Total formulas: {len(formulas)}",
            "",
        ]

        by_domain = {}
        for formula_id, formula_data in formulas.items():
            domain = formula_data.get("domain", "general")
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append((formula_id, formula_data))

        for domain, domain_formulas in sorted(by_domain.items()):
            lines.append(f"## {domain.title()}")
            lines.append("")

            for formula_id, formula_data in domain_formulas:
                latex = formula_data.get("latex", "")
                plain = formula_data.get("plain_text", "")
                page = formula_data.get("source_page", "?")
                formula_type = formula_data.get("formula_type", "unknown")

                lines.append(f"### {formula_id}")
                lines.append("")
                lines.append(f"$$\n{latex}\n$$")
                lines.append("")
                if plain:
                    lines.append(f"*{plain}*")
                    lines.append("")
                lines.append(f"- **Type:** {formula_type}")
                lines.append(f"- **Page:** {page}")

                symbolic = formula_data.get("symbolic", {})
                if symbolic and symbolic.get("variables"):
                    vars_str = ", ".join(symbolic["variables"])
                    lines.append(f"- **Variables:** {vars_str}")

                lines.append("")

        with open(output_path, 'w') as f:
            f.write("\n".join(lines))

    def _generate_concept_index(self, knowledge_base: dict, output_path: Path):
        """Generate concept index."""
        chunks = knowledge_base.get("chunks", {})

        concepts = {}
        for chunk_id, chunk_data in chunks.items():
            chunk_concepts = chunk_data.get("metadata", {}).get("concepts", [])
            pages = chunk_data.get("metadata", {}).get("page_numbers", [])

            for concept in chunk_concepts:
                if concept not in concepts:
                    concepts[concept] = {"pages": set(), "chunk_ids": []}
                concepts[concept]["pages"].update(pages)
                concepts[concept]["chunk_ids"].append(chunk_id)

        lines = [
            "# Concept Index",
            "",
            f"Total concepts: {len(concepts)}",
            "",
        ]

        for concept in sorted(concepts.keys()):
            data = concepts[concept]
            pages = sorted(data["pages"])
            pages_str = ", ".join(str(p) for p in pages[:5])
            if len(pages) > 5:
                pages_str += f" (and {len(pages) - 5} more)"

            lines.append(f"- **{concept}**: Pages {pages_str}")

        with open(output_path, 'w') as f:
            f.write("\n".join(lines))

    def _generate_chapter_summaries(self, knowledge_base: dict, output_path: Path):
        """Generate chapter summaries using LLM."""
        chunks = knowledge_base.get("chunks", {})

        chapters = []
        for chunk_id, chunk_data in chunks.items():
            if chunk_data.get("chunk_type") == "chapter":
                chapters.append((chunk_id, chunk_data))

        lines = [
            "# Chapter Summaries",
            "",
        ]

        for chapter_id, chapter_data in chapters:
            heading = chapter_data.get("metadata", {}).get("heading", "Unknown Chapter")
            content = chapter_data.get("content", "")

            lines.append(f"## {heading}")
            lines.append("")

            if self.llm and content:
                try:
                    summary = self.llm.summarize(content, max_length=150)
                    lines.append(summary)
                except Exception as e:
                    logger.warning(f"Summary generation failed: {e}")
                    lines.append(content[:300] + "..." if len(content) > 300 else content)
            else:
                lines.append(content[:300] + "..." if len(content) > 300 else content)

            lines.append("")

        with open(output_path, 'w') as f:
            f.write("\n".join(lines))

    def _generate_searchable_index(self, knowledge_base: dict, output_path: Path):
        """Generate searchable JSON index."""
        index = {
            "document_id": knowledge_base.get("document_id"),
            "generated": datetime.now().isoformat(),
            "entries": []
        }

        chunks = knowledge_base.get("chunks", {})
        for chunk_id, chunk_data in chunks.items():
            entry = {
                "id": chunk_id,
                "type": "chunk",
                "content_preview": chunk_data.get("content", "")[:200],
                "pages": chunk_data.get("metadata", {}).get("page_numbers", []),
                "keywords": chunk_data.get("metadata", {}).get("keywords", []),
                "concepts": chunk_data.get("metadata", {}).get("concepts", []),
                "chunk_type": chunk_data.get("chunk_type"),
                "heading": chunk_data.get("metadata", {}).get("heading")
            }
            index["entries"].append(entry)

        formulas = knowledge_base.get("formulas", {})
        for formula_id, formula_data in formulas.items():
            entry = {
                "id": formula_id,
                "type": "formula",
                "latex": formula_data.get("latex"),
                "plain_text": formula_data.get("plain_text"),
                "page": formula_data.get("source_page"),
                "domain": formula_data.get("domain"),
                "variables": formula_data.get("symbolic", {}).get("variables", [])
            }
            index["entries"].append(entry)

        save_json(index, output_path)

    def _generate_metadata(self, knowledge_base: dict, stats: dict, output_path: Path):
        """Generate metadata summary."""
        lines = [
            "# Document Metadata",
            "",
            f"**Document ID:** {knowledge_base.get('document_id', 'unknown')}",
            f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Statistics",
            "",
            f"- **Total chunks:** {stats.get('total_chunks', 0)}",
            f"- **Total formulas:** {stats.get('total_formulas', 0)}",
            f"- **Average chunk length:** {stats.get('avg_chunk_length', 0):.0f} characters",
            f"- **Chunks with formulas:** {stats.get('chunks_with_formulas', 0)}",
            "",
            "## Chunk Types",
            ""
        ]

        for chunk_type, count in stats.get("chunk_types", {}).items():
            lines.append(f"- {chunk_type}: {count}")

        lines.extend([
            "",
            "## Domains",
            ""
        ])

        for domain, count in stats.get("domains", {}).items():
            lines.append(f"- {domain}: {count}")

        with open(output_path, 'w') as f:
            f.write("\n".join(lines))

    def generate_quick_reference(
        self,
        knowledge_base: dict,
        output_path: Path,
        max_formulas: int = 50
    ):
        """Generate quick reference card."""
        formulas = knowledge_base.get("formulas", {})

        lines = [
            "# Quick Reference Card",
            "",
            "## Key Formulas",
            ""
        ]

        sorted_formulas = sorted(
            formulas.items(),
            key=lambda x: (x[1].get("domain", "z"), x[0])
        )[:max_formulas]

        current_domain = None
        for formula_id, formula_data in sorted_formulas:
            domain = formula_data.get("domain", "general")
            if domain != current_domain:
                lines.append(f"\n### {domain.title()}\n")
                current_domain = domain

            latex = formula_data.get("latex", "")
            plain = formula_data.get("plain_text", "")

            lines.append(f"- ${latex}$")
            if plain:
                lines.append(f"  *{plain}*")

        with open(output_path, 'w') as f:
            f.write("\n".join(lines))
