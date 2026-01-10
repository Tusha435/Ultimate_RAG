"""Build responses from retrieval results."""

from typing import Optional, Any
from loguru import logger

from ultimate_rag.models.query import Query, RetrievalResult, QueryResult, QueryIntent
from ultimate_rag.generation.llm_interface import LLMInterface
from ultimate_rag.config import Config, GenerationConfig


class ResponseBuilder:
    """Build responses from retrieval results."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.gen_config = self.config.generation

        self.llm = None
        if self.gen_config.llm_provider != "none":
            self.llm = LLMInterface(self.gen_config)

    def build_response(
        self,
        retrieval_result: RetrievalResult,
        use_llm: bool = True
    ) -> QueryResult:
        """Build a complete response from retrieval results."""
        query = retrieval_result.query

        if use_llm and self.llm:
            answer = self._generate_llm_response(retrieval_result)
            answer_type = "generated"
        else:
            answer = self._build_structured_response(retrieval_result)
            answer_type = "structured"

        sources = self._format_sources(retrieval_result)
        formulas_used = [r.formula_id for r in retrieval_result.formula_results if r.formula_id]
        diagrams_used = [r.diagram_id for r in retrieval_result.diagram_results if r.diagram_id]

        confidence_breakdown = {
            "retrieval": retrieval_result.confidence,
            "text_coverage": len(retrieval_result.text_results) / max(1, self.config.retrieval.default_top_k),
            "formula_coverage": min(1.0, len(formulas_used) / 3) if query.has_formula_query else 1.0
        }

        return QueryResult(
            query=query,
            retrieval=retrieval_result,
            answer=answer,
            answer_type=answer_type,
            sources=sources,
            formulas_used=formulas_used,
            diagrams_used=diagrams_used,
            confidence_breakdown=confidence_breakdown,
            processing_time_ms=retrieval_result.retrieval_time_ms,
            model_used=self.gen_config.llm_model if use_llm and self.llm else None
        )

    def _generate_llm_response(self, retrieval_result: RetrievalResult) -> str:
        """Generate response using LLM."""
        prompt = self._build_prompt(retrieval_result)

        try:
            response = self.llm.generate(prompt)
            return response
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._build_structured_response(retrieval_result)

    def _build_prompt(self, retrieval_result: RetrievalResult) -> str:
        """Build prompt for LLM."""
        query = retrieval_result.query
        context = retrieval_result.expanded_context or ""

        intent_instructions = self._get_intent_instructions(query.intent)

        prompt = f"""You are an expert assistant for technical subjects including Physics, Mathematics, Chemistry, and Data Science.

Answer the following question using ONLY the provided context. If the context doesn't contain enough information, say so.

{intent_instructions}

Question: {query.raw_text}

Context:
{context}

Requirements:
1. Be accurate and precise
2. Include relevant formulas in LaTeX format when applicable
3. Reference specific sources (page numbers, sections) when possible
4. If formulas are involved, explain each variable
5. Keep the answer focused and concise

Answer:"""

        return prompt

    def _get_intent_instructions(self, intent: QueryIntent) -> str:
        """Get specific instructions based on query intent."""
        instructions = {
            QueryIntent.DEFINITION: "Provide a clear, concise definition. Include formal mathematical notation if applicable.",
            QueryIntent.EXPLANATION: "Explain the concept step by step. Use analogies if helpful.",
            QueryIntent.DERIVATION: "Show the derivation step by step. Explain each mathematical transformation.",
            QueryIntent.EXAMPLE: "Provide a concrete example with detailed working.",
            QueryIntent.FORMULA_LOOKUP: "Provide the formula in LaTeX format. Explain each variable and its units.",
            QueryIntent.PROOF: "Present the proof with clear logical steps.",
            QueryIntent.COMPARISON: "Compare and contrast the concepts systematically.",
            QueryIntent.APPLICATION: "Explain how to apply this concept with practical examples.",
            QueryIntent.VISUALIZATION: "Describe the visual representation and what it shows.",
            QueryIntent.CALCULATION: "Show the calculation with all steps.",
            QueryIntent.GENERAL: "Provide a comprehensive answer addressing the question."
        }

        return instructions.get(intent, instructions[QueryIntent.GENERAL])

    def _build_structured_response(self, retrieval_result: RetrievalResult) -> str:
        """Build structured response without LLM."""
        query = retrieval_result.query
        parts = []

        if query.intent == QueryIntent.FORMULA_LOOKUP:
            parts.append(self._format_formula_response(retrieval_result))
        elif query.intent == QueryIntent.DEFINITION:
            parts.append(self._format_definition_response(retrieval_result))
        else:
            parts.append(self._format_general_response(retrieval_result))

        if retrieval_result.formula_results:
            parts.append("\n**Relevant Formulas:**")
            for result in retrieval_result.formula_results[:3]:
                if result.content:
                    parts.append(f"  - ${result.content}$")

        parts.append("\n**Sources:**")
        for i, result in enumerate(retrieval_result.results[:5]):
            source_line = f"  {i+1}. "
            if result.page_number:
                source_line += f"Page {result.page_number}"
            if result.section:
                source_line += f", {result.section}"
            source_line += f" (relevance: {result.score:.2f})"
            parts.append(source_line)

        return "\n".join(parts)

    def _format_formula_response(self, retrieval_result: RetrievalResult) -> str:
        """Format response for formula lookup queries."""
        parts = ["**Formula Response:**\n"]

        for result in retrieval_result.formula_results[:3]:
            if result.content:
                parts.append(f"$$\n{result.content}\n$$")

                if result.page_number:
                    parts.append(f"\n*Source: Page {result.page_number}*\n")

        if retrieval_result.text_results:
            parts.append("\n**Context:**")
            for result in retrieval_result.text_results[:2]:
                if result.content:
                    preview = result.content[:300] + "..." if len(result.content) > 300 else result.content
                    parts.append(f"\n{preview}")

        return "\n".join(parts)

    def _format_definition_response(self, retrieval_result: RetrievalResult) -> str:
        """Format response for definition queries."""
        parts = ["**Definition:**\n"]

        for result in retrieval_result.text_results[:2]:
            if result.content:
                parts.append(result.content)
                if result.page_number:
                    parts.append(f"\n*Source: Page {result.page_number}*\n")

        return "\n".join(parts)

    def _format_general_response(self, retrieval_result: RetrievalResult) -> str:
        """Format general response."""
        parts = ["**Answer:**\n"]

        for result in retrieval_result.text_results[:3]:
            if result.content:
                parts.append(result.content)
                parts.append("")

        return "\n".join(parts)

    def _format_sources(self, retrieval_result: RetrievalResult) -> list[dict]:
        """Format sources for citation."""
        sources = []

        for result in retrieval_result.results[:10]:
            source = {
                "type": result.source_type,
                "score": round(result.score, 3),
                "page": result.page_number,
                "section": result.section
            }

            if result.content:
                source["preview"] = result.content[:100] + "..." if len(result.content) > 100 else result.content

            if result.chunk_id:
                source["chunk_id"] = result.chunk_id
            if result.formula_id:
                source["formula_id"] = result.formula_id
            if result.diagram_id:
                source["diagram_id"] = result.diagram_id

            sources.append(source)

        return sources

    def format_for_display(self, query_result: QueryResult) -> str:
        """Format query result for display."""
        lines = [
            "=" * 60,
            f"Question: {query_result.query.raw_text}",
            "=" * 60,
            "",
            query_result.answer,
            "",
            "-" * 60,
            "Sources:"
        ]

        for i, source in enumerate(query_result.sources[:5]):
            source_line = f"  [{i+1}] {source['type'].title()}"
            if source.get('page'):
                source_line += f" - Page {source['page']}"
            if source.get('section'):
                source_line += f", {source['section']}"
            lines.append(source_line)

        if query_result.formulas_used:
            lines.append("")
            lines.append(f"Formulas used: {len(query_result.formulas_used)}")

        lines.append("")
        lines.append(f"Confidence: {query_result.confidence_breakdown.get('retrieval', 0):.1%}")
        lines.append(f"Processing time: {query_result.processing_time_ms:.0f}ms")

        return "\n".join(lines)
