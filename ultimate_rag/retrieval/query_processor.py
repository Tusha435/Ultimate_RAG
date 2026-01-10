"""Query processing and understanding."""

import re
from typing import Optional, Any
from loguru import logger

from ultimate_rag.models.query import Query, QueryType, QueryIntent
from ultimate_rag.config import RetrievalConfig


class QueryProcessor:
    """Process and understand user queries."""

    def __init__(self, config: Optional[RetrievalConfig] = None):
        self.config = config or RetrievalConfig()
        self._embedding_model = None

        self.intent_patterns = {
            QueryIntent.DEFINITION: [
                r'what\s+is\s+(?:a|an|the)?\s*',
                r'define\s+',
                r'definition\s+of\s+',
                r'meaning\s+of\s+',
            ],
            QueryIntent.EXPLANATION: [
                r'explain\s+',
                r'how\s+does\s+',
                r'why\s+does\s+',
                r'describe\s+',
            ],
            QueryIntent.DERIVATION: [
                r'derive\s+',
                r'derivation\s+of\s+',
                r'how\s+(?:to|do\s+you)\s+derive\s+',
                r'show\s+(?:the\s+)?derivation',
            ],
            QueryIntent.EXAMPLE: [
                r'example\s+of\s+',
                r'give\s+(?:me\s+)?(?:an?\s+)?example',
                r'show\s+(?:me\s+)?(?:an?\s+)?example',
                r'for\s+example',
            ],
            QueryIntent.FORMULA_LOOKUP: [
                r'formula\s+for\s+',
                r'equation\s+for\s+',
                r'what\s+(?:is\s+)?the\s+formula',
                r'find\s+(?:the\s+)?formula',
            ],
            QueryIntent.PROOF: [
                r'prove\s+',
                r'proof\s+of\s+',
                r'show\s+that\s+',
                r'demonstrate\s+',
            ],
            QueryIntent.COMPARISON: [
                r'compare\s+',
                r'difference\s+between\s+',
                r'how\s+(?:does|do)\s+.*\s+differ\s+from',
                r'vs\.?\s+',
            ],
            QueryIntent.APPLICATION: [
                r'application\s+of\s+',
                r'how\s+(?:to|do\s+I)\s+use\s+',
                r'when\s+(?:to|do\s+I)\s+use\s+',
                r'apply\s+',
            ],
            QueryIntent.VISUALIZATION: [
                r'show\s+(?:me\s+)?(?:the\s+)?diagram',
                r'visualize\s+',
                r'graph\s+of\s+',
                r'plot\s+',
                r'draw\s+',
            ],
            QueryIntent.CALCULATION: [
                r'calculate\s+',
                r'compute\s+',
                r'find\s+(?:the\s+)?value',
                r'solve\s+',
                r'evaluate\s+',
            ],
        }

        self.domain_keywords = {
            "physics": [
                "force", "energy", "momentum", "velocity", "acceleration",
                "mass", "field", "wave", "quantum", "relativity", "electric",
                "magnetic", "thermodynamics", "mechanics", "optics"
            ],
            "mathematics": [
                "function", "derivative", "integral", "limit", "series",
                "matrix", "vector", "theorem", "proof", "equation",
                "calculus", "algebra", "geometry", "topology"
            ],
            "chemistry": [
                "molecule", "atom", "bond", "reaction", "compound",
                "element", "electron", "orbital", "acid", "base",
                "oxidation", "equilibrium", "concentration"
            ],
            "data_science": [
                "model", "algorithm", "dataset", "training", "prediction",
                "regression", "classification", "neural", "learning",
                "feature", "accuracy", "loss", "optimization"
            ],
            "statistics": [
                "probability", "distribution", "variance", "mean",
                "hypothesis", "confidence", "correlation", "sample",
                "significance", "bayesian", "regression"
            ]
        }

    def process(self, raw_query: str, context: Optional[str] = None) -> Query:
        """Process a raw query into a structured Query object."""
        processed_text = self._preprocess(raw_query)

        query_type = self._detect_query_type(processed_text)
        intent = self._detect_intent(processed_text)

        keywords = self._extract_keywords(processed_text)
        entities = self._extract_entities(processed_text)
        formulas = self._extract_formulas(raw_query)

        domain = self._detect_domain(processed_text, keywords)

        query = Query(
            raw_text=raw_query,
            processed_text=processed_text,
            query_type=query_type,
            intent=intent,
            keywords=keywords,
            entities=entities,
            formulas=formulas,
            domain_hint=domain,
            context=context
        )

        return query

    def _preprocess(self, text: str) -> str:
        """Preprocess query text."""
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)

        if not text.endswith(('?', '.', '!')):
            text = text.rstrip('.,;:')

        return text

    def _detect_query_type(self, text: str) -> QueryType:
        """Detect the type of query."""
        if self._contains_formula(text):
            return QueryType.FORMULA

        if any(word in text.lower() for word in ['diagram', 'graph', 'plot', 'figure', 'image', 'picture']):
            return QueryType.DIAGRAM

        if self._is_keyword_query(text):
            return QueryType.KEYWORD

        has_formula = bool(self._extract_formulas(text))
        has_text = len(text.split()) > 2

        if has_formula and has_text:
            return QueryType.MIXED

        return QueryType.TEXT

    def _detect_intent(self, text: str) -> QueryIntent:
        """Detect the intent of the query."""
        text_lower = text.lower()

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return intent

        return QueryIntent.GENERAL

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract important keywords from query."""
        text_lower = text.lower()

        stopwords = {
            'what', 'is', 'the', 'a', 'an', 'how', 'why', 'when', 'where',
            'which', 'who', 'does', 'do', 'can', 'could', 'would', 'should',
            'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'this', 'that', 'these', 'those', 'it', 'its', 'of', 'for', 'to',
            'in', 'on', 'at', 'by', 'with', 'from', 'about', 'into', 'between'
        }

        words = re.findall(r'\b[a-zA-Z]+\b', text_lower)
        keywords = [w for w in words if w not in stopwords and len(w) > 2]

        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        keywords.extend([pn.lower() for pn in proper_nouns])

        return list(set(keywords))

    def _extract_entities(self, text: str) -> list[str]:
        """Extract named entities from query."""
        entities = []

        scientist_pattern = r"\b([A-Z][a-z]+(?:'s)?)\s+(?:law|equation|theorem|principle|constant|rule)"
        matches = re.findall(scientist_pattern, text)
        entities.extend(matches)

        concept_patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:formula|equation|theorem)',
            r'\b(Newton|Einstein|Maxwell|Schrödinger|Heisenberg|Planck|Boltzmann|Gauss|Euler|Leibniz)\b',
        ]

        for pattern in concept_patterns:
            matches = re.findall(pattern, text)
            entities.extend(matches)

        return list(set(entities))

    def _extract_formulas(self, text: str) -> list[str]:
        """Extract formula-like patterns from query."""
        formulas = []

        patterns = [
            r'\$\$(.+?)\$\$',
            r'\$([^$]+?)\$',
            r'\\begin\{equation\}(.+?)\\end\{equation\}',
            r'\\\[(.+?)\\\]',
            r'\\\((.+?)\\\)',
            r'[A-Za-z]+\s*=\s*[^,\.\s]+',
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            formulas.extend(matches)

        return formulas

    def _detect_domain(self, text: str, keywords: list[str]) -> Optional[str]:
        """Detect the likely domain of the query."""
        text_lower = text.lower()
        all_words = set(keywords + text_lower.split())

        domain_scores = {}
        for domain, domain_kws in self.domain_keywords.items():
            score = sum(1 for kw in domain_kws if kw in all_words)
            if score > 0:
                domain_scores[domain] = score

        if domain_scores:
            return max(domain_scores, key=domain_scores.get)

        return None

    def _contains_formula(self, text: str) -> bool:
        """Check if text contains formula indicators."""
        formula_chars = ['=', '\\', '^', '_', '∫', '∑', '∂', '∇', '→', '≈', '≠']
        return any(c in text for c in formula_chars)

    def _is_keyword_query(self, text: str) -> bool:
        """Check if this is a simple keyword query."""
        words = text.split()
        if len(words) <= 3:
            text_lower = text.lower()
            for patterns in self.intent_patterns.values():
                for pattern in patterns:
                    if re.search(pattern, text_lower):
                        return False
            return True
        return False

    def expand_query(self, query: Query) -> Query:
        """Expand query with synonyms and related terms."""
        expansions = {
            "derivative": ["differentiation", "d/dx", "rate of change"],
            "integral": ["integration", "antiderivative", "area under curve"],
            "momentum": ["p", "mv", "linear momentum"],
            "energy": ["E", "kinetic", "potential", "work"],
            "force": ["F", "newton", "N", "push", "pull"],
            "velocity": ["v", "speed", "rate"],
            "acceleration": ["a", "deceleration"],
        }

        expanded_keywords = list(query.keywords)
        for keyword in query.keywords:
            if keyword in expansions:
                expanded_keywords.extend(expansions[keyword])

        return query.model_copy(update={"keywords": list(set(expanded_keywords))})

    def add_embedding(self, query: Query, embedding_index: Any) -> Query:
        """Add embedding to query."""
        embedding = embedding_index.embed_text(query.processed_text or query.raw_text)
        return query.with_embedding(embedding.tolist())
