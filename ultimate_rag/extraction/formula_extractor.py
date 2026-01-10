"""Formula extraction and LaTeX conversion."""

import re
import time
from pathlib import Path
from typing import Optional, Any
from loguru import logger

from ultimate_rag.extraction.base import BaseExtractor, ExtractionResult
from ultimate_rag.models.formula import Formula, FormulaType, SymbolicForm
from ultimate_rag.config import ExtractionConfig
from ultimate_rag.utils.math_utils import (
    extract_variables, extract_operators, extract_functions,
    classify_formula, latex_to_sympy, normalize_latex
)


class FormulaExtractor(BaseExtractor):
    """Extract and parse mathematical formulas."""

    def __init__(self, config: Optional[ExtractionConfig] = None):
        super().__init__(config)
        self.config = config or ExtractionConfig()
        self._pix2tex_model = None
        self._latex_patterns = self._compile_patterns()

    def _compile_patterns(self) -> dict:
        """Compile regex patterns for formula detection."""
        return {
            'display_math': re.compile(r'\$\$(.+?)\$\$', re.DOTALL),
            'inline_math': re.compile(r'\$([^$]+?)\$'),
            'latex_env': re.compile(
                r'\\begin\{(equation|align|gather|multline)\*?\}(.+?)\\end\{\1\*?\}',
                re.DOTALL
            ),
            'bracketed': re.compile(r'\\\[(.+?)\\\]', re.DOTALL),
            'parens': re.compile(r'\\\((.+?)\\\)'),

            'physics_formulas': re.compile(
                r'[A-Za-z]+\s*=\s*[^,\.\n]{3,}(?:[\+\-\*/\^]|\\frac|\\sqrt|\\int)',
                re.MULTILINE
            ),
            'greek_formulas': re.compile(
                r'(?:\\(?:alpha|beta|gamma|delta|epsilon|theta|lambda|mu|sigma|omega|pi|nabla|partial))[^a-zA-Z].*?(?:=|\\approx)',
            ),
            'chemical_formulas': re.compile(
                r'(?:[A-Z][a-z]?\d*)+\s*(?:->|→|\\rightarrow)\s*(?:[A-Z][a-z]?\d*)+'
            ),
        }

    def _setup(self):
        """Initialize formula extraction models."""
        if self.config.formula_backend == "pix2tex":
            try:
                from pix2tex.cli import LatexOCR
                self._pix2tex_model = LatexOCR()
                logger.info("pix2tex model loaded")
            except ImportError:
                logger.warning("pix2tex not available, falling back to regex extraction")
            except Exception as e:
                logger.warning(f"pix2tex loading failed: {e}")

    def extract(self, source: Any, **kwargs) -> ExtractionResult:
        """Extract formulas from text or image."""
        self.initialize()
        start_time = time.time()

        try:
            if isinstance(source, (str, Path)) and Path(source).suffix.lower() in ['.png', '.jpg', '.jpeg']:
                formulas = self._extract_from_image(Path(source))
            elif isinstance(source, bytes):
                formulas = self._extract_from_image_bytes(source)
            elif isinstance(source, str):
                formulas = self._extract_from_text(source)
            else:
                return ExtractionResult(
                    success=False,
                    error=f"Unsupported source type: {type(source)}"
                )

            processing_time = (time.time() - start_time) * 1000

            return ExtractionResult(
                success=True,
                data=formulas,
                processing_time_ms=processing_time,
                metadata={"formula_count": len(formulas)}
            )

        except Exception as e:
            logger.error(f"Formula extraction failed: {e}")
            return ExtractionResult(success=False, error=str(e))

    def _extract_from_text(self, text: str) -> list[Formula]:
        """Extract formulas from text using regex patterns."""
        formulas = []
        seen_latex = set()

        for name, pattern in self._latex_patterns.items():
            if name in ['physics_formulas', 'greek_formulas', 'chemical_formulas']:
                continue

            for match in pattern.finditer(text):
                if match.lastindex:
                    latex = match.group(match.lastindex).strip()
                else:
                    latex = match.group(0).strip()

                if latex and latex not in seen_latex and len(latex) > 2:
                    seen_latex.add(latex)
                    formula = self._create_formula(latex)
                    if formula:
                        formulas.append(formula)

        for pattern in [self._latex_patterns['physics_formulas'],
                       self._latex_patterns['chemical_formulas']]:
            for match in pattern.finditer(text):
                latex = match.group(0).strip()
                normalized = normalize_latex(latex)
                if normalized not in seen_latex and len(latex) > 3:
                    seen_latex.add(normalized)
                    formula = self._create_formula(latex)
                    if formula:
                        formulas.append(formula)

        return formulas

    def _extract_from_image(self, image_path: Path) -> list[Formula]:
        """Extract formula from image using OCR."""
        if not self._pix2tex_model:
            logger.warning("No image OCR model available")
            return []

        try:
            from PIL import Image
            img = Image.open(image_path)
            latex = self._pix2tex_model(img)

            if latex:
                formula = self._create_formula(latex)
                if formula:
                    formula.metadata["source_image"] = str(image_path)
                    return [formula]

        except Exception as e:
            logger.error(f"Image formula extraction failed: {e}")

        return []

    def _extract_from_image_bytes(self, image_bytes: bytes) -> list[Formula]:
        """Extract formula from image bytes."""
        if not self._pix2tex_model:
            return []

        try:
            from PIL import Image
            from io import BytesIO

            img = Image.open(BytesIO(image_bytes))
            latex = self._pix2tex_model(img)

            if latex:
                formula = self._create_formula(latex)
                if formula:
                    return [formula]

        except Exception as e:
            logger.error(f"Image bytes formula extraction failed: {e}")

        return []

    def _create_formula(self, latex: str) -> Optional[Formula]:
        """Create a Formula object from LaTeX string."""
        latex = latex.strip()
        if not latex or len(latex) < 2:
            return None

        formula_type_str = classify_formula(latex)
        formula_type = FormulaType(formula_type_str) if formula_type_str in FormulaType.__members__.values() else FormulaType.UNKNOWN

        symbolic = self._parse_symbolic(latex)

        from ultimate_rag.utils.math_utils import formula_to_plain_text
        plain_text = formula_to_plain_text(latex)

        domain = self._detect_domain(latex)

        return Formula(
            id="",
            latex=latex,
            formula_type=formula_type,
            symbolic=symbolic,
            plain_text=plain_text,
            domain=domain
        )

    def _parse_symbolic(self, latex: str) -> SymbolicForm:
        """Parse LaTeX into symbolic form."""
        variables = extract_variables(latex)
        operators = extract_operators(latex)
        functions = extract_functions(latex)

        sympy_expr = latex_to_sympy(latex)

        return SymbolicForm(
            sympy_expr=sympy_expr,
            variables=variables,
            operators=operators,
            functions=functions,
            is_valid=sympy_expr is not None
        )

    def _detect_domain(self, latex: str) -> str:
        """Detect domain of formula."""
        physics_indicators = [
            '\\vec', '\\nabla', '\\partial t', 'F', 'm', 'a', 'v', 'E', 'p',
            '\\hbar', 'c', 'G', 'k_B', '\\epsilon_0', '\\mu_0',
            '\\psi', '\\phi', '\\Phi', '\\omega', 'H', 'L'
        ]

        chemistry_indicators = [
            '\\ce{', '\\rightarrow', 'mol', 'pH', 'K_a', 'K_b', 'K_p',
            '\\Delta H', '\\Delta G', '\\Delta S', 'R', 'T'
        ]

        math_indicators = [
            '\\lim', '\\sum', '\\prod', '\\int', 'dx', 'dy',
            '\\infty', '\\forall', '\\exists', '\\in', '\\subset'
        ]

        stats_indicators = [
            'P(', '\\mathbb{E}', '\\sigma', '\\mu', 'Var', 'Cov',
            '\\hat', '\\bar', 'n!', '\\binom'
        ]

        for ind in chemistry_indicators:
            if ind in latex:
                return "chemistry"

        for ind in physics_indicators:
            if ind in latex:
                return "physics"

        for ind in stats_indicators:
            if ind in latex:
                return "statistics"

        for ind in math_indicators:
            if ind in latex:
                return "mathematics"

        return "general"

    def extract_from_document(
        self,
        document: Any,
        extract_from_images: bool = True
    ) -> list[Formula]:
        """Extract all formulas from a document."""
        self.initialize()
        all_formulas = []

        for page in document.pages:
            for block in page.text_blocks:
                result = self.extract(block.content)
                if result.success and result.data:
                    for formula in result.data:
                        formula.source_page = page.page_number
                        all_formulas.append(formula)

            if extract_from_images:
                for img_block in page.image_blocks:
                    if img_block.image_bytes:
                        result = self.extract(img_block.image_bytes)
                        if result.success and result.data:
                            for formula in result.data:
                                formula.source_page = page.page_number
                                formula.bbox = img_block.bbox
                                all_formulas.append(formula)

        unique_formulas = {}
        for formula in all_formulas:
            normalized = normalize_latex(formula.latex)
            if normalized not in unique_formulas:
                unique_formulas[normalized] = formula

        return list(unique_formulas.values())
