"""Formula models for mathematical expression handling."""

from typing import Optional, Any
from pydantic import BaseModel, Field
from enum import Enum
import hashlib


class FormulaType(str, Enum):
    """Types of mathematical formulas."""

    EQUATION = "equation"
    INLINE = "inline"
    DEFINITION = "definition"
    THEOREM = "theorem"
    DERIVATION = "derivation"
    IDENTITY = "identity"
    INEQUALITY = "inequality"
    INTEGRAL = "integral"
    DERIVATIVE = "derivative"
    LIMIT = "limit"
    SUMMATION = "summation"
    MATRIX = "matrix"
    CHEMICAL = "chemical"
    UNKNOWN = "unknown"


class SymbolicForm(BaseModel):
    """Symbolic representation of a formula."""

    sympy_expr: Optional[str] = None
    variables: list[str] = Field(default_factory=list)
    constants: list[str] = Field(default_factory=list)
    operators: list[str] = Field(default_factory=list)
    functions: list[str] = Field(default_factory=list)
    is_valid: bool = False
    parse_error: Optional[str] = None

    def get_signature(self) -> str:
        """Get a normalized signature for similarity comparison."""
        parts = sorted(self.operators) + sorted(self.functions)
        return ":".join(parts)


class Formula(BaseModel):
    """Multi-representation formula object."""

    id: str
    latex: str
    formula_type: FormulaType = FormulaType.UNKNOWN
    symbolic: Optional[SymbolicForm] = None
    plain_text: Optional[str] = None
    description: Optional[str] = None
    visual_embedding: Optional[list[float]] = None
    text_embedding: Optional[list[float]] = None

    source_page: Optional[int] = None
    source_chunk_id: Optional[str] = None
    bbox: Optional[Any] = None
    confidence: float = 1.0

    dependencies: list[str] = Field(default_factory=list)
    dependents: list[str] = Field(default_factory=list)
    related_concepts: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    domain: Optional[str] = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context):
        """Generate ID if not provided."""
        if not self.id:
            self.id = f"F_{hashlib.md5(self.latex.encode()).hexdigest()[:8]}"

    @property
    def normalized_latex(self) -> str:
        """Get normalized LaTeX for comparison."""
        import re
        normalized = self.latex.strip()
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = re.sub(r'\\left|\\right', '', normalized)
        return normalized

    def latex_similarity(self, other: "Formula") -> float:
        """Calculate latex string similarity with another formula."""
        from difflib import SequenceMatcher
        return SequenceMatcher(
            None, self.normalized_latex, other.normalized_latex
        ).ratio()

    def symbolic_similarity(self, other: "Formula") -> float:
        """Calculate symbolic similarity with another formula."""
        if not self.symbolic or not other.symbolic:
            return 0.0

        if not self.symbolic.is_valid or not other.symbolic.is_valid:
            return 0.0

        sig1 = set(self.symbolic.get_signature().split(":"))
        sig2 = set(other.symbolic.get_signature().split(":"))

        if not sig1 and not sig2:
            return 1.0
        if not sig1 or not sig2:
            return 0.0

        intersection = len(sig1 & sig2)
        union = len(sig1 | sig2)
        return intersection / union if union > 0 else 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return self.model_dump()

    @classmethod
    def from_latex(cls, latex: str, **kwargs) -> "Formula":
        """Create formula from LaTeX string."""
        formula_id = kwargs.pop("id", f"F_{hashlib.md5(latex.encode()).hexdigest()[:8]}")
        return cls(id=formula_id, latex=latex, **kwargs)


class FormulaCollection(BaseModel):
    """Collection of formulas with relationship tracking."""

    formulas: dict[str, Formula] = Field(default_factory=dict)
    dependency_graph: dict[str, list[str]] = Field(default_factory=dict)
    domain_index: dict[str, list[str]] = Field(default_factory=dict)

    def add(self, formula: Formula):
        """Add a formula to the collection."""
        self.formulas[formula.id] = formula

        for dep_id in formula.dependencies:
            if dep_id in self.formulas:
                self.formulas[dep_id].dependents.append(formula.id)

        if formula.domain:
            if formula.domain not in self.domain_index:
                self.domain_index[formula.domain] = []
            self.domain_index[formula.domain].append(formula.id)

    def get(self, formula_id: str) -> Optional[Formula]:
        """Get formula by ID."""
        return self.formulas.get(formula_id)

    def find_similar(self, formula: Formula, threshold: float = 0.7) -> list[tuple[Formula, float]]:
        """Find similar formulas."""
        results = []
        for f in self.formulas.values():
            if f.id == formula.id:
                continue
            symbolic_sim = formula.symbolic_similarity(f)
            latex_sim = formula.latex_similarity(f)
            combined = 0.6 * symbolic_sim + 0.4 * latex_sim
            if combined >= threshold:
                results.append((f, combined))
        return sorted(results, key=lambda x: x[1], reverse=True)

    def find_by_variable(self, variable: str) -> list[Formula]:
        """Find formulas containing a specific variable."""
        results = []
        for formula in self.formulas.values():
            if formula.symbolic and variable in formula.symbolic.variables:
                results.append(formula)
        return results

    def get_dependency_chain(self, formula_id: str) -> list[str]:
        """Get all formulas in dependency chain."""
        if formula_id not in self.formulas:
            return []

        visited = set()
        chain = []

        def dfs(fid: str):
            if fid in visited:
                return
            visited.add(fid)
            formula = self.formulas.get(fid)
            if formula:
                for dep in formula.dependencies:
                    dfs(dep)
                chain.append(fid)

        dfs(formula_id)
        return chain
