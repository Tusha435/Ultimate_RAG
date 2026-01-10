"""Formula index with symbolic search capabilities."""

import sqlite3
import json
from pathlib import Path
from typing import Optional, Any
from loguru import logger

from ultimate_rag.models.formula import Formula, FormulaCollection
from ultimate_rag.config import IndexingConfig
from ultimate_rag.utils.math_utils import normalize_latex, extract_variables, extract_operators


class FormulaIndex:
    """Index formulas for symbolic and semantic search."""

    def __init__(self, config: Optional[IndexingConfig] = None):
        self.config = config or IndexingConfig()
        self._conn: Optional[sqlite3.Connection] = None
        self._embedding_index = None
        self._initialized = False

    def initialize(self, db_path: Optional[Path] = None):
        """Initialize the formula index database."""
        if self._initialized:
            return

        db_path = db_path or Path(":memory:")

        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_tables()

        self._initialized = True
        logger.info(f"Formula index initialized: {db_path}")

    def _create_tables(self):
        """Create database tables."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS formulas (
                id TEXT PRIMARY KEY,
                latex TEXT NOT NULL,
                normalized_latex TEXT,
                plain_text TEXT,
                formula_type TEXT,
                domain TEXT,
                source_page INTEGER,
                source_chunk_id TEXT,
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS formula_variables (
                formula_id TEXT,
                variable TEXT,
                FOREIGN KEY (formula_id) REFERENCES formulas(id)
            );

            CREATE TABLE IF NOT EXISTS formula_operators (
                formula_id TEXT,
                operator TEXT,
                FOREIGN KEY (formula_id) REFERENCES formulas(id)
            );

            CREATE TABLE IF NOT EXISTS formula_functions (
                formula_id TEXT,
                function_name TEXT,
                FOREIGN KEY (formula_id) REFERENCES formulas(id)
            );

            CREATE TABLE IF NOT EXISTS formula_dependencies (
                formula_id TEXT,
                depends_on TEXT,
                FOREIGN KEY (formula_id) REFERENCES formulas(id),
                FOREIGN KEY (depends_on) REFERENCES formulas(id)
            );

            CREATE INDEX IF NOT EXISTS idx_variables ON formula_variables(variable);
            CREATE INDEX IF NOT EXISTS idx_operators ON formula_operators(operator);
            CREATE INDEX IF NOT EXISTS idx_domain ON formulas(domain);
            CREATE INDEX IF NOT EXISTS idx_type ON formulas(formula_type);
        """)
        self._conn.commit()

    def add_formula(self, formula: Formula):
        """Add a single formula to the index."""
        self.initialize()

        normalized = normalize_latex(formula.latex)

        self._conn.execute("""
            INSERT OR REPLACE INTO formulas
            (id, latex, normalized_latex, plain_text, formula_type, domain,
             source_page, source_chunk_id, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            formula.id,
            formula.latex,
            normalized,
            formula.plain_text,
            formula.formula_type.value if hasattr(formula.formula_type, 'value') else formula.formula_type,
            formula.domain,
            formula.source_page,
            formula.source_chunk_id,
            json.dumps(formula.metadata)
        ))

        self._conn.execute("DELETE FROM formula_variables WHERE formula_id = ?", (formula.id,))
        self._conn.execute("DELETE FROM formula_operators WHERE formula_id = ?", (formula.id,))
        self._conn.execute("DELETE FROM formula_functions WHERE formula_id = ?", (formula.id,))

        if formula.symbolic:
            for var in formula.symbolic.variables:
                self._conn.execute(
                    "INSERT INTO formula_variables (formula_id, variable) VALUES (?, ?)",
                    (formula.id, var)
                )

            for op in formula.symbolic.operators:
                self._conn.execute(
                    "INSERT INTO formula_operators (formula_id, operator) VALUES (?, ?)",
                    (formula.id, op)
                )

            for func in formula.symbolic.functions:
                self._conn.execute(
                    "INSERT INTO formula_functions (formula_id, function_name) VALUES (?, ?)",
                    (formula.id, func)
                )

        for dep_id in formula.dependencies:
            self._conn.execute(
                "INSERT INTO formula_dependencies (formula_id, depends_on) VALUES (?, ?)",
                (formula.id, dep_id)
            )

        self._conn.commit()

    def add_formulas(self, formulas: list[Formula]):
        """Add multiple formulas to the index."""
        for formula in formulas:
            self.add_formula(formula)

    def search_by_latex(
        self,
        query_latex: str,
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Search formulas by LaTeX similarity."""
        self.initialize()

        normalized_query = normalize_latex(query_latex)

        cursor = self._conn.execute("""
            SELECT id, latex, normalized_latex
            FROM formulas
        """)

        results = []
        for row in cursor:
            sim = self._latex_similarity(normalized_query, row['normalized_latex'])
            if sim > 0.1:
                results.append((row['id'], sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def search_by_variable(
        self,
        variable: str,
        top_k: int = 20
    ) -> list[tuple[str, float]]:
        """Find formulas containing a specific variable."""
        self.initialize()

        cursor = self._conn.execute("""
            SELECT DISTINCT f.id
            FROM formulas f
            JOIN formula_variables v ON f.id = v.formula_id
            WHERE v.variable = ? OR v.variable LIKE ?
        """, (variable, f"{variable}_%"))

        results = [(row['id'], 1.0) for row in cursor]
        return results[:top_k]

    def search_by_operator(
        self,
        operator: str,
        top_k: int = 20
    ) -> list[tuple[str, float]]:
        """Find formulas using specific operators."""
        self.initialize()

        cursor = self._conn.execute("""
            SELECT DISTINCT f.id
            FROM formulas f
            JOIN formula_operators o ON f.id = o.formula_id
            WHERE o.operator = ?
        """, (operator,))

        results = [(row['id'], 1.0) for row in cursor]
        return results[:top_k]

    def search_by_domain(
        self,
        domain: str,
        top_k: int = 50
    ) -> list[tuple[str, float]]:
        """Find formulas in a specific domain."""
        self.initialize()

        cursor = self._conn.execute("""
            SELECT id FROM formulas WHERE domain = ?
        """, (domain,))

        results = [(row['id'], 1.0) for row in cursor]
        return results[:top_k]

    def search_symbolic(
        self,
        variables: Optional[list[str]] = None,
        operators: Optional[list[str]] = None,
        domain: Optional[str] = None,
        top_k: int = 20
    ) -> list[tuple[str, float]]:
        """Search formulas by symbolic properties."""
        self.initialize()

        query = "SELECT DISTINCT f.id FROM formulas f"
        joins = []
        conditions = []
        params = []

        if variables:
            joins.append("JOIN formula_variables v ON f.id = v.formula_id")
            var_placeholders = ",".join("?" * len(variables))
            conditions.append(f"v.variable IN ({var_placeholders})")
            params.extend(variables)

        if operators:
            joins.append("JOIN formula_operators o ON f.id = o.formula_id")
            op_placeholders = ",".join("?" * len(operators))
            conditions.append(f"o.operator IN ({op_placeholders})")
            params.extend(operators)

        if domain:
            conditions.append("f.domain = ?")
            params.append(domain)

        query += " " + " ".join(joins)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        cursor = self._conn.execute(query, params)
        results = [(row['id'], 1.0) for row in cursor]
        return results[:top_k]

    def search_similar(
        self,
        formula_id: str,
        top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Find formulas similar to a given formula."""
        self.initialize()

        source = self.get_formula(formula_id)
        if not source:
            return []

        cursor = self._conn.execute("""
            SELECT v.variable FROM formula_variables v WHERE v.formula_id = ?
        """, (formula_id,))
        source_vars = set(row['variable'] for row in cursor)

        cursor = self._conn.execute("""
            SELECT o.operator FROM formula_operators o WHERE o.formula_id = ?
        """, (formula_id,))
        source_ops = set(row['operator'] for row in cursor)

        results = []
        cursor = self._conn.execute("SELECT id FROM formulas WHERE id != ?", (formula_id,))

        for row in cursor:
            other_id = row['id']

            var_cursor = self._conn.execute(
                "SELECT variable FROM formula_variables WHERE formula_id = ?",
                (other_id,)
            )
            other_vars = set(r['variable'] for r in var_cursor)

            op_cursor = self._conn.execute(
                "SELECT operator FROM formula_operators WHERE formula_id = ?",
                (other_id,)
            )
            other_ops = set(r['operator'] for r in op_cursor)

            var_sim = len(source_vars & other_vars) / max(len(source_vars | other_vars), 1)
            op_sim = len(source_ops & other_ops) / max(len(source_ops | other_ops), 1)

            similarity = 0.6 * var_sim + 0.4 * op_sim

            if similarity > 0.2:
                results.append((other_id, similarity))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_formula(self, formula_id: str) -> Optional[dict]:
        """Get formula by ID."""
        self.initialize()

        cursor = self._conn.execute(
            "SELECT * FROM formulas WHERE id = ?",
            (formula_id,)
        )
        row = cursor.fetchone()

        if not row:
            return None

        return dict(row)

    def get_dependencies(self, formula_id: str) -> list[str]:
        """Get formulas that this formula depends on."""
        cursor = self._conn.execute(
            "SELECT depends_on FROM formula_dependencies WHERE formula_id = ?",
            (formula_id,)
        )
        return [row['depends_on'] for row in cursor]

    def get_dependents(self, formula_id: str) -> list[str]:
        """Get formulas that depend on this formula."""
        cursor = self._conn.execute(
            "SELECT formula_id FROM formula_dependencies WHERE depends_on = ?",
            (formula_id,)
        )
        return [row['formula_id'] for row in cursor]

    def _latex_similarity(self, latex1: str, latex2: str) -> float:
        """Calculate similarity between two LaTeX strings."""
        from difflib import SequenceMatcher
        return SequenceMatcher(None, latex1, latex2).ratio()

    def save(self, path: Path):
        """Save index to disk."""
        if str(self._conn.execute("PRAGMA database_list").fetchone()[2]) == ":memory:":
            import shutil
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

            backup_conn = sqlite3.connect(str(path))
            self._conn.backup(backup_conn)
            backup_conn.close()
            logger.info(f"Saved formula index to {path}")
        else:
            self._conn.commit()

    def load(self, path: Path):
        """Load index from disk."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Formula index not found: {path}")
            return

        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialized = True
        logger.info(f"Loaded formula index from {path}")

    @property
    def size(self) -> int:
        """Get number of indexed formulas."""
        if not self._conn:
            return 0
        cursor = self._conn.execute("SELECT COUNT(*) FROM formulas")
        return cursor.fetchone()[0]
