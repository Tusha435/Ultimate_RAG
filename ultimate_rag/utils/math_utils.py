"""Mathematical and symbolic utility functions."""

import re
from typing import Optional
from loguru import logger


def latex_to_sympy(latex: str) -> Optional[str]:
    """Convert LaTeX to SymPy expression string."""
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import srepr
        expr = parse_latex(latex)
        return srepr(expr)
    except Exception as e:
        logger.debug(f"LaTeX parse failed: {e}")
        return None


def sympy_to_latex(sympy_str: str) -> Optional[str]:
    """Convert SymPy expression to LaTeX."""
    try:
        from sympy import sympify, latex
        expr = sympify(sympy_str)
        return latex(expr)
    except Exception as e:
        logger.debug(f"SymPy to LaTeX failed: {e}")
        return None


def extract_variables(latex: str) -> list[str]:
    """Extract variable names from LaTeX expression."""
    var_pattern = r'(?<![\\a-zA-Z])([a-zA-Z](?:_\{?[a-zA-Z0-9]+\}?)?)'
    excluded = {
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
        'log', 'ln', 'exp', 'sqrt', 'frac', 'sum',
        'int', 'lim', 'max', 'min', 'det', 'dim',
        'arg', 'deg', 'gcd', 'lcm', 'mod', 'ker',
        'Re', 'Im', 'Pr', 'inf', 'sup', 'and', 'or'
    }

    matches = re.findall(var_pattern, latex)
    variables = []
    for match in matches:
        base = match.split('_')[0]
        if base.lower() not in excluded and len(base) <= 2:
            variables.append(match)
    return list(set(variables))


def extract_operators(latex: str) -> list[str]:
    """Extract mathematical operators from LaTeX."""
    operators = []

    op_patterns = {
        'derivative': r'\\frac\{d',
        'partial': r'\\partial',
        'integral': r'\\int',
        'sum': r'\\sum',
        'product': r'\\prod',
        'limit': r'\\lim',
        'gradient': r'\\nabla',
        'divergence': r'\\nabla\s*\\cdot',
        'curl': r'\\nabla\s*\\times',
        'laplacian': r'\\nabla\^2|\\Delta',
        'cross_product': r'\\times',
        'dot_product': r'\\cdot',
        'factorial': r'!',
    }

    for op_name, pattern in op_patterns.items():
        if re.search(pattern, latex):
            operators.append(op_name)

    return operators


def extract_functions(latex: str) -> list[str]:
    """Extract function names from LaTeX."""
    functions = []

    func_patterns = [
        r'\\(sin|cos|tan|cot|sec|csc)',
        r'\\(sinh|cosh|tanh|coth)',
        r'\\(arcsin|arccos|arctan)',
        r'\\(log|ln|exp)',
        r'\\(sqrt|cbrt)',
        r'\\(det|tr|rank)',
        r'\\(min|max|sup|inf)',
        r'\\(gcd|lcm)',
        r'([A-Z][a-z]*)\s*\('
    ]

    for pattern in func_patterns:
        matches = re.findall(pattern, latex)
        functions.extend(matches)

    return list(set(functions))


def normalize_latex(latex: str) -> str:
    """Normalize LaTeX for comparison."""
    normalized = latex.strip()
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'\\left|\\right', '', normalized)
    normalized = re.sub(r'\{([^{}])\}', r'\1', normalized)
    normalized = re.sub(r'\\,|\\;|\\:|\\!', '', normalized)
    return normalized


def is_valid_latex(latex: str) -> bool:
    """Check if LaTeX string is syntactically valid."""
    brace_count = 0
    for char in latex:
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
        if brace_count < 0:
            return False

    if brace_count != 0:
        return False

    backslash_pattern = r'\\[a-zA-Z]+|\\[^a-zA-Z]'
    invalid_pattern = r'\\[a-zA-Z]*[^a-zA-Z\s\{\}\[\]\(\)\^_\\]'

    return not bool(re.search(invalid_pattern, latex))


def classify_formula(latex: str) -> str:
    """Classify formula type based on LaTeX content."""
    latex_lower = latex.lower()

    if re.search(r'\\int|\\iint|\\iiint|\\oint', latex):
        return "integral"
    if re.search(r'\\frac\{d|\\frac\{\\partial', latex):
        return "derivative"
    if re.search(r'\\lim', latex):
        return "limit"
    if re.search(r'\\sum', latex):
        return "summation"
    if re.search(r'\\prod', latex):
        return "product"
    if re.search(r'\\begin\{matrix\}|\\begin\{pmatrix\}|\\begin\{bmatrix\}', latex):
        return "matrix"
    if re.search(r'\\ce\{|->|\\rightarrow.*\\ce', latex):
        return "chemical"
    if re.search(r'=', latex) and not re.search(r'\\neq|\\leq|\\geq|<|>', latex):
        return "equation"
    if re.search(r'\\leq|\\geq|<|>|\\neq', latex):
        return "inequality"

    return "unknown"


def extract_constants(latex: str) -> list[str]:
    """Extract physical/mathematical constants."""
    constants = []

    const_patterns = {
        'pi': r'\\pi',
        'e': r'\\mathrm\{e\}|(?<![a-zA-Z])e(?![a-zA-Z])',
        'i': r'\\mathrm\{i\}',
        'hbar': r'\\hbar',
        'planck': r'h(?![a-zA-Z])',
        'c': r'c(?![a-zA-Z])',
        'G': r'G(?![a-zA-Z])',
        'epsilon_0': r'\\epsilon_0|\\varepsilon_0',
        'mu_0': r'\\mu_0',
        'k_B': r'k_B',
        'N_A': r'N_A',
        'R': r'R(?![a-zA-Z])',
        'alpha': r'\\alpha',
        'infinity': r'\\infty',
    }

    for const_name, pattern in const_patterns.items():
        if re.search(pattern, latex):
            constants.append(const_name)

    return constants


def formula_to_plain_text(latex: str) -> str:
    """Convert LaTeX formula to plain text description."""
    text = latex
    replacements = [
        (r'\\frac\{([^}]+)\}\{([^}]+)\}', r'\1 divided by \2'),
        (r'\\sqrt\{([^}]+)\}', r'square root of \1'),
        (r'\\sum', 'sum of'),
        (r'\\int', 'integral of'),
        (r'\\partial', 'partial'),
        (r'\\nabla', 'gradient'),
        (r'\\cdot', 'dot'),
        (r'\\times', 'cross'),
        (r'\^2', ' squared'),
        (r'\^3', ' cubed'),
        (r'\^', ' to the power of '),
        (r'_', ' subscript '),
        (r'\\alpha', 'alpha'),
        (r'\\beta', 'beta'),
        (r'\\gamma', 'gamma'),
        (r'\\delta', 'delta'),
        (r'\\epsilon', 'epsilon'),
        (r'\\theta', 'theta'),
        (r'\\lambda', 'lambda'),
        (r'\\mu', 'mu'),
        (r'\\pi', 'pi'),
        (r'\\sigma', 'sigma'),
        (r'\\omega', 'omega'),
        (r'\\infty', 'infinity'),
        (r'\\pm', 'plus or minus'),
        (r'\\leq', 'less than or equal to'),
        (r'\\geq', 'greater than or equal to'),
        (r'\\neq', 'not equal to'),
        (r'=', ' equals '),
        (r'\\{|\\}|\{|\}', ''),
        (r'\\', ''),
    ]

    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text)

    text = re.sub(r'\s+', ' ', text).strip()
    return text
