"""Conservative SymPy-based equivalence checks for simple expressions."""

import re

from sympy import E, pi, simplify, sqrt, symbols, sympify

from math_learning_agent.models import SympyCheckResult


_ALLOWED_CHARACTERS = re.compile(r"^[0-9A-Za-z+\-*/^().\s]+$")
_IDENTIFIERS = re.compile(r"[A-Za-z]+")
_ALLOWED_NAMED_IDENTIFIERS = {"E", "pi", "sqrt"}
_MAX_EXPRESSION_LENGTH = 200


def _normalize_math_expression(text: str) -> str:
    """Normalize a deliberately small set of safe display-math forms."""

    expression = text.strip()
    if len(expression) >= 2 and expression.startswith("$") and expression.endswith("$"):
        expression = expression[1:-1].strip()
    if (
        len(expression) >= 4
        and expression.startswith(r"\(")
        and expression.endswith(r"\)")
    ):
        expression = expression[2:-2].strip()

    # Only simple, non-nested LaTeX square roots are supported.
    expression = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", expression)
    # Support a parenthesized radicand or one numeric/single-letter atom.
    expression = re.sub(r"√\s*(\([^()]*\))", r"sqrt\1", expression)
    expression = re.sub(
        r"√\s*([A-Za-z]|\d+(?:\.\d+)?)",
        r"sqrt(\1)",
        expression,
    )
    # The only implicit multiplication added is a number or closing
    # parenthesis immediately followed by a normalized square root.
    expression = re.sub(r"(?<=[0-9)])(?=sqrt\()", "*", expression)
    return expression.replace("^", "**")


def _parse_simple_expression(value: str):
    """Parse a deliberately small, non-LaTeX expression language."""

    expression = _normalize_math_expression(value)
    if not expression or len(expression) > _MAX_EXPRESSION_LENGTH:
        raise ValueError("The answer is empty or too long for the basic checker.")
    if not _ALLOWED_CHARACTERS.fullmatch(expression):
        raise ValueError("The answer contains unsupported notation or text.")

    identifier_names = set(_IDENTIFIERS.findall(expression))
    unsupported_names = {
        name
        for name in identifier_names
        if len(name) != 1 and name not in _ALLOWED_NAMED_IDENTIFIERS
    }
    if unsupported_names:
        raise ValueError("The answer contains unsupported names or natural language.")

    symbol_names = sorted(name for name in identifier_names if len(name) == 1)
    local_dict = {name: symbols(name) for name in symbol_names}
    local_dict.update({"E": E, "pi": pi, "sqrt": sqrt})
    return sympify(expression, locals=local_dict, evaluate=False)


def check_answer_equivalence(
    student_answer: str,
    reference_answer: str,
) -> SympyCheckResult:
    """Check basic expression equivalence without leaking parse errors."""

    try:
        student_expression = _parse_simple_expression(student_answer)
        reference_expression = _parse_simple_expression(reference_answer)
        difference = simplify(student_expression - reference_expression)
        if difference == 0 or difference.equals(0) is True:
            return SympyCheckResult(
                status="equivalent",
                reason="SymPy simplified the difference between both answers to zero.",
            )
        if difference.equals(0) is False:
            return SympyCheckResult(
                status="not_equivalent",
                reason="SymPy parsed both answers and determined they are not equivalent.",
            )
        return SympyCheckResult(
            status="unsupported",
            reason="SymPy parsed the answers but could not determine equivalence reliably.",
        )
    except Exception as exc:
        return SympyCheckResult(
            status="unsupported",
            reason=f"The basic SymPy checker could not parse this answer format: {exc}",
        )
