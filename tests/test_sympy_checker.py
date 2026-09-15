"""Deterministic tests for the conservative SymPy checker."""

from math_learning_agent.tools import check_answer_equivalence


def test_simple_equivalent_expressions() -> None:
    result = check_answer_equivalence("x + x", "2*x")
    assert result.status == "equivalent"


def test_clearly_non_equivalent_expressions() -> None:
    result = check_answer_equivalence("x + 1", "x + 2")
    assert result.status == "not_equivalent"


def test_natural_language_is_unsupported() -> None:
    result = check_answer_equivalence("答案是两个实数", "2")
    assert result.status == "unsupported"


def test_parse_failure_does_not_escape() -> None:
    result = check_answer_equivalence("x + (", "x")
    assert result.status == "unsupported"
    assert result.reason


def test_unicode_square_root_matches_function_form() -> None:
    assert check_answer_equivalence("1+√2", "1+sqrt(2)").status == "equivalent"


def test_numeric_coefficient_before_unicode_square_root() -> None:
    assert (
        check_answer_equivalence("3+2√2", "3+2*sqrt(2)").status
        == "equivalent"
    )
    assert (
        check_answer_equivalence("4√3/3", "4*sqrt(3)/3").status
        == "equivalent"
    )


def test_simple_latex_square_root_and_outer_math_delimiters() -> None:
    assert check_answer_equivalence(r"\sqrt{2}", "sqrt(2)").status == "equivalent"
    assert check_answer_equivalence(r"\(√x\)", "sqrt(x)").status == "equivalent"
    assert check_answer_equivalence("$ √2 $", "sqrt(2)").status == "equivalent"


def test_natural_language_with_square_root_remains_unsupported() -> None:
    assert check_answer_equivalence("答案是 √2", "sqrt(2)").status == "unsupported"
