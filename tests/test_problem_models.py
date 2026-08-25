"""Tests for Phase 2 Pydantic models."""

import pytest
from pydantic import ValidationError

from math_learning_agent.models import ProblemAnalysis, WrongProblemInput


def test_wrong_problem_optional_fields_default_to_none() -> None:
    problem = WrongProblemInput(problem_text="求函数 f(x)=x^2 的导数。")

    assert problem.student_answer is None
    assert problem.correct_answer is None
    assert problem.student_solution is None


@pytest.mark.parametrize("difficulty", [0, 6])
def test_problem_analysis_rejects_difficulty_outside_range(
    difficulty: int,
) -> None:
    with pytest.raises(ValidationError):
        ProblemAnalysis(
            knowledge_points=["导数"],
            question_type="计算题",
            difficulty=difficulty,
            problem_summary="求函数在指定点的导数。",
        )
