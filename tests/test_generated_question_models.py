"""Tests for Phase 5 generated-question quality models."""

import pytest
from pydantic import ValidationError

from math_learning_agent.models import (
    GeneratedQuestion,
    SolverResult,
    VerificationResult,
)


@pytest.mark.parametrize("difficulty", [0, 6])
def test_generated_question_rejects_invalid_difficulty(difficulty: int) -> None:
    with pytest.raises(ValidationError):
        GeneratedQuestion(
            question_text="求导。",
            knowledge_points=["导数"],
            difficulty=difficulty,
            reference_answer="0",
            reference_solution="计算可得。",
            generation_reason="巩固求导。",
        )


def test_solver_result_requires_answer_and_solution() -> None:
    result = SolverResult(answer="0", solution="独立计算得到 0。")
    assert result.answer == "0"

    with pytest.raises(ValidationError):
        SolverResult.model_validate({"answer": "0"})


def test_verification_result_rejects_inconsistent_pass() -> None:
    with pytest.raises(ValidationError, match="passed"):
        VerificationResult(
            passed=True,
            answer_correct=False,
            solution_correct=True,
            knowledge_match=True,
            difficulty_match=True,
            issues=[],
            feedback="答案不一致。",
        )

    with pytest.raises(ValidationError, match="issues"):
        VerificationResult(
            passed=True,
            answer_correct=True,
            solution_correct=True,
            knowledge_match=True,
            difficulty_match=True,
            issues=["题目有歧义"],
            feedback="需要修正。",
        )
