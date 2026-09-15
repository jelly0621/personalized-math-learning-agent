"""Tests for Phase 6 training-attempt models."""

import pytest
from pydantic import ValidationError

from math_learning_agent.models import (
    StudentResponse,
    SympyCheckResult,
    TrainingEvaluation,
)


def test_student_response_requires_positive_question_id() -> None:
    with pytest.raises(ValidationError):
        StudentResponse(question_id=0, student_answer="2*x")

    response = StudentResponse(question_id=1, student_answer="2*x")
    assert response.student_solution is None


def test_sympy_check_result_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        SympyCheckResult(status="error", reason="invalid")


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_training_evaluation_rejects_invalid_confidence(
    confidence: float,
) -> None:
    with pytest.raises(ValidationError):
        TrainingEvaluation(
            is_correct=False,
            error_reason="计算错误",
            feedback="请检查计算。",
            related_knowledge_points=["代数运算"],
            confidence=confidence,
        )
