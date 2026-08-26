"""Tests for Phase 4 training-plan models."""

import pytest
from pydantic import ValidationError

from math_learning_agent.models import TrainingFocus, TrainingPlan


@pytest.mark.parametrize(
    ("question_count", "target_difficulty"),
    [(0, 2), (11, 2), (2, 0), (2, 6)],
)
def test_training_focus_rejects_out_of_range_values(
    question_count: int,
    target_difficulty: int,
) -> None:
    with pytest.raises(ValidationError):
        TrainingFocus(
            knowledge_point="导数",
            question_count=question_count,
            target_difficulty=target_difficulty,
            focus_error_types=["calculation_error"],
            reason="需要巩固。",
        )


def test_training_plan_accepts_matching_question_total() -> None:
    plan = TrainingPlan(
        total_questions=6,
        focus_items=[
            TrainingFocus(
                knowledge_point="导数",
                question_count=4,
                target_difficulty=2,
                focus_error_types=["formula_error"],
                reason="错题次数较多。",
            ),
            TrainingFocus(
                knowledge_point="数列",
                question_count=2,
                target_difficulty=1,
                focus_error_types=["calculation_error"],
                reason="进行基础复习。",
            ),
        ],
        plan_reason="优先训练高频错题知识点。",
    )

    assert plan.total_questions == 6
    assert len(plan.focus_items) == 2


def test_training_plan_rejects_mismatched_question_total() -> None:
    with pytest.raises(ValidationError, match="question_count"):
        TrainingPlan(
            total_questions=6,
            focus_items=[
                TrainingFocus(
                    knowledge_point="导数",
                    question_count=5,
                    target_difficulty=2,
                    focus_error_types=[],
                    reason="需要巩固。",
                )
            ],
            plan_reason="基础训练。",
        )
