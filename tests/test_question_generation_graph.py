"""Tests for Phase 5 conditional routing and deterministic nodes."""

import sqlite3

import pytest

from math_learning_agent.graph.question_generation_graph import (
    accept_question,
    generate_question,
    mark_failed,
    route_after_verification,
)
from math_learning_agent.models import GeneratedQuestion, VerificationResult


GENERATED_QUESTION_JSON = """
{
  "question_text": "求 f(x)=x^2-2x 在 x=1 处的导数。",
  "knowledge_points": ["导数的计算"],
  "difficulty": 2,
  "reference_answer": "0",
  "reference_solution": "f'(x)=2x-2，所以 f'(1)=0。",
  "generation_reason": "训练导数公式和代入计算。"
}
"""


class FixedLLMClient:
    def invoke(self, messages) -> str:
        return GENERATED_QUESTION_JSON


def _verification(passed: bool) -> VerificationResult:
    return VerificationResult(
        passed=passed,
        answer_correct=passed,
        solution_correct=passed,
        knowledge_match=passed,
        difficulty_match=passed,
        issues=[] if passed else ["答案不一致"],
        feedback="通过。" if passed else "请重新检查答案。",
    )


def _base_state() -> dict:
    return {
        "knowledge_point": "导数的计算",
        "target_difficulty": 2,
        "focus_error_types": ["calculation_error"],
        "generation_attempts": 1,
        "max_generation_attempts": 3,
    }


def test_route_after_verification_accepts_pass() -> None:
    state = {**_base_state(), "verification_result": _verification(True)}
    assert route_after_verification(state) == "accept_question"


def test_route_after_verification_retries_failed_check() -> None:
    state = {**_base_state(), "verification_result": _verification(False)}
    assert route_after_verification(state) == "generate_question"


def test_route_after_verification_stops_at_attempt_limit() -> None:
    state = {
        **_base_state(),
        "generation_attempts": 3,
        "verification_result": _verification(False),
    }
    assert route_after_verification(state) == "mark_failed"


def test_generate_question_increments_attempts() -> None:
    state = {**_base_state(), "generation_attempts": 1}
    update = generate_question(state, llm_client=FixedLLMClient())

    assert update["generation_attempts"] == 2
    assert isinstance(update["generated_question"], GeneratedQuestion)


def test_accepted_question_requires_pass_and_failed_state_has_none(
    tmp_path,
) -> None:
    question = GeneratedQuestion.model_validate_json(GENERATED_QUESTION_JSON)
    passed_state = {
        **_base_state(),
        "generated_question": question,
        "verification_result": _verification(True),
    }
    accepted = accept_question(
        passed_state,
        database_path=tmp_path / "accepted.db",
    )

    assert accepted["accepted_question"] == question
    assert accepted["generation_status"] == "accepted"

    failed_state = {
        **_base_state(),
        "generated_question": question,
        "verification_result": _verification(False),
    }
    with pytest.raises(ValueError, match="PASS"):
        accept_question(
            failed_state,
            database_path=tmp_path / "failed.db",
        )

    failed = mark_failed(failed_state)
    assert failed["accepted_question"] is None
    assert failed["generation_status"] == "failed"

    with sqlite3.connect(tmp_path / "accepted.db") as connection:
        saved_count = connection.execute(
            "SELECT COUNT(*) FROM generated_questions"
        ).fetchone()[0]
    assert saved_count == 1
    assert not (tmp_path / "failed.db").exists()
