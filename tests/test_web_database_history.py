"""Temporary-database tests for the Phase 8 history readers."""

from math_learning_agent.db import (
    get_recent_generated_questions,
    get_recent_training_attempts,
    save_generated_question,
    save_training_attempt,
)
from math_learning_agent.models import (
    GeneratedQuestion,
    StudentResponse,
    SympyCheckResult,
    TrainingEvaluation,
    VerificationResult,
)


def test_recent_web_history_is_compact_and_does_not_leak_answers(tmp_path) -> None:
    database_path = tmp_path / "web-history.db"
    question = GeneratedQuestion(
        question_text="求 x+x。",
        knowledge_points=["代数"],
        difficulty=1,
        reference_answer="2*x",
        reference_solution="合并同类项。",
        generation_reason="fixture",
    )
    verification = VerificationResult(
        passed=True,
        answer_correct=True,
        solution_correct=True,
        knowledge_match=True,
        difficulty_match=True,
        issues=[],
        feedback="pass",
    )
    question_id = save_generated_question(
        question, verification, "代数", 1, 1, database_path
    )
    save_training_attempt(
        question_id,
        StudentResponse(question_id=question_id, student_answer="2*x"),
        TrainingEvaluation(
            is_correct=True,
            error_reason="correct",
            feedback="good",
            related_knowledge_points=["代数"],
            confidence=1.0,
        ),
        SympyCheckResult(status="equivalent", reason="same"),
        database_path,
    )

    questions = get_recent_generated_questions(10, database_path)
    attempts = get_recent_training_attempts(10, database_path)

    assert questions[0]["id"] == question_id
    assert "reference_answer" not in questions[0]
    assert "reference_solution" not in questions[0]
    assert attempts[0]["is_correct"] is True
    assert attempts[0]["related_knowledge_points"] == ["代数"]
