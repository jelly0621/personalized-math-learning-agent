"""Tests for Phase 6 persistence and training mastery migration."""

import json
import sqlite3

from math_learning_agent.db import (
    get_generated_question,
    get_knowledge_mastery,
    get_latest_generated_question,
    initialize_database,
    save_generated_question,
    save_training_attempt,
    update_training_mastery,
)
from math_learning_agent.models import (
    GeneratedQuestion,
    StudentResponse,
    SympyCheckResult,
    TrainingEvaluation,
    VerificationResult,
)


def _question(label: str = "A") -> GeneratedQuestion:
    return GeneratedQuestion(
        question_text=f"题目 {label}",
        knowledge_points=["导数", "导数"],
        difficulty=2,
        reference_answer="2*x",
        reference_solution="使用求导公式得到 2*x。",
        generation_reason="训练导数。",
    )


def _passed_verification() -> VerificationResult:
    return VerificationResult(
        passed=True,
        answer_correct=True,
        solution_correct=True,
        knowledge_match=True,
        difficulty_match=True,
        issues=[],
        feedback="通过。",
    )


def _save_question(database_path, label: str = "A") -> int:
    return save_generated_question(
        question=_question(label),
        verification=_passed_verification(),
        target_knowledge_point="导数",
        target_difficulty=2,
        generation_attempts=1,
        database_path=database_path,
    )


def test_generated_question_can_be_loaded_by_id_and_latest(tmp_path) -> None:
    database_path = tmp_path / "questions.db"
    first_id = _save_question(database_path, "first")
    second_id = _save_question(database_path, "second")

    first = get_generated_question(first_id, database_path)
    latest = get_latest_generated_question(database_path)

    assert first.question_text == "题目 first"
    assert latest is not None
    assert latest[0] == second_id
    assert latest[1].question_text == "题目 second"


def test_missing_generated_question_has_clear_error(tmp_path) -> None:
    database_path = tmp_path / "missing.db"

    try:
        get_generated_question(1, database_path)
    except LookupError as exc:
        assert "was not found" in str(exc)
    else:
        raise AssertionError("Expected a missing generated question error.")


def test_save_training_attempt_persists_evaluation(tmp_path) -> None:
    database_path = tmp_path / "attempt.db"
    question_id = _save_question(database_path)
    response = StudentResponse(
        question_id=question_id,
        student_answer="x+x",
        student_solution="化简得到 2*x。",
    )
    evaluation = TrainingEvaluation(
        is_correct=True,
        error_reason="答案正确",
        feedback="结果正确。",
        related_knowledge_points=["导数"],
        confidence=0.98,
    )
    sympy_check = SympyCheckResult(
        status="equivalent",
        reason="Difference simplified to zero.",
    )

    attempt_id = save_training_attempt(
        question_id,
        response,
        evaluation,
        sympy_check,
        database_path,
    )

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT question_id, student_answer, is_correct,
                   related_knowledge_points, evaluation_confidence, sympy_status
            FROM training_attempts WHERE id = ?
            """,
            (attempt_id,),
        ).fetchone()

    assert row is not None
    assert row[0] == question_id
    assert row[1] == "x+x"
    assert row[2] == 1
    assert json.loads(row[3]) == ["导数"]
    assert row[4] == 0.98
    assert row[5] == "equivalent"


def test_update_training_mastery_counts_correct_and_incorrect_once(
    tmp_path,
) -> None:
    database_path = tmp_path / "mastery.db"

    update_training_mastery(["导数", "导数"], True, database_path)
    update_training_mastery(["导数", "导数"], False, database_path)
    profile = get_knowledge_mastery(database_path)

    assert len(profile) == 1
    assert profile[0]["wrong_count"] == 0
    assert profile[0]["practice_count"] == 2
    assert profile[0]["correct_count"] == 1
    assert profile[0]["last_practiced_at"]


def test_old_knowledge_mastery_schema_migrates_without_data_loss(
    tmp_path,
) -> None:
    database_path = tmp_path / "legacy.db"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE knowledge_mastery (
                knowledge_point TEXT PRIMARY KEY,
                wrong_count INTEGER NOT NULL DEFAULT 0,
                last_wrong_at TEXT,
                updated_at TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO knowledge_mastery (
                knowledge_point, wrong_count, last_wrong_at, updated_at
            ) VALUES ('函数', 4, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            """
        )

    initialize_database(database_path)
    initialize_database(database_path)

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(knowledge_mastery)")
        }
        row = connection.execute(
            """
            SELECT wrong_count, practice_count, correct_count, last_practiced_at
            FROM knowledge_mastery WHERE knowledge_point = '函数'
            """
        ).fetchone()

    assert {"practice_count", "correct_count", "last_practiced_at"}.issubset(
        columns
    )
    assert row == (4, 0, 0, None)
