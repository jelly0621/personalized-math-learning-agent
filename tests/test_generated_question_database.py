"""Tests for idempotent generated-question persistence."""

import sqlite3

import pytest

from math_learning_agent.db import initialize_database, save_generated_question
from math_learning_agent.models import GeneratedQuestion, VerificationResult


def _question() -> GeneratedQuestion:
    return GeneratedQuestion(
        question_text="求 f(x)=x^2 的导数。",
        knowledge_points=["导数"],
        difficulty=1,
        reference_answer="2x",
        reference_solution="使用幂函数求导公式得到 2x。",
        generation_reason="训练基础求导。",
    )


def _verification(passed: bool) -> VerificationResult:
    return VerificationResult(
        passed=passed,
        answer_correct=passed,
        solution_correct=passed,
        knowledge_match=passed,
        difficulty_match=passed,
        issues=[] if passed else ["未通过"],
        feedback="通过。" if passed else "需要重新生成。",
    )


def test_generated_questions_table_is_idempotent(tmp_path) -> None:
    database_path = tmp_path / "generated.db"
    initialize_database(database_path)
    initialize_database(database_path)

    with sqlite3.connect(database_path) as connection:
        tables = connection.execute(
            """
            SELECT COUNT(*) FROM sqlite_master
            WHERE type = 'table' AND name = 'generated_questions'
            """
        ).fetchone()[0]

    assert tables == 1


def test_only_passed_question_can_be_saved(tmp_path) -> None:
    database_path = tmp_path / "pass_only.db"

    with pytest.raises(ValueError, match="passed"):
        save_generated_question(
            question=_question(),
            verification=_verification(False),
            target_knowledge_point="导数",
            target_difficulty=1,
            generation_attempts=1,
            database_path=database_path,
        )
    assert not database_path.exists()

    question_id = save_generated_question(
        question=_question(),
        verification=_verification(True),
        target_knowledge_point="导数",
        target_difficulty=1,
        generation_attempts=1,
        database_path=database_path,
    )

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT id, target_knowledge_point, generation_attempts
            FROM generated_questions
            """
        ).fetchone()
    assert row == (question_id, "导数", 1)
