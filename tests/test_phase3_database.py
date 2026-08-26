"""Tests for the Phase 3 SQLite migration and knowledge profile."""

import json
import sqlite3

from math_learning_agent.db import (
    get_knowledge_mastery,
    initialize_database,
    save_wrong_problem,
    update_knowledge_mastery,
)
from math_learning_agent.models import (
    ErrorDiagnosis,
    ProblemAnalysis,
    WrongProblemInput,
)


def _create_phase_2_database(database_path) -> None:
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE wrong_problems (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                problem_text TEXT NOT NULL,
                student_answer TEXT,
                correct_answer TEXT,
                student_solution TEXT,
                knowledge_points TEXT NOT NULL,
                question_type TEXT NOT NULL,
                difficulty INTEGER NOT NULL CHECK (difficulty BETWEEN 1 AND 5),
                problem_summary TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            INSERT INTO wrong_problems (
                problem_text,
                knowledge_points,
                question_type,
                difficulty,
                problem_summary
            ) VALUES (?, ?, ?, ?, ?)
            """,
            ("旧错题", '["函数"]', "计算题", 2, "Phase 2 数据"),
        )


def test_initialize_database_upgrades_phase_2_schema_without_data_loss(
    tmp_path,
) -> None:
    database_path = tmp_path / "phase_2.db"
    _create_phase_2_database(database_path)

    initialize_database(database_path)
    initialize_database(database_path)

    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(wrong_problems)")
        }
        old_row = connection.execute(
            "SELECT problem_text FROM wrong_problems WHERE id = 1"
        ).fetchone()
        profile_table = connection.execute(
            """
            SELECT name FROM sqlite_master
            WHERE type = 'table' AND name = 'knowledge_mastery'
            """
        ).fetchone()

    assert {
        "error_type",
        "error_reason",
        "related_knowledge_points",
        "error_confidence",
    }.issubset(columns)
    assert old_row == ("旧错题",)
    assert profile_table == ("knowledge_mastery",)


def test_save_wrong_problem_persists_error_diagnosis(tmp_path) -> None:
    database_path = tmp_path / "diagnosis.db"
    problem = WrongProblemInput(
        problem_text="求函数在 x=1 处的导数。",
        student_answer="2",
        correct_answer="0",
        student_solution="遗漏了常数项的导数。",
    )
    analysis = ProblemAnalysis(
        knowledge_points=["导数"],
        question_type="计算题",
        difficulty=2,
        problem_summary="计算指定点处的导数。",
    )
    diagnosis = ErrorDiagnosis(
        error_type="formula_error",
        error_reason="求导公式使用不完整。",
        related_knowledge_points=["求导公式"],
        confidence=0.9,
    )

    problem_id = save_wrong_problem(
        problem,
        analysis,
        database_path,
        diagnosis=diagnosis,
    )

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT error_type, error_reason, related_knowledge_points,
                   error_confidence
            FROM wrong_problems
            WHERE id = ?
            """,
            (problem_id,),
        ).fetchone()

    assert row is not None
    assert row[0] == diagnosis.error_type
    assert row[1] == diagnosis.error_reason
    assert json.loads(row[2]) == diagnosis.related_knowledge_points
    assert row[3] == diagnosis.confidence


def test_knowledge_mastery_inserts_then_increments(tmp_path) -> None:
    database_path = tmp_path / "profile.db"

    update_knowledge_mastery(["导数"], [], database_path)
    first_profile = get_knowledge_mastery(database_path)
    update_knowledge_mastery(["导数"], [], database_path)
    second_profile = get_knowledge_mastery(database_path)

    assert first_profile[0]["wrong_count"] == 1
    assert first_profile[0]["last_wrong_at"]
    assert second_profile[0]["wrong_count"] == 2


def test_duplicate_points_in_one_problem_count_once(tmp_path) -> None:
    database_path = tmp_path / "deduplication.db"

    update_knowledge_mastery(
        ["导数", "函数", "导数"],
        ["导数", "函数"],
        database_path,
    )
    profile = get_knowledge_mastery(database_path)

    assert {item["knowledge_point"]: item["wrong_count"] for item in profile} == {
        "函数": 1,
        "导数": 1,
    }


def test_get_knowledge_mastery_orders_by_wrong_count_descending(
    tmp_path,
) -> None:
    database_path = tmp_path / "ordering.db"
    update_knowledge_mastery(["函数", "导数"], [], database_path)
    update_knowledge_mastery(["导数"], [], database_path)

    profile = get_knowledge_mastery(database_path)

    assert [item["knowledge_point"] for item in profile] == ["导数", "函数"]
    assert [item["wrong_count"] for item in profile] == [2, 1]
