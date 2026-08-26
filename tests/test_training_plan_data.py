"""Tests for recent-history loading used by the training planner graph."""

import json
import sqlite3

from math_learning_agent.db import (
    get_recent_wrong_problems,
    initialize_database,
    update_knowledge_mastery,
)
from math_learning_agent.graph.training_plan_graph import load_student_profile


def _insert_wrong_problem(
    database_path,
    *,
    knowledge_point: str,
    difficulty: int,
    error_type: str,
    created_at: str,
) -> int:
    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO wrong_problems (
                problem_text,
                knowledge_points,
                question_type,
                difficulty,
                problem_summary,
                error_type,
                related_knowledge_points,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "测试题",
                json.dumps([knowledge_point], ensure_ascii=False),
                "计算题",
                difficulty,
                "测试摘要",
                error_type,
                json.dumps([knowledge_point], ensure_ascii=False),
                created_at,
            ),
        )
        assert cursor.lastrowid is not None
        return cursor.lastrowid


def test_get_recent_wrong_problems_orders_by_time_then_id(tmp_path) -> None:
    database_path = tmp_path / "recent.db"
    _insert_wrong_problem(
        database_path,
        knowledge_point="集合",
        difficulty=1,
        error_type="concept_error",
        created_at="2026-08-01 10:00:00",
    )
    second_id = _insert_wrong_problem(
        database_path,
        knowledge_point="函数",
        difficulty=2,
        error_type="calculation_error",
        created_at="2026-08-02 10:00:00",
    )
    third_id = _insert_wrong_problem(
        database_path,
        knowledge_point="导数",
        difficulty=3,
        error_type="formula_error",
        created_at="2026-08-02 10:00:00",
    )

    recent = get_recent_wrong_problems(limit=2, database_path=database_path)

    assert [item["id"] for item in recent] == [third_id, second_id]
    assert recent[0]["knowledge_points"] == ["导数"]


def test_load_student_profile_reads_temporary_database(tmp_path) -> None:
    database_path = tmp_path / "profile.db"
    update_knowledge_mastery(["导数"], [], database_path)
    _insert_wrong_problem(
        database_path,
        knowledge_point="导数",
        difficulty=2,
        error_type="formula_error",
        created_at="2026-08-02 10:00:00",
    )

    update = load_student_profile(
        {"total_questions": 6},
        database_path=database_path,
    )

    assert update["knowledge_mastery"][0]["knowledge_point"] == "导数"
    assert update["recent_wrong_problems"][0]["error_type"] == "formula_error"


def test_load_student_profile_handles_empty_database(tmp_path) -> None:
    database_path = tmp_path / "empty.db"

    update = load_student_profile(
        {"total_questions": 6},
        database_path=database_path,
    )

    assert update == {
        "knowledge_mastery": [],
        "recent_wrong_problems": [],
    }
