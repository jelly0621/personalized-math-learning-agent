"""Tests for the SQLite wrong-problem data layer."""

import json
import sqlite3

from math_learning_agent.db import initialize_database, save_wrong_problem
from math_learning_agent.models import ProblemAnalysis, WrongProblemInput


def test_initialize_and_save_wrong_problem(tmp_path) -> None:
    database_path = tmp_path / "test_math_agent.db"
    initialize_database(database_path)

    problem = WrongProblemInput(
        problem_text="求函数 f(x)=x^2 的导数。",
        student_answer="x",
        correct_answer="2x",
        student_solution="直接写出答案。",
    )
    analysis = ProblemAnalysis(
        knowledge_points=["导数", "幂函数求导"],
        question_type="计算题",
        difficulty=1,
        problem_summary="求一个二次函数的导数。",
    )

    problem_id = save_wrong_problem(problem, analysis, database_path)

    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """
            SELECT problem_text, student_answer, correct_answer,
                   student_solution, knowledge_points, question_type,
                   difficulty, problem_summary, created_at
            FROM wrong_problems
            WHERE id = ?
            """,
            (problem_id,),
        ).fetchone()

    assert row is not None
    assert row[0] == problem.problem_text
    assert row[1] == problem.student_answer
    assert row[2] == problem.correct_answer
    assert row[3] == problem.student_solution
    assert json.loads(row[4]) == analysis.knowledge_points
    assert row[5] == analysis.question_type
    assert row[6] == analysis.difficulty
    assert row[7] == analysis.problem_summary
    assert row[8]
