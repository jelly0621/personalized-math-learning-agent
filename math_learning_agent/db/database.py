"""Minimal SQLite data layer for wrong problems."""

import json
from pathlib import Path
import sqlite3

from math_learning_agent.config import PROJECT_ROOT
from math_learning_agent.models import ProblemAnalysis, WrongProblemInput


DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "math_agent.db"


def initialize_database(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
    """Create the database directory and wrong_problems table when needed."""

    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS wrong_problems (
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


def save_wrong_problem(
    problem: WrongProblemInput,
    analysis: ProblemAnalysis,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> int:
    """Save a wrong problem and its analysis, then return its database ID."""

    path = Path(database_path)
    initialize_database(path)

    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO wrong_problems (
                problem_text,
                student_answer,
                correct_answer,
                student_solution,
                knowledge_points,
                question_type,
                difficulty,
                problem_summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                problem.problem_text,
                problem.student_answer,
                problem.correct_answer,
                problem.student_solution,
                json.dumps(analysis.knowledge_points, ensure_ascii=False),
                analysis.question_type,
                analysis.difficulty,
                analysis.problem_summary,
            ),
        )

    if cursor.lastrowid is None:
        raise RuntimeError("SQLite did not return an ID for the saved problem.")
    return cursor.lastrowid
