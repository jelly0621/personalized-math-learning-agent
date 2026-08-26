"""Minimal SQLite data layer for wrong problems."""

import json
from pathlib import Path
import sqlite3

from math_learning_agent.config import PROJECT_ROOT
from math_learning_agent.models import (
    ErrorDiagnosis,
    ProblemAnalysis,
    WrongProblemInput,
)


DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "math_agent.db"

_WRONG_PROBLEM_PHASE_3_COLUMNS = {
    "error_type": "TEXT",
    "error_reason": "TEXT",
    "related_knowledge_points": "TEXT",
    "error_confidence": "REAL",
}


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
                error_type TEXT,
                error_reason TEXT,
                related_knowledge_points TEXT,
                error_confidence REAL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        existing_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(wrong_problems)")
        }
        for column_name, column_type in _WRONG_PROBLEM_PHASE_3_COLUMNS.items():
            if column_name not in existing_columns:
                connection.execute(
                    f"ALTER TABLE wrong_problems ADD COLUMN {column_name} {column_type}"
                )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS knowledge_mastery (
                knowledge_point TEXT PRIMARY KEY,
                wrong_count INTEGER NOT NULL DEFAULT 0,
                last_wrong_at TEXT,
                updated_at TEXT
            )
            """
        )


def save_wrong_problem(
    problem: WrongProblemInput,
    analysis: ProblemAnalysis,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    diagnosis: ErrorDiagnosis | None = None,
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
                problem_summary,
                error_type,
                error_reason,
                related_knowledge_points,
                error_confidence
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                diagnosis.error_type if diagnosis else None,
                diagnosis.error_reason if diagnosis else None,
                (
                    json.dumps(
                        diagnosis.related_knowledge_points,
                        ensure_ascii=False,
                    )
                    if diagnosis
                    else None
                ),
                diagnosis.confidence if diagnosis else None,
            ),
        )

    if cursor.lastrowid is None:
        raise RuntimeError("SQLite did not return an ID for the saved problem.")
    return cursor.lastrowid


def update_knowledge_mastery(
    knowledge_points: list[str],
    related_knowledge_points: list[str],
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
    """Increment each unique knowledge point once for the current problem."""

    path = Path(database_path)
    initialize_database(path)
    unique_points = list(
        dict.fromkeys(
            point.strip()
            for point in [*knowledge_points, *related_knowledge_points]
            if point.strip()
        )
    )

    with sqlite3.connect(path) as connection:
        for knowledge_point in unique_points:
            connection.execute(
                """
                INSERT INTO knowledge_mastery (
                    knowledge_point,
                    wrong_count,
                    last_wrong_at,
                    updated_at
                ) VALUES (?, 1, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(knowledge_point) DO UPDATE SET
                    wrong_count = wrong_count + 1,
                    last_wrong_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (knowledge_point,),
            )


def get_knowledge_mastery(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, object]]:
    """Return all knowledge points ordered by wrong count descending."""

    path = Path(database_path)
    initialize_database(path)

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT knowledge_point, wrong_count, last_wrong_at, updated_at
            FROM knowledge_mastery
            ORDER BY wrong_count DESC, knowledge_point ASC
            """
        ).fetchall()

    return [dict(row) for row in rows]


def get_recent_wrong_problems(
    limit: int = 20,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, object]]:
    """Return recent wrong-problem context ordered newest first."""

    if limit < 1:
        raise ValueError("limit must be at least 1.")

    path = Path(database_path)
    initialize_database(path)

    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, knowledge_points, difficulty, error_type,
                   related_knowledge_points, created_at
            FROM wrong_problems
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    recent_problems: list[dict[str, object]] = []
    for row in rows:
        recent_problems.append(
            {
                "id": row["id"],
                "knowledge_points": json.loads(row["knowledge_points"]),
                "difficulty": row["difficulty"],
                "error_type": row["error_type"],
                "related_knowledge_points": (
                    json.loads(row["related_knowledge_points"])
                    if row["related_knowledge_points"]
                    else []
                ),
                "created_at": row["created_at"],
            }
        )
    return recent_problems
