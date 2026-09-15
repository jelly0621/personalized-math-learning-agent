"""Minimal SQLite data layer for wrong problems."""

import json
import os
from pathlib import Path
import sqlite3

from math_learning_agent.config import PROJECT_ROOT
from math_learning_agent.models import (
    ErrorDiagnosis,
    GeneratedQuestion,
    ProblemAnalysis,
    StudentResponse,
    SympyCheckResult,
    TrainingEvaluation,
    VerificationResult,
    WrongProblemInput,
)


def get_default_database_path() -> Path:
    """Return the deployment override or the project-local SQLite path."""

    configured_path = os.getenv("DATABASE_PATH", "").strip()
    if configured_path:
        path = Path(configured_path)
        if not path.is_absolute():
            raise RuntimeError("DATABASE_PATH must be an absolute path.")
        return path
    return PROJECT_ROOT / "data" / "math_agent.db"


DEFAULT_DATABASE_PATH = get_default_database_path()

_WRONG_PROBLEM_PHASE_3_COLUMNS = {
    "error_type": "TEXT",
    "error_reason": "TEXT",
    "related_knowledge_points": "TEXT",
    "error_confidence": "REAL",
}

_KNOWLEDGE_MASTERY_PHASE_6_COLUMNS = {
    "practice_count": "INTEGER NOT NULL DEFAULT 0",
    "correct_count": "INTEGER NOT NULL DEFAULT 0",
    "last_practiced_at": "TEXT",
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
                practice_count INTEGER NOT NULL DEFAULT 0,
                correct_count INTEGER NOT NULL DEFAULT 0,
                last_practiced_at TEXT,
                updated_at TEXT
            )
            """
        )
        mastery_columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(knowledge_mastery)")
        }
        for column_name, column_definition in (
            _KNOWLEDGE_MASTERY_PHASE_6_COLUMNS.items()
        ):
            if column_name not in mastery_columns:
                connection.execute(
                    "ALTER TABLE knowledge_mastery "
                    f"ADD COLUMN {column_name} {column_definition}"
                )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS generated_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_text TEXT NOT NULL,
                knowledge_points TEXT NOT NULL,
                difficulty INTEGER NOT NULL CHECK (difficulty BETWEEN 1 AND 5),
                reference_answer TEXT NOT NULL,
                reference_solution TEXT NOT NULL,
                generation_reason TEXT NOT NULL,
                target_knowledge_point TEXT NOT NULL,
                target_difficulty INTEGER NOT NULL
                    CHECK (target_difficulty BETWEEN 1 AND 5),
                generation_attempts INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS training_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER NOT NULL,
                student_answer TEXT NOT NULL,
                student_solution TEXT,
                is_correct INTEGER NOT NULL,
                error_reason TEXT NOT NULL,
                feedback TEXT NOT NULL,
                related_knowledge_points TEXT NOT NULL,
                evaluation_confidence REAL NOT NULL,
                sympy_status TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
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
            SELECT knowledge_point, wrong_count, last_wrong_at,
                   practice_count, correct_count, last_practiced_at, updated_at
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


def save_generated_question(
    question: GeneratedQuestion,
    verification: VerificationResult,
    target_knowledge_point: str,
    target_difficulty: int,
    generation_attempts: int,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> int:
    """Save a generated question only after a successful verification."""

    if not verification.passed:
        raise ValueError("Only a passed generated question can be saved.")
    if not 1 <= target_difficulty <= 5:
        raise ValueError("target_difficulty must be between 1 and 5.")
    if generation_attempts < 1:
        raise ValueError("generation_attempts must be at least 1.")

    path = Path(database_path)
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO generated_questions (
                question_text,
                knowledge_points,
                difficulty,
                reference_answer,
                reference_solution,
                generation_reason,
                target_knowledge_point,
                target_difficulty,
                generation_attempts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                question.question_text,
                json.dumps(question.knowledge_points, ensure_ascii=False),
                question.difficulty,
                question.reference_answer,
                question.reference_solution,
                question.generation_reason,
                target_knowledge_point,
                target_difficulty,
                generation_attempts,
            ),
        )

    if cursor.lastrowid is None:
        raise RuntimeError("SQLite did not return an ID for the generated question.")
    return cursor.lastrowid


def _row_to_generated_question(row: sqlite3.Row) -> GeneratedQuestion:
    return GeneratedQuestion(
        question_text=row["question_text"],
        knowledge_points=json.loads(row["knowledge_points"]),
        difficulty=row["difficulty"],
        reference_answer=row["reference_answer"],
        reference_solution=row["reference_solution"],
        generation_reason=row["generation_reason"],
    )


def get_generated_question(
    question_id: int,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> GeneratedQuestion:
    """Load one accepted generated question by its database ID."""

    if question_id < 1:
        raise ValueError("question_id must be at least 1.")

    path = Path(database_path)
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT question_text, knowledge_points, difficulty,
                   reference_answer, reference_solution, generation_reason
            FROM generated_questions
            WHERE id = ?
            """,
            (question_id,),
        ).fetchone()

    if row is None:
        raise LookupError(f"Generated question {question_id} was not found.")
    return _row_to_generated_question(row)


def get_latest_generated_question(
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> tuple[int, GeneratedQuestion] | None:
    """Return the ID and content of the most recently accepted question."""

    path = Path(database_path)
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT id, question_text, knowledge_points, difficulty,
                   reference_answer, reference_solution, generation_reason
            FROM generated_questions
            ORDER BY created_at DESC, id DESC
            LIMIT 1
            """
        ).fetchone()

    if row is None:
        return None
    return row["id"], _row_to_generated_question(row)


def get_recent_generated_questions(
    limit: int = 10,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, object]]:
    """Return recent generated-question metadata without reference answers."""

    if limit < 1:
        raise ValueError("limit must be at least 1.")
    path = Path(database_path)
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, question_text, knowledge_points, difficulty,
                   target_knowledge_point, target_difficulty,
                   generation_attempts, created_at
            FROM generated_questions
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            **dict(row),
            "knowledge_points": json.loads(row["knowledge_points"]),
        }
        for row in rows
    ]


def get_recent_training_attempts(
    limit: int = 10,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> list[dict[str, object]]:
    """Return compact recent training-attempt history."""

    if limit < 1:
        raise ValueError("limit must be at least 1.")
    path = Path(database_path)
    initialize_database(path)
    with sqlite3.connect(path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT id, question_id, is_correct, feedback,
                   related_knowledge_points, sympy_status, created_at
            FROM training_attempts
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [
        {
            **dict(row),
            "is_correct": bool(row["is_correct"]),
            "related_knowledge_points": json.loads(
                row["related_knowledge_points"]
            ),
        }
        for row in rows
    ]


def save_training_attempt(
    question_id: int,
    student_response: StudentResponse,
    evaluation: TrainingEvaluation,
    sympy_check: SympyCheckResult,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> int:
    """Persist one evaluated response to an accepted generated question."""

    if question_id != student_response.question_id:
        raise ValueError("question_id must match student_response.question_id.")

    path = Path(database_path)
    get_generated_question(question_id, path)
    with sqlite3.connect(path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO training_attempts (
                question_id,
                student_answer,
                student_solution,
                is_correct,
                error_reason,
                feedback,
                related_knowledge_points,
                evaluation_confidence,
                sympy_status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                question_id,
                student_response.student_answer,
                student_response.student_solution,
                int(evaluation.is_correct),
                evaluation.error_reason,
                evaluation.feedback,
                json.dumps(
                    evaluation.related_knowledge_points,
                    ensure_ascii=False,
                ),
                evaluation.confidence,
                sympy_check.status,
            ),
        )

    if cursor.lastrowid is None:
        raise RuntimeError("SQLite did not return an ID for the training attempt.")
    return cursor.lastrowid


def update_training_mastery(
    knowledge_points: list[str],
    is_correct: bool,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> None:
    """Update training-only practice statistics for unique knowledge points."""

    path = Path(database_path)
    initialize_database(path)
    unique_points = list(
        dict.fromkeys(point.strip() for point in knowledge_points if point.strip())
    )
    correct_increment = int(is_correct)

    with sqlite3.connect(path) as connection:
        for knowledge_point in unique_points:
            connection.execute(
                """
                INSERT INTO knowledge_mastery (
                    knowledge_point,
                    wrong_count,
                    practice_count,
                    correct_count,
                    last_practiced_at,
                    updated_at
                ) VALUES (?, 0, 1, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                ON CONFLICT(knowledge_point) DO UPDATE SET
                    practice_count = practice_count + 1,
                    correct_count = correct_count + excluded.correct_count,
                    last_practiced_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (knowledge_point, correct_increment),
            )
