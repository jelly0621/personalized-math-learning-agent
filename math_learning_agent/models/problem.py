"""Data models for manually entered wrong problems."""

from pydantic import BaseModel, Field


class WrongProblemInput(BaseModel):
    """Original problem information entered by a student."""

    problem_text: str
    student_answer: str | None = None
    correct_answer: str | None = None
    student_solution: str | None = None


class ProblemAnalysis(BaseModel):
    """Basic structural analysis returned by the LLM."""

    knowledge_points: list[str]
    question_type: str
    difficulty: int = Field(ge=1, le=5)
    problem_summary: str
