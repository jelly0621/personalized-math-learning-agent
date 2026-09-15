"""Models for a student's response and its training evaluation."""

from typing import Literal

from pydantic import BaseModel, Field


class StudentResponse(BaseModel):
    """A student's real response to an accepted generated question."""

    question_id: int = Field(ge=1)
    student_answer: str
    student_solution: str | None = None


class SympyCheckResult(BaseModel):
    """Deterministic mathematical-equivalence result from the SymPy tool."""

    status: Literal["equivalent", "not_equivalent", "unsupported"]
    reason: str


class TrainingEvaluation(BaseModel):
    """LLM evaluation and teaching feedback for one training response."""

    is_correct: bool
    error_reason: str
    feedback: str
    related_knowledge_points: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
