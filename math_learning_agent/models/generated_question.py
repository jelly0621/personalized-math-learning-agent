"""Structured models for generated questions and their quality checks."""

from typing import Self

from pydantic import BaseModel, Field, model_validator


class GeneratedQuestion(BaseModel):
    """A training question proposed by the Question Generator."""

    question_text: str
    knowledge_points: list[str]
    difficulty: int = Field(ge=1, le=5)
    reference_answer: str
    reference_solution: str
    generation_reason: str


class SolverResult(BaseModel):
    """An independent solution produced from the question text only."""

    answer: str
    solution: str


class VerificationResult(BaseModel):
    """Quality checks comparing the generated question and independent solution."""

    passed: bool
    answer_correct: bool
    solution_correct: bool
    knowledge_match: bool
    difficulty_match: bool
    issues: list[str]
    feedback: str

    @model_validator(mode="after")
    def validate_passed_consistency(self) -> Self:
        checks_passed = all(
            (
                self.answer_correct,
                self.solution_correct,
                self.knowledge_match,
                self.difficulty_match,
            )
        )
        if self.passed and (not checks_passed or self.issues):
            raise ValueError(
                "passed can only be true when every verification check passes "
                "and issues is empty."
            )
        return self
