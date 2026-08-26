"""Structured models for a personalized training plan."""

from typing import Self

from pydantic import BaseModel, Field, model_validator


class TrainingFocus(BaseModel):
    """Training allocation for one knowledge point."""

    knowledge_point: str
    question_count: int = Field(ge=1, le=10)
    target_difficulty: int = Field(ge=1, le=5)
    focus_error_types: list[str]
    reason: str


class TrainingPlan(BaseModel):
    """A complete plan whose focus allocations match its total question count."""

    total_questions: int = Field(ge=1, le=20)
    focus_items: list[TrainingFocus]
    plan_reason: str

    @model_validator(mode="after")
    def validate_question_count_total(self) -> Self:
        allocated_questions = sum(
            item.question_count for item in self.focus_items
        )
        if allocated_questions != self.total_questions:
            raise ValueError(
                "The sum of focus_items question_count must equal "
                "total_questions."
            )
        return self
