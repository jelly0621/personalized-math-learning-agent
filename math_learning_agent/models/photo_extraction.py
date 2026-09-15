"""Structured output from textbook-photo extraction only."""

from pydantic import BaseModel, Field


class ExtractedProblem(BaseModel):
    """One complete problem visibly present in an uploaded image."""

    problem_index: int = Field(ge=1)
    source_image_index: int = Field(default=1, ge=1)
    problem_text: str = Field(min_length=1)
    printed_answer: str | None = None
    printed_solution: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)


class PhotoExtractionResult(BaseModel):
    """Problems and uncertainties extracted from one or more images."""

    problems: list[ExtractedProblem]
    warnings: list[str]
