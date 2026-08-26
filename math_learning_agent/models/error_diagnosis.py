"""Data model for diagnosing why a student answered incorrectly."""

from pydantic import BaseModel, Field


class ErrorDiagnosis(BaseModel):
    """Structured evidence-based diagnosis of a wrong answer."""

    error_type: str
    error_reason: str
    related_knowledge_points: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
