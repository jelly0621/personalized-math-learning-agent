"""Pydantic data models used by MathLearningAgent."""

from .error_diagnosis import ErrorDiagnosis
from .generated_question import (
    GeneratedQuestion,
    SolverResult,
    VerificationResult,
)
from .problem import ProblemAnalysis, WrongProblemInput
from .photo_extraction import ExtractedProblem, PhotoExtractionResult
from .training import TrainingFocus, TrainingPlan
from .training_attempt import (
    StudentResponse,
    SympyCheckResult,
    TrainingEvaluation,
)

__all__ = [
    "ErrorDiagnosis",
    "ExtractedProblem",
    "GeneratedQuestion",
    "ProblemAnalysis",
    "PhotoExtractionResult",
    "SolverResult",
    "StudentResponse",
    "SympyCheckResult",
    "TrainingEvaluation",
    "TrainingFocus",
    "TrainingPlan",
    "VerificationResult",
    "WrongProblemInput",
]
