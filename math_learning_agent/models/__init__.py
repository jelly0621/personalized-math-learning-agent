"""Pydantic data models used by MathLearningAgent."""

from .error_diagnosis import ErrorDiagnosis
from .problem import ProblemAnalysis, WrongProblemInput
from .training import TrainingFocus, TrainingPlan

__all__ = [
    "ErrorDiagnosis",
    "ProblemAnalysis",
    "TrainingFocus",
    "TrainingPlan",
    "WrongProblemInput",
]
