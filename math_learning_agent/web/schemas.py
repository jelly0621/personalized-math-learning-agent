"""Explicit HTTP schemas for the local Web MVP."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class APIModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class WrongProblemRequest(APIModel):
    problem_text: str = Field(min_length=1)
    student_answer: str | None = None
    correct_answer: str | None = None
    student_solution: str | None = None


class WrongProblemResponse(APIModel):
    saved_problem_id: int
    knowledge_points: list[str]
    question_type: str
    difficulty: int
    problem_summary: str
    error_type: str
    error_reason: str
    related_knowledge_points: list[str]
    error_confidence: float


class ProfileItem(APIModel):
    knowledge_point: str
    wrong_count: int
    practice_count: int
    correct_count: int
    last_wrong_at: str | None
    last_practiced_at: str | None


class TrainingPlanRequest(APIModel):
    total_questions: int = Field(default=6, ge=1, le=20)


class TrainingFocusResponse(APIModel):
    knowledge_point: str
    question_count: int
    target_difficulty: int
    focus_error_types: list[str]
    reason: str


class TrainingPlanResponse(APIModel):
    total_questions: int
    focus_items: list[TrainingFocusResponse]
    plan_reason: str


class GenerateQuestionRequest(APIModel):
    knowledge_point: str = Field(min_length=1)
    target_difficulty: int = Field(ge=1, le=5)
    focus_error_types: list[str] = Field(default_factory=list)


class GeneratedQuestionResponse(APIModel):
    """Student-safe response: reference answer fields intentionally absent."""

    question_id: int | None
    question_text: str | None
    knowledge_points: list[str]
    difficulty: int | None
    generation_attempts: int
    generation_status: Literal["accepted", "failed"]


class TrainingFeedbackRequest(APIModel):
    question_id: int = Field(ge=1)
    student_answer: str = Field(min_length=1)
    student_solution: str | None = None


class TrainingFeedbackResponse(APIModel):
    is_correct: bool
    error_reason: str
    feedback: str
    related_knowledge_points: list[str]
    confidence: float
    sympy_status: Literal["equivalent", "not_equivalent", "unsupported"]
    sympy_reason: str
    training_attempt_id: int
    reference_answer: str
    reference_solution: str


class PhotoImportProblemRequest(APIModel):
    """One user-confirmed problem; student work must remain absent."""

    problem_text: str = Field(min_length=1)
    correct_answer: str | None = None
    student_answer: None = None
    student_solution: None = None


class PhotoImportRequest(APIModel):
    problems: list[PhotoImportProblemRequest] = Field(min_length=1)


class PhotoImportResult(APIModel):
    index: int
    status: Literal["succeeded", "failed"]
    saved_problem_id: int | None = None
    problem_text: str
    knowledge_points: list[str] | None = None
    question_type: str | None = None
    difficulty: int | None = None
    problem_summary: str | None = None
    error_type: str | None = None
    error_reason: str | None = None
    related_knowledge_points: list[str] | None = None
    error_confidence: float | None = None
    error: str | None = None


class PhotoImportResponse(APIModel):
    total: int
    succeeded: int
    failed: int
    results: list[PhotoImportResult]


class EvaluationResponse(APIModel):
    available: bool
    evaluation_label: str = "Pilot Evaluation"
    run_id: str | None = None
    dataset_name: str | None = None
    case_count: int | None = None
    topic_distribution: dict[str, int] | None = None
    solver: dict[str, Any] | None = None
    planner: dict[str, Any] | None = None
    generator: dict[str, Any] | None = None
    verifier_challenge: dict[str, Any] | None = None
    observability: dict[str, Any] | None = None
    limitation: str
