"""Small, explicit data models for Phase 7 evaluation artifacts."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class EvalCase(BaseModel):
    """One immutable pilot case with source and seed labels kept separate."""

    model_config = ConfigDict(frozen=True)

    id: str
    source_type: Literal["real_student_weak_problem"]
    student_status: Literal["unable_to_solve"]
    problem_text: str
    gold_answer: str
    gold_answer_source: Literal["printed_textbook_answer"]
    auto_checkable: bool
    topic_group: str
    knowledge_points_seed: tuple[str, ...]
    label_provenance: Literal[
        "manual_seed_label_from_textbook_structure_not_strict_gold"
    ]
    student_answer: None = None
    student_solution: None = None
    error_type_ground_truth: None = None
    notes: str


class EvalDataset(BaseModel):
    """Immutable collection of evaluation cases."""

    model_config = ConfigDict(frozen=True)

    dataset_name: str
    description: str
    cases: tuple[EvalCase, ...]

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "EvalDataset":
        case_ids = [case.id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Evaluation case IDs must be unique.")
        return self


class EvalTraceEvent(BaseModel):
    """One local observability event for a component invocation."""

    run_id: str
    dataset_name: str
    timestamp: datetime
    case_id: str | None
    component: str
    attempt: int = Field(ge=1)
    status: Literal["succeeded", "failed", "skipped"]
    latency_ms: float | None
    error: str | None
    important_outputs: dict[str, Any]
    token_usage: dict[str, int] | None = None


class EvalCaseResult(BaseModel):
    """Structured result for one case/component pair."""

    case_id: str
    component: str
    status: Literal["succeeded", "failed", "skipped"]
    outputs: dict[str, Any]
    error: str | None = None
    manual_review_required: bool = False


class EvalSummary(BaseModel):
    """Machine-readable summary for one pilot run."""

    run_id: str
    dataset_name: str
    started_at: datetime
    finished_at: datetime
    dataset_case_count: int
    metrics: dict[str, Any]
    total_llm_calls: int
    failed_llm_calls: int
    average_llm_latency_ms: float | None
    stopped_by_budget: bool
    limitations: list[str]
