"""Deterministic loading and validation for Phase 7 pilot data."""

import json
from pathlib import Path

from evals.models import EvalDataset


DEFAULT_DATASET_PATH = Path(__file__).parent / "datasets" / "real_student_weak_v1.json"
EXPECTED_PILOT_CASE_COUNT = 10


def load_eval_dataset(
    path: str | Path = DEFAULT_DATASET_PATH,
) -> EvalDataset:
    """Load the source JSON without mutating it and validate its schema."""

    dataset_path = Path(path)
    with dataset_path.open("r", encoding="utf-8") as file:
        raw_data = json.load(file)

    dataset = EvalDataset.model_validate(raw_data)
    if dataset.dataset_name == "real_student_weak_v1":
        if len(dataset.cases) != EXPECTED_PILOT_CASE_COUNT:
            raise ValueError(
                "real_student_weak_v1 must contain exactly 10 evaluation cases."
            )
    return dataset


def validate_dataset_integrity(dataset: EvalDataset) -> dict[str, object]:
    """Return deterministic integrity facts after enforcing pilot invariants."""

    case_ids = [case.id for case in dataset.cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Evaluation case IDs must be unique.")

    for case in dataset.cases:
        if case.student_status != "unable_to_solve":
            raise ValueError(f"{case.id} has an invalid student_status.")
        if case.student_answer is not None or case.student_solution is not None:
            raise ValueError(f"{case.id} must not contain invented student work.")
        if case.error_type_ground_truth is not None:
            raise ValueError(f"{case.id} must not contain an inferred error ground truth.")
        if case.auto_checkable and not case.gold_answer.strip():
            raise ValueError(f"{case.id} is auto-checkable but has no gold answer.")

    topic_distribution: dict[str, int] = {}
    for case in dataset.cases:
        topic_distribution[case.topic_group] = (
            topic_distribution.get(case.topic_group, 0) + 1
        )

    return {
        "dataset_case_count": len(dataset.cases),
        "unique_ids": True,
        "topic_distribution": topic_distribution,
        "student_work_absent": True,
        "error_type_ground_truth_absent": True,
    }
