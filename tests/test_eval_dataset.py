"""Offline integrity tests for the Phase 7 source dataset."""

from evals.dataset import load_eval_dataset, validate_dataset_integrity


def test_pilot_dataset_loads_exactly_ten_unique_cases() -> None:
    dataset = load_eval_dataset()

    assert len(dataset.cases) == 10
    assert len({case.id for case in dataset.cases}) == 10


def test_source_facts_and_seed_labels_remain_explicitly_separate() -> None:
    case = load_eval_dataset().cases[0]

    assert case.gold_answer_source == "printed_textbook_answer"
    assert case.label_provenance == (
        "manual_seed_label_from_textbook_structure_not_strict_gold"
    )
    assert case.knowledge_points_seed


def test_unable_to_solve_does_not_create_student_work_or_error_type() -> None:
    dataset = load_eval_dataset()

    assert all(case.student_status == "unable_to_solve" for case in dataset.cases)
    assert all(case.student_answer is None for case in dataset.cases)
    assert all(case.student_solution is None for case in dataset.cases)
    assert all(case.error_type_ground_truth is None for case in dataset.cases)


def test_dataset_integrity_reports_expected_facts() -> None:
    facts = validate_dataset_integrity(load_eval_dataset())

    assert facts["dataset_case_count"] == 10
    assert facts["unique_ids"] is True
    assert facts["student_work_absent"] is True
    assert facts["error_type_ground_truth_absent"] is True
