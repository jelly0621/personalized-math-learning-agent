"""Deterministic metric tests; no LLM client is used."""

import pytest

from evals.metrics import (
    create_verified_corrupted_answer,
    planner_weak_cluster_metrics,
    solver_symbolic_metrics,
    surface_similarity_metrics,
)
from math_learning_agent.models import TrainingFocus, TrainingPlan
from math_learning_agent.tools import check_answer_equivalence


def test_solver_symbolic_metrics_keep_unsupported_separate_from_incorrect() -> None:
    metrics = solver_symbolic_metrics(
        ["equivalent", "not_equivalent", "unsupported", "equivalent"],
        auto_checkable_count=4,
    )

    assert metrics["solver_symbolically_verified_correct"] == 2
    assert metrics["solver_symbolically_verified_incorrect"] == 1
    assert metrics["solver_symbolic_unsupported"] == 1
    assert metrics["solver_symbolic_coverage"] == pytest.approx(0.75)


def test_solver_call_failure_is_not_counted_as_symbolic_incorrect() -> None:
    metrics = solver_symbolic_metrics(["failed", "unsupported"], 2)

    assert metrics["solver_symbolically_verified_incorrect"] == 0
    assert metrics["solver_failed_cases"] == 1
    assert metrics["solver_symbolic_coverage"] == 0.0


def test_planner_metric_uses_only_small_alias_normalization() -> None:
    plan = TrainingPlan(
        total_questions=6,
        focus_items=[
            TrainingFocus(
                knowledge_point="均值不等式",
                question_count=4,
                target_difficulty=3,
                focus_error_types=[],
                reason="fixture",
            ),
            TrainingFocus(
                knowledge_point="数列",
                question_count=2,
                target_difficulty=2,
                focus_error_types=[],
                reason="fixture",
            ),
        ],
        plan_reason="fixture",
    )

    metrics = planner_weak_cluster_metrics(plan, ["基本不等式"])

    assert metrics["planner_weak_cluster_allocation_ratio"] == pytest.approx(4 / 6)
    assert metrics["planner_top_focus_in_weak_cluster"] is True


def test_surface_similarity_is_an_explicit_near_copy_signal_only() -> None:
    metrics = surface_similarity_metrics(
        "已知 a,b 都是正数，求 8a/(7a+9b) + 8b/(9a+7b) 的最小值。",
        ["已知 a,b 都是正数，求 5a/(2a+3b) + 5b/(3a+2b) 的最小值。"],
    )

    assert metrics["surface_near_copy_flag"] is True


def test_corrupted_answer_is_accepted_only_after_symbolic_inequality() -> None:
    corrupted = create_verified_corrupted_answer("sqrt(2)+1")

    assert corrupted is not None
    assert check_answer_equivalence(corrupted, "sqrt(2)+1").status == "not_equivalent"
