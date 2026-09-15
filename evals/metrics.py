"""Deterministic, honestly named metrics for the Phase 7 pilot."""

from collections.abc import Sequence
from difflib import SequenceMatcher
import re

from math_learning_agent.models import TrainingPlan
from math_learning_agent.tools import check_answer_equivalence


_KNOWLEDGE_ALIASES = {
    "基本不等式": "基本不等式",
    "均值不等式": "基本不等式",
    "基本不等式求最值": "基本不等式",
    "结构统一": "结构变换",
    "结构变换": "结构变换",
    "配凑": "结构变换",
    "换元": "结构变换",
    "分式结构变换": "结构变换",
    "分式结构统一": "结构变换",
    "1的代换": "结构变换",
    "整体换元": "结构变换",
}


def normalize_knowledge_point(value: str) -> str:
    """Apply the pilot's intentionally small deterministic alias map."""

    normalized = value.strip()
    return _KNOWLEDGE_ALIASES.get(normalized, normalized)


def solver_symbolic_metrics(
    statuses: Sequence[str],
    auto_checkable_count: int,
) -> dict[str, int | float]:
    """Count only explicit SymPy outcomes; unsupported is not incorrect."""

    equivalent = sum(status == "equivalent" for status in statuses)
    not_equivalent = sum(status == "not_equivalent" for status in statuses)
    unsupported = sum(status == "unsupported" for status in statuses)
    failed = sum(status == "failed" for status in statuses)
    if len(statuses) != auto_checkable_count:
        raise ValueError("statuses must contain one result per auto-checkable case.")
    coverage = (
        (equivalent + not_equivalent) / auto_checkable_count
        if auto_checkable_count
        else 0.0
    )
    return {
        "solver_symbolically_verified_correct": equivalent,
        "solver_symbolically_verified_incorrect": not_equivalent,
        "solver_symbolic_unsupported": unsupported,
        "solver_failed_cases": failed,
        "solver_symbolic_coverage": coverage,
    }


def planner_weak_cluster_metrics(
    plan: TrainingPlan,
    weak_seed_labels: Sequence[str],
) -> dict[str, float | bool]:
    """Measure allocation to the known pilot cluster, not general personalization."""

    weak_cluster = {
        normalize_knowledge_point(label) for label in weak_seed_labels
    }
    allocated = sum(
        item.question_count
        for item in plan.focus_items
        if normalize_knowledge_point(item.knowledge_point) in weak_cluster
    )
    top_item = max(plan.focus_items, key=lambda item: item.question_count)
    return {
        "planner_weak_cluster_allocation_ratio": allocated
        / plan.total_questions,
        "planner_top_focus_in_weak_cluster": (
            normalize_knowledge_point(top_item.knowledge_point) in weak_cluster
        ),
    }


def normalize_surface_text(value: str) -> str:
    """Normalize digits and formatting for a surface-only near-copy signal."""

    digits_normalized = re.sub(r"\d+(?:\.\d+)?", "N", value.lower())
    return "".join(character for character in digits_normalized if character.isalnum())


def surface_similarity_metrics(
    generated_text: str,
    real_seed_texts: Sequence[str],
    threshold: float = 0.85,
) -> dict[str, float | bool]:
    """Return surface similarity only; this is not semantic plagiarism detection."""

    normalized_generated = normalize_surface_text(generated_text)
    similarities = [
        SequenceMatcher(
            None,
            normalized_generated,
            normalize_surface_text(seed_text),
        ).ratio()
        for seed_text in real_seed_texts
    ]
    maximum = max(similarities, default=0.0)
    return {
        "max_surface_similarity_to_real_seed": maximum,
        "surface_near_copy_flag": maximum > threshold,
    }


def create_verified_corrupted_answer(gold_answer: str) -> str | None:
    """Create a deterministic +1 corruption only when SymPy proves inequality."""

    candidate = f"({gold_answer})+1"
    result = check_answer_equivalence(candidate, gold_answer)
    return candidate if result.status == "not_equivalent" else None


def generator_pilot_metrics(
    records: Sequence[dict[str, object]],
) -> dict[str, int | float]:
    """Aggregate generation and LLM-verifier-judged targeting outcomes."""

    accepted = [record for record in records if record["generation_status"] == "accepted"]
    attempts = [int(record["generation_attempts"]) for record in records]
    verifier_records = [record for record in records if record.get("verification_result")]
    return {
        "generator_accepted_count": len(accepted),
        "generator_failed_count": len(records) - len(accepted),
        "generator_accept_rate": len(accepted) / len(records) if records else 0.0,
        "average_generation_attempts": (
            sum(attempts) / len(attempts) if attempts else 0.0
        ),
        "verifier_rejection_count": sum(
            max(int(record["generation_attempts"]) - 1, 0) for record in records
        ),
        "verifier_judged_knowledge_match_rate": (
            sum(
                bool(record["verification_result"]["knowledge_match"])
                for record in verifier_records
            )
            / len(verifier_records)
            if verifier_records
            else 0.0
        ),
        "verifier_judged_difficulty_match_rate": (
            sum(
                bool(record["verification_result"]["difficulty_match"])
                for record in verifier_records
            )
            / len(verifier_records)
            if verifier_records
            else 0.0
        ),
        "surface_near_copy_flags": sum(
            bool(record.get("surface_near_copy_flag")) for record in records
        ),
    }


def verifier_challenge_metrics(
    positive_passes: Sequence[bool],
    corrupted_rejections: Sequence[bool],
) -> dict[str, float | int]:
    """Aggregate the synthetic corruption challenge without calling it accuracy."""

    return {
        "verifier_challenge_case_count": len(positive_passes),
        "positive_accept_rate": (
            sum(positive_passes) / len(positive_passes)
            if positive_passes
            else 0.0
        ),
        "corrupted_answer_rejection_rate": (
            sum(corrupted_rejections) / len(corrupted_rejections)
            if corrupted_rejections
            else 0.0
        ),
    }
