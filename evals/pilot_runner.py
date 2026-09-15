"""Bounded Phase 7 pilot runner with isolated storage and local observability."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import gc
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any
from uuid import uuid4

from evals.dataset import load_eval_dataset, validate_dataset_integrity
from evals.metrics import (
    create_verified_corrupted_answer,
    generator_pilot_metrics,
    planner_weak_cluster_metrics,
    solver_symbolic_metrics,
    surface_similarity_metrics,
    verifier_challenge_metrics,
)
from evals.models import EvalCaseResult, EvalSummary, EvalTraceEvent
from evals.tracing import LocalEvalWriter
from math_learning_agent.agents import (
    solve_generated_question_with_llm,
    verify_generated_question_with_llm,
)
from math_learning_agent.db import save_wrong_problem, update_knowledge_mastery
from math_learning_agent.graph.question_generation_graph import (
    build_question_generation_graph,
)
from math_learning_agent.graph.training_plan_graph import build_training_plan_graph
from math_learning_agent.llm import LLMClient
from math_learning_agent.models import GeneratedQuestion, ProblemAnalysis, WrongProblemInput
from math_learning_agent.tools import check_answer_equivalence


DEFAULT_OUTPUT_ROOT = Path("outputs") / "evals"


class BudgetExceeded(RuntimeError):
    """Raised before a call that would exceed the configured LLM-call budget."""


@dataclass
class APIBudget:
    """A process-local hard guard around all pilot LLM calls."""

    max_llm_calls: int
    total_llm_calls: int = 0
    failed_llm_calls: int = 0
    stopped_by_budget: bool = False
    total_latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.max_llm_calls < 1:
            raise ValueError("max_llm_calls must be at least 1.")

    def reserve_call(self) -> None:
        if self.total_llm_calls >= self.max_llm_calls:
            self.stopped_by_budget = True
            raise BudgetExceeded(
                f"LLM call budget exhausted ({self.max_llm_calls} calls)."
            )
        self.total_llm_calls += 1

    def record_completion(self, latency_ms: float, *, failed: bool) -> None:
        self.total_latency_ms += latency_ms
        if failed:
            self.failed_llm_calls += 1

    @property
    def average_latency_ms(self) -> float | None:
        if self.total_llm_calls == 0:
            return None
        return self.total_latency_ms / self.total_llm_calls


class TracedBudgetedLLMClient:
    """Duck-typed LLMClient wrapper that records one event per invoke call."""

    def __init__(
        self,
        base_client: Any,
        budget: APIBudget,
        writer: LocalEvalWriter,
        *,
        run_id: str,
        dataset_name: str,
        component: str,
        case_id: str | None = None,
    ) -> None:
        self._base_client = base_client
        self._budget = budget
        self._writer = writer
        self._run_id = run_id
        self._dataset_name = dataset_name
        self._component = component
        self._case_id = case_id
        self._attempt = 0

    def invoke(self, messages: list[dict[str, str]]) -> str:
        self._attempt += 1
        try:
            self._budget.reserve_call()
        except BudgetExceeded as exc:
            self._write_event("skipped", None, str(exc), {})
            raise
        started = perf_counter()
        try:
            response = self._base_client.invoke(messages)
        except Exception as exc:
            latency = (perf_counter() - started) * 1000
            self._budget.record_completion(latency, failed=True)
            self._write_event("failed", latency, str(exc), {})
            raise

        latency = (perf_counter() - started) * 1000
        self._budget.record_completion(latency, failed=False)
        self._write_event(
            "succeeded",
            latency,
            None,
            {"response_character_count": len(response)},
        )
        return response

    def _write_event(
        self,
        status: str,
        latency_ms: float | None,
        error: str | None,
        important_outputs: dict[str, object],
    ) -> None:
        self._writer.write_trace(
            EvalTraceEvent(
                run_id=self._run_id,
                dataset_name=self._dataset_name,
                timestamp=datetime.now(timezone.utc),
                case_id=self._case_id,
                component=self._component,
                attempt=self._attempt,
                status=status,
                latency_ms=latency_ms,
                error=error,
                important_outputs=important_outputs,
                # The current LLMClient returns text only and exposes no usage object.
                token_usage=None,
            )
        )


@dataclass(frozen=True)
class PilotRunArtifacts:
    """Paths and summary returned to the CLI and offline tests."""

    output_directory: Path
    summary: EvalSummary


def seed_isolated_eval_database(cases: tuple[Any, ...], database_path: Path) -> None:
    """Create an evaluation-only profile from transparent fixture labels."""

    for case in cases:
        analysis = ProblemAnalysis(
            knowledge_points=list(case.knowledge_points_seed),
            question_type=case.topic_group,
            difficulty=3,
            problem_summary="Phase 7 evaluation fixture; not an LLM analysis.",
        )
        save_wrong_problem(
            WrongProblemInput(problem_text=case.problem_text),
            analysis,
            database_path=database_path,
        )
        update_knowledge_mastery(
            list(case.knowledge_points_seed),
            [],
            database_path=database_path,
        )


def _new_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"phase7_{timestamp}_{uuid4().hex[:8]}"


def _limitations() -> list[str]:
    return [
        "The pilot contains only 10 cases and is not a representative benchmark.",
        "The cases are concentrated in one weak-topic cluster, so planner results do not establish general personalization quality.",
        "No student answer or solution is available; error-diagnosis accuracy cannot be evaluated.",
        "Knowledge-point labels are manual seed labels, not strict gold annotations.",
        "No difficulty gold labels exist, so difficulty quality is only verifier-judged.",
        "Generator alignment metrics reuse the LLM verifier and are not independent expert judgments.",
        "The verifier challenge is synthetic and does not establish verifier accuracy.",
        "Surface similarity is not semantic copy or plagiarism detection.",
        "There is no ablation baseline, so this pilot does not prove that Solver, Verifier, or SymPy causally improves the end-to-end system.",
    ]


def _empty_metrics(dataset_metrics: dict[str, object]) -> dict[str, object]:
    return {
        "dataset": dataset_metrics,
        "solver": solver_symbolic_metrics([], 0),
        "planner": {
            "planner_weak_cluster_allocation_ratio": 0.0,
            "planner_top_focus_in_weak_cluster": False,
        },
        "generator": generator_pilot_metrics([]),
        "verifier_challenge": verifier_challenge_metrics([], []),
    }


def run_phase7_eval(
    *,
    dry_run: bool,
    suite: str = "pilot",
    max_cases: int = 10,
    max_llm_calls: int = 50,
    max_focuses: int = 3,
    challenge_cases: int = 5,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    llm_client: Any | None = None,
) -> PilotRunArtifacts:
    """Run validation only or the bounded real-LLM Phase 7 pilot."""

    if suite != "pilot":
        raise ValueError("Only the 'pilot' suite exists in Phase 7.")
    if not 1 <= max_cases <= 10:
        raise ValueError("max_cases must be between 1 and 10.")
    if not 1 <= max_focuses <= 3:
        raise ValueError("max_focuses must be between 1 and 3.")
    if challenge_cases < 0:
        raise ValueError("challenge_cases cannot be negative.")

    started_at = datetime.now(timezone.utc)
    run_id = _new_run_id()
    dataset = load_eval_dataset()
    dataset_snapshot = dataset.model_dump_json()
    dataset_metrics = validate_dataset_integrity(dataset)
    selected_cases = dataset.cases[:max_cases]
    output_directory = Path(output_root) / run_id
    writer = LocalEvalWriter(output_directory)
    budget = APIBudget(max_llm_calls=max_llm_calls)
    metrics = _empty_metrics(dataset_metrics)

    writer.write_result(
        EvalCaseResult(
            case_id="dataset",
            component="dataset_validation",
            status="succeeded",
            outputs={
                **dataset_metrics,
                "selected_case_count": len(selected_cases),
                "dry_run": dry_run,
                "max_llm_calls": max_llm_calls,
                "max_focuses": max_focuses,
                "challenge_cases": challenge_cases,
            },
        )
    )
    writer.write_trace(
        EvalTraceEvent(
            run_id=run_id,
            dataset_name=dataset.dataset_name,
            timestamp=datetime.now(timezone.utc),
            case_id=None,
            component="dataset_validation",
            attempt=1,
            status="succeeded",
            latency_ms=None,
            error=None,
            important_outputs={"case_count": len(dataset.cases)},
            token_usage=None,
        )
    )

    with TemporaryDirectory(prefix="math-agent-phase7-") as temporary_directory:
        eval_database = Path(temporary_directory) / "pilot.db"
        seed_isolated_eval_database(selected_cases, eval_database)
        writer.write_result(
            EvalCaseResult(
                case_id="profile_fixture",
                component="isolated_database",
                status="succeeded",
                outputs={
                    "database_scope": "temporary_evaluation_only",
                    "fixture_label_provenance": (
                        "manual_seed_label_from_textbook_structure_not_strict_gold"
                    ),
                },
            )
        )

        if not dry_run:
            base_client = llm_client or LLMClient()
            solver_statuses, solver_results = _run_solver_experiment(
                selected_cases, base_client, budget, writer, run_id, dataset.dataset_name
            )
            auto_count = sum(case.auto_checkable for case in selected_cases)
            metrics["solver"] = solver_symbolic_metrics(
                solver_statuses, auto_count
            )

            plan = _run_planner_experiment(
                base_client,
                budget,
                writer,
                run_id,
                dataset.dataset_name,
                eval_database,
            )
            generator_records: list[dict[str, object]] = []
            if plan is not None:
                weak_labels = [
                    label
                    for case in selected_cases
                    for label in case.knowledge_points_seed
                ]
                metrics["planner"] = planner_weak_cluster_metrics(plan, weak_labels)
                generator_records = _run_generator_experiment(
                    plan,
                    selected_cases,
                    base_client,
                    budget,
                    writer,
                    run_id,
                    dataset.dataset_name,
                    eval_database,
                    max_focuses,
                )
                metrics["generator"] = generator_pilot_metrics(generator_records)

            positive, negative = _run_verifier_challenge(
                selected_cases,
                solver_results,
                base_client,
                budget,
                writer,
                run_id,
                dataset.dataset_name,
                challenge_cases,
            )
            metrics["verifier_challenge"] = verifier_challenge_metrics(
                positive, negative
            )

        # sqlite3 context managers commit/rollback but do not close the
        # connection object themselves.  Collect any short-lived connection
        # objects before Windows tries to remove the temporary database.
        gc.collect()

    if dataset.model_dump_json() != dataset_snapshot:
        raise RuntimeError("The immutable source evaluation dataset was modified.")

    summary = EvalSummary(
        run_id=run_id,
        dataset_name=dataset.dataset_name,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc),
        dataset_case_count=len(dataset.cases),
        metrics=metrics,
        total_llm_calls=budget.total_llm_calls,
        failed_llm_calls=budget.failed_llm_calls,
        average_llm_latency_ms=budget.average_latency_ms,
        stopped_by_budget=budget.stopped_by_budget,
        limitations=_limitations(),
    )
    writer.write_summary(summary)
    return PilotRunArtifacts(output_directory=output_directory, summary=summary)


def _client(
    base_client: Any,
    budget: APIBudget,
    writer: LocalEvalWriter,
    run_id: str,
    dataset_name: str,
    component: str,
    case_id: str | None = None,
) -> TracedBudgetedLLMClient:
    return TracedBudgetedLLMClient(
        base_client,
        budget,
        writer,
        run_id=run_id,
        dataset_name=dataset_name,
        component=component,
        case_id=case_id,
    )


def _run_solver_experiment(
    cases: tuple[Any, ...],
    base_client: Any,
    budget: APIBudget,
    writer: LocalEvalWriter,
    run_id: str,
    dataset_name: str,
) -> tuple[list[str], dict[str, Any]]:
    statuses: list[str] = []
    solver_results: dict[str, Any] = {}
    for case in cases:
        if budget.stopped_by_budget:
            if case.auto_checkable:
                statuses.append("failed")
            continue
        try:
            result = solve_generated_question_with_llm(
                case.problem_text,
                _client(
                    base_client, budget, writer, run_id, dataset_name, "solver", case.id
                ),
            )
            solver_results[case.id] = result
            outputs: dict[str, object] = result.model_dump()
            manual_review = not case.auto_checkable
            if case.auto_checkable:
                sympy_result = check_answer_equivalence(result.answer, case.gold_answer)
                statuses.append(sympy_result.status)
                outputs["sympy_check"] = sympy_result.model_dump()
                writer.write_trace(
                    EvalTraceEvent(
                        run_id=run_id,
                        dataset_name=dataset_name,
                        timestamp=datetime.now(timezone.utc),
                        case_id=case.id,
                        component="sympy",
                        attempt=1,
                        status="succeeded",
                        latency_ms=None,
                        error=None,
                        important_outputs=sympy_result.model_dump(),
                        token_usage=None,
                    )
                )
            else:
                outputs["sympy_check"] = {
                    "status": "not_applicable",
                    "reason": "The printed answer is not a single supported expression.",
                }
            writer.write_result(
                EvalCaseResult(
                    case_id=case.id,
                    component="solver",
                    status="succeeded",
                    outputs=outputs,
                    manual_review_required=manual_review
                    or outputs["sympy_check"]["status"] == "unsupported",
                )
            )
        except BudgetExceeded as exc:
            if case.auto_checkable:
                statuses.append("failed")
            writer.write_result(
                EvalCaseResult(
                    case_id=case.id,
                    component="solver",
                    status="skipped",
                    outputs={},
                    error=str(exc),
                    manual_review_required=True,
                )
            )
        except Exception as exc:
            if case.auto_checkable:
                statuses.append("failed")
            writer.write_result(
                EvalCaseResult(
                    case_id=case.id,
                    component="solver",
                    status="failed",
                    outputs={},
                    error=str(exc),
                    manual_review_required=True,
                )
            )
    return statuses, solver_results


def _run_planner_experiment(
    base_client: Any,
    budget: APIBudget,
    writer: LocalEvalWriter,
    run_id: str,
    dataset_name: str,
    database_path: Path,
):
    if budget.stopped_by_budget:
        return None
    try:
        graph = build_training_plan_graph(
            database_path=database_path,
            llm_client=_client(
                base_client, budget, writer, run_id, dataset_name, "planner"
            ),
        )
        state = graph.invoke({"total_questions": 6})
        plan = state["training_plan"]
        writer.write_result(
            EvalCaseResult(
                case_id="student_profile",
                component="planner",
                status="succeeded",
                outputs=plan.model_dump(),
            )
        )
        return plan
    except Exception as exc:
        writer.write_result(
            EvalCaseResult(
                case_id="student_profile",
                component="planner",
                status="skipped" if isinstance(exc, BudgetExceeded) else "failed",
                outputs={},
                error=str(exc),
                manual_review_required=True,
            )
        )
        return None


def _run_generator_experiment(
    plan: Any,
    cases: tuple[Any, ...],
    base_client: Any,
    budget: APIBudget,
    writer: LocalEvalWriter,
    run_id: str,
    dataset_name: str,
    database_path: Path,
    max_focuses: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    seed_texts = [case.problem_text for case in cases]
    for index, focus in enumerate(plan.focus_items[:max_focuses], start=1):
        if budget.stopped_by_budget:
            break
        case_id = f"focus_{index}"
        try:
            graph = build_question_generation_graph(
                database_path=database_path,
                generator_client=_client(
                    base_client, budget, writer, run_id, dataset_name, "generator", case_id
                ),
                solver_client=_client(
                    base_client, budget, writer, run_id, dataset_name, "generated_solver", case_id
                ),
                verifier_client=_client(
                    base_client, budget, writer, run_id, dataset_name, "generator_verifier", case_id
                ),
            )
            state = graph.invoke(
                {
                    "knowledge_point": focus.knowledge_point,
                    "target_difficulty": focus.target_difficulty,
                    "focus_error_types": focus.focus_error_types,
                    "generation_context": {
                        "source": "phase7_isolated_profile",
                        "reason": focus.reason,
                    },
                    "max_generation_attempts": 3,
                }
            )
            verification = state.get("verification_result")
            record: dict[str, object] = {
                "generation_status": state["generation_status"],
                "generation_attempts": state.get("generation_attempts", 0),
                "verification_result": (
                    verification.model_dump() if verification is not None else None
                ),
            }
            generated = state.get("generated_question")
            if generated is not None:
                record.update(
                    surface_similarity_metrics(generated.question_text, seed_texts)
                )
                record["generated_question"] = generated.model_dump()
            records.append(record)
            writer.write_result(
                EvalCaseResult(
                    case_id=case_id,
                    component="generator_pipeline",
                    status="succeeded",
                    outputs=record,
                    manual_review_required=bool(record.get("surface_near_copy_flag")),
                )
            )
        except Exception as exc:
            writer.write_result(
                EvalCaseResult(
                    case_id=case_id,
                    component="generator_pipeline",
                    status="skipped" if isinstance(exc, BudgetExceeded) else "failed",
                    outputs={},
                    error=str(exc),
                    manual_review_required=True,
                )
            )
            if isinstance(exc, BudgetExceeded):
                break
    return records


def _run_verifier_challenge(
    cases: tuple[Any, ...],
    solver_results: dict[str, Any],
    base_client: Any,
    budget: APIBudget,
    writer: LocalEvalWriter,
    run_id: str,
    dataset_name: str,
    challenge_cases: int,
) -> tuple[list[bool], list[bool]]:
    positive_passes: list[bool] = []
    corrupted_rejections: list[bool] = []
    eligible = _select_verifier_challenge_cases(
        cases, solver_results, challenge_cases
    )
    for case in eligible:
        if budget.stopped_by_budget:
            break
        solver_result = solver_results[case.id]
        question = GeneratedQuestion(
            question_text=case.problem_text,
            knowledge_points=list(case.knowledge_points_seed),
            difficulty=3,
            reference_answer=case.gold_answer,
            reference_solution=solver_result.solution,
            generation_reason="Synthetic positive verifier challenge fixture.",
        )
        corrupted_answer = create_verified_corrupted_answer(case.gold_answer)
        try:
            positive = verify_generated_question_with_llm(
                question,
                solver_result,
                case.knowledge_points_seed[0],
                3,
                _client(
                    base_client,
                    budget,
                    writer,
                    run_id,
                    dataset_name,
                    "verifier_challenge_positive",
                    case.id,
                ),
            )
            positive_passes.append(positive.passed)
            if corrupted_answer is None:
                continue
            corrupted = question.model_copy(
                update={"reference_answer": corrupted_answer}
            )
            negative = verify_generated_question_with_llm(
                corrupted,
                solver_result,
                case.knowledge_points_seed[0],
                3,
                _client(
                    base_client,
                    budget,
                    writer,
                    run_id,
                    dataset_name,
                    "verifier_challenge_corrupted",
                    case.id,
                ),
            )
            corrupted_rejections.append(not negative.passed)
            writer.write_result(
                EvalCaseResult(
                    case_id=case.id,
                    component="verifier_challenge",
                    status="succeeded",
                    outputs={
                        "positive_passed": positive.passed,
                        "corrupted_answer": corrupted_answer,
                        "corrupted_rejected": not negative.passed,
                        "challenge_type": "synthetic_plus_one_corruption",
                    },
                )
            )
        except Exception as exc:
            writer.write_result(
                EvalCaseResult(
                    case_id=case.id,
                    component="verifier_challenge",
                    status="skipped" if isinstance(exc, BudgetExceeded) else "failed",
                    outputs={},
                    error=str(exc),
                    manual_review_required=True,
                )
            )
            if isinstance(exc, BudgetExceeded):
                break
    return positive_passes, corrupted_rejections


def _select_verifier_challenge_cases(
    cases: tuple[Any, ...],
    solver_results: dict[str, Any],
    max_cases: int,
) -> list[Any]:
    """Select only symbolically grounded cases with a safe corruption."""

    return [
        case
        for case in cases
        if case.auto_checkable
        and case.id in solver_results
        and check_answer_equivalence(
            solver_results[case.id].answer, case.gold_answer
        ).status
        == "equivalent"
        and create_verified_corrupted_answer(case.gold_answer) is not None
    ][:max_cases]
