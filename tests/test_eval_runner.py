"""Offline tests for isolation, tracing, budget, and dry-run behavior."""

import json
from pathlib import Path
import sqlite3

import pytest

from evals.dataset import load_eval_dataset
from evals.pilot_runner import (
    APIBudget,
    BudgetExceeded,
    TracedBudgetedLLMClient,
    _select_verifier_challenge_cases,
    run_phase7_eval,
    seed_isolated_eval_database,
)
from math_learning_agent.models import SolverResult
from scripts.run_phase7_eval import build_parser
from evals.tracing import LocalEvalWriter


class FailingIfCalledClient:
    def invoke(self, messages: list[dict[str, str]]) -> str:
        raise AssertionError("Dry-run must not call an LLM.")


class TextClient:
    def invoke(self, messages: list[dict[str, str]]) -> str:
        return "ok"


def test_isolated_profile_database_uses_the_requested_temporary_path(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "eval-only.db"

    seed_isolated_eval_database(load_eval_dataset().cases, database_path)

    with sqlite3.connect(database_path) as connection:
        problem_count = connection.execute(
            "SELECT COUNT(*) FROM wrong_problems"
        ).fetchone()[0]
        error_count = connection.execute(
            "SELECT COUNT(*) FROM wrong_problems WHERE error_type IS NOT NULL"
        ).fetchone()[0]
    assert problem_count == 10
    assert error_count == 0


def test_budget_stops_before_call_overflow_and_preserves_counts() -> None:
    budget = APIBudget(max_llm_calls=1)
    budget.reserve_call()
    budget.record_completion(10.0, failed=False)

    with pytest.raises(BudgetExceeded):
        budget.reserve_call()

    assert budget.total_llm_calls == 1
    assert budget.stopped_by_budget is True


def test_traced_client_writes_jsonl_without_token_usage(tmp_path: Path) -> None:
    writer = LocalEvalWriter(tmp_path)
    client = TracedBudgetedLLMClient(
        TextClient(),
        APIBudget(max_llm_calls=1),
        writer,
        run_id="test-run",
        dataset_name="test-data",
        component="solver",
        case_id="R01",
    )

    assert client.invoke([{"role": "user", "content": "test"}]) == "ok"
    event = json.loads(writer.traces_path.read_text(encoding="utf-8"))
    assert event["component"] == "solver"
    assert event["token_usage"] is None
    assert event["status"] == "succeeded"


def test_dry_run_creates_all_artifacts_without_llm_calls(tmp_path: Path) -> None:
    artifacts = run_phase7_eval(
        dry_run=True,
        output_root=tmp_path,
        llm_client=FailingIfCalledClient(),
    )

    assert artifacts.summary.dataset_case_count == 10
    assert artifacts.summary.total_llm_calls == 0
    assert artifacts.summary.stopped_by_budget is False
    assert (artifacts.output_directory / "results.jsonl").is_file()
    assert (artifacts.output_directory / "traces.jsonl").is_file()
    assert (artifacts.output_directory / "summary.json").is_file()
    assert (artifacts.output_directory / "summary.md").is_file()


def test_challenge_selection_is_bounded_and_symbolically_grounded() -> None:
    cases = load_eval_dataset().cases
    solver_results = {
        case.id: SolverResult(answer=case.gold_answer, solution="fixture")
        for case in cases
        if case.auto_checkable
    }
    solver_results["R02"] = SolverResult(answer="999", solution="wrong fixture")

    selected = _select_verifier_challenge_cases(cases, solver_results, 5)

    assert len(selected) == 5
    assert all(case.auto_checkable for case in selected)
    assert "R02" not in {case.id for case in selected}


def test_phase71_cli_defaults_to_five_challenges_and_finite_budget() -> None:
    args = build_parser().parse_args([])

    assert args.challenge_cases == 5
    assert args.max_llm_calls == 50
