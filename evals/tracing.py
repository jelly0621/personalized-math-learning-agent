"""Local JSONL tracing and summary artifact writing for Phase 7."""

import json
from pathlib import Path

from evals.models import EvalCaseResult, EvalSummary, EvalTraceEvent


class LocalEvalWriter:
    """Append durable local artifacts as each experiment step completes."""

    def __init__(self, output_directory: str | Path) -> None:
        self.output_directory = Path(output_directory)
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.results_path = self.output_directory / "results.jsonl"
        self.traces_path = self.output_directory / "traces.jsonl"
        self.summary_json_path = self.output_directory / "summary.json"
        self.summary_markdown_path = self.output_directory / "summary.md"
        self.results_path.touch(exist_ok=True)
        self.traces_path.touch(exist_ok=True)

    @staticmethod
    def _append_json_line(path: Path, payload: dict[str, object]) -> None:
        with path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def write_trace(self, event: EvalTraceEvent) -> None:
        self._append_json_line(
            self.traces_path,
            event.model_dump(mode="json"),
        )

    def write_result(self, result: EvalCaseResult) -> None:
        self._append_json_line(
            self.results_path,
            result.model_dump(mode="json"),
        )

    def write_summary(self, summary: EvalSummary) -> None:
        with self.summary_json_path.open("w", encoding="utf-8") as file:
            json.dump(
                summary.model_dump(mode="json"),
                file,
                ensure_ascii=False,
                indent=2,
            )
            file.write("\n")
        self.summary_markdown_path.write_text(
            render_summary_markdown(summary),
            encoding="utf-8",
        )


def _format_ratio(value: object) -> str:
    return f"{float(value):.3f}"


def render_summary_markdown(summary: EvalSummary) -> str:
    """Render an honestly scoped, human-readable Pilot Evaluation summary."""

    metrics = summary.metrics
    dataset = metrics.get("dataset", {})
    solver = metrics.get("solver", {})
    planner = metrics.get("planner", {})
    generator = metrics.get("generator", {})
    challenge = metrics.get("verifier_challenge", {})

    topic_distribution = dataset.get("topic_distribution", {})
    topic_lines = "\n".join(
        f"- {topic}: {count}" for topic, count in topic_distribution.items()
    ) or "- No topics recorded"
    limitation_lines = "\n".join(
        f"{index}. {limitation}"
        for index, limitation in enumerate(summary.limitations, start=1)
    )

    return f"""# Phase 7 Pilot Evaluation Summary

Run ID: `{summary.run_id}`

Dataset: `{summary.dataset_name}`

## Dataset

- Case count: {summary.dataset_case_count}
- Topic distribution:
{topic_lines}

## Solver

- Symbolically verified correct: {solver.get('solver_symbolically_verified_correct', 0)}
- Symbolically verified incorrect: {solver.get('solver_symbolically_verified_incorrect', 0)}
- Symbolic unsupported: {solver.get('solver_symbolic_unsupported', 0)}
- Solver call failures: {solver.get('solver_failed_cases', 0)}
- Symbolic coverage: {_format_ratio(solver.get('solver_symbolic_coverage', 0.0))}

Unsupported cases require manual review and are not counted as incorrect.

## Planner

- Weak-cluster allocation ratio: {_format_ratio(planner.get('planner_weak_cluster_allocation_ratio', 0.0))}
- Top focus in weak cluster: {planner.get('planner_top_focus_in_weak_cluster', False)}

This only measures response to the pilot's concentrated weak cluster.

## Generator

- Accepted count: {generator.get('generator_accepted_count', 0)}
- Failed count: {generator.get('generator_failed_count', 0)}
- Average attempts: {float(generator.get('average_generation_attempts', 0.0)):.2f}
- Verifier-judged knowledge match rate: {_format_ratio(generator.get('verifier_judged_knowledge_match_rate', 0.0))}
- Verifier-judged difficulty match rate: {_format_ratio(generator.get('verifier_judged_difficulty_match_rate', 0.0))}
- Surface near-copy flags: {generator.get('surface_near_copy_flags', 0)}

Surface similarity detects only highly similar text, not semantic copying.

## Verifier Challenge

- Positive accept rate: {_format_ratio(challenge.get('positive_accept_rate', 0.0))}
- Corrupted-answer rejection rate: {_format_ratio(challenge.get('corrupted_answer_rejection_rate', 0.0))}

This is a synthetic corruption challenge derived from real questions and reuses Solver output; it is not an independent Verifier accuracy benchmark.

## Observability

- Total LLM calls: {summary.total_llm_calls}
- Failed LLM calls: {summary.failed_llm_calls}
- Average LLM latency (ms): {summary.average_llm_latency_ms}
- Stopped by budget: {summary.stopped_by_budget}
- Token usage: unavailable (`null`) because the current LLMClient returns text only.

## Limitations

{limitation_lines}
"""
