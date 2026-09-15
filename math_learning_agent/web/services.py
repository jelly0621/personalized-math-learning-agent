"""Thin adapters between FastAPI routes and existing graphs/data functions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from math_learning_agent.config import PROJECT_ROOT
from math_learning_agent.agents import extract_problems_from_images
from math_learning_agent.db import (
    get_knowledge_mastery,
    get_recent_generated_questions,
    get_recent_training_attempts,
    get_recent_wrong_problems,
)
from math_learning_agent.graph.question_generation_graph import question_generation_graph
from math_learning_agent.graph.training_feedback_graph import training_feedback_graph
from math_learning_agent.graph.training_plan_graph import training_plan_graph
from math_learning_agent.graph.wrong_problem_graph import wrong_problem_graph
from math_learning_agent.web.schemas import (
    GenerateQuestionRequest,
    GeneratedQuestionResponse,
    PhotoImportRequest,
    PhotoImportResponse,
    PhotoImportResult,
    TrainingFeedbackRequest,
    TrainingFeedbackResponse,
    TrainingPlanRequest,
    WrongProblemRequest,
)


EVALUATION_ROOT = PROJECT_ROOT / "outputs" / "evals"
PACKAGED_EVALUATION_SUMMARY = (
    PROJECT_ROOT / "docs" / "evaluation" / "pilot_summary.json"
)
PILOT_LIMITATION = (
    "当前仅基于 10 道真实学生薄弱题的小规模实验，不代表完整高中数学 Benchmark。"
)


class WebServices:
    """Use production graphs and the fixed project database/output paths."""

    def __init__(self, vision_client: Any | None = None) -> None:
        self._vision_client = vision_client

    def extract_photo_problems(
        self,
        images: list[tuple[bytes, str]],
    ):
        """Extract only; this path never invokes a persistence graph."""

        return extract_problems_from_images(
            images,
            vision_client=self._vision_client,
        )

    def import_photo_problems(
        self,
        payload: PhotoImportRequest,
    ) -> PhotoImportResponse:
        """Import confirmed problems independently without batch rollback."""

        results: list[PhotoImportResult] = []
        for index, problem in enumerate(payload.problems, start=1):
            try:
                imported = self.create_wrong_problem(
                    WrongProblemRequest(
                        problem_text=problem.problem_text,
                        correct_answer=problem.correct_answer,
                        student_answer=None,
                        student_solution=None,
                    )
                )
                results.append(
                    PhotoImportResult(
                        index=index,
                        status="succeeded",
                        problem_text=problem.problem_text,
                        **imported,
                    )
                )
            except Exception as exc:
                error = (
                    "LLM configuration is unavailable."
                    if "LLM configuration" in str(exc)
                    else "This problem could not be analyzed."
                )
                results.append(
                    PhotoImportResult(
                        index=index,
                        status="failed",
                        problem_text=problem.problem_text,
                        error=error,
                    )
                )

        succeeded = sum(result.status == "succeeded" for result in results)
        return PhotoImportResponse(
            total=len(results),
            succeeded=succeeded,
            failed=len(results) - succeeded,
            results=results,
        )

    def create_wrong_problem(self, payload: WrongProblemRequest) -> dict[str, object]:
        state = wrong_problem_graph.invoke(payload.model_dump())
        return {
            key: state[key]
            for key in (
                "saved_problem_id",
                "knowledge_points",
                "question_type",
                "difficulty",
                "problem_summary",
                "error_type",
                "error_reason",
                "related_knowledge_points",
                "error_confidence",
            )
        }

    def get_profile(self) -> list[dict[str, object]]:
        rows = get_knowledge_mastery()
        return [
            {
                key: row.get(key)
                for key in (
                    "knowledge_point",
                    "wrong_count",
                    "practice_count",
                    "correct_count",
                    "last_wrong_at",
                    "last_practiced_at",
                )
            }
            for row in rows
        ]

    def create_training_plan(self, payload: TrainingPlanRequest) -> dict[str, object]:
        state = training_plan_graph.invoke(
            {"total_questions": payload.total_questions}
        )
        return state["training_plan"].model_dump()

    def generate_question(
        self, payload: GenerateQuestionRequest
    ) -> GeneratedQuestionResponse:
        state = question_generation_graph.invoke(
            {
                **payload.model_dump(),
                "max_generation_attempts": 3,
            }
        )
        question = state.get("accepted_question")
        return GeneratedQuestionResponse(
            question_id=state.get("saved_question_id"),
            question_text=question.question_text if question else None,
            knowledge_points=question.knowledge_points if question else [],
            difficulty=question.difficulty if question else None,
            generation_attempts=state.get("generation_attempts", 0),
            generation_status=state["generation_status"],
        )

    def submit_training_feedback(
        self, payload: TrainingFeedbackRequest
    ) -> TrainingFeedbackResponse:
        state = training_feedback_graph.invoke(payload.model_dump())
        evaluation = state["training_evaluation"]
        sympy_check = state["sympy_check"]
        question = state["generated_question"]
        return TrainingFeedbackResponse(
            is_correct=evaluation.is_correct,
            error_reason=evaluation.error_reason,
            feedback=evaluation.feedback,
            related_knowledge_points=evaluation.related_knowledge_points,
            confidence=evaluation.confidence,
            sympy_status=sympy_check.status,
            sympy_reason=sympy_check.reason,
            training_attempt_id=state["training_attempt_id"],
            reference_answer=question.reference_answer,
            reference_solution=question.reference_solution,
        )

    def get_wrong_problem_history(self, limit: int) -> list[dict[str, object]]:
        return get_recent_wrong_problems(limit=limit)

    def get_generated_question_history(self, limit: int) -> list[dict[str, object]]:
        return get_recent_generated_questions(limit=limit)

    def get_training_attempt_history(self, limit: int) -> list[dict[str, object]]:
        return get_recent_training_attempts(limit=limit)

    def get_latest_evaluation(self) -> dict[str, object]:
        for summary_path in self._real_eval_summaries_newest_first():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                metrics = summary["metrics"]
                solver = metrics["solver"]
                planner = metrics["planner"]
                generator = metrics["generator"]
                challenge = metrics["verifier_challenge"]
                return {
                    "available": True,
                    "evaluation_label": "Pilot Evaluation",
                    "run_id": summary["run_id"],
                    "dataset_name": summary["dataset_name"],
                    "case_count": summary["dataset_case_count"],
                    "topic_distribution": metrics["dataset"][
                        "topic_distribution"
                    ],
                    "solver": {
                        "verified_correct": solver[
                            "solver_symbolically_verified_correct"
                        ],
                        "verified_incorrect": solver[
                            "solver_symbolically_verified_incorrect"
                        ],
                        "unsupported": solver["solver_symbolic_unsupported"],
                        "symbolic_coverage": solver["solver_symbolic_coverage"],
                    },
                    "planner": {
                        "weak_cluster_allocation_ratio": planner[
                            "planner_weak_cluster_allocation_ratio"
                        ]
                    },
                    "generator": {
                        "accepted_count": generator["generator_accepted_count"],
                        "failed_count": generator["generator_failed_count"],
                        "accept_rate": generator["generator_accept_rate"],
                        "average_attempts": generator[
                            "average_generation_attempts"
                        ],
                    },
                    "verifier_challenge": {
                        "positive_accept_rate": challenge["positive_accept_rate"],
                        "corrupted_answer_rejection_rate": challenge[
                            "corrupted_answer_rejection_rate"
                        ],
                        "challenge_case_count": challenge[
                            "verifier_challenge_case_count"
                        ],
                    },
                    "observability": {
                        "total_llm_calls": summary["total_llm_calls"],
                        "failed_llm_calls": summary["failed_llm_calls"],
                    },
                    "limitation": PILOT_LIMITATION,
                }
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                continue

        try:
            packaged = json.loads(
                PACKAGED_EVALUATION_SUMMARY.read_text(encoding="utf-8")
            )
            return {
                "available": True,
                "evaluation_label": "Pilot Evaluation",
                "dataset_name": packaged["dataset_name"],
                "case_count": packaged["case_count"],
                "topic_distribution": packaged["topic_distribution"],
                "solver": packaged["solver"],
                "planner": packaged["planner"],
                "generator": packaged["generator"],
                "verifier_challenge": packaged["verifier_challenge"],
                "observability": packaged["observability"],
                "limitation": " ".join(packaged["limitations"]),
            }
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
        return {
            "available": False,
            "evaluation_label": "Pilot Evaluation",
            "limitation": PILOT_LIMITATION,
        }

    @staticmethod
    def _real_eval_summaries_newest_first() -> list[Path]:
        if not EVALUATION_ROOT.is_dir():
            return []
        candidates: list[Path] = []
        for directory in EVALUATION_ROOT.iterdir():
            summary_path = directory / "summary.json"
            results_path = directory / "results.jsonl"
            if not summary_path.is_file() or not results_path.is_file():
                continue
            try:
                first_result = json.loads(
                    results_path.read_text(encoding="utf-8").splitlines()[0]
                )
            except (IndexError, OSError, ValueError, json.JSONDecodeError):
                continue
            if first_result.get("outputs", {}).get("dry_run") is False:
                candidates.append(summary_path)
        return sorted(
            candidates,
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
