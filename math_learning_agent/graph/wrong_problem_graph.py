"""Analyze, diagnose, persist, and profile a wrong problem."""

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from math_learning_agent.agents import (
    analyze_problem_with_llm,
    diagnose_error_with_llm,
)
from math_learning_agent.db import save_wrong_problem, update_knowledge_mastery
from math_learning_agent.models import (
    ErrorDiagnosis,
    ProblemAnalysis,
    WrongProblemInput,
)


class WrongProblemState(TypedDict):
    """State shared by the analysis and persistence nodes."""

    problem_text: str
    student_answer: NotRequired[str | None]
    correct_answer: NotRequired[str | None]
    student_solution: NotRequired[str | None]
    knowledge_points: NotRequired[list[str]]
    question_type: NotRequired[str]
    difficulty: NotRequired[int]
    problem_summary: NotRequired[str]
    error_type: NotRequired[str]
    error_reason: NotRequired[str]
    related_knowledge_points: NotRequired[list[str]]
    error_confidence: NotRequired[float]
    saved_problem_id: NotRequired[int]


def analyze_problem(state: WrongProblemState) -> dict[str, object]:
    """Run the basic LLM problem-understanding step."""

    analysis = analyze_problem_with_llm(
        problem_text=state["problem_text"],
        student_answer=state.get("student_answer"),
        correct_answer=state.get("correct_answer"),
        student_solution=state.get("student_solution"),
    )
    return analysis.model_dump()


def diagnose_error(state: WrongProblemState) -> dict[str, object]:
    """Diagnose the wrong answer from the available evidence."""

    diagnosis = diagnose_error_with_llm(
        problem_text=state["problem_text"],
        student_answer=state.get("student_answer"),
        correct_answer=state.get("correct_answer"),
        student_solution=state.get("student_solution"),
        knowledge_points=state["knowledge_points"],
        question_type=state["question_type"],
        difficulty=state["difficulty"],
    )
    return {
        "error_type": diagnosis.error_type,
        "error_reason": diagnosis.error_reason,
        "related_knowledge_points": diagnosis.related_knowledge_points,
        "error_confidence": diagnosis.confidence,
    }


def save_problem(state: WrongProblemState) -> dict[str, int]:
    """Persist the current input and analysis fields to SQLite."""

    problem = WrongProblemInput(
        problem_text=state["problem_text"],
        student_answer=state.get("student_answer"),
        correct_answer=state.get("correct_answer"),
        student_solution=state.get("student_solution"),
    )
    analysis = ProblemAnalysis(
        knowledge_points=state["knowledge_points"],
        question_type=state["question_type"],
        difficulty=state["difficulty"],
        problem_summary=state["problem_summary"],
    )
    diagnosis = ErrorDiagnosis(
        error_type=state["error_type"],
        error_reason=state["error_reason"],
        related_knowledge_points=state["related_knowledge_points"],
        confidence=state["error_confidence"],
    )
    problem_id = save_wrong_problem(problem, analysis, diagnosis=diagnosis)
    return {"saved_problem_id": problem_id}


def update_student_profile(state: WrongProblemState) -> dict[str, object]:
    """Update deterministic wrong-count statistics for unique knowledge points."""

    update_knowledge_mastery(
        knowledge_points=state["knowledge_points"],
        related_knowledge_points=state["related_knowledge_points"],
    )
    return {}


def build_wrong_problem_graph():
    """Build the linear Phase 3 wrong-problem workflow."""

    builder = StateGraph(WrongProblemState)
    builder.add_node("analyze_problem", analyze_problem)
    builder.add_node("diagnose_error", diagnose_error)
    builder.add_node("save_problem", save_problem)
    builder.add_node("update_student_profile", update_student_profile)
    builder.add_edge(START, "analyze_problem")
    builder.add_edge("analyze_problem", "diagnose_error")
    builder.add_edge("diagnose_error", "save_problem")
    builder.add_edge("save_problem", "update_student_profile")
    builder.add_edge("update_student_profile", END)
    return builder.compile()


wrong_problem_graph = build_wrong_problem_graph()
