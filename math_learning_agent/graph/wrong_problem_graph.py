"""Phase 2 workflow: analyze a wrong problem, then save it to SQLite."""

from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from math_learning_agent.agents import analyze_problem_with_llm
from math_learning_agent.db import save_wrong_problem
from math_learning_agent.models import ProblemAnalysis, WrongProblemInput


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
    problem_id = save_wrong_problem(problem, analysis)
    return {"saved_problem_id": problem_id}


def build_wrong_problem_graph():
    """Build START -> analyze_problem -> save_problem -> END."""

    builder = StateGraph(WrongProblemState)
    builder.add_node("analyze_problem", analyze_problem)
    builder.add_node("save_problem", save_problem)
    builder.add_edge(START, "analyze_problem")
    builder.add_edge("analyze_problem", "save_problem")
    builder.add_edge("save_problem", END)
    return builder.compile()


wrong_problem_graph = build_wrong_problem_graph()
