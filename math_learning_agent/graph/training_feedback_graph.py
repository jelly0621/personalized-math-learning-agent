"""Linear Phase 6 workflow for evaluating a real student response."""

from pathlib import Path
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from math_learning_agent.agents import evaluate_student_response_with_llm
from math_learning_agent.db import (
    DEFAULT_DATABASE_PATH,
    get_generated_question,
    save_training_attempt,
    update_training_mastery,
)
from math_learning_agent.llm import LLMClient
from math_learning_agent.models import (
    GeneratedQuestion,
    StudentResponse,
    SympyCheckResult,
    TrainingEvaluation,
)
from math_learning_agent.tools import check_answer_equivalence


class TrainingFeedbackState(TypedDict):
    """State shared by the Phase 6 training feedback nodes."""

    question_id: int
    student_answer: str
    student_solution: NotRequired[str | None]
    generated_question: NotRequired[GeneratedQuestion]
    student_response: NotRequired[StudentResponse]
    sympy_check: NotRequired[SympyCheckResult]
    training_evaluation: NotRequired[TrainingEvaluation]
    training_attempt_id: NotRequired[int]


def load_training_question(
    state: TrainingFeedbackState,
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, object]:
    """Load the accepted question and construct the student's response."""

    question = get_generated_question(state["question_id"], database_path)
    response = StudentResponse(
        question_id=state["question_id"],
        student_answer=state["student_answer"],
        student_solution=state.get("student_solution"),
    )
    return {
        "generated_question": question,
        "student_response": response,
    }


def check_answer_with_sympy(
    state: TrainingFeedbackState,
) -> dict[str, SympyCheckResult]:
    """Run the deterministic mathematical-equivalence tool."""

    result = check_answer_equivalence(
        student_answer=state["student_response"].student_answer,
        reference_answer=state["generated_question"].reference_answer,
    )
    return {"sympy_check": result}


def evaluate_student_response(
    state: TrainingFeedbackState,
    *,
    llm_client: LLMClient | None = None,
) -> dict[str, TrainingEvaluation]:
    """Use the LLM to evaluate the response with SymPy evidence."""

    evaluation = evaluate_student_response_with_llm(
        generated_question=state["generated_question"],
        student_response=state["student_response"],
        sympy_check=state["sympy_check"],
        llm_client=llm_client,
    )
    return {"training_evaluation": evaluation}


def save_training_result(
    state: TrainingFeedbackState,
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, int]:
    """Persist the evaluated training attempt."""

    attempt_id = save_training_attempt(
        question_id=state["question_id"],
        student_response=state["student_response"],
        evaluation=state["training_evaluation"],
        sympy_check=state["sympy_check"],
        database_path=database_path,
    )
    return {"training_attempt_id": attempt_id}


def update_training_profile(
    state: TrainingFeedbackState,
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, object]:
    """Update deterministic practice statistics without calling an LLM."""

    update_training_mastery(
        knowledge_points=state["generated_question"].knowledge_points,
        is_correct=state["training_evaluation"].is_correct,
        database_path=database_path,
    )
    return {}


def build_training_feedback_graph():
    """Build the linear Phase 6 training feedback workflow."""

    builder = StateGraph(TrainingFeedbackState)
    builder.add_node("load_training_question", load_training_question)
    builder.add_node("check_answer_with_sympy", check_answer_with_sympy)
    builder.add_node("evaluate_student_response", evaluate_student_response)
    builder.add_node("save_training_result", save_training_result)
    builder.add_node("update_training_profile", update_training_profile)
    builder.add_edge(START, "load_training_question")
    builder.add_edge("load_training_question", "check_answer_with_sympy")
    builder.add_edge("check_answer_with_sympy", "evaluate_student_response")
    builder.add_edge("evaluate_student_response", "save_training_result")
    builder.add_edge("save_training_result", "update_training_profile")
    builder.add_edge("update_training_profile", END)
    return builder.compile()


training_feedback_graph = build_training_feedback_graph()
