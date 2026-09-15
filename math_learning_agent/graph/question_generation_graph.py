"""Conditional quality loop for generating one targeted math question."""

from pathlib import Path
from typing import Literal, NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from math_learning_agent.agents import (
    generate_question_with_llm,
    solve_generated_question_with_llm,
    verify_generated_question_with_llm,
)
from math_learning_agent.db import DEFAULT_DATABASE_PATH, save_generated_question
from math_learning_agent.llm import LLMClient
from math_learning_agent.models import (
    GeneratedQuestion,
    SolverResult,
    VerificationResult,
)


class QuestionGenerationState(TypedDict):
    """State for the bounded question generation and verification loop."""

    knowledge_point: str
    target_difficulty: int
    focus_error_types: list[str]
    generation_context: NotRequired[dict[str, object] | None]
    generated_question: NotRequired[GeneratedQuestion]
    solver_result: NotRequired[SolverResult]
    verification_result: NotRequired[VerificationResult]
    generation_attempts: NotRequired[int]
    max_generation_attempts: NotRequired[int]
    accepted_question: NotRequired[GeneratedQuestion | None]
    saved_question_id: NotRequired[int]
    generation_status: NotRequired[Literal["accepted", "failed"]]


def generate_question(
    state: QuestionGenerationState,
    *,
    llm_client: LLMClient | None = None,
) -> dict[str, object]:
    """Generate or regenerate a question and increment the attempt counter."""

    max_attempts = state.get("max_generation_attempts", 3)
    if max_attempts < 1:
        raise ValueError("max_generation_attempts must be at least 1.")

    previous_verification = state.get("verification_result")
    previous_feedback = (
        previous_verification.feedback if previous_verification else None
    )
    question = generate_question_with_llm(
        knowledge_point=state["knowledge_point"],
        target_difficulty=state["target_difficulty"],
        focus_error_types=state["focus_error_types"],
        generation_context=state.get("generation_context"),
        previous_feedback=previous_feedback,
        llm_client=llm_client,
    )
    return {
        "generated_question": question,
        "generation_attempts": state.get("generation_attempts", 0) + 1,
    }


def solve_question(
    state: QuestionGenerationState,
    *,
    llm_client: LLMClient | None = None,
) -> dict[str, SolverResult]:
    """Give only question_text to the independent solver."""

    result = solve_generated_question_with_llm(
        question_text=state["generated_question"].question_text,
        llm_client=llm_client,
    )
    return {"solver_result": result}


def verify_question(
    state: QuestionGenerationState,
    *,
    llm_client: LLMClient | None = None,
) -> dict[str, VerificationResult]:
    """Verify correctness and alignment without modifying the question."""

    result = verify_generated_question_with_llm(
        generated_question=state["generated_question"],
        solver_result=state["solver_result"],
        target_knowledge_point=state["knowledge_point"],
        target_difficulty=state["target_difficulty"],
        llm_client=llm_client,
    )
    return {"verification_result": result}


def route_after_verification(
    state: QuestionGenerationState,
) -> Literal["accept_question", "generate_question", "mark_failed"]:
    """Route deterministically after verification without calling an LLM."""

    verification = state.get("verification_result")
    if verification is None:
        raise ValueError("verification_result is required for routing.")
    if verification.passed:
        return "accept_question"
    if state.get("generation_attempts", 0) < state.get(
        "max_generation_attempts", 3
    ):
        return "generate_question"
    return "mark_failed"


def accept_question(
    state: QuestionGenerationState,
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, object]:
    """Accept and persist the current question after a PASS result."""

    verification = state["verification_result"]
    if not verification.passed:
        raise ValueError("A question cannot be accepted without verification PASS.")
    question = state["generated_question"]
    question_id = save_generated_question(
        question=question,
        verification=verification,
        target_knowledge_point=state["knowledge_point"],
        target_difficulty=state["target_difficulty"],
        generation_attempts=state["generation_attempts"],
        database_path=database_path,
    )
    return {
        "accepted_question": question,
        "saved_question_id": question_id,
        "generation_status": "accepted",
    }


def mark_failed(state: QuestionGenerationState) -> dict[str, object]:
    """Finish without persisting a question after exhausting all attempts."""

    return {
        "accepted_question": None,
        "generation_status": "failed",
    }


def build_question_generation_graph(
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    generator_client: LLMClient | None = None,
    solver_client: LLMClient | None = None,
    verifier_client: LLMClient | None = None,
):
    """Build the bounded conditional generation, solve, and verify loop."""

    def generate_node(state: QuestionGenerationState) -> dict[str, object]:
        return generate_question(state, llm_client=generator_client)

    def solve_node(state: QuestionGenerationState) -> dict[str, SolverResult]:
        return solve_question(state, llm_client=solver_client)

    def verify_node(
        state: QuestionGenerationState,
    ) -> dict[str, VerificationResult]:
        return verify_question(state, llm_client=verifier_client)

    def accept_node(state: QuestionGenerationState) -> dict[str, object]:
        return accept_question(state, database_path=database_path)

    builder = StateGraph(QuestionGenerationState)
    builder.add_node("generate_question", generate_node)
    builder.add_node("solve_question", solve_node)
    builder.add_node("verify_question", verify_node)
    builder.add_node("accept_question", accept_node)
    builder.add_node("mark_failed", mark_failed)
    builder.add_edge(START, "generate_question")
    builder.add_edge("generate_question", "solve_question")
    builder.add_edge("solve_question", "verify_question")
    builder.add_conditional_edges(
        "verify_question",
        route_after_verification,
        {
            "accept_question": "accept_question",
            "generate_question": "generate_question",
            "mark_failed": "mark_failed",
        },
    )
    builder.add_edge("accept_question", END)
    builder.add_edge("mark_failed", END)
    return builder.compile()


question_generation_graph = build_question_generation_graph()
