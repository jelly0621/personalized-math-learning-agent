"""Independent Phase 4 workflow for personalized training planning."""

from pathlib import Path
from typing import NotRequired, TypedDict

from langgraph.graph import END, START, StateGraph

from math_learning_agent.agents import plan_training_with_llm
from math_learning_agent.db import (
    DEFAULT_DATABASE_PATH,
    get_knowledge_mastery,
    get_recent_wrong_problems,
)
from math_learning_agent.llm import LLMClient
from math_learning_agent.models import TrainingPlan


class TrainingPlanState(TypedDict):
    """State for loading history and creating a training plan."""

    total_questions: int
    knowledge_mastery: NotRequired[list[dict[str, object]]]
    recent_wrong_problems: NotRequired[list[dict[str, object]]]
    training_plan: NotRequired[TrainingPlan]


def load_student_profile(
    state: TrainingPlanState,
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
) -> dict[str, object]:
    """Load exact profile and recent-problem data without calling an LLM."""

    return {
        "knowledge_mastery": get_knowledge_mastery(database_path),
        "recent_wrong_problems": get_recent_wrong_problems(
            database_path=database_path
        ),
    }


def create_training_plan(
    state: TrainingPlanState,
    *,
    llm_client: LLMClient | None = None,
) -> dict[str, TrainingPlan]:
    """Create a validated training plan from the history in state."""

    plan = plan_training_with_llm(
        knowledge_mastery=state["knowledge_mastery"],
        recent_wrong_problems=state["recent_wrong_problems"],
        total_questions=state["total_questions"],
        llm_client=llm_client,
    )
    return {"training_plan": plan}


def build_training_plan_graph(
    *,
    database_path: str | Path = DEFAULT_DATABASE_PATH,
    llm_client: LLMClient | None = None,
):
    """Build START -> load_student_profile -> create_training_plan -> END."""

    def load_profile_node(state: TrainingPlanState) -> dict[str, object]:
        return load_student_profile(state, database_path=database_path)

    def create_plan_node(state: TrainingPlanState) -> dict[str, TrainingPlan]:
        return create_training_plan(state, llm_client=llm_client)

    builder = StateGraph(TrainingPlanState)
    builder.add_node("load_student_profile", load_profile_node)
    builder.add_node("create_training_plan", create_plan_node)
    builder.add_edge(START, "load_student_profile")
    builder.add_edge("load_student_profile", "create_training_plan")
    builder.add_edge("create_training_plan", END)
    return builder.compile()


training_plan_graph = build_training_plan_graph()
