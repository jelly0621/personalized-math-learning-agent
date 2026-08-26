"""Manually generate a Phase 4 personalized training plan."""

from math_learning_agent.graph.training_plan_graph import training_plan_graph


def main() -> None:
    final_state = training_plan_graph.invoke({"total_questions": 6})
    plan = final_state["training_plan"]

    print("Training Plan")
    print("- total_questions:", plan.total_questions)
    for item in plan.focus_items:
        print("\nFocus Item")
        print("- knowledge_point:", item.knowledge_point)
        print("- question_count:", item.question_count)
        print("- target_difficulty:", item.target_difficulty)
        print("- focus_error_types:", item.focus_error_types)
        print("- reason:", item.reason)
    print("\n- plan_reason:", plan.plan_reason)


if __name__ == "__main__":
    main()
