"""Manually run the complete Phase 3 wrong-problem workflow."""

from math_learning_agent.db import get_knowledge_mastery
from math_learning_agent.graph.wrong_problem_graph import wrong_problem_graph


def main() -> None:
    final_state = wrong_problem_graph.invoke(
        {
            "problem_text": "已知函数 f(x)=x^2-2x，求函数在 x=1 处的导数。",
            "student_answer": "2",
            "correct_answer": "0",
            "student_solution": "f'(x)=2x，所以 f'(1)=2。",
        }
    )

    print("Problem Analysis:")
    print("- knowledge_points:", final_state["knowledge_points"])
    print("- question_type:", final_state["question_type"])
    print("- difficulty:", final_state["difficulty"])

    print("\nError Diagnosis:")
    print("- error_type:", final_state["error_type"])
    print("- error_reason:", final_state["error_reason"])
    print("- confidence:", final_state["error_confidence"])
    print(
        "- related_knowledge_points:",
        final_state["related_knowledge_points"],
    )

    print("\nSaved:")
    print("- saved_problem_id:", final_state["saved_problem_id"])

    print("\nStudent Profile:")
    for item in get_knowledge_mastery()[:5]:
        print(
            f"- {item['knowledge_point']}: "
            f"wrong_count={item['wrong_count']}, "
            f"last_wrong_at={item['last_wrong_at']}"
        )


if __name__ == "__main__":
    main()
