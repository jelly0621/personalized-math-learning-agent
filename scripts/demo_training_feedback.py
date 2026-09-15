"""Interactive Phase 6 demo using the latest accepted generated question."""

from math_learning_agent.db import get_latest_generated_question
from math_learning_agent.graph.training_feedback_graph import (
    training_feedback_graph,
)


def main() -> None:
    latest = get_latest_generated_question()
    if latest is None:
        print("请先运行 Phase 5 demo 生成一道人机验证通过的训练题。")
        return

    question_id, question = latest
    print("question_id:", question_id)
    print("question_text:", question.question_text)
    student_answer = input("student_answer: ").strip()
    student_solution_input = input("student_solution (optional): ").strip()
    student_solution = student_solution_input or None

    final_state = training_feedback_graph.invoke(
        {
            "question_id": question_id,
            "student_answer": student_answer,
            "student_solution": student_solution,
        }
    )
    evaluation = final_state["training_evaluation"]
    sympy_check = final_state["sympy_check"]

    print("\nTraining Evaluation")
    print("- is_correct:", evaluation.is_correct)
    print("- error_reason:", evaluation.error_reason)
    print("- feedback:", evaluation.feedback)
    print("- related_knowledge_points:", evaluation.related_knowledge_points)
    print("- confidence:", evaluation.confidence)

    print("\nSymPy")
    print("- status:", sympy_check.status)
    print("- reason:", sympy_check.reason)

    print("\nSaved")
    print("- training_attempt_id:", final_state["training_attempt_id"])


if __name__ == "__main__":
    main()
