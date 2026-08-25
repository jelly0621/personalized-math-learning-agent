"""Manually run the Phase 2 wrong-problem workflow."""

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

    print("knowledge_points:", final_state["knowledge_points"])
    print("question_type:", final_state["question_type"])
    print("difficulty:", final_state["difficulty"])
    print("problem_summary:", final_state["problem_summary"])
    print("saved_problem_id:", final_state["saved_problem_id"])


if __name__ == "__main__":
    main()
