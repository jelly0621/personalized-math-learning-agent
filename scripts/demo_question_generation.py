"""Manually run the Phase 5 question-generation quality loop."""

from math_learning_agent.graph.question_generation_graph import (
    question_generation_graph,
)


def main() -> None:
    final_state = question_generation_graph.invoke(
        {
            "knowledge_point": "导数的计算",
            "target_difficulty": 2,
            "focus_error_types": ["calculation_error"],
            "generation_attempts": 0,
            "max_generation_attempts": 3,
        }
    )

    print("generation_status:", final_state["generation_status"])
    print("generation_attempts:", final_state["generation_attempts"])

    accepted_question = final_state.get("accepted_question")
    if accepted_question is not None:
        print("question_text:", accepted_question.question_text)
        print("difficulty:", accepted_question.difficulty)
        print("reference_answer:", accepted_question.reference_answer)
        print("reference_solution:", accepted_question.reference_solution)

    verification = final_state["verification_result"]
    print("Verifier:")
    print("- passed:", verification.passed)
    print("- answer_correct:", verification.answer_correct)
    print("- solution_correct:", verification.solution_correct)
    print("- knowledge_match:", verification.knowledge_match)
    print("- difficulty_match:", verification.difficulty_match)
    print("- issues:", verification.issues)
    print("- feedback:", verification.feedback)


if __name__ == "__main__":
    main()
