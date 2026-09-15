"""Offline node-level tests for the Phase 6 training feedback graph."""

from math_learning_agent.db import (
    get_knowledge_mastery,
    save_generated_question,
)
from math_learning_agent.graph.training_feedback_graph import (
    check_answer_with_sympy,
    evaluate_student_response,
    load_training_question,
    save_training_result,
    update_training_profile,
)
from math_learning_agent.models import GeneratedQuestion, VerificationResult


class FakeLLMClient:
    def invoke(self, messages) -> str:
        return """
        {
          "is_correct": true,
          "error_reason": "答案正确",
          "feedback": "答案与参考结果等价。",
          "related_knowledge_points": ["导数"],
          "confidence": 0.99
        }
        """


def _save_generated_question(database_path) -> int:
    question = GeneratedQuestion(
        question_text="化简 x+x。",
        knowledge_points=["代数运算", "代数运算"],
        difficulty=1,
        reference_answer="2*x",
        reference_solution="合并同类项得到 2*x。",
        generation_reason="训练代数化简。",
    )
    verification = VerificationResult(
        passed=True,
        answer_correct=True,
        solution_correct=True,
        knowledge_match=True,
        difficulty_match=True,
        issues=[],
        feedback="通过。",
    )
    return save_generated_question(
        question,
        verification,
        "代数运算",
        1,
        1,
        database_path,
    )


def test_training_feedback_nodes_use_temp_database_and_fake_llm(
    tmp_path,
) -> None:
    database_path = tmp_path / "feedback.db"
    question_id = _save_generated_question(database_path)
    state = {
        "question_id": question_id,
        "student_answer": "x+x",
        "student_solution": "合并同类项。",
    }

    state.update(
        load_training_question(state, database_path=database_path)
    )
    state.update(check_answer_with_sympy(state))
    state.update(
        evaluate_student_response(state, llm_client=FakeLLMClient())
    )
    state.update(save_training_result(state, database_path=database_path))
    update_training_profile(state, database_path=database_path)

    assert state["sympy_check"].status == "equivalent"
    assert state["training_evaluation"].is_correct is True
    assert state["training_attempt_id"] >= 1
    profile = get_knowledge_mastery(database_path)
    assert profile[0]["practice_count"] == 1
    assert profile[0]["correct_count"] == 1
