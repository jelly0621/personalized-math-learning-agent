"""Offline tests for answer-evaluation JSON parsing and prompt input."""

import pytest

from math_learning_agent.agents.answer_evaluator import (
    evaluate_student_response_with_llm,
    parse_training_evaluation,
)
from math_learning_agent.models import (
    GeneratedQuestion,
    StudentResponse,
    SympyCheckResult,
)


EVALUATION_JSON = """
{
  "is_correct": true,
  "error_reason": "答案正确",
  "feedback": "结果正确，继续保持。",
  "related_knowledge_points": ["导数"],
  "confidence": 0.98
}
"""


class RecordingLLMClient:
    def __init__(self) -> None:
        self.messages = None

    def invoke(self, messages) -> str:
        self.messages = messages
        return EVALUATION_JSON


def _question() -> GeneratedQuestion:
    return GeneratedQuestion(
        question_text="求 f(x)=x^2 的导数。",
        knowledge_points=["导数"],
        difficulty=1,
        reference_answer="2*x",
        reference_solution="根据幂函数求导公式，f'(x)=2*x。",
        generation_reason="训练基础求导。",
    )


def test_answer_evaluator_uses_allowed_context_without_network() -> None:
    client = RecordingLLMClient()
    evaluation = evaluate_student_response_with_llm(
        generated_question=_question(),
        student_response=StudentResponse(
            question_id=1,
            student_answer="x+x",
            student_solution="使用求导公式。",
        ),
        sympy_check=SympyCheckResult(
            status="equivalent",
            reason="Difference simplified to zero.",
        ),
        llm_client=client,
    )

    prompt = client.messages[0]["content"]
    assert evaluation.is_correct is True
    assert "x+x" in prompt
    assert "equivalent" in prompt
    assert "generation_reason" not in prompt


def test_training_evaluation_parser_reports_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_training_evaluation("not json")
