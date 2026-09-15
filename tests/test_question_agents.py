"""Offline tests for Phase 5 LLM response parsing and solver isolation."""

import pytest

from math_learning_agent.agents.question_generator import (
    parse_generated_question,
)
from math_learning_agent.agents.question_solver import (
    parse_solver_result,
    solve_generated_question_with_llm,
)
from math_learning_agent.agents.question_verifier import (
    parse_verification_result,
)


class RecordingLLMClient:
    def __init__(self, response: str) -> None:
        self.response = response
        self.messages = None

    def invoke(self, messages) -> str:
        self.messages = messages
        return self.response


def test_solver_receives_question_text_only() -> None:
    client = RecordingLLMClient(
        '{"answer": "0", "solution": "独立求解得到 0。"}'
    )

    result = solve_generated_question_with_llm(
        question_text="求 f(x)=x^2-2x 在 x=1 处的导数。",
        llm_client=client,
    )

    prompt = client.messages[0]["content"]
    assert result.answer == "0"
    assert "reference_answer" not in prompt
    assert "reference_solution" not in prompt
    assert "generation_reason" not in prompt
    assert "verification" not in prompt.lower()


@pytest.mark.parametrize(
    ("parser", "expected_message"),
    [
        (parse_generated_question, "not valid JSON"),
        (parse_solver_result, "not valid JSON"),
        (parse_verification_result, "not valid JSON"),
    ],
)
def test_question_agent_parsers_report_invalid_json(
    parser,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        parser("not json")
