"""Offline tests for the Phase 4 training planner."""

import pytest

from math_learning_agent.agents.training_planner import (
    parse_training_plan,
    plan_training_with_llm,
)


class FakeLLMClient:
    """Small local substitute that returns a fixed JSON response."""

    def __init__(self, response: str) -> None:
        self.response = response

    def invoke(self, messages) -> str:
        assert messages[0]["role"] == "user"
        return self.response


VALID_EMPTY_HISTORY_PLAN = """
{
  "total_questions": 6,
  "focus_items": [
    {
      "knowledge_point": "函数基础",
      "question_count": 3,
      "target_difficulty": 1,
      "focus_error_types": [],
      "reason": "用于探索函数基础掌握情况。"
    },
    {
      "knowledge_point": "代数运算",
      "question_count": 3,
      "target_difficulty": 1,
      "focus_error_types": [],
      "reason": "用于探索基础运算掌握情况。"
    }
  ],
  "plan_reason": "当前缺少个人历史数据，因此该计划属于初始探索性训练。"
}
"""


def test_parse_training_plan_accepts_valid_json() -> None:
    plan = parse_training_plan(VALID_EMPTY_HISTORY_PLAN)

    assert plan.total_questions == 6
    assert sum(item.question_count for item in plan.focus_items) == 6


def test_parse_training_plan_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_training_plan("not json")


def test_empty_history_returns_exploratory_plan_without_network() -> None:
    plan = plan_training_with_llm(
        knowledge_mastery=[],
        recent_wrong_problems=[],
        total_questions=6,
        llm_client=FakeLLMClient(VALID_EMPTY_HISTORY_PLAN),
    )

    assert "当前缺少个人历史数据" in plan.plan_reason
    assert plan.total_questions == 6


def test_empty_history_rejects_plan_that_claims_personalization() -> None:
    misleading_response = VALID_EMPTY_HISTORY_PLAN.replace(
        "当前缺少个人历史数据，因此该计划属于初始探索性训练。",
        "根据学生历史薄弱点制定。",
    )

    with pytest.raises(ValueError, match="initial exploratory plan"):
        plan_training_with_llm(
            knowledge_mastery=[],
            recent_wrong_problems=[],
            total_questions=6,
            llm_client=FakeLLMClient(misleading_response),
        )
