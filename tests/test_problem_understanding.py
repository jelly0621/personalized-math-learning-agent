"""Offline tests for parsing structured LLM output."""

import pytest

from math_learning_agent.agents.problem_understanding import parse_problem_analysis


def test_parse_problem_analysis_accepts_valid_json_code_fence() -> None:
    response = """```json
{
  "knowledge_points": ["函数", "导数"],
  "question_type": "计算题",
  "difficulty": 2,
  "problem_summary": "计算函数在指定点的导数。"
}
```"""

    analysis = parse_problem_analysis(response)

    assert analysis.knowledge_points == ["函数", "导数"]
    assert analysis.difficulty == 2


def test_parse_problem_analysis_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_problem_analysis("not json")
