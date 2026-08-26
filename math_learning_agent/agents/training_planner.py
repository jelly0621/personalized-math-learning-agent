"""LLM-based personalized training planning from stored student history."""

import json
from typing import Any

from pydantic import ValidationError

from math_learning_agent.llm import LLMClient
from math_learning_agent.models import TrainingPlan


EMPTY_HISTORY_REASON = (
    "当前缺少个人历史数据，因此该计划属于初始探索性训练。"
)


def _remove_markdown_code_fence(text: str) -> str:
    """Remove an optional Markdown JSON code fence from an LLM response."""

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_training_plan(response_text: str) -> TrainingPlan:
    """Parse and validate a JSON training plan returned by the LLM."""

    try:
        payload: Any = json.loads(_remove_markdown_code_fence(response_text))
    except json.JSONDecodeError as exc:
        raise ValueError("The LLM training plan is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("The LLM training plan JSON must be an object.")

    try:
        return TrainingPlan.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            "The LLM response does not match the required TrainingPlan schema."
        ) from exc


def plan_training_with_llm(
    knowledge_mastery: list[dict[str, object]],
    recent_wrong_problems: list[dict[str, object]],
    total_questions: int = 6,
    llm_client: LLMClient | None = None,
) -> TrainingPlan:
    """Create a plan from exact history loaded by the deterministic data layer."""

    if not 1 <= total_questions <= 20:
        raise ValueError("total_questions must be between 1 and 20.")

    planning_context = {
        "requested_total_questions": total_questions,
        "knowledge_mastery": knowledge_mastery,
        "recent_wrong_problems": recent_wrong_problems,
    }
    prompt = f"""
你只负责根据已提供的学生历史制定个性化训练计划，决定练什么、练几道、目标难度和重点错误类型。不要生成任何具体数学题，不要重新分析原始数学题，不要修改数据库。

规划要求：
- 优先关注 wrong_count 较高的知识点，同时参考最近错题，不能只看累计次数。
- 根据历史 difficulty 选择保守且合理的 target_difficulty。
- 根据高频 error_type 设置 focus_error_types。
- 不要虚构历史中不存在的严重薄弱知识点。
- 如果两类历史数据都为空，生成保守的高中数学基础探索计划，并在 plan_reason 中明确写出：“当前缺少个人历史数据，因此该计划属于初始探索性训练。”不要假装了解学生薄弱点。
- total_questions 必须等于 {total_questions}。
- 所有 focus_items 的 question_count 总和必须严格等于 {total_questions}。
- question_count 必须是 1 到 10 的整数，target_difficulty 必须是 1 到 5 的整数。

请返回且只返回一个 JSON 对象，结构必须严格如下：
{{
  "total_questions": {total_questions},
  "focus_items": [
    {{
      "knowledge_point": "知识点",
      "question_count": 3,
      "target_difficulty": 2,
      "focus_error_types": ["calculation_error"],
      "reason": "安排原因"
    }}
  ],
  "plan_reason": "整体规划原因"
}}

不要使用 Markdown 代码块，不要添加 JSON 以外的文字。

学生历史：
{json.dumps(planning_context, ensure_ascii=False)}
""".strip()

    client = llm_client or LLMClient()
    response_text = client.invoke([{"role": "user", "content": prompt}])
    plan = parse_training_plan(response_text)
    if plan.total_questions != total_questions:
        raise ValueError(
            "The LLM training plan total_questions does not match the requested "
            "total_questions."
        )
    if (
        not knowledge_mastery
        and not recent_wrong_problems
        and EMPTY_HISTORY_REASON not in plan.plan_reason
    ):
        raise ValueError(
            "A training plan created without student history must clearly state "
            "that it is an initial exploratory plan."
        )
    return plan
