"""LLM-based generation of one targeted high-school mathematics question."""

import json
from typing import Any

from pydantic import ValidationError

from math_learning_agent.llm import LLMClient
from math_learning_agent.models import GeneratedQuestion


def _remove_markdown_code_fence(text: str) -> str:
    """Remove an optional Markdown JSON code fence from an LLM response."""

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_generated_question(response_text: str) -> GeneratedQuestion:
    """Parse and validate a generated-question JSON response."""

    try:
        payload: Any = json.loads(_remove_markdown_code_fence(response_text))
    except json.JSONDecodeError as exc:
        raise ValueError("The generated question is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("The generated question JSON must be an object.")

    try:
        return GeneratedQuestion.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            "The LLM response does not match the required GeneratedQuestion schema."
        ) from exc


def generate_question_with_llm(
    knowledge_point: str,
    target_difficulty: int,
    focus_error_types: list[str],
    generation_context: dict[str, object] | None = None,
    previous_feedback: str | None = None,
    llm_client: LLMClient | None = None,
) -> GeneratedQuestion:
    """Generate one targeted question, optionally using verifier feedback."""

    generation_input = {
        "knowledge_point": knowledge_point,
        "target_difficulty": target_difficulty,
        "focus_error_types": focus_error_types,
        "generation_context": generation_context,
        "previous_verifier_feedback": previous_feedback,
    }
    prompt = f"""
你是高中数学训练题生成器。请生成一道全新的训练题，必须针对指定 knowledge_point，难度尽量符合 target_difficulty。

要求：
- focus_error_types 非空时，题目设计应有针对性，但不要故意诱导学生犯错。
- 不允许只是机械替换已有错题中的数字。
- 不要引用学生个人隐私、历史答案或身份信息。
- previous_verifier_feedback 非空时，必须根据上一轮反馈重新生成并修正问题。
- 必须给出参考答案、完整参考解析和生成原因。
- 只生成一道题。

请返回且只返回一个 JSON 对象，结构必须严格如下：
{{
  "question_text": "题目正文",
  "knowledge_points": ["知识点"],
  "difficulty": 2,
  "reference_answer": "参考答案",
  "reference_solution": "完整参考解析",
  "generation_reason": "针对训练目标的原因"
}}

difficulty 必须是 1 到 5 的整数。不要使用 Markdown 代码块，不要添加 JSON 以外的文字。

生成目标与上下文：
{json.dumps(generation_input, ensure_ascii=False)}
""".strip()

    client = llm_client or LLMClient()
    response_text = client.invoke([{"role": "user", "content": prompt}])
    return parse_generated_question(response_text)
