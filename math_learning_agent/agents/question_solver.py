"""Independent LLM solver that receives generated question text only."""

import json
from typing import Any

from pydantic import ValidationError

from math_learning_agent.llm import LLMClient
from math_learning_agent.models import SolverResult


def _remove_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_solver_result(response_text: str) -> SolverResult:
    """Parse and validate an independent-solver JSON response."""

    try:
        payload: Any = json.loads(_remove_markdown_code_fence(response_text))
    except json.JSONDecodeError as exc:
        raise ValueError("The solver result is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("The solver result JSON must be an object.")

    try:
        return SolverResult.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            "The LLM response does not match the required SolverResult schema."
        ) from exc


def solve_generated_question_with_llm(
    question_text: str,
    llm_client: LLMClient | None = None,
) -> SolverResult:
    """Solve independently without access to generator answers or verification."""

    prompt = f"""
你是独立的高中数学解题者。请只根据下面的题目正文独立求解，不要假设存在任何参考答案或外部解析。

请返回且只返回一个 JSON 对象：
{{
  "answer": "最终答案",
  "solution": "完整、可检查的解题过程"
}}

不要使用 Markdown 代码块，不要添加 JSON 以外的文字。

题目正文：
{question_text}
""".strip()

    client = llm_client or LLMClient()
    response_text = client.invoke([{"role": "user", "content": prompt}])
    return parse_solver_result(response_text)
