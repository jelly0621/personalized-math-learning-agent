"""Basic LLM-based structural understanding of a math problem."""

import json
from typing import Any

from pydantic import ValidationError

from math_learning_agent.llm import LLMClient
from math_learning_agent.models import ProblemAnalysis, WrongProblemInput


def _remove_markdown_code_fence(text: str) -> str:
    """Remove an optional Markdown JSON code fence from an LLM response."""

    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_problem_analysis(response_text: str) -> ProblemAnalysis:
    """Parse and validate a JSON analysis returned by the LLM."""

    try:
        payload: Any = json.loads(_remove_markdown_code_fence(response_text))
    except json.JSONDecodeError as exc:
        raise ValueError("The LLM response is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("The LLM response JSON must be an object.")

    try:
        return ProblemAnalysis.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            "The LLM response does not match the required ProblemAnalysis schema."
        ) from exc


def analyze_problem_with_llm(
    problem_text: str,
    student_answer: str | None = None,
    correct_answer: str | None = None,
    student_solution: str | None = None,
    llm_client: LLMClient | None = None,
) -> ProblemAnalysis:
    """Ask the LLM for basic structure only, without diagnosing student errors."""

    problem = WrongProblemInput(
        problem_text=problem_text,
        student_answer=student_answer,
        correct_answer=correct_answer,
        student_solution=student_solution,
    )
    prompt = f"""
你只负责理解一道高中数学题的基本结构，不要分析学生错因，不要提供学习建议，也不要生成新题。

请根据输入返回且只返回一个 JSON 对象，结构必须严格如下：
{{
  "knowledge_points": ["知识点1", "知识点2"],
  "question_type": "题型",
  "difficulty": 1,
  "problem_summary": "题目的一句简要概括"
}}

要求：
- difficulty 必须是 1 到 5 的整数。
- knowledge_points 必须是字符串数组。
- 不要使用 Markdown 代码块，不要添加 JSON 以外的文字。

题目信息：
{problem.model_dump_json(exclude_none=True)}
""".strip()

    client = llm_client or LLMClient()
    response_text = client.invoke([{"role": "user", "content": prompt}])
    return parse_problem_analysis(response_text)
