"""Evidence-based LLM diagnosis for a student's wrong answer."""

import json
from typing import Any

from pydantic import ValidationError

from math_learning_agent.llm import LLMClient
from math_learning_agent.models import ErrorDiagnosis


ERROR_TYPES = (
    "concept_error",
    "calculation_error",
    "formula_error",
    "condition_omission",
    "incomplete_case_analysis",
    "reasoning_error",
    "problem_misunderstanding",
    "method_selection_error",
    "unknown",
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


def parse_error_diagnosis(response_text: str) -> ErrorDiagnosis:
    """Parse and validate a JSON error diagnosis returned by the LLM."""

    try:
        payload: Any = json.loads(_remove_markdown_code_fence(response_text))
    except json.JSONDecodeError as exc:
        raise ValueError("The LLM error diagnosis is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("The LLM error diagnosis JSON must be an object.")

    try:
        return ErrorDiagnosis.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            "The LLM response does not match the required ErrorDiagnosis schema."
        ) from exc


def diagnose_error_with_llm(
    problem_text: str,
    student_answer: str | None,
    correct_answer: str | None,
    student_solution: str | None,
    knowledge_points: list[str],
    question_type: str,
    difficulty: int,
    llm_client: LLMClient | None = None,
) -> ErrorDiagnosis:
    """Diagnose why the answer is wrong using only the available evidence."""

    diagnosis_input = {
        "problem_text": problem_text,
        "student_answer": student_answer,
        "correct_answer": correct_answer,
        "student_solution": student_solution,
        "knowledge_points": knowledge_points,
        "question_type": question_type,
        "difficulty": difficulty,
    }
    allowed_types = ", ".join(ERROR_TYPES)
    prompt = f"""
你只负责分析学生为什么做错，不要重新进行题目结构分析，不要重新概括题目，也不要提供训练计划或生成新题。

请依据题目、学生答案、正确答案和学生解题过程中的明确证据进行诊断。禁止在证据不足时猜测：如果 student_answer、correct_answer 或 student_solution 不足以判断错因，请返回 error_type 为 "unknown"，并在 error_reason 中明确说明证据不足。

请返回且只返回一个 JSON 对象，结构必须严格如下：
{{
  "error_type": "unknown",
  "error_reason": "诊断理由",
  "related_knowledge_points": ["相关知识点"],
  "confidence": 0.0
}}

要求：
- error_type 优先从以下类型中选择：{allowed_types}。
- confidence 必须是 0.0 到 1.0 之间的数字。
- related_knowledge_points 必须是字符串数组。
- 不要使用 Markdown 代码块，不要添加 JSON 以外的文字。

已有信息：
{json.dumps(diagnosis_input, ensure_ascii=False)}
""".strip()

    client = llm_client or LLMClient()
    response_text = client.invoke([{"role": "user", "content": prompt}])
    return parse_error_diagnosis(response_text)
