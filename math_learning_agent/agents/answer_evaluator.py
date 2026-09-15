"""LLM evaluation of a student response with deterministic SymPy evidence."""

import json
from typing import Any

from pydantic import ValidationError

from math_learning_agent.llm import LLMClient
from math_learning_agent.models import (
    GeneratedQuestion,
    StudentResponse,
    SympyCheckResult,
    TrainingEvaluation,
)


def _remove_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_training_evaluation(response_text: str) -> TrainingEvaluation:
    """Parse and validate an answer-evaluation JSON response."""

    try:
        payload: Any = json.loads(_remove_markdown_code_fence(response_text))
    except json.JSONDecodeError as exc:
        raise ValueError("The training evaluation is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("The training evaluation JSON must be an object.")

    try:
        return TrainingEvaluation.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            "The LLM response does not match the required TrainingEvaluation schema."
        ) from exc


def evaluate_student_response_with_llm(
    generated_question: GeneratedQuestion,
    student_response: StudentResponse,
    sympy_check: SympyCheckResult,
    llm_client: LLMClient | None = None,
) -> TrainingEvaluation:
    """Evaluate correctness and return concise teaching feedback."""

    evaluation_input = {
        "question_text": generated_question.question_text,
        "knowledge_points": generated_question.knowledge_points,
        "reference_answer": generated_question.reference_answer,
        "reference_solution": generated_question.reference_solution,
        "student_answer": student_response.student_answer,
        "student_solution": student_response.student_solution,
        "sympy_check": sympy_check.model_dump(),
    }
    prompt = f"""
你是高中数学训练作答评价器。请结合题目、参考答案、参考解析、学生实际作答和 SymPy 检查结果，判断学生是否真正答对并给出简洁教学反馈。

规则：
- SymPy status 为 equivalent 是答案数学等价的强证据。
- SymPy status 为 not_equivalent 是答案不一致的强证据，但仍需结合题型和答案形式判断。
- SymPy status 为 unsupported 只表示工具无法判断，绝不代表学生答错。
- 正确时 is_correct=true，error_reason 可明确写“答案正确”，feedback 给出简洁正向反馈。
- 错误时明确说明具体问题，并给出 related_knowledge_points。
- 不要虚构学生没有写出的推理步骤。
- 不要生成新题，不要修改数据库。

请返回且只返回一个 JSON 对象：
{{
  "is_correct": true,
  "error_reason": "答案正确",
  "feedback": "简洁教学反馈",
  "related_knowledge_points": ["相关知识点"],
  "confidence": 0.9
}}

confidence 必须在 0 到 1 之间。不要使用 Markdown 代码块，不要添加 JSON 以外的文字。

评价材料：
{json.dumps(evaluation_input, ensure_ascii=False)}
""".strip()

    client = llm_client or LLMClient()
    response_text = client.invoke([{"role": "user", "content": prompt}])
    return parse_training_evaluation(response_text)
