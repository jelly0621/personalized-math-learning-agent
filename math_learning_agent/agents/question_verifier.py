"""LLM verifier for generated-question correctness and target alignment."""

import json
from typing import Any

from pydantic import ValidationError

from math_learning_agent.llm import LLMClient
from math_learning_agent.models import (
    GeneratedQuestion,
    SolverResult,
    VerificationResult,
)


def _remove_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_verification_result(response_text: str) -> VerificationResult:
    """Parse and validate a verifier JSON response."""

    try:
        payload: Any = json.loads(_remove_markdown_code_fence(response_text))
    except json.JSONDecodeError as exc:
        raise ValueError("The verification result is not valid JSON.") from exc

    if not isinstance(payload, dict):
        raise ValueError("The verification result JSON must be an object.")

    try:
        return VerificationResult.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            "The LLM response does not match the required VerificationResult schema."
        ) from exc


def verify_generated_question_with_llm(
    generated_question: GeneratedQuestion,
    solver_result: SolverResult,
    target_knowledge_point: str,
    target_difficulty: int,
    llm_client: LLMClient | None = None,
) -> VerificationResult:
    """Judge quality and alignment without modifying the generated question."""

    verification_input = {
        "generated_question": generated_question.model_dump(),
        "independent_solver_result": solver_result.model_dump(),
        "target_knowledge_point": target_knowledge_point,
        "target_difficulty": target_difficulty,
    }
    prompt = f"""
你是高中数学训练题质量验证者。你只负责检查并反馈，不要修改题目，也不要生成替代题目。

请检查：
1. Generator reference_answer 与 Independent Solver answer 是否一致或数学上等价。
2. Generator reference_solution 是否正确、合理。
3. Independent Solver solution 是否正确、合理。
4. 题目是否确实考查目标 knowledge_point。
5. 实际难度是否大致符合 target_difficulty。
6. 是否存在条件不足、歧义、矛盾或不可解情况。

只有 answer_correct、solution_correct、knowledge_match、difficulty_match 全部为 true，且 issues 为空时，passed 才能为 true。Verifier 只能判断并提供 feedback。

请返回且只返回一个 JSON 对象：
{{
  "passed": false,
  "answer_correct": false,
  "solution_correct": false,
  "knowledge_match": true,
  "difficulty_match": true,
  "issues": ["具体问题"],
  "feedback": "给下一轮 Generator 的明确改进反馈"
}}

不要使用 Markdown 代码块，不要添加 JSON 以外的文字。

验证材料：
{json.dumps(verification_input, ensure_ascii=False)}
""".strip()

    client = llm_client or LLMClient()
    response_text = client.invoke([{"role": "user", "content": prompt}])
    return parse_verification_result(response_text)
