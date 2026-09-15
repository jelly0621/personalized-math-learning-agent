"""Vision-only extraction of math problems from textbook images."""

import json
from typing import Any, Sequence

from pydantic import ValidationError

from math_learning_agent.llm import VisionLLMClient
from math_learning_agent.models import ExtractedProblem, PhotoExtractionResult


EXTRACTION_PROMPT = """
你是数学教材图片结构化识别器。你的唯一任务是从当前图片中识别完整、可独立作答的数学题目。

图片可能包含多道题、例题、变式、选择题、填空题、教材解析、教材答案和手写批注。

要求：
1. 一张图中可能有多道题，必须分别提取。
2. problem_text 必须保留图片中清晰可见的完整题目条件。
3. 数学表达式尽可能使用可读 LaTeX 或明确数学文本。
4. printed_answer 仅在图片明确印刷或展示教材答案时填写，否则为 null。
5. printed_solution 仅在图片明确印刷或展示教材解析时填写，否则为 null。
6. 手写批注不要混入题目正文。
7. 不要把反思、总结或知识点说明当作题目。
8. 不要自己求解题目，不要推导或补写缺失答案。
9. 不要补充图片中看不清的条件；不确定内容写入 warnings。
10. 只返回 JSON，不要使用 Markdown 代码块或添加说明文字。

返回结构：
{
  "problems": [
    {
      "problem_index": 1,
      "problem_text": "完整题目正文",
      "printed_answer": null,
      "printed_solution": null,
      "confidence": 0.9
    }
  ],
  "warnings": []
}
""".strip()


def _remove_markdown_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) >= 2 and lines[-1].strip() == "```":
        return "\n".join(lines[1:-1]).strip()
    return stripped


def parse_photo_extraction_result(response_text: str) -> PhotoExtractionResult:
    """Parse JSON and validate the Vision response without silent fallback."""

    try:
        payload: Any = json.loads(_remove_markdown_code_fence(response_text))
    except json.JSONDecodeError as exc:
        raise ValueError("The Vision response is not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError("The Vision response JSON must be an object.")
    try:
        return PhotoExtractionResult.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(
            "The Vision response does not match the photo extraction schema."
        ) from exc


def extract_problems_from_images(
    images: Sequence[tuple[bytes, str]],
    vision_client: VisionLLMClient | None = None,
) -> PhotoExtractionResult:
    """Extract each image independently, then merge and globally re-index."""

    if not images:
        raise ValueError("At least one image is required.")
    client = vision_client or VisionLLMClient()
    problems: list[ExtractedProblem] = []
    warnings: list[str] = []
    failures: list[str] = []

    for source_image_index, (image_bytes, mime_type) in enumerate(images, start=1):
        try:
            response_text = client.invoke_image(
                image_bytes=image_bytes,
                mime_type=mime_type,
                prompt=EXTRACTION_PROMPT,
            )
            extracted = parse_photo_extraction_result(response_text)
        except Exception as exc:
            message = f"Image {source_image_index} extraction failed: {exc}"
            failures.append(message)
            warnings.append(message)
            continue

        warnings.extend(
            f"Image {source_image_index}: {warning}"
            for warning in extracted.warnings
        )
        for problem in extracted.problems:
            problems.append(
                problem.model_copy(
                    update={
                        "problem_index": len(problems) + 1,
                        "source_image_index": source_image_index,
                    }
                )
            )

    if failures and not problems:
        raise ValueError(
            "No image could be extracted successfully. " + "; ".join(failures)
        )
    return PhotoExtractionResult(problems=problems, warnings=warnings)
