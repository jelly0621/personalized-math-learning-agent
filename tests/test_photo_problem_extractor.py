"""Offline extraction tests using a fake Vision client."""

import pytest

from math_learning_agent.agents.photo_problem_extractor import (
    EXTRACTION_PROMPT,
    extract_problems_from_images,
    parse_photo_extraction_result,
)


THREE_PROBLEMS_JSON = """
{
  "problems": [
    {"problem_index": 1, "problem_text": "题目一", "printed_answer": "1", "printed_solution": null, "confidence": 0.95},
    {"problem_index": 2, "problem_text": "题目二", "printed_answer": null, "printed_solution": null, "confidence": 0.85},
    {"problem_index": 3, "problem_text": "题目三", "printed_answer": "3", "printed_solution": "教材解析", "confidence": 0.75}
  ],
  "warnings": ["题目三有一处字符较模糊"]
}
"""


class FakeVisionClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = iter(responses)
        self.calls = []

    def invoke_image(self, image_bytes, mime_type, prompt) -> str:
        self.calls.append((image_bytes, mime_type, prompt))
        return next(self.responses)


def test_vision_json_parses_three_problems() -> None:
    result = parse_photo_extraction_result(THREE_PROBLEMS_JSON)

    assert len(result.problems) == 3
    assert result.problems[1].printed_answer is None


def test_invalid_vision_json_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_photo_extraction_result("not-json")


def test_images_are_called_separately_and_results_are_reindexed() -> None:
    client = FakeVisionClient([THREE_PROBLEMS_JSON, THREE_PROBLEMS_JSON])

    result = extract_problems_from_images(
        [(b"image-one", "image/png"), (b"image-two", "image/jpeg")],
        vision_client=client,
    )

    assert len(client.calls) == 2
    assert len(result.problems) == 6
    assert [problem.problem_index for problem in result.problems] == list(range(1, 7))
    assert result.problems[3].source_image_index == 2
    assert "error_type" not in EXTRACTION_PROMPT
    assert "不要自己求解题目" in EXTRACTION_PROMPT


def test_one_bad_image_does_not_discard_a_successful_image() -> None:
    client = FakeVisionClient(["invalid", THREE_PROBLEMS_JSON])

    result = extract_problems_from_images(
        [(b"bad", "image/png"), (b"good", "image/png")],
        vision_client=client,
    )

    assert len(result.problems) == 3
    assert result.problems[0].source_image_index == 2
    assert any("Image 1 extraction failed" in warning for warning in result.warnings)


def test_all_invalid_images_do_not_silently_return_empty_result() -> None:
    with pytest.raises(ValueError, match="No image could be extracted"):
        extract_problems_from_images(
            [(b"bad", "image/png")],
            vision_client=FakeVisionClient(["invalid"]),
        )
