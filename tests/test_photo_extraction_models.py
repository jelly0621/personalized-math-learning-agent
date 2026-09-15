"""Validation tests for photo-only extraction models."""

import pytest
from pydantic import ValidationError

from math_learning_agent.models import ExtractedProblem, PhotoExtractionResult


def test_photo_extraction_result_accepts_multiple_problems_and_null_answer() -> None:
    result = PhotoExtractionResult(
        problems=[
            ExtractedProblem(
                problem_index=index,
                source_image_index=1,
                problem_text=f"题目 {index}",
                printed_answer=None,
                printed_solution=None,
                confidence=0.9,
            )
            for index in range(1, 4)
        ],
        warnings=[],
    )

    assert len(result.problems) == 3
    assert result.problems[0].printed_answer is None


def test_photo_extraction_models_enforce_indexes_text_and_confidence() -> None:
    with pytest.raises(ValidationError):
        ExtractedProblem(
            problem_index=0,
            source_image_index=1,
            problem_text="",
            confidence=1.2,
        )
