"""Offline tests for the ErrorDiagnosis model and JSON parser."""

import pytest
from pydantic import ValidationError

from math_learning_agent.agents.error_diagnosis import parse_error_diagnosis
from math_learning_agent.models import ErrorDiagnosis


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_error_diagnosis_rejects_invalid_confidence(confidence: float) -> None:
    with pytest.raises(ValidationError):
        ErrorDiagnosis(
            error_type="calculation_error",
            error_reason="代入时计算错误。",
            related_knowledge_points=["导数"],
            confidence=confidence,
        )


def test_parse_error_diagnosis_accepts_valid_json() -> None:
    diagnosis = parse_error_diagnosis(
        """
        {
          "error_type": "formula_error",
          "error_reason": "遗漏了常数项的导数。",
          "related_knowledge_points": ["导数公式"],
          "confidence": 0.9
        }
        """
    )

    assert diagnosis.error_type == "formula_error"
    assert diagnosis.confidence == 0.9


def test_parse_error_diagnosis_rejects_invalid_json() -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        parse_error_diagnosis("not json")
