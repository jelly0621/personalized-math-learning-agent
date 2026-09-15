"""Tests for the sanitized packaged Pilot Evaluation fallback."""

from math_learning_agent.web.services import WebServices


def test_packaged_pilot_is_used_when_runtime_outputs_are_absent(
    monkeypatch, tmp_path
) -> None:
    monkeypatch.setattr(
        "math_learning_agent.web.services.EVALUATION_ROOT",
        tmp_path / "missing-outputs",
    )

    result = WebServices().get_latest_evaluation()

    assert result["available"] is True
    assert result["dataset_name"] == "real_student_weak_v1"
    assert result["case_count"] == 10
    assert result["solver"]["symbolic_coverage"] == 5 / 9
    assert result["verifier_challenge"]["challenge_case_count"] == 1
    assert result["observability"] == {
        "total_llm_calls": 22,
        "failed_llm_calls": 0,
    }
    assert "before the later symbolic normalization patch" in result["limitation"]
