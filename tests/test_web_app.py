"""Offline HTTP-contract tests for the Phase 8 Web MVP."""

from pathlib import Path

from fastapi.testclient import TestClient

from math_learning_agent.web.app import create_app
from math_learning_agent.web.services import WebServices


class FakeWebServices:
    def get_profile(self):
        return [
            {
                "knowledge_point": "基本不等式",
                "wrong_count": 3,
                "practice_count": 2,
                "correct_count": 1,
                "last_wrong_at": None,
                "last_practiced_at": None,
            }
        ]

    def create_wrong_problem(self, payload):
        return {
            "saved_problem_id": 1,
            "knowledge_points": ["函数"],
            "question_type": "解答题",
            "difficulty": 2,
            "problem_summary": "fixture",
            "error_type": "unknown",
            "error_reason": "未提供作答过程",
            "related_knowledge_points": ["函数"],
            "error_confidence": 0.4,
        }

    def create_training_plan(self, payload):
        return {
            "total_questions": payload.total_questions,
            "focus_items": [
                {
                    "knowledge_point": "函数",
                    "question_count": payload.total_questions,
                    "target_difficulty": 2,
                    "focus_error_types": [],
                    "reason": "fixture",
                }
            ],
            "plan_reason": "fixture",
        }

    def generate_question(self, payload):
        return {
            "question_id": 7,
            "question_text": "求函数的定义域。",
            "knowledge_points": ["函数"],
            "difficulty": 2,
            "generation_attempts": 1,
            "generation_status": "accepted",
            "reference_answer": "secret answer",
            "reference_solution": "secret solution",
        }

    def submit_training_feedback(self, payload):
        return {
            "is_correct": True,
            "error_reason": "答案正确",
            "feedback": "继续保持",
            "related_knowledge_points": ["函数"],
            "confidence": 0.95,
            "sympy_status": "equivalent",
            "sympy_reason": "equivalent fixture",
            "training_attempt_id": 9,
            "reference_answer": "x > 0",
            "reference_solution": "由定义域条件可得。",
        }

    def get_wrong_problem_history(self, limit):
        return []

    def get_generated_question_history(self, limit):
        return []

    def get_training_attempt_history(self, limit):
        return []

    def get_latest_evaluation(self):
        return {
            "available": False,
            "evaluation_label": "Pilot Evaluation",
            "limitation": "small pilot",
        }


def _client() -> TestClient:
    return TestClient(create_app(FakeWebServices()))


def test_health_and_root_are_available() -> None:
    client = _client()
    assert client.get("/api/health").json() == {
        "status": "ok",
        "app": "MathLearningAgent",
    }
    root = client.get("/")
    assert root.status_code == 200
    assert "个性化数学学习 Agent" in root.text


def test_profile_uses_injected_service_without_database() -> None:
    response = _client().get("/api/profile")
    assert response.status_code == 200
    assert response.json()[0]["knowledge_point"] == "基本不等式"


def test_wrong_problem_and_training_plan_schema_validation() -> None:
    client = _client()
    assert client.post("/api/wrong-problems", json={}).status_code == 422
    assert client.post(
        "/api/wrong-problems", json={"problem_text": ""}
    ).status_code == 422
    assert client.post(
        "/api/training-plan", json={"total_questions": 0}
    ).status_code == 422


def test_generated_question_never_exposes_reference_material() -> None:
    response = _client().post(
        "/api/questions/generate",
        json={
            "knowledge_point": "函数",
            "target_difficulty": 2,
            "focus_error_types": [],
        },
    )
    body = response.json()
    assert response.status_code == 200
    assert "reference_answer" not in body
    assert "reference_solution" not in body


def test_feedback_can_expose_reference_material_after_submission() -> None:
    response = _client().post(
        "/api/training-feedback",
        json={"question_id": 7, "student_answer": "x > 0"},
    )
    assert response.status_code == 200
    assert response.json()["reference_answer"] == "x > 0"


def test_evaluation_without_run_is_safe(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        "math_learning_agent.web.services.EVALUATION_ROOT", tmp_path / "missing"
    )
    monkeypatch.setattr(
        "math_learning_agent.web.services.PACKAGED_EVALUATION_SUMMARY",
        tmp_path / "missing-summary.json",
    )
    response = TestClient(create_app(WebServices())).get("/api/evaluation/latest")
    assert response.status_code == 200
    assert response.json()["available"] is False


def test_service_exception_does_not_leak_traceback_or_key() -> None:
    class FailingServices(FakeWebServices):
        def create_wrong_problem(self, payload):
            raise RuntimeError("Missing required LLM configuration: sk-secret-value")

    response = TestClient(
        create_app(FailingServices()), raise_server_exceptions=False
    ).post("/api/wrong-problems", json={"problem_text": "test"})
    assert response.status_code == 503
    assert response.json() == {"detail": "LLM configuration is unavailable."}
    assert "secret" not in response.text
    assert "Traceback" not in response.text


def test_static_css_and_javascript_load() -> None:
    client = _client()
    css = client.get("/static/styles.css")
    javascript = client.get("/static/app.js")
    assert css.status_code == 200
    assert "--primary" in css.text
    assert javascript.status_code == 200
    assert "/api/questions/generate" in javascript.text
