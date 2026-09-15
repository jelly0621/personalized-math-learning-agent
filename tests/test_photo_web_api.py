"""Offline upload and confirmed-import API tests."""

from fastapi.testclient import TestClient

import math_learning_agent.web.services as services_module
from math_learning_agent.web.app import create_app
from math_learning_agent.web.services import WebServices


PNG_BYTES = b"\x89PNG\r\n\x1a\nfixture"


class FakeVisionClient:
    def invoke_image(self, image_bytes, mime_type, prompt) -> str:
        return """
        {
          "problems": [
            {"problem_index": 1, "problem_text": "题目一", "printed_answer": null, "printed_solution": null, "confidence": 0.9},
            {"problem_index": 2, "problem_text": "题目二", "printed_answer": "2", "printed_solution": null, "confidence": 0.8},
            {"problem_index": 3, "problem_text": "题目三", "printed_answer": null, "printed_solution": "解析", "confidence": 0.7}
          ],
          "warnings": []
        }
        """


def _extract_client() -> TestClient:
    return TestClient(create_app(WebServices(vision_client=FakeVisionClient())))


def test_photo_extract_rejects_non_image_and_mime_spoof() -> None:
    client = _extract_client()
    non_image = client.post(
        "/api/photo/extract",
        files={"images": ("notes.txt", b"hello", "text/plain")},
    )
    spoofed = client.post(
        "/api/photo/extract",
        files={"images": ("fake.png", b"not a png", "image/png")},
    )

    assert non_image.status_code == 415
    assert spoofed.status_code == 415


def test_photo_extract_rejects_more_than_five_images() -> None:
    files = [
        ("images", (f"image-{index}.png", PNG_BYTES, "image/png"))
        for index in range(6)
    ]

    response = _extract_client().post("/api/photo/extract", files=files)

    assert response.status_code == 400


def test_photo_extract_rejects_an_image_larger_than_ten_mb() -> None:
    oversized = b"\x89PNG\r\n\x1a\n" + b"0" * (10 * 1024 * 1024)

    response = _extract_client().post(
        "/api/photo/extract",
        files={"images": ("large.png", oversized, "image/png")},
    )

    assert response.status_code == 413


def test_photo_extract_returns_multiple_problems_without_using_wrong_graph(
    monkeypatch,
) -> None:
    class ForbiddenGraph:
        def invoke(self, state):
            raise AssertionError("Extraction must not write a wrong problem.")

    monkeypatch.setattr(services_module, "wrong_problem_graph", ForbiddenGraph())

    response = _extract_client().post(
        "/api/photo/extract",
        files={"images": ("page.png", PNG_BYTES, "image/png")},
    )

    assert response.status_code == 200
    assert len(response.json()["problems"]) == 3


class BatchImportServices(WebServices):
    def __init__(self) -> None:
        super().__init__()
        self.received = []

    def create_wrong_problem(self, payload):
        self.received.append(payload)
        if "失败" in payload.problem_text:
            raise RuntimeError("fixture failure")
        return {
            "saved_problem_id": len(self.received),
            "knowledge_points": ["基本不等式"],
            "question_type": "解答题",
            "difficulty": 2,
            "problem_summary": "fixture",
            "error_type": "unknown",
            "error_reason": "缺少学生作答证据",
            "related_knowledge_points": ["基本不等式"],
            "error_confidence": 0.2,
        }


def test_photo_import_processes_three_and_continues_after_one_failure() -> None:
    services = BatchImportServices()
    client = TestClient(create_app(services))

    response = client.post(
        "/api/photo/import",
        json={
            "problems": [
                {"problem_text": "题目一", "correct_answer": "1"},
                {"problem_text": "这题失败", "correct_answer": None},
                {"problem_text": "题目三", "correct_answer": "3"},
            ]
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["total"] == 3
    assert body["succeeded"] == 2
    assert body["failed"] == 1
    assert [item["status"] for item in body["results"]] == [
        "succeeded",
        "failed",
        "succeeded",
    ]
    assert body["results"][0]["error_type"] == "unknown"
    assert all(item.student_answer is None for item in services.received)
    assert all(item.student_solution is None for item in services.received)


def test_photo_import_schema_rejects_fabricated_student_work() -> None:
    response = TestClient(create_app(BatchImportServices())).post(
        "/api/photo/import",
        json={
            "problems": [
                {
                    "problem_text": "题目",
                    "student_answer": "fabricated",
                    "student_solution": "fabricated",
                }
            ]
        },
    )

    assert response.status_code == 422
