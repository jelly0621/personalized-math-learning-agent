"""HTTP Basic Authentication tests without external services."""

import pytest
from fastapi.testclient import TestClient

from math_learning_agent.web.app import create_app


class EmptyServices:
    def get_profile(self):
        return []


def _clear_credentials(monkeypatch) -> None:
    monkeypatch.delenv("APP_ACCESS_USERNAME", raising=False)
    monkeypatch.delenv("APP_ACCESS_PASSWORD", raising=False)


def test_unset_credentials_keep_local_access_open(monkeypatch) -> None:
    _clear_credentials(monkeypatch)

    assert TestClient(create_app(EmptyServices())).get("/").status_code == 200


def test_configured_credentials_protect_root_and_api(monkeypatch) -> None:
    monkeypatch.setenv("APP_ACCESS_USERNAME", "demo-user")
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "strong-test-password")
    client = TestClient(create_app(EmptyServices()))

    root = client.get("/")
    api = client.get("/api/profile")

    assert root.status_code == 401
    assert root.headers["www-authenticate"].startswith("Basic")
    assert api.status_code == 401
    assert "strong-test-password" not in root.text
    assert "strong-test-password" not in api.text


def test_correct_credentials_allow_root_and_api(monkeypatch) -> None:
    monkeypatch.setenv("APP_ACCESS_USERNAME", "demo-user")
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "strong-test-password")
    client = TestClient(create_app(EmptyServices()))

    assert client.get(
        "/", auth=("demo-user", "strong-test-password")
    ).status_code == 200
    assert client.get(
        "/api/profile", auth=("demo-user", "strong-test-password")
    ).status_code == 200


def test_health_never_requires_authentication(monkeypatch) -> None:
    monkeypatch.setenv("APP_ACCESS_USERNAME", "demo-user")
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "strong-test-password")

    response = TestClient(create_app(EmptyServices())).get("/api/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_partial_credentials_fail_configuration_without_leaking_password(
    monkeypatch,
) -> None:
    monkeypatch.delenv("APP_ACCESS_USERNAME", raising=False)
    monkeypatch.setenv("APP_ACCESS_PASSWORD", "strong-test-password")

    with pytest.raises(RuntimeError) as exc_info:
        create_app(EmptyServices())
    assert "must either both be set" in str(exc_info.value)
    assert "strong-test-password" not in str(exc_info.value)
