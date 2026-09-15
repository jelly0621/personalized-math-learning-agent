"""Offline tests for deployment-time host and SQLite configuration."""

from pathlib import Path

import pytest

from math_learning_agent.config import PROJECT_ROOT
from math_learning_agent.db.database import (
    get_default_database_path,
    initialize_database,
)
from scripts.run_web_app import get_server_address


def test_database_path_keeps_local_default(monkeypatch) -> None:
    monkeypatch.delenv("DATABASE_PATH", raising=False)

    assert get_default_database_path() == PROJECT_ROOT / "data" / "math_agent.db"


def test_database_path_uses_absolute_override_and_creates_parent(
    monkeypatch, tmp_path: Path
) -> None:
    database_path = tmp_path / "nested" / "railway" / "math_agent.db"
    monkeypatch.setenv("DATABASE_PATH", str(database_path))

    assert get_default_database_path() == database_path
    initialize_database(get_default_database_path())
    assert database_path.is_file()


def test_database_path_rejects_relative_override(monkeypatch) -> None:
    monkeypatch.setenv("DATABASE_PATH", "relative/math_agent.db")

    with pytest.raises(RuntimeError, match="absolute path"):
        get_default_database_path()


def test_server_address_keeps_local_defaults(monkeypatch) -> None:
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.delenv("PORT", raising=False)

    assert get_server_address() == ("127.0.0.1", 8000)


def test_server_address_uses_railway_port_and_public_host(monkeypatch) -> None:
    monkeypatch.delenv("HOST", raising=False)
    monkeypatch.setenv("PORT", "9123")

    assert get_server_address() == ("0.0.0.0", 9123)


def test_server_address_honors_explicit_host(monkeypatch) -> None:
    monkeypatch.setenv("HOST", "127.0.0.2")
    monkeypatch.setenv("PORT", "8123")

    assert get_server_address() == ("127.0.0.2", 8123)
