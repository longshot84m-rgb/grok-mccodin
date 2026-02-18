"""Shared test fixtures."""

from __future__ import annotations

import pytest

from grok_mccodin.config import Config


@pytest.fixture()
def config() -> Config:
    """A Config with dummy credentials for testing."""
    return Config(
        grok_api_key="test-key-123",
        grok_model="grok-3",
        giphy_api_key="giphy-test-key",
        x_api_key="x-key",
        x_api_secret="x-secret",
        x_access_token="x-token",
        x_access_secret="x-access-secret",
    )


@pytest.fixture()
def tmp_project(tmp_path):
    """Create a minimal project structure in a temp dir."""
    (tmp_path / "main.py").write_text('print("hello")\n')
    (tmp_path / "utils.py").write_text("def add(a, b):\n    return a + b\n")
    sub = tmp_path / "src"
    sub.mkdir()
    (sub / "app.py").write_text("# app\n")
    return tmp_path
