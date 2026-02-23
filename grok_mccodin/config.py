"""Configuration management — loads from .env and environment variables."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


def _safe_int(env_var: str, default: int) -> int:
    """Parse an env var as int, falling back to default on bad values."""
    raw = os.getenv(env_var)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer for %s=%r, using default %d", env_var, raw, default)
        return default


def _find_dotenv() -> Path | None:
    """Walk up from cwd looking for a .env file."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        candidate = parent / ".env"
        if candidate.is_file():
            return candidate
    return None


@dataclass(slots=True)
class Config:
    """Runtime configuration populated from environment variables."""

    # Grok / xAI
    grok_api_key: str = ""
    grok_base_url: str = "https://api.x.ai/v1"
    grok_model: str = "grok-3"

    # X / Twitter (OAuth 1.0a)
    x_api_key: str = ""
    x_api_secret: str = ""
    x_access_token: str = ""
    x_access_secret: str = ""

    # Giphy
    giphy_api_key: str = ""

    # Database
    db_path: str = "project.db"

    # Runtime
    safe_lock: bool = False
    log_file: str = "grok_mccodin_log.json"
    working_dir: str = field(default_factory=lambda: str(Path.cwd()))

    # Memory / context management
    memory_dir: str = "~/.grok_mccodin/sessions"
    token_budget: int = 6000
    keep_recent: int = 10
    memory_top_k: int = 3

    @classmethod
    def from_env(cls) -> "Config":
        """Build a Config from the process environment (loads .env first)."""
        dotenv_path = _find_dotenv()
        if dotenv_path:
            load_dotenv(dotenv_path)

        return cls(
            grok_api_key=os.getenv("GROK_API_KEY", ""),
            grok_base_url=os.getenv("GROK_BASE_URL", "https://api.x.ai/v1"),
            grok_model=os.getenv("GROK_MODEL", "grok-3"),
            x_api_key=os.getenv("X_API_KEY", ""),
            x_api_secret=os.getenv("X_API_SECRET", ""),
            x_access_token=os.getenv("X_ACCESS_TOKEN", ""),
            x_access_secret=os.getenv("X_ACCESS_SECRET", ""),
            giphy_api_key=os.getenv("GIPHY_API_KEY", ""),
            db_path=os.getenv("DB_PATH", "project.db"),
            memory_dir=os.getenv("GROK_MEMORY_DIR", "~/.grok_mccodin/sessions"),
            token_budget=_safe_int("GROK_TOKEN_BUDGET", 6000),
            keep_recent=_safe_int("GROK_KEEP_RECENT", 10),
            memory_top_k=_safe_int("GROK_MEMORY_TOP_K", 3),
        )

    @property
    def has_grok_key(self) -> bool:
        return bool(self.grok_api_key)

    @property
    def has_x_credentials(self) -> bool:
        return all([self.x_api_key, self.x_api_secret, self.x_access_token, self.x_access_secret])

    @property
    def has_giphy_key(self) -> bool:
        return bool(self.giphy_api_key)
