"""Application configuration.

Every knob the system has lives here, sourced from environment variables and
`.env`. Nothing else in the codebase reads `os.environ` directly -- that rule
keeps configuration discoverable and makes tests able to override settings by
constructing a `Settings` object rather than mutating global state.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root, i.e. the directory containing `backend/` and `frontend/`.
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Typed view over the environment.

    Field names map to upper-case environment variables of the same name
    (``llm_provider`` <- ``LLM_PROVIDER``).
    """

    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env", REPO_ROOT / "backend" / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---------------------------------------------------------------- LLM ---
    llm_provider: Literal["gemini", "mock"] = "mock"
    gemini_api_key: str = ""
    llm_model: str = "gemini-2.5-flash"
    llm_fast_model: str = "gemini-2.5-flash-lite"
    llm_temperature: float = 0.2
    embedding_model: str = "gemini-embedding-001"

    # ----------------------------------------------------------- database ---
    database_url: str = "postgresql+asyncpg://agent:agent@localhost:5433/browser_agent"

    # ------------------------------------------------------------ browser ---
    browser_headless: bool = False
    browser_channel: str = "chromium"
    browser_viewport_width: int = 1440
    browser_viewport_height: int = 900
    browser_profile_dir: Path = REPO_ROOT / "data" / "profiles"
    browser_default_profile: str = "default"
    browser_action_timeout_ms: int = 15_000

    # -------------------------------------------------------------- agent ---
    agent_max_steps: int = 40
    #: Total failed actions tolerated in one run before giving up.
    agent_max_failures: int = 8
    #: How many times the agent may stop and rethink before giving up.
    agent_max_reflections: int = 3
    agent_step_timeout_s: int = 120

    # ------------------------------------------------------------- vision ---
    vision_mode: Literal["auto", "always", "never"] = "auto"

    # ------------------------------------------------------------- search ---
    tavily_api_key: str = ""

    # ------------------------------------------------------------- safety ---
    allowed_domains: list[str] = Field(default_factory=list)
    blocked_domains: list[str] = Field(default_factory=list)
    require_approval: bool = True

    # ------------------------------------------------------------- server ---
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    log_level: str = "INFO"

    # ------------------------------------------------------------ derived ---
    data_dir: Path = REPO_ROOT / "data"

    @field_validator("allowed_domains", "blocked_domains", "cors_origins", mode="before")
    @classmethod
    def _split_csv(cls, value: object) -> object:
        """Accept ``a.com,b.com`` from the environment as a list.

        pydantic-settings would otherwise try to JSON-decode list-typed fields,
        which makes for an unfriendly `.env` file.
        """
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @property
    def screenshot_dir(self) -> Path:
        """Where per-task screenshots are written and served from."""
        return self.data_dir / "screenshots"

    @property
    def download_dir(self) -> Path:
        """Where the browser drops downloaded files."""
        return self.data_dir / "downloads"

    def ensure_directories(self) -> None:
        """Create every directory the app writes to. Called once at startup."""
        for directory in (
            self.data_dir,
            self.screenshot_dir,
            self.download_dir,
            self.browser_profile_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Process-wide settings singleton.

    Cached so that importing modules can call this freely; tests that need a
    different configuration should call ``get_settings.cache_clear()``.
    """
    return Settings()
