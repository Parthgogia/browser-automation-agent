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
        # Without this, pydantic-settings JSON-decodes every list-typed field
        # at the *source* level, before any validator runs -- so a perfectly
        # ordinary `ALLOWED_DOMAINS=` line dies with "Expecting value: line 1
        # column 1". Decoding off means the raw string reaches `_split_csv`,
        # which is what lets `.env` use comma-separated lists like a human
        # would write them.
        enable_decoding=False,
    )

    # ---------------------------------------------------------------- LLM ---
    llm_provider: Literal[
        "gemini", "mock", "free", "adaptive", "groq", "openrouter", "ollama"
    ] = "mock"
    gemini_api_key: str = ""
    gemini_api_key_2: str = ""
    gemini_api_key_3: str = ""
    #: Floating aliases rather than pinned versions. Google retires specific
    #: model IDs ("no longer available to new users") without warning, which
    #: breaks a setup that worked last month; the aliases keep tracking the
    #: current generation. Pin an exact ID here if you need reproducibility
    #: more than you need it to keep working.
    llm_model: str = "gemini-flash-latest"
    llm_fast_model: str = "gemini-flash-lite-latest"
    llm_temperature: float = 0.2
    embedding_model: str = "gemini-embedding-001"

    # `adaptive` (and legacy `free`) selects local or hosted providers based on
    # task complexity. Hosted providers without keys are skipped.
    groq_api_key: str = ""
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "openai/gpt-oss-120b"
    groq_fast_model: str = "openai/gpt-oss-20b"
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    # OpenRouter's free router chooses a free model that fits tool/image input.
    openrouter_model: str = "openrouter/free"
    openrouter_fast_model: str = "openrouter/free"
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_muse_model: str = "meta/muse-glimmer-30b"
    nvidia_glm_model: str = "z-ai/glm-5.3-flash"
    nvidia_nemotron_model: str = "nvidia/nemotron-3.5-lightning-30b-a3b"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3.5:9b"
    ollama_fast_model: str = "qwen3.5:4b"
    ollama_embedding_model: str = "nomic-embed-text:latest"
    llm_fallback_cooldown_s: int = 60

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
    #: Where a freshly launched browser lands, before the agent's first look.
    #: `about:blank` has no elements at all, which makes the first observation
    #: useless to the model and trips the vision fallback into spending a
    #: screenshot to show it nothing. Set to "about:blank" for the old
    #: behaviour; failure to load is never fatal.
    browser_start_url: str = "https://search.brave.com/"

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
        """Accept ``a.com,b.com`` -- or an empty string -- as a list.

        Paired with ``enable_decoding=False`` above. An empty or
        whitespace-only value yields an empty list, which is what an
        unset-but-present `.env` line means.
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
