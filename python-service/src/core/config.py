"""Configuration settings for SMS AI Service."""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve .env against the service root so settings load regardless of the
# working directory uvicorn was started from.
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


class Settings(BaseSettings):
    """Application settings."""

    model_config = SettingsConfigDict(env_file=ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str

    # LLM Provider settings
    # "auto" will pick the first provider with a configured API key
    llm_provider: str = "auto"  # Options: "auto", "claude", "openai", "gemini"
    anthropic_api_key: str = ""
    openai_api_key: str = ""
    google_api_key: str = ""

    # Model configurations (big/small per provider)
    claude_model: str = "claude-sonnet-4-5-20250929"
    claude_small_model: str = "claude-haiku-4-5-20251001"
    openai_model: str = "gpt-5.2-2025-12-11"
    openai_small_model: str = "gpt-5-mini-2025-08-07"
    gemini_model: str = "gemini-3-flash-preview"
    gemini_small_model: str = "gemini-3-flash-preview"

    # Analysis settings
    max_tokens: int = 8000
    temperature: float = 0.1
    # Multi-LLM screening: when true, each provider evaluates all criteria of a source in
    # one call, so the document is sent once per provider instead of once per criterion.
    # Off by default until its decisions have been compared with per-criterion screening.
    screening_batch_criteria: bool = False

    # Open-access PDF retrieval
    # Unpaywall requires a contact email; OpenAlex uses it for its polite pool.
    oa_contact_email: str = ""
    # Optional; without it Semantic Scholar shares one throttled pool with
    # every anonymous caller and answers 429 for most batch lookups.
    semantic_scholar_api_key: str = ""
    pdf_retrieval_timeout_seconds: float = 30.0
    pdf_retrieval_max_mb: int = 50

    # Publisher APIs, all optional. Each resolver is skipped unless its key is
    # set, and only ever runs for records that belong to that publisher.
    # One Elsevier key (dev.elsevier.com) covers both the Scopus Search API
    # (title -> DOI lookup) and the ScienceDirect Article Retrieval API, which
    # serves the PDF for open-access articles.
    elsevier_api_key: str = Field(
        default="", validation_alias=AliasChoices("ELSEVIER_API_KEY", "SCOPUS_API_KEY")
    )
    # Institutional token issued by Elsevier to a library. Needed for paywalled
    # articles: the API ignores IP-based entitlement, so the campus network or
    # VPN does not help without it.
    elsevier_inst_token: str = ""
    # IEEE Xplore metadata API key from developer.ieee.org.
    ieee_api_key: str = ""

    # E2E test mode — uses FakeProvider instead of real LLM APIs
    e2e_test_mode: bool = False


settings = Settings()
