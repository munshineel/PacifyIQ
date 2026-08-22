"""Application configuration.

Typed settings loaded from environment variables and .env, with sensible
defaults. Import `settings` anywhere; never read os.environ directly.

    from src.config.settings import settings
    print(settings.db_path)
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = two levels up from this file (src/config/settings.py)
ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """All runtime configuration in one typed object."""

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_prefix="PACIFYIQ_",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- paths -------------------------------------------------------
    root: Path = ROOT
    data_dir: Path = ROOT / "data"

    # --- LLM (used from Phase 1 onward) ------------------------------
    groq_api_key: str = Field(default="", description="Set PACIFYIQ_GROQ_API_KEY in .env")
    llm_model: str = "llama-3.3-70b-versatile"
    vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct"
    embedding_model: str = "nomic-embed-text-v1_5"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 3

    # --- retrieval (Phase 5-7) ---------------------------------------
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    retrieval_strategy: str = "hybrid"  # dense | bm25 | hybrid | hybrid_rerank

    # --- confidence / escalation (Phase 12) --------------------------
    min_retrieval_similarity: float = 0.35
    escalation_confidence_threshold: float = 0.60

    # --- reference date ----------------------------------------------
    # Fixed so eligibility calculations are reproducible across runs.
    # Set to None in production to use the real current date.
    reference_date: str = "2026-08-21"

    # --- derived paths -----------------------------------------------
    @property
    def db_path(self) -> Path:
        return self.data_dir / "db" / "pacify.db"

    @property
    def documents_dir(self) -> Path:
        return self.data_dir / "documents"

    @property
    def eval_dir(self) -> Path:
        return self.data_dir / "eval"

    @property
    def intents_dir(self) -> Path:
        return self.data_dir / "intents"

    @property
    def tickets_csv(self) -> Path:
        return self.data_dir / "tickets" / "ticket_history.csv"

    @property
    def sql_dir(self) -> Path:
        return self.root / "sql"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"


settings = Settings()


if __name__ == "__main__":
    print(f"root          {settings.root}")
    print(f"db_path       {settings.db_path}   exists={settings.db_path.exists()}")
    print(f"documents     {settings.documents_dir}  exists={settings.documents_dir.exists()}")
    print(f"llm_model     {settings.llm_model}")
    print(f"api key set   {bool(settings.groq_api_key)}")
