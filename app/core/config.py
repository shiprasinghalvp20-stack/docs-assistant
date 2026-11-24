"""Application settings, loaded from environment variables and the .env file.

Priority (highest wins):  real environment variables  >  .env file  >  defaults below.

Usage:
    from app.core.config import get_settings
    settings = get_settings()
    settings.chroma_dir          # Path("data/chroma")
    settings.llm_api_key()       # real key string - pass it to the LLM client, never print it
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,  # "GROQ_API_KEY=" (blank) counts as "not set"
        extra="ignore",         # unknown variables in .env don't crash the app
    )

    # --- LLM ---
    # Literal = only these exact values are allowed; anything else fails at startup.
    llm_provider: Literal["groq", "gemini"] = "groq"
    groq_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None

    # --- Storage ---
    upload_dir: Path = Path("data/uploads")
    chroma_dir: Path = Path("data/chroma")

    # --- Embeddings + vector store ---
    # Changing the model requires re-indexing: vectors from different models can't be mixed.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    collection_name: str = "docs"

    # --- Retrieval ---
    top_k: int = Field(default=4, gt=0, le=50)  # max chunks returned per question
    # Cosine similarity cut-off (-1..1). Calibrated for all-MiniLM-L6-v2 on the sample
    # handbook: on-topic best scores 0.31-0.51, off-topic <= 0.17. Re-measure if the model changes.
    min_score: float = Field(default=0.25, ge=-1.0, le=1.0)

    # --- Chunking (measured in characters) ---
    # Field(gt=0) / ge=0 = pydantic rejects zero or negative values at startup.
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=120, ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be smaller than "
                f"CHUNK_SIZE ({self.chunk_size})."
            )
        return self

    def llm_api_key(self) -> str:
        """Return the API key for the selected provider, or raise a clear error.

        Raises instead of returning None so a missing key is caught the moment
        the LLM is needed, with a message that says exactly what to fix.
        The error message never contains the key itself.
        """
        env_name, secret = {
            "groq": ("GROQ_API_KEY", self.groq_api_key),
            "gemini": ("GOOGLE_API_KEY", self.google_api_key),
        }[self.llm_provider]

        value = secret.get_secret_value().strip() if secret else ""
        if not value or value.startswith("your-"):
            raise ValueError(
                f"LLM_PROVIDER is '{self.llm_provider}' but {env_name} is not set. "
                f"Add it to your .env file (see .env.example)."
            )
        return value


@lru_cache
def get_settings() -> Settings:
    """Create Settings once and reuse it (reading .env on every call would be wasteful).

    In tests, call get_settings.cache_clear() if you change environment variables.
    """
    return Settings()
