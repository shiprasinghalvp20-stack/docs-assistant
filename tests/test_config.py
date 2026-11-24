from pathlib import Path

import pytest
from pydantic import ValidationError

from app.core.config import Settings, get_settings

ENV_VARS = [
    "LLM_PROVIDER", "GROQ_API_KEY", "GOOGLE_API_KEY", "UPLOAD_DIR", "CHROMA_DIR",
    "CHUNK_SIZE", "CHUNK_OVERLAP", "EMBEDDING_MODEL", "COLLECTION_NAME",
    "TOP_K", "MIN_SCORE",
]
FAKE_KEY = "test-fake-key-123"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Remove real settings from the environment so every test starts from zero."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def make_settings(**kwargs) -> Settings:
    """Build Settings WITHOUT reading the developer's real .env file."""
    return Settings(_env_file=None, **kwargs)


def test_defaults_when_nothing_is_set():
    s = make_settings()
    assert s.llm_provider == "groq"
    assert s.groq_api_key is None
    assert s.upload_dir == Path("data/uploads")
    assert s.chroma_dir == Path("data/chroma")
    assert s.chunk_size == 800
    assert s.chunk_overlap == 120
    assert s.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
    assert s.collection_name == "docs"
    assert s.top_k == 4
    assert s.min_score == 0.25


def test_environment_variables_override_defaults(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("CHROMA_DIR", "/tmp/my_chroma")
    s = make_settings()
    assert s.llm_provider == "gemini"
    assert s.chroma_dir == Path("/tmp/my_chroma")  # string converted to Path


def test_values_are_read_from_env_file(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text(f"LLM_PROVIDER=gemini\nGOOGLE_API_KEY={FAKE_KEY}\n")
    s = Settings(_env_file=env_file)
    assert s.llm_provider == "gemini"
    assert s.llm_api_key() == FAKE_KEY


def test_invalid_provider_fails_at_startup(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gorq")  # typo
    with pytest.raises(ValidationError):
        make_settings()


def test_api_key_is_hidden_when_printed(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    s = make_settings()
    assert FAKE_KEY not in str(s)
    assert FAKE_KEY not in repr(s)
    assert FAKE_KEY not in str(s.groq_api_key)


def test_llm_api_key_returns_key_for_selected_provider(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", FAKE_KEY)
    assert make_settings().llm_api_key() == FAKE_KEY


@pytest.mark.parametrize("value", ["", "   ", "your-groq-key-here"])
def test_missing_or_placeholder_key_gives_clear_error(monkeypatch, value):
    monkeypatch.setenv("GROQ_API_KEY", value)
    with pytest.raises(ValueError, match="GROQ_API_KEY is not set"):
        make_settings().llm_api_key()


def test_get_settings_returns_same_object():
    assert get_settings() is get_settings()


def test_chunk_settings_are_converted_to_int(monkeypatch):
    monkeypatch.setenv("CHUNK_SIZE", "500")
    monkeypatch.setenv("CHUNK_OVERLAP", "50")
    s = make_settings()
    assert (s.chunk_size, s.chunk_overlap) == (500, 50)


@pytest.mark.parametrize(
    "size, overlap",
    [("0", "0"), ("-5", "0"), ("100", "-1"), ("100", "100"), ("100", "200"), ("abc", "10")],
)
def test_invalid_chunk_settings_fail_at_startup(monkeypatch, size, overlap):
    monkeypatch.setenv("CHUNK_SIZE", size)
    monkeypatch.setenv("CHUNK_OVERLAP", overlap)
    with pytest.raises(ValidationError):
        make_settings()


@pytest.mark.parametrize("name, value", [
    ("TOP_K", "0"), ("TOP_K", "51"), ("TOP_K", "abc"),
    ("MIN_SCORE", "1.5"), ("MIN_SCORE", "-2"), ("MIN_SCORE", "high"),
])
def test_invalid_retrieval_settings_fail_at_startup(monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValidationError):
        make_settings()
