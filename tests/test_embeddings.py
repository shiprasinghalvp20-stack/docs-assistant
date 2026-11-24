import langchain_huggingface

from app.core.config import get_settings
from app.services.embeddings import get_embeddings


def test_get_embeddings_uses_configured_model_without_loading_it(monkeypatch):
    """Replace the real class so no model is downloaded or loaded."""
    captured = {}

    class FakeHF:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", FakeHF)
    monkeypatch.setenv("EMBEDDING_MODEL", "my-org/my-model")
    get_settings.cache_clear()
    get_embeddings.cache_clear()
    try:
        assert isinstance(get_embeddings(), FakeHF)
        assert captured["model"] == "my-org/my-model"
        assert captured["encode_kwargs"] == {"normalize_embeddings": True}
        assert captured["model_kwargs"] == {"local_files_only": True}  # cache first, no network
    finally:
        get_settings.cache_clear()
        get_embeddings.cache_clear()


def test_get_embeddings_downloads_when_model_not_cached(monkeypatch):
    calls = []

    class FakeHF:
        def __init__(self, **kwargs):
            calls.append(kwargs)
            if kwargs.get("model_kwargs", {}).get("local_files_only"):
                raise OSError("not in cache")

    monkeypatch.setattr(langchain_huggingface, "HuggingFaceEmbeddings", FakeHF)
    get_settings.cache_clear()
    get_embeddings.cache_clear()
    try:
        assert isinstance(get_embeddings(), FakeHF)
        assert len(calls) == 2                      # cache attempt, then download
        assert "model_kwargs" not in calls[1]
    finally:
        get_settings.cache_clear()
        get_embeddings.cache_clear()
