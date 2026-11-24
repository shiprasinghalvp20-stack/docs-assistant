"""Embedding model: turns text into vectors.

Uses a free local sentence-transformers model: downloaded once from Hugging Face,
then loaded from the local cache with no network calls (chunks never leave the machine).
Everything else in the app depends only on LangChain's `Embeddings` interface
(embed_documents / embed_query), so the model can be swapped in this one file.
"""
import logging
from functools import lru_cache

from langchain_core.embeddings import Embeddings

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def get_embeddings() -> Embeddings:
    """Load the embedding model once and reuse it (loading takes a few seconds)."""
    # Imported here so the app (and tests) don't load PyTorch until embeddings are needed.
    from langchain_huggingface import HuggingFaceEmbeddings

    model = get_settings().embedding_model
    # Unit-length vectors: cosine similarity then equals the dot product.
    encode_kwargs = {"normalize_embeddings": True}
    try:
        # Cache first: no network request when the model is already downloaded.
        return HuggingFaceEmbeddings(
            model=model, model_kwargs={"local_files_only": True}, encode_kwargs=encode_kwargs
        )
    except OSError:
        logger.info("Embedding model '%s' not in local cache - downloading once.", model)
        return HuggingFaceEmbeddings(model=model, encode_kwargs=encode_kwargs)
