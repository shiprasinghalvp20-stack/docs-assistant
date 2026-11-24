"""ChromaDB wrapper: store chunk vectors with their text and metadata, on disk.

All ChromaDB-specific code lives here, so swapping to another vector DB
(pgvector, Qdrant, ...) means rewriting only this file.
"""
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)

BATCH_SIZE = 256  # embed + write this many chunks at a time (keeps memory flat for big PDFs)


class EmbeddingModelMismatchError(Exception):
    """The collection was built with a different embedding model than the one configured."""


class VectorStore:
    def __init__(
        self,
        persist_dir: str | Path,
        embeddings: Embeddings,
        embedding_model: str,
        collection_name: str = "docs",
    ):
        self.embeddings = embeddings
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),  # don't send usage data
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"embedding_model": embedding_model},
            configuration={"hnsw": {"space": "cosine"}},  # default is L2; cosine suits text
            embedding_function=None,  # we always pass vectors ourselves
        )
        # get_or_create silently keeps the OLD metadata if the collection exists,
        # so we must check ourselves that the model matches.
        stored_model = (self._collection.metadata or {}).get("embedding_model")
        if stored_model != embedding_model:
            raise EmbeddingModelMismatchError(
                f"Collection '{collection_name}' was built with '{stored_model}', but "
                f"EMBEDDING_MODEL is '{embedding_model}'. Vectors from different models "
                f"can't be mixed: delete {persist_dir} and re-index, or switch the model back."
            )

    # ---------- write ----------

    def add_chunks(self, chunks: list[Document], extra_metadata: dict | None = None) -> int:
        """Embed chunks and upsert them, using each chunk's `chunk_id` as its database id.

        Upsert = insert, or overwrite if the id already exists -> the same chunk is never
        stored twice. Returns the number of chunks written.
        """
        extra = extra_metadata or {}
        for start in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[start : start + BATCH_SIZE]
            texts = [c.page_content for c in batch]
            self._collection.upsert(
                ids=[c.metadata["chunk_id"] for c in batch],
                embeddings=self.embeddings.embed_documents(texts),
                documents=texts,
                metadatas=[{**c.metadata, **extra} for c in batch],
            )
        return len(chunks)

    def delete_source(self, source: str) -> int:
        """Remove every chunk that came from one file. Returns how many were removed."""
        ids = self._collection.get(where={"source": source}, include=[])["ids"]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    # ---------- read ----------

    def count(self) -> int:
        """Total number of stored chunks (= vectors)."""
        return self._collection.count()

    def count_source(self, source: str) -> int:
        return len(self._collection.get(where={"source": source}, include=[])["ids"])

    def get_index_info(self, source: str) -> dict | None:
        """How a file was indexed: {file_hash, chunk_size, chunk_overlap}, or None if absent."""
        result = self._collection.get(where={"source": source}, limit=1, include=["metadatas"])
        if not result["ids"]:
            return None
        meta = result["metadatas"][0]
        return {k: meta.get(k) for k in ("file_hash", "chunk_size", "chunk_overlap")}

    def list_sources(self) -> list[str]:
        """Names of all indexed files."""
        metadatas = self._collection.get(include=["metadatas"])["metadatas"]
        return sorted({m["source"] for m in metadatas})

    def get_chunks(self, source: str) -> list[Document]:
        """All stored chunks of one file, in page/chunk order (handy for inspection)."""
        result = self._collection.get(where={"source": source}, include=["documents", "metadatas"])
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(result["documents"], result["metadatas"])
        ]
        return sorted(docs, key=lambda d: (d.metadata["page"], d.metadata["chunk_index"]))
