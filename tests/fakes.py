"""Fake embedding model and fake LLM shared by several test files (no downloads, no API)."""
import math
import re
import zlib

from langchain_core.messages import AIMessage


class BagOfWordsEmbeddings:
    """Fake model where texts sharing words get similar vectors.

    Real models understand meaning; this one only counts words - but that's enough
    to test ranking, k and the score threshold deterministically.
    """

    VOCAB = ["leave", "days", "annual", "sick", "remote", "home", "work", "vpn",
             "expense", "meal", "allowance", "travel", "receipts", "cricket", "cup"]
    EXTRA_SLOTS = 64  # separate dimensions for texts containing no VOCAB word

    def _vec(self, text: str) -> list[float]:
        words = re.findall(r"[a-z]+", text.lower())
        v = [float(words.count(w)) for w in self.VOCAB] + [0.0] * self.EXTRA_SLOTS
        norm = math.sqrt(sum(x * x for x in v))
        if norm:
            return [x / norm for x in v]
        # No known words: put the text in its own extra slot (stable hash of the text), so it is
        # orthogonal to every text that has known words (score 0) instead of colliding with one.
        v[len(self.VOCAB) + zlib.crc32(text.encode()) % self.EXTRA_SLOTS] = 1.0
        return v

    def embed_documents(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeLLM:
    """Scripted stand-in for Gemini: returns `reply` (or raises `error`) and records the prompt."""

    def __init__(self, reply=None, error: Exception | None = None):
        self.reply, self.error, self.calls = reply, error, []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return AIMessage(content=self.reply)


class FakeAPIError(Exception):
    """Mimics google.genai errors, which carry an HTTP status in `.code`."""

    def __init__(self, code: int, message: str):
        super().__init__(f"{code} {message}")
        self.code = code
