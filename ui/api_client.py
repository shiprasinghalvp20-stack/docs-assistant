"""HTTP client for the Docs Assistant API, used by the Streamlit UI.

No Streamlit code here: it's plain Python, so it can be tested with fake HTTP responses.
Every failure becomes an ApiError whose `.message` is friendly enough to show to users.
"""
import os
from dataclasses import dataclass, field

import httpx

DEFAULT_API_URL = "http://localhost:8000"


def get_api_url() -> str:
    """API base URL from the API_URL environment variable (default: local API)."""
    return (os.environ.get("API_URL") or DEFAULT_API_URL).rstrip("/")


# ---------- results ----------

@dataclass
class UploadResult:
    filename: str
    status: str             # indexed | replaced | unchanged | no_text
    pages_with_text: int
    chunks: int
    message: str


@dataclass
class Source:
    source: str
    page: int
    score: float
    text: str


@dataclass
class AskResult:
    question: str
    answer: str
    found: bool
    citations: list[tuple[str, int]] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    model: str | None = None


# ---------- errors (message = what the user sees) ----------

class ApiError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class ApiUnreachable(ApiError):
    """The API isn't running or can't be reached."""


class RateLimited(ApiError):
    """HTTP 429: the LLM's rate limit was hit."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class LLMUnavailable(ApiError):
    """HTTP 503, or the request timed out: the LLM is overloaded or slow."""


class InvalidRequest(ApiError):
    """HTTP 400 / 413 / 422: the input was rejected (bad file, too large, empty question...)."""


class ServerError(ApiError):
    """HTTP 502 / other 5xx: a problem on the server side."""


# ---------- helpers ----------

def citation_label(source: str, page: int) -> str:
    return f"[{source}, p.{page}]"


def escape_markdown(text: str) -> str:
    """Stop Streamlit from turning '$' amounts into LaTeX maths (e.g. '$5 and $10')."""
    return text.replace("$", "\\$")


def _detail(response: httpx.Response) -> str:
    """Pull FastAPI's error message out of a response ('detail' may be a string or a list)."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return response.text[:200] or f"HTTP {response.status_code}"
    if isinstance(detail, list) and detail:            # 422 validation errors
        first = detail[0]
        return str(first.get("msg", first)) if isinstance(first, dict) else str(first)
    return str(detail) if detail else f"HTTP {response.status_code}"


# ---------- client ----------

class DocsApiClient:
    def __init__(self, base_url: str | None = None, timeout: float = 120.0,
                 transport: httpx.BaseTransport | None = None):
        # `transport` lets tests plug in fake responses (httpx.MockTransport) - no network.
        self.base_url = (base_url or get_api_url()).rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout, transport=transport)

    def health(self) -> bool:
        try:
            return self._client.get("/health", timeout=3.0).status_code == 200
        except httpx.HTTPError:
            return False

    def list_documents(self) -> list[tuple[str, int]]:
        data = self._request("GET", "/documents")
        return [(d["filename"], d["chunks"]) for d in data["documents"]]

    def upload(self, filename: str, content: bytes) -> UploadResult:
        data = self._request("POST", "/documents", files={"file": (filename, content, "application/pdf")})
        return UploadResult(data["filename"], data["status"], data["pages_with_text"],
                            data["chunks"], data["message"])

    def ask(self, question: str) -> AskResult:
        data = self._request("POST", "/ask", json={"question": question})
        return AskResult(
            question=data["question"],
            answer=data["answer"],
            found=data["found"],
            citations=[(c["source"], c["page"]) for c in data["citations"]],
            sources=[Source(s["source"], s["page"], s["score"], s["text"]) for s in data["sources"]],
            model=data.get("model"),
        )

    def _request(self, method: str, path: str, **kwargs) -> dict:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise LLMUnavailable("The request took too long. The AI model may be busy - please try again.") from exc
        except httpx.TransportError as exc:  # connection refused, DNS failure, ...
            raise ApiUnreachable(
                f"Can't reach the Docs Assistant API at {self.base_url}. Is it running?"
            ) from exc

        code = response.status_code
        if code < 400:
            return response.json()
        if code == 429:
            retry = response.headers.get("retry-after")
            retry_after = int(retry) if retry and retry.isdigit() else None
            wait = f"{retry_after} seconds" if retry_after else "a minute"
            raise RateLimited(f"The AI model's free-tier limit was reached. Please wait {wait} and try again.",
                              retry_after)
        if code == 503:
            raise LLMUnavailable("The AI model is temporarily overloaded. Please try again in a moment.")
        if code in (400, 413, 422):
            raise InvalidRequest(_detail(response))
        if code == 502:
            raise ServerError(f"The server couldn't use the AI model: {_detail(response)}")
        raise ServerError(f"Unexpected server error (HTTP {code}). Please try again.")
