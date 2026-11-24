"""LLM client + error handling.

The rest of the app only needs an object with `.invoke(messages)` returning a message
with `.text`, so tests can pass a fake and the provider can be swapped here.
"""
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """A problem calling the LLM. The message is safe to show to users (never has the key)."""


class LLMUnavailableError(LLMError):
    """Temporary problem (overloaded, rate-limited, timeout): trying again later may work."""


class LLMRateLimitError(LLMUnavailableError):
    """The provider's quota/rate limit was hit (HTTP 429)."""


@lru_cache
def get_chat_model(model: str):
    """Create a chat model client for the configured provider (created once per model)."""
    from app.core.config import get_settings

    settings = get_settings()
    if settings.llm_provider != "gemini":
        raise LLMError(f"LLM_PROVIDER '{settings.llm_provider}' is not supported yet; use 'gemini'.")

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        api_key=settings.llm_api_key(),  # read via settings only; never logged
        timeout=settings.llm_timeout,
        max_retries=settings.llm_max_retries,
        max_output_tokens=1024,
        # temperature left at the model default: Google recommends this for Gemini 3 models.
    )


def to_llm_error(exc: Exception, model: str) -> LLMError:
    """Turn any provider exception into a clear, user-safe LLMError.

    We deliberately don't copy the raw exception text into the message shown to users.
    """
    code = getattr(exc, "code", None)
    text = str(exc).upper()
    name = type(exc).__name__.lower()

    if code == 429 or "RESOURCE_EXHAUSTED" in text or "429" in text:
        return LLMRateLimitError(
            f"Rate limit reached for {model} (free tier). Please wait a minute and try again."
        )
    if code in (500, 502, 503, 504) or "UNAVAILABLE" in text or "timeout" in name or "timed out" in text.lower():
        return LLMUnavailableError(f"{model} is temporarily unavailable or slow. Please try again shortly.")
    if code in (400, 401, 403) and ("API KEY" in text or "API_KEY" in text or "PERMISSION" in text):
        return LLMError("The LLM provider rejected the API key. Check GOOGLE_API_KEY in your .env file.")
    if code == 404:
        return LLMError(f"Model '{model}' was not found. Check LLM_MODEL in your .env file.")
    return LLMError(f"The LLM request failed ({type(exc).__name__}). Please try again.")
