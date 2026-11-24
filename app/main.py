"""Application entry point.

Run with:  uvicorn app.main:app --reload
"""
from fastapi import FastAPI

app = FastAPI(
    title="Docs Assistant",
    description="Upload PDFs and ask questions with file + page citations.",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict:
    """Liveness check - used by humans, tests, and later by Docker."""
    return {"status": "ok"}
