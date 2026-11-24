"""Application entry point.

Run with:  uvicorn app.main:app --reload
Then open: http://127.0.0.1:8000/docs
"""
import logging

from fastapi import FastAPI

from app.api.routes import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # hide HTTP request logs from client libraries

app = FastAPI(
    title="Docs Assistant",
    description="Upload PDFs and ask questions. Answers cite the file and page they came from.",
    version="0.1.0",
)
app.include_router(router)


@app.get("/health", tags=["system"])
def health() -> dict:
    """Liveness check - used by humans, tests, and later by Docker."""
    return {"status": "ok"}
