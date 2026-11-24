"""Request/response models: the API's contract. FastAPI uses them to validate input
(-> 422 on bad data) and to generate the interactive /docs page."""
from typing import Literal

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    filename: str = Field(description="Safe name the file was saved and indexed under")
    status: Literal["indexed", "replaced", "unchanged", "no_text"]
    pages_with_text: int
    chunks: int
    message: str

    model_config = {"json_schema_extra": {"examples": [{
        "filename": "company_handbook.pdf", "status": "indexed", "pages_with_text": 3,
        "chunks": 3, "message": "Indexed 3 chunks from 3 pages."}]}}


class DocumentInfo(BaseModel):
    filename: str
    chunks: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total_chunks: int


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, description="Your question in plain English")
    k: int | None = Field(default=None, ge=1, le=20, description="Max chunks to retrieve (default: TOP_K)")

    model_config = {
        "str_strip_whitespace": True,  # "   " becomes "" and fails min_length -> 422
        "json_schema_extra": {"examples": [{"question": "How many holidays do I get?"}]},
    }


class CitationOut(BaseModel):
    source: str
    page: int


class SourceOut(BaseModel):
    source: str
    page: int
    score: float = Field(description="Cosine similarity of the chunk to the question (-1..1)")
    text: str = Field(description="The chunk text the answer was based on")


class AskResponse(BaseModel):
    question: str
    answer: str
    found: bool = Field(description="False when the answer is not in the documents")
    citations: list[CitationOut]
    sources: list[SourceOut]
    model: str | None = Field(description="LLM that answered; null if the LLM wasn't called")


class ErrorResponse(BaseModel):
    detail: str
