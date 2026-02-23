from typing import Any

from pydantic import BaseModel, Field


class IngestResponse(BaseModel):
    filename: str
    chunks_added: int
    collection_name: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, description="User question about uploaded documents.")
    k: int = Field(4, ge=1, le=20, description="Number of retrieved chunks to use.")


class SourceSnippet(BaseModel):
    source: str
    page: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceSnippet]
