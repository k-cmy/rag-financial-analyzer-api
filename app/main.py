from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import get_settings
from app.schemas import IngestResponse, QueryRequest, QueryResponse
from app.services.rag_service import RAGService

settings = get_settings()
rag_service = RAGService(settings=settings)

app = FastAPI(
    title="RAG-Powered Financial Document Analyzer API",
    version="1.0.0",
    description="Upload financial reports, index them into Chroma, then query with a Gemini-backed RAG pipeline.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    extension = file.filename.lower().rsplit(".", maxsplit=1)
    if len(extension) < 2 or extension[-1] not in {"pdf", "txt", "md"}:
        raise HTTPException(status_code=400, detail="Only PDF, TXT, and MD files are supported.")

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    result = rag_service.ingest_document(filename=file.filename, file_bytes=file_bytes)
    return IngestResponse(**result)


@app.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest) -> QueryResponse:
    result = rag_service.query(question=payload.question, k=payload.k)
    return QueryResponse(**result)
