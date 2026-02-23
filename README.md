# RAG-Powered Financial Document Analyzer API

A backend API that ingests financial documents (PDF/TXT/MD), chunks and embeds them into ChromaDB, and answers questions using a Gemini LLM through a Retrieval-Augmented Generation (RAG) pipeline.

## Tech Stack

- Python
- FastAPI (inference API)
- LangChain (orchestration framework)
- ChromaDB (vector database)
- Gemini API (`gemini-3-flash-preview` by default)
- Docker / Docker Compose

## Project Structure

```text
rag-financial-analyzer-api/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── schemas.py
│   └── services/
│       └── rag_service.py
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Quick Start (Local)

1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure environment variables:

```bash
copy .env.example .env
```

Then set your real `GEMINI_API_KEY` in `.env`.

3. Run the API:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

4. Open docs:

- Swagger UI: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

## Docker Run

1. Create `.env` from the sample and set `GEMINI_API_KEY`.
2. Build and run:

```bash
docker compose up --build
```

API will be available at `http://localhost:8000`.

## API Endpoints

### `POST /ingest`

Upload a financial document to index.

- Supported file types: `pdf`, `txt`, `md`
- Form field: `file`

Example:

```bash
curl -X POST "http://localhost:8000/ingest" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@qbe_esg_report.pdf"
```

### `POST /query`

Ask a question about indexed documents.

Example request:

```json
{
  "question": "What are the key ESG initiatives mentioned in the report?",
  "k": 4
}
```

Example:

```bash
curl -X POST "http://localhost:8000/query" \
  -H "accept: application/json" \
  -H "Content-Type: application/json" \
  -d "{\"question\":\"What are the key ESG initiatives mentioned in the report?\",\"k\":4}"
```

## Notes

- ChromaDB data persists in `./chroma_db`.
- If answers are weak, increase `k` or tune `CHUNK_SIZE` and `CHUNK_OVERLAP`.
- This baseline is single-tenant and document-agnostic. You can extend it with per-user or per-document filtering later.
