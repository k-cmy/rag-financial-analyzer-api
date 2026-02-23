import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str
    gemini_model: str
    gemini_embedding_model: str
    chroma_persist_dir: str
    chroma_collection_name: str
    chunk_size: int
    chunk_overlap: int
    max_retrieval_k: int


def get_settings() -> Settings:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required. Set it in your environment or .env file.")

    return Settings(
        gemini_api_key=api_key,
        gemini_model=os.getenv("GEMINI_MODEL", "gemini-3-flash-preview"),
        gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/text-embedding-004"),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
        chroma_collection_name=os.getenv("CHROMA_COLLECTION_NAME", "financial_docs"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
        max_retrieval_k=int(os.getenv("MAX_RETRIEVAL_K", "8")),
    )
