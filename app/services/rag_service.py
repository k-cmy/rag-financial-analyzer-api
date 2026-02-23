import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config import Settings


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.gemini_api_key,
        )
        self.llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.gemini_api_key,
            temperature=0,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.vector_store = Chroma(
            collection_name=settings.chroma_collection_name,
            persist_directory=settings.chroma_persist_dir,
            embedding_function=self.embeddings,
        )

    def _extract_answer_text(self, model_response: Any) -> str:
        content = getattr(model_response, "content", model_response)

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        text_parts.append(text)
                        continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    text_parts.append(text)
            if text_parts:
                return "\n".join(text_parts).strip()

        return str(content)

    def _load_documents(self, filename: str, file_bytes: bytes) -> list[Document]:
        suffix = Path(filename).suffix.lower()
        source_name = Path(filename).name

        if suffix == ".pdf":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(file_bytes)
                tmp_path = tmp_file.name

            try:
                docs = PyPDFLoader(tmp_path).load()
            finally:
                Path(tmp_path).unlink(missing_ok=True)

            for doc in docs:
                doc.metadata["source"] = source_name
            return docs

        text = file_bytes.decode("utf-8", errors="ignore")
        return [Document(page_content=text, metadata={"source": source_name})]

    def ingest_document(self, filename: str, file_bytes: bytes) -> dict[str, Any]:
        docs = self._load_documents(filename=filename, file_bytes=file_bytes)
        chunks = self.splitter.split_documents(docs)

        if not chunks:
            return {
                "filename": filename,
                "chunks_added": 0,
                "collection_name": self.settings.chroma_collection_name,
            }

        ids = [str(uuid4()) for _ in chunks]
        self.vector_store.add_documents(chunks, ids=ids)
        if hasattr(self.vector_store, "persist"):
            self.vector_store.persist()

        return {
            "filename": filename,
            "chunks_added": len(chunks),
            "collection_name": self.settings.chroma_collection_name,
        }

    def query(self, question: str, k: int) -> dict[str, Any]:
        safe_k = min(k, self.settings.max_retrieval_k)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": safe_k})
        docs = retriever.invoke(question)

        if not docs:
            return {
                "answer": "I could not find relevant content in the indexed documents.",
                "sources": [],
            }

        context_blocks: list[str] = []
        sources: list[dict[str, Any]] = []
        seen_keys: set[tuple[str, int | None]] = set()

        for doc in docs:
            source = str(doc.metadata.get("source", "unknown"))
            page_raw = doc.metadata.get("page")
            page = int(page_raw) if isinstance(page_raw, int) else None
            key = (source, page)
            if key not in seen_keys:
                seen_keys.add(key)
                sources.append({"source": source, "page": page, "metadata": dict(doc.metadata)})
            context_blocks.append(doc.page_content)

        context = "\n\n---\n\n".join(context_blocks)
        prompt = (
            "You are a financial document analyst.\n"
            "Answer using only the context below.\n"
            "If the answer is not in the context, say you do not know.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context}"
        )
        model_response = self.llm.invoke(prompt)
        answer = self._extract_answer_text(model_response)

        return {"answer": answer, "sources": sources}
