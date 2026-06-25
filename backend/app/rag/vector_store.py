import hashlib
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import settings
from app.utils.errors import AIServiceError, IndexMissingError
from app.utils.text import limit_text

try:
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_openai import OpenAIEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:  # Dependencies are installed from backend/requirements.txt.
    FAISS = None
    Embeddings = None
    OpenAIEmbeddings = None
    RecursiveCharacterTextSplitter = None

    @dataclass
    class Document:  # type: ignore[no-redef]
        page_content: str
        metadata: dict[str, Any]


BaseEmbeddings = Embeddings if Embeddings is not None else object


class LocalHashEmbeddings(BaseEmbeddings):
    """Deterministic local embeddings for development without an embeddings API."""

    def __init__(self, dimensions: int = 384) -> None:
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9']{2,}", text.lower())
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


@dataclass
class IndexedSession:
    store: Any
    source_url: str
    title: str | None
    chunks_indexed: int


class SimpleVectorStore:
    def __init__(self, documents: list[Document], embeddings: LocalHashEmbeddings) -> None:
        self.documents = documents
        self.embeddings = embeddings
        self.vectors = embeddings.embed_documents([doc.page_content for doc in documents])

    def similarity_search(self, query: str, k: int = 5) -> list[Document]:
        query_vector = self.embeddings.embed_query(query)
        scored = [
            (self._cosine(query_vector, vector), document)
            for vector, document in zip(self.vectors, self.documents, strict=False)
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored[:k]]

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right, strict=False))


class VectorStoreService:
    def __init__(self, ai_service) -> None:
        self.ai_service = ai_service
        self.sessions: dict[str, IndexedSession] = {}
        settings.faiss_storage_dir.mkdir(parents=True, exist_ok=True)

    def index_page(self, session_id: str, page) -> dict[str, Any]:
        safe_session_id = self._safe_session_id(session_id)
        chunks = self._split_text(page.text)
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source_url": page.url,
                    "title": page.title,
                    "chunk_index": index,
                },
            )
            for index, chunk in enumerate(chunks)
            if chunk.strip()
        ]

        if not documents:
            raise AIServiceError("No text chunks were available to index.")

        embeddings = self._build_embeddings()
        if FAISS is not None:
            store = FAISS.from_documents(documents, embeddings)
            if settings.persist_faiss:
                store.save_local(str(self._session_path(safe_session_id)))
        else:
            store = SimpleVectorStore(documents, embeddings)

        self.sessions[safe_session_id] = IndexedSession(
            store=store,
            source_url=page.url,
            title=page.title,
            chunks_indexed=len(documents),
        )
        return {
            "status": "indexed",
            "session_id": safe_session_id,
            "chunks_indexed": len(documents),
            "source_url": page.url,
            "title": page.title,
        }

    def answer(self, session_id: str, question: str) -> dict[str, Any]:
        safe_session_id = self._safe_session_id(session_id)
        session = self.sessions.get(safe_session_id) or self._load_session(safe_session_id)
        if not session:
            raise IndexMissingError(
                "No FAISS index was found for this page. Index the page before chatting.",
            )

        docs = session.store.similarity_search(question, k=5)
        context = "\n\n".join(doc.page_content for doc in docs)
        answer = self.ai_service.answer_question(context, question)
        return {
            "answer": answer,
            "sources": [
                {
                    "source_url": doc.metadata.get("source_url"),
                    "title": doc.metadata.get("title"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "snippet": limit_text(doc.page_content, 280),
                }
                for doc in docs
            ],
        }

    def _build_embeddings(self):
        if (
            settings.embeddings_provider == "openai"
            and settings.openai_api_key
            and OpenAIEmbeddings is not None
        ):
            try:
                return OpenAIEmbeddings(
                    api_key=settings.openai_api_key,
                    model=settings.openai_embedding_model,
                )
            except TypeError:
                return OpenAIEmbeddings(
                    openai_api_key=settings.openai_api_key,
                    model=settings.openai_embedding_model,
                )
        return LocalHashEmbeddings()

    @staticmethod
    def _split_text(text: str) -> list[str]:
        if RecursiveCharacterTextSplitter is None:
            chunk_size = 1500
            overlap = 200
            chunks = []
            start = 0
            while start < len(text):
                chunks.append(text[start : start + chunk_size])
                start += chunk_size - overlap
            return chunks

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)

    def _load_session(self, session_id: str) -> IndexedSession | None:
        if FAISS is None or not settings.persist_faiss:
            return None

        path = self._session_path(session_id)
        if not path.exists():
            return None

        embeddings = self._build_embeddings()
        store = FAISS.load_local(
            str(path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        docs = list(getattr(store, "docstore")._dict.values())
        first_doc = docs[0] if docs else None
        session = IndexedSession(
            store=store,
            source_url=first_doc.metadata.get("source_url") if first_doc else None,
            title=first_doc.metadata.get("title") if first_doc else None,
            chunks_indexed=len(docs),
        )
        self.sessions[session_id] = session
        return session

    @staticmethod
    def _safe_session_id(session_id: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9_-]", "-", session_id.strip())
        return cleaned[:80] or "default"

    @staticmethod
    def _session_path(session_id: str) -> Path:
        return settings.faiss_storage_dir / session_id

