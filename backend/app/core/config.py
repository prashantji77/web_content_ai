import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[2]
PROJECT_ROOT = BACKEND_DIR.parent

# Load the project-root .env first, then backend/.env (backend overrides root if both exist).
# The user's API keys live in the repository-root .env, so it must be loaded.
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(BACKEND_DIR / ".env", override=True)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    path = Path(raw)
    if path.is_absolute():
        return path
    return BACKEND_DIR / path


@dataclass(frozen=True)
class Settings:
    openrouter_api_key: str | None = os.getenv("OPENROUTER_API_KEY")
    openrouter_model: str = os.getenv(
        "OPENROUTER_MODEL",
        "openai/gpt-oss-120b:free",
    )
    openrouter_base_url: str = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )
    openrouter_site_url: str = os.getenv("OPENROUTER_SITE_URL", "http://localhost:8000")
    openrouter_app_name: str = os.getenv(
        "OPENROUTER_APP_NAME",
        "AI Web Content Summarizer",
    )
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_embedding_model: str = os.getenv(
        "OPENAI_EMBEDDING_MODEL",
        "text-embedding-3-small",
    )
    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "local").lower()
    request_timeout_seconds: int = _get_int("REQUEST_TIMEOUT_SECONDS", 20)
    llm_timeout_seconds: int = _get_int("LLM_TIMEOUT_SECONDS", 60)
    max_prompt_chars: int = _get_int("MAX_PROMPT_CHARS", 18000)
    max_pages_per_request: int = _get_int("MAX_PAGES_PER_REQUEST", 8)
    min_content_length: int = _get_int("MIN_CONTENT_LENGTH", 120)
    chunk_size: int = _get_int("CHUNK_SIZE", 800)
    chunk_overlap: int = _get_int("CHUNK_OVERLAP", 160)
    retriever_k: int = _get_int("RETRIEVER_K", 6)
    retriever_fetch_k: int = _get_int("RETRIEVER_FETCH_K", 20)
    retriever_search_type: str = os.getenv("RETRIEVER_SEARCH_TYPE", "mmr").lower()
    temperature: float = _get_float("LLM_TEMPERATURE", 0.2)
    faiss_storage_dir: Path = _get_path("FAISS_STORAGE_DIR", BACKEND_DIR / "storage" / "faiss")
    persist_faiss: bool = _get_bool("PERSIST_FAISS", True)
    user_agent: str = os.getenv(
        "USER_AGENT",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    )

    @property
    def cors_origins(self) -> list[str]:
        raw = os.getenv("CORS_ORIGINS", "*")
        origins = [origin.strip() for origin in raw.split(",") if origin.strip()]
        return origins or ["*"]


settings = Settings()
