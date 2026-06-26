import asyncio
from dataclasses import asdict

from fastapi import APIRouter, HTTPException
from starlette.concurrency import run_in_threadpool

from app.core.config import settings
from app.models.schemas import (
    BrowserPage,
    ChatRequest,
    ChatResponse,
    CompareRequest,
    CompareResponse,
    IndexPageRequest,
    IndexPageResponse,
    MultiSummaryRequest,
    MultiSummaryResponse,
    PageSummary,
    SummarizeRequest,
    SummaryResponse,
)
from app.rag.vector_store import VectorStoreService
from app.services.content_extractor import ContentExtractor, ExtractedPage
from app.services.langchain_service import LangChainService
from app.utils.errors import AIServiceError, ContentExtractionError, IndexMissingError

router = APIRouter(tags=["web-content"])

content_extractor = ContentExtractor()
ai_service = LangChainService()
vector_store = VectorStoreService(ai_service)


def _string_urls(urls: list[object]) -> list[str]:
    return [str(url) for url in urls]


def _page_from_content(url: str, title: str | None, content: str | None) -> ExtractedPage | None:
    text = (content or "").strip()
    if not text:
        return None
    if len(text) < settings.min_content_length:
        raise HTTPException(
            status_code=400,
            detail="The browser page did not contain enough readable text to summarize.",
        )
    return ExtractedPage(url=url, title=title, text=text)


def _pages_from_payload(pages: list[BrowserPage]) -> list[ExtractedPage]:
    return [
        _page_from_content(str(page.url), page.title, page.content)
        for page in pages
    ]


async def _extract_pages(urls: list[str]):
    if len(urls) > settings.max_pages_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"Use {settings.max_pages_per_request} URLs or fewer per request.",
        )

    try:
        tasks = [
            run_in_threadpool(content_extractor.extract_from_url, url) for url in urls
        ]
        return await asyncio.gather(*tasks)
    except ContentExtractionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _collect_pages(urls: list[str], pages: list[BrowserPage]) -> list[ExtractedPage]:
    submitted_pages = [page for page in _pages_from_payload(pages) if page is not None]
    scraped_pages = await _extract_pages(urls) if urls else []
    total_pages = len(submitted_pages) + len(scraped_pages)
    if total_pages > settings.max_pages_per_request:
        raise HTTPException(
            status_code=400,
            detail=f"Use {settings.max_pages_per_request} pages or fewer per request.",
        )
    return [*submitted_pages, *scraped_pages]


async def _summarize_page(page) -> PageSummary:
    summary = await run_in_threadpool(ai_service.summarize, page.text)
    return PageSummary(source_url=page.url, title=page.title, **summary)


@router.post("/summarize", response_model=SummaryResponse)
async def summarize(payload: SummarizeRequest) -> SummaryResponse:
    try:
        page = _page_from_content(str(payload.url), payload.title, payload.content)
        if page is None:
            page = (await _extract_pages([str(payload.url)]))[0]
        summary = await _summarize_page(page)
        return SummaryResponse(**summary.model_dump())
    except AIServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/multi-summary", response_model=MultiSummaryResponse)
async def multi_summary(payload: MultiSummaryRequest) -> MultiSummaryResponse:
    try:
        pages = await _collect_pages(_string_urls(payload.urls), payload.pages)
        summaries = await asyncio.gather(*[_summarize_page(page) for page in pages])
        combined_summary = await run_in_threadpool(
            ai_service.combined_summary,
            [summary.model_dump() for summary in summaries],
        )
        return MultiSummaryResponse(
            summaries=summaries,
            combined_summary=combined_summary,
        )
    except AIServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/compare", response_model=CompareResponse)
async def compare(payload: CompareRequest) -> CompareResponse:
    try:
        pages = await _collect_pages(_string_urls(payload.urls), payload.pages)
        comparison = await run_in_threadpool(
            ai_service.compare,
            [asdict(page) for page in pages],
        )
        return CompareResponse(**comparison)
    except AIServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/index-page", response_model=IndexPageResponse)
async def index_page(payload: IndexPageRequest) -> IndexPageResponse:
    try:
        page = _page_from_content(str(payload.url), payload.title, payload.content)
        if page is None:
            page = (await _extract_pages([str(payload.url)]))[0]
        result = await run_in_threadpool(
            vector_store.index_page,
            payload.session_id,
            page,
        )
        return IndexPageResponse(**result)
    except AIServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    try:
        result = await run_in_threadpool(
            vector_store.answer,
            payload.session_id,
            payload.question,
        )
        return ChatResponse(**result)
    except IndexMissingError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AIServiceError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
