from pydantic import BaseModel, Field, HttpUrl


class SummarizeRequest(BaseModel):
    url: HttpUrl


class MultiSummaryRequest(BaseModel):
    urls: list[HttpUrl] = Field(..., min_length=1, max_length=10)


class CompareRequest(BaseModel):
    urls: list[HttpUrl] = Field(..., min_length=2, max_length=10)


class IndexPageRequest(BaseModel):
    url: HttpUrl
    session_id: str = Field("default", min_length=1, max_length=80)


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=2, max_length=2000)
    session_id: str = Field("default", min_length=1, max_length=80)


class PageSummary(BaseModel):
    source_url: str
    title: str | None = None
    short_summary: str
    detailed_summary: str
    key_points: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)


class SummaryResponse(PageSummary):
    pass


class MultiSummaryResponse(BaseModel):
    summaries: list[PageSummary]
    combined_summary: str


class CompareResponse(BaseModel):
    similarities: list[str] = Field(default_factory=list)
    differences: list[str] = Field(default_factory=list)
    conclusion: str


class IndexPageResponse(BaseModel):
    status: str
    session_id: str
    chunks_indexed: int
    source_url: str
    title: str | None = None


class SourceChunk(BaseModel):
    source_url: str | None = None
    title: str | None = None
    chunk_index: int | None = None
    snippet: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceChunk] = Field(default_factory=list)

