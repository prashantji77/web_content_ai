from pydantic import BaseModel, Field, HttpUrl, model_validator


class BrowserPage(BaseModel):
    url: HttpUrl
    title: str | None = None
    content: str = Field(..., min_length=1)


class SummarizeRequest(BaseModel):
    url: HttpUrl
    title: str | None = None
    content: str | None = None


class MultiSummaryRequest(BaseModel):
    urls: list[HttpUrl] = Field(default_factory=list, max_length=10)
    pages: list[BrowserPage] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_input(self):
        if not self.urls and not self.pages:
            raise ValueError("Provide at least one URL or browser page.")
        return self


class CompareRequest(BaseModel):
    urls: list[HttpUrl] = Field(default_factory=list, max_length=10)
    pages: list[BrowserPage] = Field(default_factory=list, max_length=10)

    @model_validator(mode="after")
    def validate_input(self):
        if len(self.urls) + len(self.pages) < 2:
            raise ValueError("Provide at least two URLs or browser pages to compare.")
        return self


class IndexPageRequest(BaseModel):
    url: HttpUrl
    session_id: str = Field("default", min_length=1, max_length=80)
    title: str | None = None
    content: str | None = None


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
