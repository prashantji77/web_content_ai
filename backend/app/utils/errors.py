class AppError(Exception):
    """Base application error."""


class ContentExtractionError(AppError):
    """Raised when a webpage cannot be downloaded or cleaned."""


class AIServiceError(AppError):
    """Raised when LLM, embedding, or vector processing fails."""


class IndexMissingError(AppError):
    """Raised when chat is requested before a page is indexed."""

