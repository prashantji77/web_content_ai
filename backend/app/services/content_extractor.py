import re
from dataclasses import dataclass
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from app.core.config import settings
from app.utils.errors import ContentExtractionError
from app.utils.text import normalize_whitespace


NOISY_SELECTOR_RE = re.compile(
    r"(^|[-_\s])(ad|ads|advert|sponsor|promo|cookie|banner|newsletter|popup)([-_\s]|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ExtractedPage:
    url: str
    title: str | None
    text: str


class ContentExtractor:
    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": settings.user_agent})

    def extract_from_url(self, url: str) -> ExtractedPage:
        self._validate_url(url)
        html = self._download(url)
        soup = BeautifulSoup(html, "html.parser")

        for tag in soup(["script", "style", "noscript", "svg", "canvas", "iframe"]):
            tag.decompose()

        for tag in soup.find_all(["nav", "aside", "footer", "form"]):
            tag.decompose()

        for tag in soup.find_all(True):
            classes = tag.get("class") or []
            marker = " ".join([tag.get("id", ""), *classes])
            if marker and NOISY_SELECTOR_RE.search(marker):
                tag.decompose()

        title = normalize_whitespace(soup.title.get_text(" ", strip=True)) if soup.title else None
        container = soup.find("article") or soup.find("main") or soup.body or soup
        raw_text = container.get_text("\n", strip=True)
        text = self._clean_text(raw_text)

        if len(text) < settings.min_content_length:
            raise ContentExtractionError(
                "The page did not contain enough readable text to summarize.",
            )

        return ExtractedPage(url=url, title=title, text=text)

    def _download(self, url: str) -> str:
        try:
            response = self.session.get(
                url,
                timeout=settings.request_timeout_seconds,
                allow_redirects=True,
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise ContentExtractionError("The webpage request timed out.") from exc
        except requests.RequestException as exc:
            raise ContentExtractionError(f"Could not download webpage: {exc}") from exc

        content_type = response.headers.get("content-type", "").lower()
        if content_type and not any(
            allowed in content_type for allowed in ("text/html", "application/xhtml+xml")
        ):
            raise ContentExtractionError(
                f"Unsupported content type: {content_type.split(';')[0]}",
            )

        return response.text

    @staticmethod
    def _validate_url(url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ContentExtractionError("Please provide a valid http or https URL.")

    @staticmethod
    def _clean_text(raw_text: str) -> str:
        lines = [normalize_whitespace(line) for line in raw_text.splitlines()]
        useful_lines = [line for line in lines if len(line) > 1]
        return normalize_whitespace("\n".join(useful_lines))

