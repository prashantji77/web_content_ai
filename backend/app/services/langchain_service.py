import json
import re
import time
from typing import Any

import requests

from app.core.config import settings
from app.utils.errors import AIServiceError
from app.utils.text import extract_keywords, extractive_summary, limit_text, relevant_sentences

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:  # Dependencies are installed from backend/requirements.txt.
    PromptTemplate = None


SUMMARY_PROMPT = """You are an expert content analyst.
Return valid JSON only with this schema:
{{
  "short_summary": "one concise paragraph",
  "detailed_summary": "a thorough but readable summary",
  "key_points": ["point 1", "point 2", "point 3"],
  "keywords": ["keyword 1", "keyword 2"]
}}

Content:
{content}
"""

COMPARISON_PROMPT = """Compare the following webpages.
Return valid JSON only with this schema:
{{
  "similarities": ["similarity 1", "similarity 2"],
  "differences": ["difference 1", "difference 2"],
  "conclusion": "final conclusion"
}}

Content:
{content}
"""

COMBINED_SUMMARY_PROMPT = """You are synthesizing research notes from multiple webpages.
Write a combined summary that highlights the overall topic, major findings, and any important disagreements.

Page summaries:
{content}
"""

RAG_PROMPT = """Answer the question only from the provided context.
If information is unavailable, say:
"Information not found in the provided webpage."

Context:
{context}

Question:
{question}
"""


class LangChainService:
    def __init__(self) -> None:
        self.llm = self._build_llm()

    @property
    def is_llm_configured(self) -> bool:
        return bool(self.llm)

    def summarize(self, content: str) -> dict[str, Any]:
        if not self.llm:
            return self._fallback_summary(content)

        prompt = self._format_prompt(
            SUMMARY_PROMPT,
            content=limit_text(content, settings.max_prompt_chars),
        )
        message = self._invoke(prompt)
        try:
            data = self._parse_json(message)
        except AIServiceError:
            data = self._summary_from_llm_text(message, content)
        return self._normalize_summary(data, content)

    def combined_summary(self, summaries: list[dict[str, Any]]) -> str:
        content = json.dumps(summaries, indent=2)
        if not self.llm:
            joined = " ".join(summary.get("short_summary", "") for summary in summaries)
            return extractive_summary(joined, sentence_count=5)

        prompt = self._format_prompt(
            COMBINED_SUMMARY_PROMPT,
            content=limit_text(content, settings.max_prompt_chars),
        )
        return self._invoke(prompt).strip()

    def compare(self, pages: list[dict[str, Any]]) -> dict[str, Any]:
        if not self.llm:
            return self._fallback_comparison(pages)

        content = "\n\n".join(
            f"URL: {page['url']}\nTitle: {page.get('title') or 'Untitled'}\n"
            f"Text:\n{limit_text(page['text'], 6000)}"
            for page in pages
        )
        prompt = self._format_prompt(COMPARISON_PROMPT, content=content)
        message = self._invoke(prompt)
        try:
            data = self._parse_json(message)
        except AIServiceError:
            data = self._comparison_from_llm_text(message)
        return {
            "similarities": self._string_list(data.get("similarities")),
            "differences": self._string_list(data.get("differences")),
            "conclusion": str(data.get("conclusion") or "No conclusion was generated."),
        }

    def answer_question(self, context: str, question: str) -> str:
        if not self.llm:
            matches = relevant_sentences(context, question, max_sentences=4)
            if not matches:
                return "Information not found in the provided webpage."
            return " ".join(matches)

        prompt = self._format_prompt(
            RAG_PROMPT,
            context=limit_text(context, settings.max_prompt_chars),
            question=question,
        )
        return self._invoke(prompt).strip()

    def _build_llm(self):
        if not settings.openrouter_api_key:
            return None
        return True

    @staticmethod
    def _headers() -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {settings.openrouter_api_key}",
            "Content-Type": "application/json",
            # OpenRouter attributes requests to an app via the "X-Title" header (shown in
            # the Activity dashboard). The previous "X-OpenRouter-Title" header is ignored.
            "X-Title": settings.openrouter_app_name,
        }
        if settings.openrouter_site_url:
            headers["HTTP-Referer"] = settings.openrouter_site_url
        return headers

    @staticmethod
    def _format_prompt(template: str, **values: str) -> str:
        if PromptTemplate is None:
            return template.format(**values)
        return PromptTemplate.from_template(template).format(**values)

    def _invoke(self, prompt: str) -> str:
        url = f"{settings.openrouter_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": settings.openrouter_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": settings.temperature,
        }

        # Free OpenRouter models are heavily rate limited (HTTP 429). Retry a few times with
        # backoff so transient throttling doesn't surface as a hard failure.
        max_attempts = 4
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            try:
                response = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=settings.llm_timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
                break
            except requests.HTTPError as exc:
                last_exc = exc
                status = exc.response.status_code if exc.response is not None else None
                if status in (429, 502, 503) and attempt < max_attempts - 1:
                    retry_after = self._retry_after_seconds(exc.response, attempt)
                    time.sleep(retry_after)
                    continue
                detail = self._response_error(exc.response)
                raise AIServiceError(f"OpenRouter request failed: {detail}") from exc
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < max_attempts - 1:
                    time.sleep(2 ** attempt)
                    continue
                raise AIServiceError(f"OpenRouter request failed: {exc}") from exc
            except ValueError as exc:
                raise AIServiceError("OpenRouter returned a non-JSON response.") from exc
        else:  # pragma: no cover - loop always breaks or raises
            raise AIServiceError(f"OpenRouter request failed: {last_exc}")

        choices = data.get("choices") or []
        if not choices:
            raise AIServiceError("OpenRouter returned no completion choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(
                item.get("text", "") if isinstance(item, dict) else str(item)
                for item in content
            )
        if not content:
            raise AIServiceError("OpenRouter returned an empty response.")
        return str(content)

    @staticmethod
    def _retry_after_seconds(response, attempt: int) -> float:
        default = min(2 ** attempt, 8)
        if response is None:
            return default
        header = response.headers.get("Retry-After")
        if header:
            try:
                return max(default, min(float(header), 15))
            except ValueError:
                return default
        return default

    @staticmethod
    def _response_error(response) -> str:
        if response is None:
            return "unknown HTTP error"
        try:
            data = response.json()
        except ValueError:
            data = {}
        message = data.get("error", {}).get("message") or data.get("detail")
        if message:
            return f"{response.status_code} {message}"
        return f"{response.status_code} {response.text[:300]}"

    @staticmethod
    def _parse_json(message: str) -> dict[str, Any]:
        cleaned = message.strip()
        cleaned = re.sub(r"^```(?:json)?", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if not match:
                raise AIServiceError("The AI response was not valid JSON.")
            parsed = json.loads(match.group(0))

        if not isinstance(parsed, dict):
            raise AIServiceError("The AI response JSON must be an object.")
        return parsed

    def _normalize_summary(self, data: dict[str, Any], content: str) -> dict[str, Any]:
        fallback = self._fallback_summary(content)
        return {
            "short_summary": str(data.get("short_summary") or fallback["short_summary"]),
            "detailed_summary": str(
                data.get("detailed_summary") or fallback["detailed_summary"],
            ),
            "key_points": self._string_list(data.get("key_points")) or fallback["key_points"],
            "keywords": self._string_list(data.get("keywords")) or fallback["keywords"],
        }

    @staticmethod
    def _summary_from_llm_text(message: str, content: str) -> dict[str, Any]:
        cleaned = message.strip()
        bullet_lines = [
            re.sub(r"^[\-*\d.)\s]+", "", line).strip()
            for line in cleaned.splitlines()
            if line.strip().startswith(("-", "*")) or re.match(r"^\d+[.)]\s+", line.strip())
        ]
        return {
            "short_summary": extractive_summary(cleaned, sentence_count=2),
            "detailed_summary": cleaned,
            "key_points": bullet_lines[:6] or relevant_sentences(cleaned, "", max_sentences=5),
            "keywords": extract_keywords(content, limit=10),
        }

    @staticmethod
    def _comparison_from_llm_text(message: str) -> dict[str, Any]:
        lines = [
            re.sub(r"^[\-*\d.)\s]+", "", line).strip()
            for line in message.splitlines()
            if line.strip()
        ]
        return {
            "similarities": lines[:3],
            "differences": lines[3:8],
            "conclusion": message.strip() or "No comparison was generated.",
        }

    @staticmethod
    def _fallback_summary(content: str) -> dict[str, Any]:
        sentences = relevant_sentences(content, "", max_sentences=5)
        return {
            "short_summary": extractive_summary(content, sentence_count=2),
            "detailed_summary": extractive_summary(content, sentence_count=6),
            "key_points": sentences[:5],
            "keywords": extract_keywords(content, limit=10),
        }

    @staticmethod
    def _fallback_comparison(pages: list[dict[str, Any]]) -> dict[str, Any]:
        keyword_sets = [set(extract_keywords(page["text"], limit=20)) for page in pages]
        common = sorted(set.intersection(*keyword_sets)) if keyword_sets else []
        differences: list[str] = []

        for index, page in enumerate(pages, start=1):
            other_keywords = set.union(
                *[keywords for i, keywords in enumerate(keyword_sets) if i != index - 1],
            )
            unique = sorted(keyword_sets[index - 1] - other_keywords)[:8]
            label = page.get("title") or page.get("url") or f"Page {index}"
            if unique:
                differences.append(f"{label}: unique focus on {', '.join(unique)}.")

        similarities = (
            [f"Shared keywords and themes: {', '.join(common[:10])}."]
            if common
            else ["The pages do not share enough obvious keywords for a reliable local comparison."]
        )
        conclusion = (
            "A full semantic comparison is available when OPENROUTER_API_KEY is configured. "
            "This local comparison is based on extracted keyword overlap."
        )
        return {
            "similarities": similarities,
            "differences": differences,
            "conclusion": conclusion,
        }

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []
