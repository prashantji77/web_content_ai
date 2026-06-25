import json
import re
from typing import Any

from app.core.config import settings
from app.utils.errors import AIServiceError
from app.utils.text import extract_keywords, extractive_summary, limit_text, relevant_sentences

try:
    from langchain_core.prompts import PromptTemplate
    from langchain_openai import ChatOpenAI
except ImportError:  # Dependencies are installed from backend/requirements.txt.
    PromptTemplate = None
    ChatOpenAI = None


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

    def summarize(self, content: str) -> dict[str, Any]:
        if not self.llm:
            return self._fallback_summary(content)

        prompt = self._format_prompt(
            SUMMARY_PROMPT,
            content=limit_text(content, settings.max_prompt_chars),
        )
        message = self._invoke(prompt)
        data = self._parse_json(message)
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
        data = self._parse_json(message)
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
        if ChatOpenAI is None:
            raise AIServiceError(
                "LangChain OpenAI integration is not installed. Run pip install -r backend/requirements.txt.",
            )

        headers = {"X-Title": settings.openrouter_app_name}
        if settings.openrouter_site_url:
            headers["HTTP-Referer"] = settings.openrouter_site_url

        kwargs = {
            "model": settings.openrouter_model,
            "api_key": settings.openrouter_api_key,
            "base_url": settings.openrouter_base_url,
            "temperature": settings.temperature,
            "timeout": settings.llm_timeout_seconds,
            "default_headers": headers,
        }
        try:
            return ChatOpenAI(**kwargs)
        except TypeError:
            return ChatOpenAI(
                model_name=settings.openrouter_model,
                openai_api_key=settings.openrouter_api_key,
                openai_api_base=settings.openrouter_base_url,
                temperature=settings.temperature,
                request_timeout=settings.llm_timeout_seconds,
                default_headers=headers,
            )

    @staticmethod
    def _format_prompt(template: str, **values: str) -> str:
        if PromptTemplate is None:
            return template.format(**values)
        return PromptTemplate.from_template(template).format(**values)

    def _invoke(self, prompt: str) -> str:
        try:
            result = self.llm.invoke(prompt)
        except Exception as exc:
            raise AIServiceError(f"OpenRouter request failed: {exc}") from exc
        return str(getattr(result, "content", result))

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
