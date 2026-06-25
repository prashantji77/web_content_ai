import re
from collections import Counter


STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "because",
    "before",
    "between",
    "could",
    "first",
    "from",
    "have",
    "into",
    "more",
    "most",
    "other",
    "over",
    "such",
    "than",
    "that",
    "their",
    "there",
    "these",
    "this",
    "through",
    "under",
    "using",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "with",
    "would",
    "your",
}


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def limit_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def split_sentences(text: str) -> list[str]:
    compact = normalize_whitespace(text)
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    return [sentence.strip() for sentence in sentences if len(sentence.strip()) > 20]


def extractive_summary(text: str, sentence_count: int = 4) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return limit_text(normalize_whitespace(text), 600)
    return " ".join(sentences[:sentence_count])


def extract_keywords(text: str, limit: int = 10) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9'-]{3,}", text.lower())
    candidates = [word.strip("-'") for word in words if word not in STOPWORDS]
    counts = Counter(candidates)
    return [word for word, _ in counts.most_common(limit)]


def relevant_sentences(text: str, question: str, max_sentences: int = 5) -> list[str]:
    sentences = split_sentences(text)
    if not question.strip():
        return sentences[:max_sentences]

    query_terms = set(extract_keywords(question, limit=12))
    if not query_terms:
        return sentences[:max_sentences]

    scored = []
    for index, sentence in enumerate(sentences):
        sentence_terms = set(extract_keywords(sentence, limit=40))
        score = len(query_terms & sentence_terms)
        if score:
            scored.append((score, -index, sentence))

    scored.sort(reverse=True)
    return [sentence for _, _, sentence in scored[:max_sentences]]

