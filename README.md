# AI Web Content Summarizer & Research Assistant

A production-ready starter project for a Chrome Extension plus FastAPI backend that summarizes webpages, compares multiple pages, extracts key insights, and supports RAG chat over indexed webpage content.

## Features

- Current webpage summarization with short summary, detailed summary, key points, and keywords.
- Multi-page summarization through `POST /multi-summary`.
- Webpage comparison through `POST /compare`.
- RAG chat using LangChain text splitting, FAISS vector search, and an OpenRouter-backed chat model.
- Copy, TXT export, Markdown export, and dark mode in the Chrome extension.
- Local development fallbacks for summarization and embeddings when API keys are not configured.

## Architecture

```text
Chrome Extension
  -> FastAPI Backend
  -> BeautifulSoup Content Extraction
  -> LangChain Prompting
  -> Recursive Text Chunking
  -> FAISS Vector Store
  -> Retriever
  -> OpenRouter LLM
  -> Response
```

## Project Structure

```text
backend/
  app/
    api/
    core/
    models/
    rag/
    services/
    utils/
    main.py
  requirements.txt
  .env.example
extension/
  manifest.json
  popup.html
  popup.css
  popup.js
  content.js
  background.js
README.md
```

## Backend Setup

Use Python 3.11 or 3.12 for the smoothest FAISS installation experience.

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `backend/.env`:

```env
OPENROUTER_API_KEY=your_openrouter_key
OPENROUTER_MODEL=openai/gpt-4o-mini
EMBEDDINGS_PROVIDER=local
```

To use OpenAI-compatible embeddings instead of the local deterministic fallback:

```env
OPENAI_API_KEY=your_openai_key
EMBEDDINGS_PROVIDER=openai
```

Run the API:

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

Open the API docs at:

```text
http://127.0.0.1:8000/docs
```

## Chrome Extension Setup

1. Start the backend on `http://127.0.0.1:8000`.
2. Open Chrome and go to `chrome://extensions`.
3. Enable Developer mode.
4. Click Load unpacked.
5. Select the `extension/` folder.
6. Open any `http` or `https` webpage and click the extension icon.

## API Documentation

### `POST /summarize`

Request:

```json
{ "url": "https://example.com" }
```

Response:

```json
{
  "source_url": "https://example.com",
  "title": "Example",
  "short_summary": "",
  "detailed_summary": "",
  "key_points": [],
  "keywords": []
}
```

### `POST /multi-summary`

Request:

```json
{ "urls": ["https://site1.com", "https://site2.com"] }
```

Response:

```json
{
  "summaries": [],
  "combined_summary": ""
}
```

### `POST /compare`

Request:

```json
{ "urls": ["https://site1.com", "https://site2.com"] }
```

Response:

```json
{
  "similarities": [],
  "differences": [],
  "conclusion": ""
}
```

### `POST /index-page`

Request:

```json
{ "url": "https://example.com", "session_id": "default" }
```

Response:

```json
{
  "status": "indexed",
  "session_id": "default",
  "chunks_indexed": 4,
  "source_url": "https://example.com",
  "title": "Example"
}
```

### `POST /chat`

Request:

```json
{ "question": "What is the article about?", "session_id": "default" }
```

Response:

```json
{
  "answer": "...",
  "sources": []
}
```

## Error Handling

The backend returns user-friendly messages for invalid URLs, empty content, API timeouts, OpenRouter failures, and missing FAISS indexes. The extension displays those messages in the popup status area.


## Notes

- Without `OPENROUTER_API_KEY`, the backend uses local extractive summaries so the app can still be tested.
- Without `OPENAI_API_KEY`, RAG uses deterministic local hashing embeddings with FAISS when installed.
- Some websites block server-side scraping. For those pages, the backend will return a clear extraction error.

