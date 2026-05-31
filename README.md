# Lumina Doc

Lumina Doc is an AI-powered document chatbot for PDF, DOCX, and EPUB files. It uses Google Gemini, LangChain, and ChromaDB to index uploaded documents and answer questions from their contents.

App title: **Lumina Doc - Chatbot Dokumen Cerdas**

## Features

- PDF, DOCX, and EPUB document loading
- Google Gemini 3.5 Flash for document question answering
- Google Generative AI embeddings with local ChromaDB storage
- Streamlit web UI with Indonesian language support
- Streaming answers in the web chat
- Staged upload progress for document processing
- Web controls for chunking, retrieval, model, temperature, and indexing limits
- CLI chatbot for terminal workflows
- CLI reuse of persisted Chroma collections for unchanged documents and chunking settings
- Numbered source citations for retrieved document evidence
- Document metadata display: filename, pages/sections, and total chunks
- Source metadata and snippets for retrieved answers

## Project Structure

```text
lumina-doc/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── main.py
├── app.py
├── core/
│   ├── __init__.py
│   ├── chatbot.py
│   ├── embedder.py
│   ├── loader.py
│   └── splitter.py
├── tests/
│   ├── test_app.py
│   ├── test_chatbot.py
│   ├── test_cli.py
│   ├── test_embedder.py
│   ├── test_helpers.py
│   ├── test_live_smoke.py
│   ├── test_loader.py
│   ├── test_model_config.py
│   ├── test_sources.py
│   └── test_splitter.py
├── data/
│   └── .gitkeep
└── utils/
    ├── __init__.py
    ├── helpers.py
    └── sources.py
```

## Requirements

- Python 3.10 or newer
- Google Gemini API key from Google AI Studio

## Installation

1. Create and activate a virtual environment:

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure environment variables:

```bash
cp .env.example .env
```

Then edit `.env`:

```env
GOOGLE_API_KEY=your_key_here
GEMINI_CHAT_MODEL=gemini-3.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-2
```

## Streamlit Usage

Run the web app:

```bash
streamlit run app.py
```

Open the local Streamlit URL, upload a PDF, DOCX, or EPUB file from the sidebar, then ask questions in Bahasa Indonesia or English.

The sidebar settings let you tune chunk size, chunk overlap, retrieval `top-k`, Gemini model names, response temperature, and document indexing limits before the file is processed.

## CLI Usage

Run with a document path:

```bash
python main.py path/to/document.pdf
```

Or start the CLI and enter the path when prompted:

```bash
python main.py
```

Use `exit`, `quit`, or `q` to close the chatbot.

Useful CLI options:

| Option | Description |
| --- | --- |
| `--chunk-size` | Maximum characters per chunk. Default: `1000` |
| `--chunk-overlap` | Characters shared between adjacent chunks. Default: `200` |
| `--retrieval-k` | Number of chunks retrieved for each question. Default: `4` |
| `--persist-dir` | Directory for persisted ChromaDB collections. Default: `chroma_db` |
| `--rebuild-index` | Recreate embeddings even if a matching persisted collection already exists |
| `--chat-model` | Override `GEMINI_CHAT_MODEL` for one run |
| `--embedding-model` | Override `GEMINI_EMBEDDING_MODEL` for one run |
| `--temperature` | Response randomness from `0` to `1`. Default: `0.2` |
| `--hide-sources` | Hide source snippets in terminal answers |
| `--max-file-size-mb` | Reject files larger than this before indexing. Default: `50`; use `0` to disable |
| `--max-pages` | Reject documents with more pages/sections than this. Default: `500`; use `0` to disable |
| `--max-chunks` | Reject documents that produce more chunks than this. Default: `1000`; use `0` to disable |
| `--debug` | Print full tracebacks for troubleshooting |

Example with retrieval tuning:

```bash
python main.py path/to/document.pdf --chunk-size 1200 --chunk-overlap 180 --retrieval-k 5
```

## Testing

Run the test suite:

```bash
python -m unittest discover -s tests
```

Optional live Gemini smoke test:

```bash
LUMINA_LIVE_TEST=1 GOOGLE_API_KEY=your_key_here python -m unittest tests.test_live_smoke
```

On Windows PowerShell:

```powershell
$env:LUMINA_LIVE_TEST="1"; $env:GOOGLE_API_KEY="your_key_here"; python -m unittest tests.test_live_smoke
```

## How It Works

1. `core/loader.py` extracts text and metadata from PDF, DOCX, or EPUB files.
2. `core/splitter.py` splits text with `RecursiveCharacterTextSplitter` using `chunk_size=1000` and `chunk_overlap=200`.
3. `core/embedder.py` creates Google Generative AI embeddings and stores vectors in ChromaDB.
4. `core/chatbot.py` creates a `RetrievalQA` chain with Gemini 3.5 Flash.
5. `app.py` and `main.py` provide Streamlit and CLI interfaces.

## Environment Variables

| Name | Description |
| --- | --- |
| `GOOGLE_API_KEY` | Google Gemini API key used by LangChain Google GenAI integrations |
| `GEMINI_CHAT_MODEL` | Optional Gemini chat model override. Defaults to `gemini-3.5-flash` |
| `GEMINI_EMBEDDING_MODEL` | Optional embedding model override. Defaults to `models/gemini-embedding-2` |

## Notes

- CLI ChromaDB data is stored in `chroma_db/` and ignored by Git.
- Streamlit uploads use a temporary in-memory Chroma collection for each processed document/settings combination.
- CLI vector collections include the document hash, embedding model, and chunking settings to avoid reusing a collection for different content or indexing parameters.
- The CLI reuses a persisted collection when it already contains vectors. Use `--rebuild-index` to force a fresh embedding pass.
- Uploaded files are processed locally.
- Answers are constrained to the uploaded document context. If the answer is not present, the assistant should say the information was not found in the document.
