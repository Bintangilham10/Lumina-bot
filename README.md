# Lumina Doc

**AI-Powered Document Chatbot** — Upload your documents and ask anything about their content.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=flat-square)
![Gemini](https://img.shields.io/badge/Google_Gemini-1.5_Flash-orange?style=flat-square&logo=google)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## Overview

Lumina Doc is an AI-powered chatbot application that enables users to upload documents in various formats (PDF, DOCX, EPUB) and interact with their content through natural language queries using **Google Gemini AI**.

This project was developed as part of a **Kerja Praktik (Internship)** initiative focused on building a prototype for a national book conversion system based on Artificial Intelligence.

---

## Features

- Multi-format document support: PDF, DOCX, and EPUB
- Natural language Q&A powered by Google Gemini 1.5 Flash
- Local vector storage using ChromaDB
- Clean and interactive web interface built with Streamlit
- Command-line interface (CLI) for terminal-based usage
- Bilingual support: Bahasa Indonesia and English

---

## Project Structure

```
lumina-doc/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── main.py               # CLI entry point
├── app.py                # Streamlit web UI entry point
├── core/
│   ├── __init__.py
│   ├── loader.py         # Document loader (PDF, DOCX, EPUB)
│   ├── splitter.py       # Text chunking
│   ├── embedder.py       # Embedding generation
│   └── chatbot.py        # QA chain logic
├── data/
│   └── .gitkeep
└── utils/
    ├── __init__.py
    └── helpers.py
```

---

## Requirements

- Python 3.10 or higher
- Google Gemini API key (free at [Google AI Studio](https://aistudio.google.com/app/apikey))

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/username/lumina-doc.git
cd lumina-doc
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

```bash
cp .env.example .env
```

Open `.env` and fill in your API key:

```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

---

## Usage

**Web Interface (Recommended)**

```bash
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`.

**Command Line Interface**

```bash
python main.py
```

---

## Example

```
Loading document...
Splitting text into 142 chunks...
Saving to vector database...

Lumina Doc is ready. Type 'exit' to quit.

You : What is the main topic of chapter one?
AI  : Chapter one discusses...

You : Who is the author of this book?
AI  : Based on the document, the author is...
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| langchain | AI orchestration framework |
| langchain-google-genai | Google Gemini integration |
| chromadb | Local vector database |
| pymupdf | PDF parsing |
| python-docx | DOCX parsing |
| ebooklib | EPUB parsing |
| streamlit | Web UI framework |
| python-dotenv | Environment variable management |

---

## Roadmap

- [x] PDF chatbot via CLI
- [x] DOCX and EPUB support
- [x] Web UI with Streamlit
- [ ] Multi-document support
- [ ] Conversation memory
- [ ] Export chat history to PDF
- [ ] Cloud deployment (Streamlit Cloud / Hugging Face Spaces)

---

## Development Context

Developed as part of the **Kerja Praktik** program for the national book conversion system prototype using Artificial Intelligence.

---

## License

This project is licensed under the [MIT License](LICENSE).
