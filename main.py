"""Command-line interface for Lumina Doc."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.chatbot import ask_question, create_qa_chain
from core.embedder import create_vector_store
from core.loader import load_document
from core.splitter import split_documents
from utils.helpers import load_environment, safe_collection_name, supported_extensions_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Lumina Doc - AI-powered document chatbot for PDF, DOCX, and EPUB files."
    )
    parser.add_argument(
        "document",
        nargs="?",
        help=f"Path to a document file ({supported_extensions_text()}).",
    )
    parser.add_argument(
        "--persist-dir",
        default="chroma_db",
        help="Directory for local ChromaDB persistence.",
    )
    return parser


def prompt_for_document() -> Path:
    while True:
        value = input("Document path: ").strip().strip('"')
        if value:
            return Path(value)
        print("Please provide a document path.")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        load_environment()
        document_path = Path(args.document) if args.document else prompt_for_document()

        print("Loading document...")
        loaded = load_document(document_path)

        print("Splitting text into chunks...")
        chunks = split_documents(loaded.documents)
        collection_name = safe_collection_name(["lumina", loaded.file_path.stem])

        print(f"Creating embeddings for {len(chunks)} chunks...")
        vector_store = create_vector_store(
            chunks=chunks,
            collection_name=collection_name,
            persist_directory=args.persist_dir,
        )
        qa_chain = create_qa_chain(vector_store)

        print("\nLumina Doc is ready. Type 'exit' or 'quit' to stop.")
        print(
            f"Document: {loaded.filename} | Type: {loaded.file_type} | "
            f"Pages/sections: {loaded.total_pages} | Chunks: {len(chunks)}\n"
        )

        while True:
            question = input("You: ").strip()
            if question.lower() in {"exit", "quit", "q"}:
                print("Goodbye.")
                return 0
            if not question:
                continue

            response = ask_question(qa_chain, question)
            print(f"AI: {response.get('result', '').strip()}\n")

    except KeyboardInterrupt:
        print("\nGoodbye.")
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
