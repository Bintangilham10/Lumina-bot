"""Command-line interface for Lumina Doc."""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from core.chatbot import ask_question, create_qa_chain
from core.embedder import create_vector_store
from core.loader import load_document
from core.splitter import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, split_documents
from utils.helpers import (
    file_sha256,
    load_environment,
    safe_collection_name,
    supported_extensions_text,
)


SOURCE_SNIPPET_LENGTH = 220


def positive_int(value: str) -> int:
    """Parse a positive integer for argparse options."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def non_negative_int(value: str) -> int:
    """Parse a non-negative integer for argparse options."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError("cannot be negative")
    return parsed


def temperature_value(value: str) -> float:
    """Parse a Gemini temperature value for argparse options."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number") from exc
    if not 0 <= parsed <= 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


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
    parser.add_argument(
        "--chunk-size",
        type=positive_int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Maximum characters per text chunk. Default: {DEFAULT_CHUNK_SIZE}.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=non_negative_int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Characters of overlap between chunks. Default: {DEFAULT_CHUNK_OVERLAP}.",
    )
    parser.add_argument(
        "--retrieval-k",
        type=positive_int,
        default=4,
        help="Number of relevant chunks to retrieve for each question. Default: 4.",
    )
    parser.add_argument(
        "--chat-model",
        help="Override GEMINI_CHAT_MODEL for this run.",
    )
    parser.add_argument(
        "--embedding-model",
        help="Override GEMINI_EMBEDDING_MODEL for this run.",
    )
    parser.add_argument(
        "--temperature",
        type=temperature_value,
        default=0.2,
        help="Gemini response temperature from 0 to 1. Default: 0.2.",
    )
    parser.add_argument(
        "--hide-sources",
        action="store_true",
        help="Do not print retrieved source snippets after each answer.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print a full traceback when an error occurs.",
    )
    return parser


def prompt_for_document() -> Path:
    while True:
        value = input("Document path: ").strip().strip('"')
        if value:
            return Path(value)
        print("Please provide a document path.")


def format_cli_snippet(text: str, max_length: int = SOURCE_SNIPPET_LENGTH) -> str:
    """Normalize and truncate a source snippet for terminal output."""
    snippet = " ".join(str(text).split())
    if max_length <= 0:
        return ""
    if len(snippet) > max_length:
        if max_length <= 3:
            return snippet[:max_length]
        snippet = f"{snippet[: max_length - 3].rstrip()}..."
    return snippet


def format_cli_sources(source_documents, max_sources: int = 4) -> list[str]:
    """Format retrieved documents as concise terminal source lines."""
    sources: list[str] = []
    seen: set[tuple[str, str, str]] = set()

    for document in source_documents:
        metadata = document.metadata or {}
        fallback_filename = Path(str(metadata.get("source") or "Document")).name
        filename = str(metadata.get("filename") or fallback_filename or "Document")
        page = str(metadata.get("page", "-"))
        section = str(metadata.get("section", "")).strip()
        key = (filename, page, section)
        if key in seen:
            continue
        seen.add(key)

        label_parts = [filename, f"page/section {page}"]
        if section and section not in {page, f"Page {page}"}:
            label_parts.append(section)

        snippet = format_cli_snippet(document.page_content)
        label = " | ".join(label_parts)
        sources.append(f"{label} - {snippet}" if snippet else label)
        if len(sources) >= max_sources:
            break

    return sources


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must be smaller than --chunk-size")

    try:
        load_environment()
        document_path = Path(args.document) if args.document else prompt_for_document()

        print("Loading document...")
        loaded = load_document(document_path)

        print("Splitting text into chunks...")
        chunks = split_documents(
            loaded.documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        document_hash = file_sha256(loaded.file_path)
        collection_name = safe_collection_name(
            ["lumina", loaded.file_path.stem, document_hash[:16]]
        )

        print(f"Creating embeddings for {len(chunks)} chunks...")
        vector_store = create_vector_store(
            chunks=chunks,
            collection_name=collection_name,
            persist_directory=args.persist_dir,
            embedding_model=args.embedding_model,
        )
        qa_chain = create_qa_chain(
            vector_store,
            k=args.retrieval_k,
            model=args.chat_model,
            temperature=args.temperature,
        )

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
            if not args.hide_sources:
                sources = format_cli_sources(
                    response.get("source_documents", []),
                    max_sources=args.retrieval_k,
                )
                if sources:
                    print("Sources:")
                    for index, source in enumerate(sources, start=1):
                        print(f"  [{index}] {source}")
                    print()

    except KeyboardInterrupt:
        print("\nGoodbye.")
        return 0
    except Exception as exc:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
