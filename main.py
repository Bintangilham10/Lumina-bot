"""Command-line interface for Lumina Doc."""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Sequence
from pathlib import Path

from core.chatbot import ask_question, create_qa_chain
from core.embedder import (
    create_vector_store,
    load_vector_store,
    resolve_embedding_model,
    vector_store_document_count,
)
from core.loader import load_document
from core.splitter import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, split_documents
from utils.helpers import (
    DEFAULT_MAX_CHUNKS,
    DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_MAX_PAGES,
    document_collection_name,
    file_sha256,
    load_environment,
    supported_extensions_text,
    validate_document_limits,
    validate_file_size,
)
from utils.sources import build_source_references, normalize_source_snippet


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


def relevance_score_value(value: str) -> float:
    """Parse a relevance score threshold for argparse options."""
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
        "--rebuild-index",
        action="store_true",
        help="Recreate embeddings even when a persisted collection already exists.",
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
        "--min-relevance-score",
        type=relevance_score_value,
        default=0.0,
        help=(
            "Minimum retrieval relevance score from 0 to 1. "
            "Default: 0 disables score filtering."
        ),
    )
    parser.add_argument(
        "--hide-sources",
        action="store_true",
        help="Do not print retrieved source snippets after each answer.",
    )
    parser.add_argument(
        "--max-file-size-mb",
        type=non_negative_int,
        default=DEFAULT_MAX_FILE_SIZE_MB,
        help=(
            "Maximum document file size in MB before indexing. "
            f"Default: {DEFAULT_MAX_FILE_SIZE_MB}. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-pages",
        type=non_negative_int,
        default=DEFAULT_MAX_PAGES,
        help=(
            "Maximum pages/sections before indexing. "
            f"Default: {DEFAULT_MAX_PAGES}. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--max-chunks",
        type=non_negative_int,
        default=DEFAULT_MAX_CHUNKS,
        help=(
            "Maximum chunks before embedding. "
            f"Default: {DEFAULT_MAX_CHUNKS}. Use 0 to disable."
        ),
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
    return normalize_source_snippet(text, max_length)


def format_cli_sources(source_documents, max_sources: int = 4) -> list[str]:
    """Format retrieved documents as concise terminal source lines."""
    sources: list[str] = []
    references = build_source_references(
        list(source_documents),
        max_sources=max_sources,
        snippet_length=SOURCE_SNIPPET_LENGTH,
    )

    for reference in references:
        label_parts = [
            f"[{reference.number}] {reference.filename}",
            f"page/section {reference.page}",
        ]
        if reference.section and reference.section not in {
            reference.page,
            f"Page {reference.page}",
        }:
            label_parts.append(reference.section)
        if reference.relevance_score is not None:
            label_parts.append(f"relevance {reference.relevance_score:.2f}")

        label = " | ".join(label_parts)
        sources.append(f"{label} - {reference.snippet}" if reference.snippet else label)

    return sources


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.chunk_overlap >= args.chunk_size:
        parser.error("--chunk-overlap must be smaller than --chunk-size")

    try:
        load_environment()
        document_path = Path(args.document) if args.document else prompt_for_document()
        persist_dir = args.persist_dir or None
        embedding_model = resolve_embedding_model(args.embedding_model)
        input_path = document_path.expanduser()
        if input_path.exists() and input_path.is_file():
            validate_file_size(input_path.stat().st_size, args.max_file_size_mb)

        print("Loading document...")
        loaded = load_document(document_path)

        print("Splitting text into chunks...")
        chunks = split_documents(
            loaded.documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
        validate_document_limits(
            total_pages=loaded.total_pages,
            total_chunks=len(chunks),
            max_pages=args.max_pages,
            max_chunks=args.max_chunks,
        )
        document_hash = file_sha256(loaded.file_path)
        collection_name = document_collection_name(
            loaded.file_path.stem,
            document_hash,
            args.chunk_size,
            args.chunk_overlap,
            embedding_model,
        )

        vector_store = None
        if persist_dir is not None and not args.rebuild_index:
            existing_store = load_vector_store(
                collection_name=collection_name,
                persist_directory=persist_dir,
                embedding_model=embedding_model,
            )
            existing_count = vector_store_document_count(existing_store)
            if existing_count > 0:
                print(
                    f"Using existing embeddings for {existing_count} chunks "
                    f"from '{persist_dir}'."
                )
                vector_store = existing_store

        if vector_store is None:
            if persist_dir is not None and args.rebuild_index:
                existing_store = load_vector_store(
                    collection_name=collection_name,
                    persist_directory=persist_dir,
                    embedding_model=embedding_model,
                )
                if vector_store_document_count(existing_store) > 0:
                    print("Rebuilding existing vector collection...")
                    existing_store.delete_collection()

            print(f"Creating embeddings for {len(chunks)} chunks...")
            vector_store = create_vector_store(
                chunks=chunks,
                collection_name=collection_name,
                persist_directory=persist_dir,
                embedding_model=embedding_model,
            )

        qa_chain = create_qa_chain(
            vector_store,
            k=args.retrieval_k,
            model=args.chat_model,
            temperature=args.temperature,
            min_relevance_score=args.min_relevance_score or None,
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
                    for source in sources:
                        print(f"  {source}")
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
