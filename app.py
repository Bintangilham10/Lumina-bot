"""Streamlit web UI for Lumina Doc."""

from __future__ import annotations

import hashlib
import time
import tempfile
from dataclasses import dataclass
from html import escape
from pathlib import Path

import streamlit as st

from core.chatbot import create_qa_chain, resolve_chat_model, stream_question
from core.embedder import create_vector_store, resolve_embedding_model
from core.loader import load_document
from core.splitter import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, split_documents
from utils.audit import (
    audit_event,
    document_text_stats,
    duration_ms,
    estimate_token_count,
)
from utils.helpers import (
    DEFAULT_MAX_CHUNKS,
    DEFAULT_MAX_FILE_SIZE_MB,
    DEFAULT_MAX_PAGES,
    document_collection_name,
    load_environment,
    validate_document_limits,
    validate_file_size,
)
from utils.security import (
    ALLOWED_CHAT_MODELS_ENV_VAR,
    ALLOWED_EMBEDDING_MODELS_ENV_VAR,
    DEFAULT_MAX_QUESTIONS_PER_MINUTE,
    DEFAULT_MAX_GLOBAL_QUESTIONS_PER_MINUTE,
    MAX_GLOBAL_QUESTIONS_PER_MINUTE_ENV_VAR,
    MAX_QUESTIONS_PER_MINUTE_ENV_VAR,
    RATE_LIMIT_WINDOW_SECONDS,
    check_global_rate_limit,
    check_rate_limit,
    configured_model_options,
    configured_password,
    int_from_env,
    verify_password,
)
from utils.sources import build_source_references, normalize_source_snippet


APP_TITLE = "Lumina Doc — Chatbot Dokumen Cerdas"
SOURCE_SNIPPET_LENGTH = 280
DEFAULT_RETRIEVAL_K = 4
DEFAULT_TEMPERATURE = 0.2
DEFAULT_MIN_RELEVANCE_SCORE = 0.0
PROCESSING_STEPS: tuple[tuple[int, str], ...] = (
    (10, "Menyimpan file sementara..."),
    (30, "Membaca isi dokumen..."),
    (55, "Memecah teks menjadi chunk..."),
    (80, "Membuat embedding dan indeks pencarian..."),
    (95, "Menyiapkan sesi tanya jawab..."),
)


@dataclass(frozen=True)
class AppSettings:
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    min_relevance_score: float
    chat_model: str
    embedding_model: str
    temperature: float
    max_file_size_mb: int
    max_pages: int
    max_chunks: int

    def cache_key(self) -> str:
        return "|".join(
            [
                str(self.chunk_size),
                str(self.chunk_overlap),
                str(self.retrieval_k),
                f"{self.min_relevance_score:g}",
                self.chat_model,
                self.embedding_model,
                f"{self.temperature:g}",
                str(self.max_file_size_mb),
                str(self.max_pages),
                str(self.max_chunks),
            ]
        )


def configure_page() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        """
        <style>
            .main .block-container {
                padding-top: 2rem;
                max-width: 1100px;
            }
            [data-testid="stSidebar"] {
                background: #f7f9fb;
            }
            .lumina-meta {
                border: 1px solid #e6eaf0;
                border-radius: 8px;
                padding: 0.85rem 1rem;
                background: #ffffff;
                margin-bottom: 0.75rem;
            }
            .lumina-source {
                color: #52616f;
                font-size: 0.88rem;
                line-height: 1.45;
                border-left: 3px solid #2f80ed;
                padding: 0.65rem 0 0.65rem 0.75rem;
                margin-bottom: 0.7rem;
                background: #f8fbff;
            }
            .lumina-source-title {
                color: #17212b;
                font-weight: 700;
                margin-bottom: 0.2rem;
            }
            .lumina-source-snippet {
                margin-top: 0.45rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def initialize_state() -> None:
    defaults = {
        "authenticated": False,
        "messages": [],
        "qa_chain": None,
        "document_meta": None,
        "processed_file_id": None,
        "question_timestamps": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_document_state() -> None:
    st.session_state.messages = []
    st.session_state.qa_chain = None
    st.session_state.document_meta = None
    st.session_state.processed_file_id = None


def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        return Path(temp_file.name)


def uploaded_file_hash(uploaded_file) -> str:
    return hashlib.sha256(uploaded_file.getbuffer()).hexdigest()


def render_processing_step(status, progress, step_index: int) -> None:
    percent, label = PROCESSING_STEPS[step_index]
    status.write(label)
    progress.progress(percent, text=label)


def process_uploaded_document(uploaded_file, settings: AppSettings) -> None:
    processing_started_at = time.perf_counter()
    file_size_attr = getattr(uploaded_file, "size", None)
    file_size = int(
        file_size_attr if file_size_attr is not None else len(uploaded_file.getbuffer())
    )
    validate_file_size(file_size, settings.max_file_size_mb)
    file_hash = uploaded_file_hash(uploaded_file)
    file_id = f"{uploaded_file.name}-{file_size}-{file_hash}-{settings.cache_key()}"
    if st.session_state.processed_file_id == file_id:
        return

    temp_path: Path | None = None
    with st.status("Memproses dokumen...", expanded=True) as status:
        progress = st.progress(0, text="Menyiapkan dokumen...")
        try:
            render_processing_step(status, progress, 0)
            temp_path = save_uploaded_file(uploaded_file)

            render_processing_step(status, progress, 1)
            loaded = load_document(temp_path)

            render_processing_step(status, progress, 2)
            chunks = split_documents(
                loaded.documents,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
            chunk_stats = document_text_stats(chunks)
            validate_document_limits(
                total_pages=loaded.total_pages,
                total_chunks=len(chunks),
                max_pages=settings.max_pages,
                max_chunks=settings.max_chunks,
            )
            collection_name = document_collection_name(
                Path(uploaded_file.name).stem,
                file_hash,
                settings.chunk_size,
                settings.chunk_overlap,
                settings.embedding_model or None,
            )

            render_processing_step(status, progress, 3)
            vector_store = create_vector_store(
                chunks,
                collection_name=collection_name,
                persist_directory=None,
                embedding_model=settings.embedding_model or None,
            )

            render_processing_step(status, progress, 4)
            st.session_state.qa_chain = create_qa_chain(
                vector_store,
                k=settings.retrieval_k,
                model=settings.chat_model or None,
                temperature=settings.temperature,
                min_relevance_score=settings.min_relevance_score or None,
            )
            st.session_state.document_meta = {
                "filename": uploaded_file.name,
                "file_type": loaded.file_type,
                "total_pages": loaded.total_pages,
                "total_chunks": len(chunks),
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "retrieval_k": settings.retrieval_k,
                "min_relevance_score": settings.min_relevance_score,
                "chat_model": settings.chat_model,
                "embedding_model": settings.embedding_model,
            }
            audit_event(
                "document_processed",
                filename=uploaded_file.name,
                file_type=loaded.file_type,
                total_pages=loaded.total_pages,
                total_chunks=len(chunks),
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                retrieval_k=settings.retrieval_k,
                min_relevance_score=settings.min_relevance_score,
                processing_duration_ms=duration_ms(processing_started_at),
                indexed_text_chars=chunk_stats["text_chars"],
                estimated_indexed_tokens=chunk_stats["estimated_tokens"],
            )
            st.session_state.processed_file_id = file_id
            st.session_state.messages = []
            progress.progress(100, text="Dokumen siap ditanyakan.")
            status.update(label="Dokumen selesai diproses.", state="complete", expanded=False)
        except Exception as exc:
            audit_event(
                "document_processing_error",
                filename=uploaded_file.name,
                error_type=type(exc).__name__,
                processing_duration_ms=duration_ms(processing_started_at),
            )
            status.update(label="Pemrosesan dokumen gagal.", state="error", expanded=True)
            raise
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def render_settings_controls() -> AppSettings:
    with st.expander("Pengaturan", expanded=False):
        chunk_size = int(
            st.number_input(
                "Ukuran chunk",
                min_value=100,
                max_value=8000,
                value=DEFAULT_CHUNK_SIZE,
                step=100,
            )
        )
        chunk_overlap = int(
            st.number_input(
                "Overlap chunk",
                min_value=0,
                max_value=max(0, chunk_size - 1),
                value=min(DEFAULT_CHUNK_OVERLAP, max(0, chunk_size - 1)),
                step=50,
            )
        )
        retrieval_k = int(
            st.number_input(
                "Top-k sumber",
                min_value=1,
                max_value=20,
                value=DEFAULT_RETRIEVAL_K,
                step=1,
            )
        )
        temperature = float(
            st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_TEMPERATURE,
                step=0.05,
            )
        )
        min_relevance_score = float(
            st.slider(
                "Minimum relevansi",
                min_value=0.0,
                max_value=1.0,
                value=DEFAULT_MIN_RELEVANCE_SCORE,
                step=0.05,
            )
        )
        chat_model_default = resolve_chat_model()
        chat_model = st.selectbox(
            "Model chat",
            options=configured_model_options(
                ALLOWED_CHAT_MODELS_ENV_VAR,
                chat_model_default,
            ),
            index=0,
        ).strip()
        embedding_model_default = resolve_embedding_model()
        embedding_model = st.selectbox(
            "Model embedding",
            options=configured_model_options(
                ALLOWED_EMBEDDING_MODELS_ENV_VAR,
                embedding_model_default,
            ),
            index=0,
        ).strip()
        max_file_size_mb = int(
            st.number_input(
                "Batas file (MB)",
                min_value=0,
                max_value=500,
                value=DEFAULT_MAX_FILE_SIZE_MB,
                step=5,
            )
        )
        max_pages = int(
            st.number_input(
                "Batas halaman/bagian",
                min_value=0,
                max_value=5000,
                value=DEFAULT_MAX_PAGES,
                step=50,
            )
        )
        max_chunks = int(
            st.number_input(
                "Batas chunk",
                min_value=0,
                max_value=10000,
                value=DEFAULT_MAX_CHUNKS,
                step=100,
            )
        )

    return AppSettings(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        retrieval_k=retrieval_k,
        min_relevance_score=min_relevance_score,
        chat_model=chat_model,
        embedding_model=embedding_model,
        temperature=temperature,
        max_file_size_mb=max_file_size_mb,
        max_pages=max_pages,
        max_chunks=max_chunks,
    )


def authenticate_session() -> bool:
    expected_password = configured_password()
    if not expected_password:
        return True
    if st.session_state.authenticated:
        return True

    st.title(APP_TITLE)
    password = st.text_input("Password", type="password")
    if st.button("Masuk", use_container_width=True):
        if verify_password(password, expected_password):
            st.session_state.authenticated = True
            audit_event("auth_success")
            st.rerun()
        audit_event("auth_failure")
        st.error("Password tidak cocok.")
    return False


def rate_limit_question() -> bool:
    now = time.time()
    max_questions = int_from_env(
        MAX_QUESTIONS_PER_MINUTE_ENV_VAR,
        DEFAULT_MAX_QUESTIONS_PER_MINUTE,
    )
    allowed, timestamps, retry_after = check_rate_limit(
        list(st.session_state.question_timestamps),
        now,
        max_questions,
        RATE_LIMIT_WINDOW_SECONDS,
    )
    st.session_state.question_timestamps = timestamps
    if not allowed:
        audit_event("question_rate_limited", retry_after_seconds=retry_after)
        st.warning(f"Terlalu banyak pertanyaan. Coba lagi dalam {retry_after} detik.")
        return False

    max_global_questions = int_from_env(
        MAX_GLOBAL_QUESTIONS_PER_MINUTE_ENV_VAR,
        DEFAULT_MAX_GLOBAL_QUESTIONS_PER_MINUTE,
    )
    global_allowed, global_retry_after = check_global_rate_limit(
        now,
        max_global_questions,
        RATE_LIMIT_WINDOW_SECONDS,
    )
    if global_allowed:
        return True

    audit_event(
        "question_global_rate_limited",
        retry_after_seconds=global_retry_after,
    )
    st.warning(
        f"Layanan sedang sibuk. Coba lagi dalam {global_retry_after} detik."
    )
    return False


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Dokumen")
        settings = render_settings_controls()
        uploaded_file = st.file_uploader(
            "Unggah PDF, DOCX, atau EPUB",
            type=["pdf", "docx", "epub"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            try:
                process_uploaded_document(uploaded_file, settings)
                st.success("Dokumen siap ditanyakan.")
            except Exception as exc:
                reset_document_state()
                st.error(f"Gagal memproses dokumen: {exc}")

        meta = st.session_state.document_meta
        if meta:
            chunk_size = meta.get("chunk_size", DEFAULT_CHUNK_SIZE)
            chunk_overlap = meta.get("chunk_overlap", DEFAULT_CHUNK_OVERLAP)
            retrieval_k = meta.get("retrieval_k", DEFAULT_RETRIEVAL_K)
            min_relevance_score = float(
                meta.get("min_relevance_score", DEFAULT_MIN_RELEVANCE_SCORE)
            )
            chat_model = escape(str(meta.get("chat_model", resolve_chat_model())))
            embedding_model = escape(
                str(meta.get("embedding_model", resolve_embedding_model()))
            )
            st.subheader("Metadata")
            st.markdown(
                f"""
                <div class="lumina-meta">
                    <strong>Nama file</strong><br>{escape(meta["filename"])}<br><br>
                    <strong>Format</strong><br>{escape(meta["file_type"])}<br><br>
                    <strong>Total halaman/bagian</strong><br>{meta["total_pages"]}<br><br>
                    <strong>Total chunk</strong><br>{meta["total_chunks"]}<br><br>
                    <strong>Chunk / overlap</strong><br>{chunk_size} / {chunk_overlap}<br><br>
                    <strong>Top-k sumber</strong><br>{retrieval_k}<br><br>
                    <strong>Minimum relevansi</strong><br>{min_relevance_score:.2f}<br><br>
                    <strong>Model chat</strong><br>{chat_model}<br><br>
                    <strong>Model embedding</strong><br>{embedding_model}
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("Bersihkan percakapan", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_chat() -> None:
    st.title(APP_TITLE)
    st.caption("Unggah dokumen, lalu tanyakan isi dokumen dalam Bahasa Indonesia atau English.")

    if st.session_state.qa_chain is None:
        st.info("Silakan unggah dokumen melalui sidebar untuk memulai.")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sumber jawaban"):
                    for source in message["sources"]:
                        st.markdown(source, unsafe_allow_html=True)

    question = st.chat_input("Tulis pertanyaan tentang dokumen...")
    if not question:
        return
    if not rate_limit_question():
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        answer_started_at = time.perf_counter()
        try:
            with st.spinner("Mencari jawaban di dokumen..."):
                answer_stream, source_documents = stream_question(
                    st.session_state.qa_chain,
                    question,
                )
                source_stats = document_text_stats(source_documents)
                sources = format_sources(source_documents)
                answer = str(st.write_stream(answer_stream)).strip()
                if sources:
                    with st.expander("Sumber jawaban"):
                        for source in sources:
                            st.markdown(source, unsafe_allow_html=True)
                audit_event(
                    "question_answered",
                    question_length=len(question),
                    question_estimated_tokens=estimate_token_count(question),
                    answer_length=len(answer),
                    answer_estimated_tokens=estimate_token_count(answer),
                    source_count=len(sources),
                    retrieved_context_chars=source_stats["text_chars"],
                    retrieved_context_estimated_tokens=source_stats["estimated_tokens"],
                    answer_duration_ms=duration_ms(answer_started_at),
                )
        except Exception as exc:
            sources = []
            answer = f"Gagal menjawab pertanyaan: {exc}"
            audit_event(
                "question_error",
                question_length=len(question),
                error_type=type(exc).__name__,
                answer_duration_ms=duration_ms(answer_started_at),
            )
            st.error(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


def format_sources(source_documents) -> list[str]:
    sources: list[str] = []
    references = build_source_references(
        list(source_documents),
        snippet_length=SOURCE_SNIPPET_LENGTH,
    )

    for reference in references:
        title = escape(f"[{reference.number}] {reference.filename}")
        page = escape(reference.page)
        section = escape(reference.section)
        section_html = f"<br>{section}" if section else ""
        score_html = (
            f"<br>Relevansi: {reference.relevance_score:.2f}"
            if reference.relevance_score is not None
            else ""
        )
        snippet_html = (
            f'<div class="lumina-source-snippet">{escape(reference.snippet)}</div>'
            if reference.snippet
            else ""
        )
        sources.append(
            '<div class="lumina-source">'
            f'<div class="lumina-source-title">{title}</div>'
            f"Halaman/bagian: {page}{section_html}{score_html}{snippet_html}</div>"
        )

    return sources


def format_source_snippet(text: str, max_length: int = SOURCE_SNIPPET_LENGTH) -> str:
    return escape(normalize_source_snippet(text, max_length))


def main() -> None:
    configure_page()
    initialize_state()

    if not authenticate_session():
        st.stop()
    try:
        load_environment()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
