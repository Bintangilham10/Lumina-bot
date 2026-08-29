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
    audit_filename_fields,
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
    DEFAULT_MAX_AUTH_ATTEMPTS_PER_MINUTE,
    DEFAULT_MAX_GLOBAL_AUTH_ATTEMPTS_PER_MINUTE,
    DEFAULT_MAX_QUESTIONS_PER_MINUTE,
    DEFAULT_MAX_GLOBAL_QUESTIONS_PER_MINUTE,
    MAX_AUTH_ATTEMPTS_PER_MINUTE_ENV_VAR,
    MAX_GLOBAL_AUTH_ATTEMPTS_PER_MINUTE_ENV_VAR,
    MAX_GLOBAL_QUESTIONS_PER_MINUTE_ENV_VAR,
    MAX_QUESTIONS_PER_MINUTE_ENV_VAR,
    RATE_LIMIT_WINDOW_SECONDS,
    active_rate_limit_timestamps,
    check_global_auth_rate_limit,
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
USER_SAFE_ERROR_MESSAGES = {
    "document_processing": (
        "Gagal memproses dokumen. Periksa format, ukuran, lalu coba lagi."
    ),
    "question_answering": (
        "Gagal menjawab pertanyaan. Coba lagi dalam beberapa saat."
    ),
}


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
        initial_sidebar_state="auto",
    )
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
            .main .block-container {
                padding-top: 2rem;
                max-width: 1240px;
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

            :root {
                --lumina-ink: #171914;
                --lumina-muted: #73776d;
                --lumina-paper: #f5f4ef;
                --lumina-white: #fffef9;
                --lumina-line: #dedfd6;
                --lumina-lime: #d8f36a;
                --lumina-lime-deep: #9aae36;
                --lumina-sidebar: #1b1d19;
            }

            html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
            .stApp { background: var(--lumina-paper); color: var(--lumina-ink); }
            [data-testid="stHeader"] { background: transparent; }
            [data-testid="stToolbar"] { opacity: 0.55; }
            .main .block-container { padding: 3.25rem 4rem 7rem; max-width: 1240px; }

            [data-testid="stSidebar"] {
                background: var(--lumina-sidebar);
                border-right: 1px solid #30332d;
            }

            [data-testid="stSidebar"] > div:first-child {
                background: var(--lumina-sidebar);
                padding: 1.4rem 1.15rem 2rem;
            }

            [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
            [data-testid="stSidebar"] label,
            [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
                color: #c9cec1;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
                background: #242720;
                border: 1px dashed #59604f;
                border-radius: 14px;
                padding: 1rem 0.9rem;
            }

            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] small { color: #899080; }
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] > div:last-child { color: #899080 !important; }
    
            [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button {
                color: var(--lumina-ink);
                background: var(--lumina-lime);
                border: 0;
                border-radius: 999px;
                font-weight: 800;
            }

            [data-testid="stSidebar"] [data-testid="stExpander"] {
                background: #22251f;
                border: 1px solid #363b31;
                border-radius: 13px;
            }

            [data-testid="stSidebar"] [data-testid="stExpander"] summary p {
                color: #f2f4e9;
                font-weight: 700;
            }

            [data-testid="stSidebar"] input,
            [data-testid="stSidebar"] [data-baseweb="select"] > div {
                background: #2a2d26;
                color: #f2f4e9;
                border-color: #4a5142;
            }

            [data-testid="stSidebar"] [data-testid="stButton"] button {
                background: transparent;
                color: #b9c0b0;
                border: 1px solid #42483b;
                border-radius: 999px;
            }

            [data-testid="stSidebar"] [data-testid="stButton"] button:hover {
                border-color: var(--lumina-lime);
                color: var(--lumina-lime);
            }

            .lumina-sidebar-brand {
                display: flex;
                align-items: center;
                gap: 0.7rem;
                padding: 0.3rem 0.25rem 2.25rem;
            }

            .lumina-brand-mark {
                display: grid;
                place-items: center;
                width: 2.2rem;
                height: 2.2rem;
                border-radius: 9px;
                background: var(--lumina-lime);
                color: var(--lumina-ink);
                font-size: 1.2rem;
                font-weight: 800;
                transform: rotate(-5deg);
            }

            .lumina-brand-name { color: #f4f7eb; font-size: 1rem; font-weight: 800; letter-spacing: -0.03em; }
            .lumina-brand-note { color: #899080; font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; }
            .lumina-sidebar-kicker { color: #737d6c; font-family: 'DM Mono', monospace; font-size: 0.65rem; letter-spacing: 0.12em; margin: 0.4rem 0 0.55rem; text-transform: uppercase; }

            .lumina-main-header {
                display: flex;
                align-items: flex-start;
                justify-content: space-between;
                gap: 2rem;
                margin: 0 auto 2.6rem;
                max-width: 930px;
            }

            .lumina-eyebrow { color: var(--lumina-lime-deep); font-family: 'DM Mono', monospace; font-size: 0.68rem; letter-spacing: 0.14em; margin-bottom: 0.8rem; text-transform: uppercase; }
            .lumina-main-header h1 { color: var(--lumina-ink); font-size: clamp(2.4rem, 4.4vw, 4.3rem); font-weight: 800; letter-spacing: -0.075em; line-height: 0.98; margin: 0; max-width: 680px; }
            .lumina-header-copy { color: var(--lumina-muted); font-size: 0.93rem; line-height: 1.65; margin: 1.15rem 0 0; max-width: 480px; }
            .lumina-header-index { border-top: 1px solid var(--lumina-line); color: var(--lumina-muted); font-family: 'DM Mono', monospace; font-size: 0.68rem; line-height: 1.5; min-width: 155px; padding-top: 0.65rem; text-align: right; }
            .lumina-header-index strong { color: var(--lumina-ink); display: block; font-family: 'Manrope', sans-serif; font-size: 0.78rem; margin-bottom: 0.18rem; }

            .lumina-empty {
                background: var(--lumina-white);
                border: 1px solid var(--lumina-line);
                border-radius: 22px;
                margin: 0 auto;
                max-width: 930px;
                overflow: hidden;
            }

            .lumina-empty-top {
                align-items: flex-end;
                background: var(--lumina-ink);
                color: #f7f8ef;
                display: flex;
                justify-content: space-between;
                min-height: 185px;
                padding: 2rem 2.15rem;
            }

            .lumina-empty-top h2 { font-size: 1.9rem; letter-spacing: -0.055em; line-height: 1.02; margin: 0; max-width: 370px; }
            .lumina-empty-number { color: var(--lumina-lime); font-family: 'DM Mono', monospace; font-size: 3.3rem; letter-spacing: -0.1em; line-height: 0.8; }
            .lumina-empty-body { color: var(--lumina-muted); font-size: 0.9rem; line-height: 1.65; padding: 1.5rem 2.15rem 1.8rem; }
            .lumina-empty-body p { margin: 0 0 1.35rem; max-width: 560px; }

            .lumina-feature-grid { border-top: 1px solid var(--lumina-line); display: grid; grid-template-columns: repeat(3, 1fr); }
            .lumina-feature { border-right: 1px solid var(--lumina-line); padding: 1.05rem 1.2rem 1.2rem; }
            .lumina-feature:last-child { border-right: 0; }
            .lumina-feature-index { color: var(--lumina-lime-deep); font-family: 'DM Mono', monospace; font-size: 0.65rem; }
            .lumina-feature strong { color: var(--lumina-ink); display: block; font-size: 0.82rem; margin: 0.55rem 0 0.25rem; }
            .lumina-feature span { color: var(--lumina-muted); display: block; font-size: 0.72rem; line-height: 1.45; }

            .lumina-document-strip {
                align-items: center;
                background: var(--lumina-white);
                border: 1px solid var(--lumina-line);
                border-radius: 14px;
                display: flex;
                justify-content: space-between;
                margin: 0 auto 1.4rem;
                max-width: 930px;
                padding: 0.85rem 1.1rem;
            }

            .lumina-document-name { color: var(--lumina-ink); font-size: 0.84rem; font-weight: 800; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
            .lumina-document-meta { color: var(--lumina-muted); font-family: 'DM Mono', monospace; font-size: 0.67rem; white-space: nowrap; }
            .lumina-status-pill { background: #eef7c9; border: 1px solid #d2e894; border-radius: 999px; color: #5e741d; font-family: 'DM Mono', monospace; font-size: 0.63rem; margin-left: 0.8rem; padding: 0.3rem 0.55rem; white-space: nowrap; }

            .lumina-meta {
                background: #242720;
                border: 1px solid #363b31;
                border-radius: 13px;
                color: #aeb7a4;
                font-size: 0.76rem;
                line-height: 1.5;
                margin-bottom: 0.75rem;
                padding: 0.9rem 1rem;
            }

            .lumina-meta strong { color: #eef2e6; font-size: 0.67rem; letter-spacing: 0.04em; text-transform: uppercase; }
            .lumina-meta-file { border-bottom: 1px solid #3b4136; display: flex; flex-direction: column; gap: 0.2rem; padding-bottom: 0.8rem; }
            .lumina-meta-file span { color: #8e9888; font-family: 'DM Mono', monospace; font-size: 0.68rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
            .lumina-meta-grid { display: grid; gap: 0.8rem 0.65rem; grid-template-columns: repeat(2, 1fr); padding: 0.85rem 0; }
            .lumina-meta-grid span { color: #7f8979; display: block; font-family: 'DM Mono', monospace; font-size: 0.61rem; line-height: 1.25; text-transform: uppercase; }
            .lumina-meta-grid strong { color: #eef2e6; display: block; font-size: 0.83rem; margin-top: 0.2rem; }
            .lumina-meta-footer { border-top: 1px solid #3b4136; color: #8e9888; font-family: 'DM Mono', monospace; font-size: 0.62rem; line-height: 1.45; padding-top: 0.75rem; word-break: break-word; }
            .lumina-ready { color: var(--lumina-lime); font-family: 'DM Mono', monospace; font-size: 0.68rem; letter-spacing: 0.04em; margin: 0.6rem 0 0.9rem; }
            .lumina-ready span { color: #8ca835; font-size: 0.9rem; margin-right: 0.25rem; }

            [data-testid="stChatMessage"] { border-top: 1px solid var(--lumina-line); margin: 0 auto; max-width: 930px; padding: 1.45rem 0; }
            [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p { color: var(--lumina-ink); font-size: 0.92rem; line-height: 1.72; }
            [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) { background: rgba(255, 254, 249, 0.55); }
            [data-testid="stChatMessageAvatarUser"] { background: var(--lumina-lime); border-radius: 9px; color: var(--lumina-ink); }
            [data-testid="stChatMessageAvatarAssistant"] { background: var(--lumina-ink); border-radius: 9px; color: var(--lumina-lime); }
            [data-testid="stChatInput"] { margin: 1rem auto 0; max-width: 930px; }
            [data-testid="stChatInput"] textarea { background: var(--lumina-white); border: 1px solid #c8cabf; border-radius: 14px; color: var(--lumina-ink); font-size: 0.88rem; }
            [data-testid="stChatInput"] textarea:focus { border-color: var(--lumina-lime-deep); box-shadow: 0 0 0 3px rgba(216, 243, 106, 0.28); }
            [data-testid="stExpander"] { border-color: var(--lumina-line); border-radius: 12px; }

            @media (max-width: 800px) {
                .main .block-container { padding: 2.2rem 1.15rem 5rem; }
                .lumina-main-header { display: block; margin-bottom: 1.8rem; }
                .lumina-header-index { margin-top: 1.5rem; text-align: left; }
                .lumina-empty-top { min-height: 150px; padding: 1.4rem; }
                .lumina-empty-body { padding: 1.35rem 1.4rem 1.5rem; }
                .lumina-feature-grid { grid-template-columns: 1fr; }
                .lumina-feature { border-bottom: 1px solid var(--lumina-line); border-right: 0; }
                .lumina-feature:last-child { border-bottom: 0; }
                .lumina-document-strip { align-items: flex-start; flex-direction: column; gap: 0.55rem; }
                .lumina-status-pill { margin-left: 0.35rem; }
            }

            @media (max-width: 520px) {
                .main .block-container { padding: 1.5rem 0.85rem 4rem; }
                .lumina-main-header h1 { font-size: clamp(2.15rem, 10vw, 3.1rem); letter-spacing: -0.07em; }
                .lumina-header-copy { font-size: 0.84rem; }
                .lumina-header-index { font-size: 0.62rem; min-width: 0; }
                .lumina-empty-top { gap: 0.7rem; min-height: 0; padding: 1.25rem; }
                .lumina-empty-top h2 { font-size: 1.55rem; }
                .lumina-empty-number { font-size: 2.5rem; }
                .lumina-empty-body { font-size: 0.82rem; padding: 1.2rem 1.25rem 1.4rem; }
                .lumina-feature { padding: 0.9rem 1rem 1rem; }
                .lumina-document-meta { font-size: 0.61rem; }
            }

            :focus-visible { outline: 3px solid rgba(154, 174, 54, 0.55); outline-offset: 2px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def user_safe_error_message(operation: str) -> str:
    """Return a generic message that does not expose internal exception details."""
    return USER_SAFE_ERROR_MESSAGES.get(
        operation,
        "Terjadi kesalahan. Coba lagi dalam beberapa saat.",
    )


def initialize_state() -> None:
    defaults = {
        "authenticated": False,
        "messages": [],
        "qa_chain": None,
        "document_meta": None,
        "processed_file_id": None,
        "auth_attempt_timestamps": [],
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


def document_embedding_cache_key(
    file_hash: str,
    settings: AppSettings,
) -> tuple[str, int, int, str]:
    """Return the inputs that determine document embeddings."""
    return (
        file_hash,
        settings.chunk_size,
        settings.chunk_overlap,
        settings.embedding_model,
    )


@st.cache_resource(show_spinner=False)
def get_cached_vector_store(
    file_hash: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    _chunks: tuple,
):
    """Create or reuse a process-local vector store for identical embeddings."""
    collection_name = document_collection_name(
        "uploaded",
        file_hash,
        chunk_size,
        chunk_overlap,
        embedding_model or None,
    )
    return create_vector_store(
        list(_chunks),
        collection_name=collection_name,
        persist_directory=None,
        embedding_model=embedding_model or None,
    )


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
            render_processing_step(status, progress, 3)
            vector_store = get_cached_vector_store(
                *document_embedding_cache_key(file_hash, settings),
                tuple(chunks),
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
                **audit_filename_fields(uploaded_file.name),
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
                **audit_filename_fields(uploaded_file.name),
                error_type=type(exc).__name__,
                processing_duration_ms=duration_ms(processing_started_at),
            )
            status.update(label="Pemrosesan dokumen gagal.", state="error", expanded=True)
            raise
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def render_settings_controls() -> AppSettings:
    with st.expander("Tuning indeks", expanded=False):
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
        now = time.time()
        allowed, timestamps, retry_after = evaluate_auth_attempt_limit(
            list(st.session_state.auth_attempt_timestamps),
            now,
            int_from_env(
                MAX_AUTH_ATTEMPTS_PER_MINUTE_ENV_VAR,
                DEFAULT_MAX_AUTH_ATTEMPTS_PER_MINUTE,
            ),
        )
        st.session_state.auth_attempt_timestamps = timestamps
        if not allowed:
            audit_event("auth_rate_limited", retry_after_seconds=retry_after)
            st.warning(
                f"Terlalu banyak percobaan masuk. Coba lagi dalam {retry_after} detik."
            )
            return False

        global_allowed, global_retry_after = check_global_auth_rate_limit(
            now,
            int_from_env(
                MAX_GLOBAL_AUTH_ATTEMPTS_PER_MINUTE_ENV_VAR,
                DEFAULT_MAX_GLOBAL_AUTH_ATTEMPTS_PER_MINUTE,
            ),
            RATE_LIMIT_WINDOW_SECONDS,
        )
        if not global_allowed:
            audit_event(
                "auth_global_rate_limited",
                retry_after_seconds=global_retry_after,
            )
            st.warning(
                f"Layanan sedang sibuk. Coba lagi dalam {global_retry_after} detik."
            )
            return False

        if not verify_password(password, expected_password):
            audit_event("auth_failure")
            st.error("Password tidak cocok.")
            return False

        st.session_state.authenticated = True
        st.session_state.auth_attempt_timestamps = []
        audit_event("auth_success")
        st.rerun()
    return False


def evaluate_auth_attempt_limit(
    attempt_timestamps: list[float],
    now: float,
    max_attempts: int,
) -> tuple[bool, list[float], int]:
    """Check password-attempt rate limits without reaching into Streamlit state."""
    return check_rate_limit(
        attempt_timestamps,
        now,
        max_attempts,
        RATE_LIMIT_WINDOW_SECONDS,
    )


def evaluate_question_rate_limit(
    session_timestamps: list[float],
    now: float,
    max_questions: int,
    max_global_questions: int,
) -> tuple[bool, list[float], int, str | None]:
    """Check question limits while avoiding session quota use on global blocks."""
    session_allowed, session_candidate, session_retry_after = check_rate_limit(
        session_timestamps,
        now,
        max_questions,
        RATE_LIMIT_WINDOW_SECONDS,
    )
    if not session_allowed:
        return False, session_candidate, session_retry_after, "session"

    global_allowed, global_retry_after = check_global_rate_limit(
        now,
        max_global_questions,
        RATE_LIMIT_WINDOW_SECONDS,
    )
    if not global_allowed:
        active_session_timestamps = active_rate_limit_timestamps(
            session_timestamps,
            now,
            RATE_LIMIT_WINDOW_SECONDS,
        )
        return False, active_session_timestamps, global_retry_after, "global"

    return True, session_candidate, 0, None


def rate_limit_question() -> bool:
    now = time.time()
    max_questions = int_from_env(
        MAX_QUESTIONS_PER_MINUTE_ENV_VAR,
        DEFAULT_MAX_QUESTIONS_PER_MINUTE,
    )
    max_global_questions = int_from_env(
        MAX_GLOBAL_QUESTIONS_PER_MINUTE_ENV_VAR,
        DEFAULT_MAX_GLOBAL_QUESTIONS_PER_MINUTE,
    )
    allowed, timestamps, retry_after, limit_scope = evaluate_question_rate_limit(
        list(st.session_state.question_timestamps),
        now,
        max_questions,
        max_global_questions,
    )
    st.session_state.question_timestamps = timestamps
    if allowed:
        return True

    if limit_scope == "session":
        audit_event("question_rate_limited", retry_after_seconds=retry_after)
        st.warning(f"Terlalu banyak pertanyaan. Coba lagi dalam {retry_after} detik.")
        return False

    audit_event(
        "question_global_rate_limited",
        retry_after_seconds=retry_after,
    )
    st.warning(
        f"Layanan sedang sibuk. Coba lagi dalam {retry_after} detik."
    )
    return False


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="lumina-sidebar-brand">
                <div class="lumina-brand-mark">L</div>
                <div>
                    <div class="lumina-brand-name">Lumina Doc</div>
                    <div class="lumina-brand-note">document intelligence</div>
                </div>
            </div>
            <div class="lumina-sidebar-kicker">01 / Workspace</div>
            """,
            unsafe_allow_html=True,
        )
        settings = render_settings_controls()
        st.markdown('<div class="lumina-sidebar-kicker">03 / Add source</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Tambah dokumen",
            type=["pdf", "docx", "epub"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            try:
                process_uploaded_document(uploaded_file, settings)
                st.markdown(
                    '<div class="lumina-ready"><span>●</span> DOCUMENT READY FOR QUESTIONS</div>',
                    unsafe_allow_html=True,
                )
            except Exception:
                reset_document_state()
                st.error(user_safe_error_message("document_processing"))

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
            st.markdown('<div class="lumina-sidebar-kicker">02 / Indexed document</div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="lumina-meta">
                    <div class="lumina-meta-file">
                        <strong>{escape(meta["filename"])}</strong>
                        <span>{escape(meta["file_type"])} document</span>
                    </div>
                    <div class="lumina-meta-grid">
                        <div><span>Pages / parts</span><strong>{meta["total_pages"]}</strong></div>
                        <div><span>Text chunks</span><strong>{meta["total_chunks"]}</strong></div>
                        <div><span>Chunk / overlap</span><strong>{chunk_size} / {chunk_overlap}</strong></div>
                        <div><span>Top-k sources</span><strong>{retrieval_k}</strong></div>
                        <div><span>Min. relevance</span><strong>{min_relevance_score:.2f}</strong></div>
                    </div>
                    <div class="lumina-meta-footer">CHAT&nbsp;&nbsp;{chat_model}<br>EMBED&nbsp;{embedding_model}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button("Bersihkan percakapan", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_main_header(meta: dict | None = None) -> None:
    if meta:
        filename = escape(str(meta.get("filename", "Dokumen aktif")))
        pages = meta.get("total_pages", "—")
        chunks = meta.get("total_chunks", "—")
        st.markdown(
            f"""
            <div class="lumina-main-header">
                <div>
                    <div class="lumina-eyebrow">Lumina Doc / Research desk</div>
                    <h1>Ajukan pertanyaan.<br>Temukan intinya.</h1>
                    <p class="lumina-header-copy">Jawaban dirangkum dari dokumen aktif, dengan jejak sumber yang bisa kamu buka kembali kapan saja.</p>
                </div>
                <div class="lumina-header-index"><strong>SESSION 01</strong>CONTEXT-FIRST<br>ANSWERING</div>
            </div>
            <div class="lumina-document-strip">
                <div class="lumina-document-name">{filename}</div>
                <div><span class="lumina-document-meta">{pages} halaman · {chunks} chunks</span><span class="lumina-status-pill">READY</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <div class="lumina-main-header">
            <div>
                <div class="lumina-eyebrow">Lumina Doc / Research desk</div>
                <h1>Baca lebih cepat.<br>Tanyakan apa saja.</h1>
                <p class="lumina-header-copy">Ruang kerja untuk memahami dokumen panjang tanpa kehilangan konteks. Unggah satu file, lalu biarkan isinya memandu percakapan.</p>
            </div>
            <div class="lumina-header-index"><strong>WORKSPACE 01</strong>PRIVATE DOCUMENT<br>ANALYSIS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_chat() -> None:
    meta = st.session_state.document_meta
    render_main_header(meta)

    if st.session_state.qa_chain is None:
        st.markdown(
            """
            <div class="lumina-empty">
                <div class="lumina-empty-top">
                    <h2>Mulai dari satu dokumen yang ingin kamu pahami.</h2>
                    <div class="lumina-empty-number">01</div>
                </div>
                <div class="lumina-empty-body">
                    <p>Pilih PDF, DOCX, atau EPUB dari panel kiri. Lumina akan mengindeks isinya secara lokal sebelum mengirim konteks yang relevan ke model.</p>
                    <div class="lumina-feature-grid">
                        <div class="lumina-feature"><div class="lumina-feature-index">A / 01</div><strong>Context-first</strong><span>Jawaban fokus pada isi dokumen aktif.</span></div>
                        <div class="lumina-feature"><div class="lumina-feature-index">B / 02</div><strong>Source trail</strong><span>Telusuri halaman dan potongan sumber.</span></div>
                        <div class="lumina-feature"><div class="lumina-feature-index">C / 03</div><strong>Three formats</strong><span>PDF, DOCX, dan EPUB siap dibaca.</span></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
            answer = user_safe_error_message("question_answering")
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
