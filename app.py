"""Streamlit web UI for Lumina Doc."""

from __future__ import annotations

import hashlib
import tempfile
from html import escape
from pathlib import Path

import streamlit as st

from core.chatbot import create_qa_chain, stream_question
from core.embedder import create_vector_store
from core.loader import load_document
from core.splitter import split_documents
from utils.helpers import load_environment, safe_collection_name
from utils.sources import build_source_references, normalize_source_snippet


APP_TITLE = "Lumina Doc — Chatbot Dokumen Cerdas"
SOURCE_SNIPPET_LENGTH = 280
PROCESSING_STEPS: tuple[tuple[int, str], ...] = (
    (10, "Menyimpan file sementara..."),
    (30, "Membaca isi dokumen..."),
    (55, "Memecah teks menjadi chunk..."),
    (80, "Membuat embedding dan indeks pencarian..."),
    (95, "Menyiapkan sesi tanya jawab..."),
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
        "messages": [],
        "qa_chain": None,
        "document_meta": None,
        "processed_file_id": None,
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


def process_uploaded_document(uploaded_file) -> None:
    file_hash = uploaded_file_hash(uploaded_file)
    file_id = f"{uploaded_file.name}-{uploaded_file.size}-{file_hash}"
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
            chunks = split_documents(loaded.documents)
            collection_name = safe_collection_name(
                ["lumina", Path(uploaded_file.name).stem, file_hash[:16]]
            )

            render_processing_step(status, progress, 3)
            vector_store = create_vector_store(
                chunks,
                collection_name=collection_name,
                persist_directory=None,
            )

            render_processing_step(status, progress, 4)
            st.session_state.qa_chain = create_qa_chain(vector_store)
            st.session_state.document_meta = {
                "filename": uploaded_file.name,
                "file_type": loaded.file_type,
                "total_pages": loaded.total_pages,
                "total_chunks": len(chunks),
            }
            st.session_state.processed_file_id = file_id
            st.session_state.messages = []
            progress.progress(100, text="Dokumen siap ditanyakan.")
            status.update(label="Dokumen selesai diproses.", state="complete", expanded=False)
        except Exception:
            status.update(label="Pemrosesan dokumen gagal.", state="error", expanded=True)
            raise
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Dokumen")
        uploaded_file = st.file_uploader(
            "Unggah PDF, DOCX, atau EPUB",
            type=["pdf", "docx", "epub"],
            accept_multiple_files=False,
        )

        if uploaded_file is not None:
            try:
                process_uploaded_document(uploaded_file)
                st.success("Dokumen siap ditanyakan.")
            except Exception as exc:
                reset_document_state()
                st.error(f"Gagal memproses dokumen: {exc}")

        meta = st.session_state.document_meta
        if meta:
            st.subheader("Metadata")
            st.markdown(
                f"""
                <div class="lumina-meta">
                    <strong>Nama file</strong><br>{escape(meta["filename"])}<br><br>
                    <strong>Format</strong><br>{escape(meta["file_type"])}<br><br>
                    <strong>Total halaman/bagian</strong><br>{meta["total_pages"]}<br><br>
                    <strong>Total chunk</strong><br>{meta["total_chunks"]}
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

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Mencari jawaban di dokumen..."):
            answer_stream, source_documents = stream_question(
                st.session_state.qa_chain,
                question,
            )
            sources = format_sources(source_documents)
            answer = st.write_stream(answer_stream).strip()
            if sources:
                with st.expander("Sumber jawaban"):
                    for source in sources:
                        st.markdown(source, unsafe_allow_html=True)

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
        snippet_html = (
            f'<div class="lumina-source-snippet">{escape(reference.snippet)}</div>'
            if reference.snippet
            else ""
        )
        sources.append(
            '<div class="lumina-source">'
            f'<div class="lumina-source-title">{title}</div>'
            f"Halaman/bagian: {page}{section_html}{snippet_html}</div>"
        )

    return sources


def format_source_snippet(text: str, max_length: int = SOURCE_SNIPPET_LENGTH) -> str:
    return escape(normalize_source_snippet(text, max_length))


def main() -> None:
    configure_page()
    initialize_state()

    try:
        load_environment()
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
