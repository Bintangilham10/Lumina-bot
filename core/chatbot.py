"""Retrieval QA chatbot logic powered by Gemini."""

from __future__ import annotations

import os
from collections.abc import Iterator
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.sources import format_source_context


CHAT_MODEL_ENV_VAR = "GEMINI_CHAT_MODEL"
DEFAULT_CHAT_MODEL = "gemini-3.5-flash"

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are Lumina Doc, a helpful AI assistant for document question answering.\n"
        "Answer in the same language as the user's question. If the question is in Bahasa "
        "Indonesia, answer in Bahasa Indonesia.\n"
        "Use only the provided document context. If the answer is not available in the "
        "context, say that the information was not found in the document.\n"
        "When source labels like [1] or [2] are present, cite the relevant source after "
        "the sentence that uses it.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)


@dataclass
class DocumentQaChain:
    """Small retrieval QA container independent of deprecated LangChain chains."""

    retriever: object
    llm: ChatGoogleGenerativeAI
    vector_store: Chroma | None = None
    retrieval_k: int = 4
    min_relevance_score: float | None = None


def resolve_chat_model(model: str | None = None) -> str:
    """Resolve the chat model from an explicit value, environment, or default."""
    resolved_model = model or os.getenv(CHAT_MODEL_ENV_VAR) or DEFAULT_CHAT_MODEL
    return resolved_model.strip() or DEFAULT_CHAT_MODEL


def create_llm(
    model: str | None = None,
    temperature: float = 0.2,
) -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model."""
    if not 0 <= temperature <= 1:
        raise ValueError("temperature must be between 0 and 1.")
    return ChatGoogleGenerativeAI(model=resolve_chat_model(model), temperature=temperature)


def create_qa_chain(
    vector_store: Chroma,
    k: int = 4,
    model: str | None = None,
    temperature: float = 0.2,
    min_relevance_score: float | None = None,
) -> DocumentQaChain:
    """Create a retrieval QA flow from a Chroma vector store."""
    if k <= 0:
        raise ValueError("retrieval k must be greater than 0.")
    min_relevance_score = _normalize_min_relevance_score(min_relevance_score)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return DocumentQaChain(
        retriever=retriever,
        llm=create_llm(model=model, temperature=temperature),
        vector_store=vector_store,
        retrieval_k=k,
        min_relevance_score=min_relevance_score,
    )


def retrieve_documents(qa_chain: DocumentQaChain, question: str) -> list[Document]:
    """Retrieve relevant source documents for a question."""
    question = _normalize_question(question)
    scored_documents = _retrieve_scored_documents(qa_chain, question)
    if scored_documents is not None:
        return scored_documents

    retriever = qa_chain.retriever
    if hasattr(retriever, "invoke"):
        return list(retriever.invoke(question))
    return list(retriever.get_relevant_documents(question))


def stream_question(
    qa_chain: DocumentQaChain,
    question: str,
) -> tuple[Iterator[str], list[Document]]:
    """Stream an answer while returning the source documents used for context."""
    question = _normalize_question(question)
    source_documents = retrieve_documents(qa_chain, question)
    if not source_documents:
        return iter([_not_found_answer(question)]), source_documents

    context = format_documents_context(source_documents)
    prompt = QA_PROMPT.format(context=context, question=question)
    llm = _chain_llm(qa_chain)

    return _stream_llm_text(llm, prompt), source_documents


def ask_question(qa_chain: DocumentQaChain, question: str) -> dict:
    """Ask a question and return the retrieval QA response."""
    question = _normalize_question(question)
    source_documents = retrieve_documents(qa_chain, question)
    if not source_documents:
        return {
            "query": question,
            "result": _not_found_answer(question),
            "source_documents": source_documents,
        }

    context = format_documents_context(source_documents)
    prompt = QA_PROMPT.format(context=context, question=question)
    response = _chain_llm(qa_chain).invoke(prompt)

    return {
        "query": question,
        "result": _chunk_text(response).strip(),
        "source_documents": source_documents,
    }


def format_documents_context(documents: list[Document]) -> str:
    """Join retrieved documents into a numbered citation context."""
    return format_source_context(documents)


def _chain_llm(qa_chain: DocumentQaChain) -> ChatGoogleGenerativeAI:
    if hasattr(qa_chain, "llm"):
        return qa_chain.llm
    try:
        return qa_chain.combine_documents_chain.llm_chain.llm
    except AttributeError as exc:
        raise ValueError("QA chain does not expose an LLM for streaming.") from exc


def _normalize_question(question: str) -> str:
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    return question


def _normalize_min_relevance_score(score: float | None) -> float | None:
    if score is None:
        return None
    if not 0 <= score <= 1:
        raise ValueError("min_relevance_score must be between 0 and 1.")
    return score if score > 0 else None


def _retrieve_scored_documents(
    qa_chain: DocumentQaChain,
    question: str,
) -> list[Document] | None:
    min_relevance_score = getattr(qa_chain, "min_relevance_score", None)
    vector_store = getattr(qa_chain, "vector_store", None)
    if min_relevance_score is None or vector_store is None:
        return None
    if not hasattr(vector_store, "similarity_search_with_relevance_scores"):
        return None

    results = vector_store.similarity_search_with_relevance_scores(
        question,
        k=getattr(qa_chain, "retrieval_k", 4),
    )
    documents: list[Document] = []
    for document, score in results:
        relevance_score = float(score)
        if relevance_score < min_relevance_score:
            continue
        documents.append(
            Document(
                page_content=document.page_content,
                metadata={
                    **(document.metadata or {}),
                    "relevance_score": relevance_score,
                },
            )
        )
    return documents


def _not_found_answer(question: str) -> str:
    if _looks_indonesian(question):
        return "Informasi tersebut tidak ditemukan di dokumen."
    return "The information was not found in the document."


def _looks_indonesian(text: str) -> bool:
    words = {word.strip(".,?!:;()[]{}\"'").lower() for word in text.split()}
    indonesian_markers = {
        "apa",
        "siapa",
        "kapan",
        "dimana",
        "mana",
        "mengapa",
        "kenapa",
        "bagaimana",
        "berapa",
        "jelaskan",
        "sebutkan",
        "dokumen",
        "dalam",
        "adalah",
        "yang",
    }
    return bool(words & indonesian_markers)


def _stream_llm_text(llm: ChatGoogleGenerativeAI, prompt: str) -> Iterator[str]:
    for chunk in llm.stream(prompt):
        text = _chunk_text(chunk)
        if text:
            yield text


def _chunk_text(chunk) -> str:
    content = getattr(chunk, "content", chunk)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "".join(parts)
    return str(content)
