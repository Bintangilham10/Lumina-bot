"""Retrieval QA chatbot logic powered by Gemini."""

from __future__ import annotations

import os
from dataclasses import dataclass
from collections.abc import Iterator

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
) -> DocumentQaChain:
    """Create a retrieval QA flow from a Chroma vector store."""
    if k <= 0:
        raise ValueError("retrieval k must be greater than 0.")
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return DocumentQaChain(
        retriever=retriever,
        llm=create_llm(model=model, temperature=temperature),
    )


def retrieve_documents(qa_chain: DocumentQaChain, question: str) -> list[Document]:
    """Retrieve relevant source documents for a question."""
    question = _normalize_question(question)
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
    context = format_documents_context(source_documents)
    prompt = QA_PROMPT.format(context=context, question=question)
    llm = _chain_llm(qa_chain)

    return _stream_llm_text(llm, prompt), source_documents


def ask_question(qa_chain: DocumentQaChain, question: str) -> dict:
    """Ask a question and return the retrieval QA response."""
    question = _normalize_question(question)
    source_documents = retrieve_documents(qa_chain, question)
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
