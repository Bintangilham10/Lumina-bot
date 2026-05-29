"""Retrieval QA chatbot logic powered by Gemini."""

from __future__ import annotations

import os
from collections.abc import Iterator

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI


CHAT_MODEL_ENV_VAR = "GEMINI_CHAT_MODEL"
DEFAULT_CHAT_MODEL = "gemini-3.5-flash"

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are Lumina Doc, a helpful AI assistant for document question answering.\n"
        "Answer in the same language as the user's question. If the question is in Bahasa "
        "Indonesia, answer in Bahasa Indonesia.\n"
        "Use only the provided document context. If the answer is not available in the "
        "context, say that the information was not found in the document.\n\n"
        "Context:\n{context}\n\n"
        "Question: {question}\n"
        "Answer:"
    ),
)


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
) -> RetrievalQA:
    """Create a RetrievalQA chain from a Chroma vector store."""
    if k <= 0:
        raise ValueError("retrieval k must be greater than 0.")
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return RetrievalQA.from_chain_type(
        llm=create_llm(model=model, temperature=temperature),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )


def retrieve_documents(qa_chain: RetrievalQA, question: str) -> list[Document]:
    """Retrieve relevant source documents for a question."""
    question = _normalize_question(question)
    retriever = qa_chain.retriever
    if hasattr(retriever, "invoke"):
        return list(retriever.invoke(question))
    return list(retriever.get_relevant_documents(question))


def stream_question(
    qa_chain: RetrievalQA,
    question: str,
) -> tuple[Iterator[str], list[Document]]:
    """Stream an answer while returning the source documents used for context."""
    question = _normalize_question(question)
    source_documents = retrieve_documents(qa_chain, question)
    context = format_documents_context(source_documents)
    prompt = QA_PROMPT.format(context=context, question=question)
    llm = _chain_llm(qa_chain)

    return _stream_llm_text(llm, prompt), source_documents


def ask_question(qa_chain: RetrievalQA, question: str) -> dict:
    """Ask a question and return the RetrievalQA response."""
    question = _normalize_question(question)
    return qa_chain.invoke({"query": question})


def format_documents_context(documents: list[Document]) -> str:
    """Join retrieved documents into the same context shape used by the QA prompt."""
    return "\n\n".join(document.page_content for document in documents)


def _chain_llm(qa_chain: RetrievalQA) -> ChatGoogleGenerativeAI:
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
