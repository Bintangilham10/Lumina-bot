"""Retrieval QA chatbot logic powered by Gemini."""

from __future__ import annotations

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI


DEFAULT_CHAT_MODEL = "gemini-1.5-flash"

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


def create_llm(
    model: str = DEFAULT_CHAT_MODEL,
    temperature: float = 0.2,
) -> ChatGoogleGenerativeAI:
    """Create the Gemini chat model."""
    return ChatGoogleGenerativeAI(model=model, temperature=temperature)


def create_qa_chain(vector_store: Chroma, k: int = 4) -> RetrievalQA:
    """Create a RetrievalQA chain from a Chroma vector store."""
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return RetrievalQA.from_chain_type(
        llm=create_llm(),
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT},
    )


def ask_question(qa_chain: RetrievalQA, question: str) -> dict:
    """Ask a question and return the RetrievalQA response."""
    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")
    return qa_chain.invoke({"query": question})
