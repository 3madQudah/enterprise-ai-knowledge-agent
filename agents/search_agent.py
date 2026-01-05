"""
Search Agent
------------
Handles precise question answering using RAG + Two-Step Answering.
"""

from rag.two_step_answering import (
    rule_based_extraction,
    generate_final_answer
)
from rag.retriever import get_retriever
from rag.context_utils import format_docs


def search_agent(
    query: str,
    vector_store,
    llm,
    department: str | None,
    keywords: list[str] | None,
    k: int = 4
) -> str:
    """
    Perform grounded RAG-based question answering.
    """

    # -----------------------------
    # Retrieve documents
    # -----------------------------
    retriever = get_retriever(
        vector_store=vector_store,
        k=k,
        department=department
    )

    docs = retriever.invoke(query)
    context = format_docs(docs)

    # -----------------------------
    # FACT MODE (Personal data)
    # -----------------------------
    if keywords is None:
        extracted = context  # 🔥 Direct grounding, no filtering

    # -----------------------------
    # RULE-BASED MODE (Policies / Knowledge)
    # -----------------------------
    else:
        extracted = rule_based_extraction(
            context=context,
            keywords=keywords
        )

    # -----------------------------
    # Final answer generation
    # -----------------------------
    final_answer = generate_final_answer(
        llm=llm,
        extracted_sentences=extracted
    )

    # -----------------------------
    # Hard guardrail: max 5 bullets
    # -----------------------------
    lines = [
        line for line in final_answer.split("\n")
        if line.strip()
    ]

    return "\n".join(lines[:5])