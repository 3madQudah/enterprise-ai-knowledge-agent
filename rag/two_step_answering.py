"""
Two-Step Answering Module
-------------------------
Hallucination-resistant RAG answering using a hybrid approach.

Step 1 (Rule-Based): Extract relevant policy sentences using keyword rules
                     to guarantee coverage (high recall).
Step 2 (LLM-Based): Generate the final answer using ONLY the extracted sentences.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re


# -------------------------------------------------
# Step 1A: Rule-Based Extraction (High Recall + Policy Guardrail)
# -------------------------------------------------
def rule_based_extraction(context: str, keywords: list[str]) -> str:
    """
    Final adaptive rule-based extraction for enterprise policies.
    Balances precision with guaranteed recall.
    """

    if not keywords:
        keywords = []

    extracted = []

    candidates = re.split(
        r'(?:\n|\r|(?<=\.)\s+|(?<=\d\.)\s+)',
        context
    )

    policy_indicators = (
        "must", "should", "required",
        "ethical", "ethically", "honestly",
        "harassment", "discrimination",
        "conflict", "conflict of interest",
        "disclose",
        "prohibited", "strictly prohibited",
        "comply", "compliance",
        "violation", "violations",
        "disciplinary", "reported"
    )

    # -------- Pass 1: Strict (precision) --------
    for chunk in candidates:
        cl = chunk.lower()
        if any(k in cl for k in keywords) and any(p in cl for p in policy_indicators):
            cleaned = chunk.strip(" -•\t")
            if cleaned and cleaned not in extracted:
                extracted.append(cleaned)

    # -------- Pass 2: Fallback (recall guarantee) --------
    if len(extracted) < 3:
        for chunk in candidates:
            cl = chunk.lower()
            if any(k in cl for k in keywords):
                cleaned = chunk.strip(" -•\t")
                if cleaned and cleaned not in extracted:
                    extracted.append(cleaned)

    return "\n".join(extracted)


# -------------------------------------------------
# Step 2: Final Answer Prompt (Strict & Grounded)
# -------------------------------------------------

FINAL_ANSWER_PROMPT = PromptTemplate.from_template(
    """
Answer the question using ONLY the extracted sentences below.

Rules:
- Maximum 5 bullet points
- Each bullet must be 20 words or less
- No document repetition
- No commentary or extra explanations
- Do NOT add new information

Extracted Sentences:
{extracted}

Final Answer:
"""
)


def generate_final_answer(llm, extracted_sentences: str) -> str:
    """
    Step 2: Generate the final grounded answer from extracted sentences.
    """
    chain = FINAL_ANSWER_PROMPT | llm | StrOutputParser()
    return chain.invoke({
        "extracted": extracted_sentences
    })