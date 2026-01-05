"""
Summary Agent
-------------
Summarizes retrieved documents.
"""

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


SUMMARY_PROMPT = PromptTemplate.from_template(
    """
Summarize the following content clearly and concisely.
Focus on key points only.
Do not add new information.

Content:
{context}

Summary:
"""
)


def summary_agent(llm, context: str) -> str:
    chain = SUMMARY_PROMPT | llm | StrOutputParser()
    return chain.invoke({"context": context})