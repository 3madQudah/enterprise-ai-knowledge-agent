"""
Query Conditioning Module
-------------------------
Transforms user questions into search-optimized queries.
"""

def condition_query(question: str) -> str:
    """
    Expand and normalize the user query for better retrieval.
    """

    keywords = {
        "code of conduct": "ethical behavior confidentiality harassment compliance",
        "leave": "annual leave vacation policy",
        "api": "authentication authorization endpoints",
        "pricing": "pricing plans subscription cost"
    }

    conditioned_query = question.lower()

    for key, expansion in keywords.items():
        if key in conditioned_query:
            conditioned_query += " " + expansion

    return conditioned_query