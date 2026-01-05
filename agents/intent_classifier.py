"""
Intent Classification Module
----------------------------
Determines how the answer should be structured.
"""

def classify_intent(question: str) -> str:
    q = question.lower()

    if "what is" in q or "define" in q:
        return "definition"
    if "policy" in q or "code of conduct" in q:
        return "policy"
    if "how" in q or "steps" in q:
        return "procedure"
    if "compare" in q:
        return "comparison"

    return "general"