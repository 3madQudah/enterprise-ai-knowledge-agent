"""
Router Agent
------------
Routes the query to the correct agent based on intent
and dynamically controls retrieval breadth.
"""

from agents.search_agent import search_agent
from agents.summary_agent import summary_agent
from agents.intent_classifier import classify_intent


def router_agent(
    query: str,
    vector_store,
    llm,
    department: str | None = None,
    keywords: list[str] | None = None,
    context: str | None = None
) -> str:
    """
    Route query to the appropriate agent.
    """

    query_lower = query.lower()
    intent = classify_intent(query)

    # -----------------------------
    # Summary intent
    # -----------------------------
    if intent == "summary" and context:
        return summary_agent(llm, context)

    # -----------------------------
    # Default SAFE values
    # -----------------------------
    k = 4

    # -----------------------------
    # Personal Profile Routing (FACT MODE)
    # -----------------------------
    if any(term in query_lower for term in ["name", "age", "phone", "email"]):
        department = "personal"
        keywords = None   # 🔥 IMPORTANT: keep None for direct fact extraction
        k = 4

    # -----------------------------
    # HR Policy / Code of Conduct
    # -----------------------------
    elif "code of conduct" in query_lower or "policy" in query_lower:
        department = "HR"
        keywords = [
            "ethical",
            "ethic",
            "harassment",
            "discrimination",
            "conflict",
            "interest",
            "confidential",
            "confidentiality",
            "disclose",
            "prohibited",
            "comply",
            "compliance",
            "violation",
            "violations",
            "disciplinary",
            "report"
        ]
        k = 12

    # -----------------------------
    # Fallback (generic search)
    # -----------------------------
    else:
        keywords = keywords or query_lower.split()

    # -----------------------------
    # Delegate to Search Agent
    # -----------------------------
    return search_agent(
        query=query,
        vector_store=vector_store,
        llm=llm,
        department=department,
        keywords=keywords,
        k=k
    )