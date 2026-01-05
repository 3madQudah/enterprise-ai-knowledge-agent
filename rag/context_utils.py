"""
Context Utilities
-----------------
Shared helpers for formatting retrieved documents.
"""

def format_docs(docs):
    """
    Convert a list of LangChain Document objects
    into a single text block.
    """
    return "\n\n".join(doc.page_content for doc in docs)