"""
Vector Store Module
-------------------
Creates and manages the FAISS vector database
used for similarity search.
"""

from langchain_community.vectorstores import FAISS


def create_vector_store(documents, embeddings):
    """
    Create FAISS vector store from documents.

    Args:
        documents (list): List of document chunks
        embeddings: Embedding model

    Returns:
        FAISS: Vector store instance
    """
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embeddings
    )
    return vector_store