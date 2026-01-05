"""
Embeddings Module
-----------------
Loads HuggingFace embedding models for vector search.
"""

from langchain_huggingface import HuggingFaceEmbeddings


def load_embedding_model(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
):
    """
    Load and return a HuggingFace embedding model.

    Args:
        model_name (str): Sentence-transformers model name

    Returns:
        HuggingFaceEmbeddings instance
    """
    return HuggingFaceEmbeddings(
        model_name=model_name
    )
