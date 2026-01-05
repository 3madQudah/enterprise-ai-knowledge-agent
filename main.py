"""
Enterprise AI Knowledge Agent
-----------------------------
Agentic RAG application using LangChain v1.x.

Responsibilities:
- Load documents + metadata
- Build vector store
- Load LLM
- Delegate all reasoning to Router Agent
"""

# -----------------------------
# Imports
# -----------------------------
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline
from rag.embeddings import load_embedding_model
from rag.vector_store import create_vector_store

from agents.router_agent import router_agent


# -----------------------------
# 1. Load Documents + Metadata
# -----------------------------
def load_documents():
    loader = DirectoryLoader(
        path="/Users/mac/Desktop/enterprise_ai_agent/data/documents",
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )

    docs = loader.load()

    for doc in docs:
        source = doc.metadata.get("source", "").lower()

        if any(k in source for k in ["code_of_conduct", "handbook", "policy"]):
            doc.metadata.update({
                "department": "HR",
                "document_type": "policy"
            })

        elif any(k in source for k in ["pricing", "sales"]):
            doc.metadata.update({
                "department": "Sales",
                "document_type": "business"
            })

        elif any(k in source for k in ["api", "technical"]):
            doc.metadata.update({
                "department": "Engineering",
                "document_type": "technical"
            })

    return docs


# -----------------------------
# 2. Split Documents
# -----------------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50
    )
    return splitter.split_documents(documents)


# -----------------------------
# 3. Load LLM
# -----------------------------
def load_llm():
    hf_pipeline = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)


# -----------------------------
# 4. Main Application
# -----------------------------
if __name__ == "__main__":

    print("Loading documents...")
    documents = load_documents()

    print("Splitting documents...")
    chunks = split_documents(documents)

    print("Loading embeddings...")
    embeddings = load_embedding_model()

    print("Creating vector store...")
    vector_store = create_vector_store(chunks, embeddings)

    print("Loading language model...")
    llm = load_llm()

    print("\n🤖 Enterprise AI Agent is ready! Type 'exit' to quit.\n")

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        final_answer = router_agent(
            query=query,
            vector_store=vector_store,
            llm=llm,
            department=None,   # Router decides
            keywords=None      # Router decides
        )

        print("\nAnswer:")
        print(final_answer)
        print("\n" + "-" * 50)