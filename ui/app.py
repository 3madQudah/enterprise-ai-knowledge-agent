"""
UI Application (Streamlit)
--------------------------
Enterprise AI Knowledge Agent UI
"""

import sys
import os
import shutil
import streamlit as st

# -----------------------------
# Add project root to PYTHONPATH
# -----------------------------
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from agents.router_agent import router_agent
from rag.embeddings import load_embedding_model
from rag.vector_store import create_vector_store

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline
from langchain_huggingface import HuggingFacePipeline

# -----------------------------
# Config
# -----------------------------
UPLOAD_DIR = "uploads/user_1"

# -----------------------------
# Helper functions
# -----------------------------
def prepare_upload_dir():
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)


def load_llm():
    hf_pipeline = pipeline(
        task="text2text-generation",
        model="google/flan-t5-base",
        max_new_tokens=256
    )
    return HuggingFacePipeline(pipeline=hf_pipeline)


# -----------------------------
# Streamlit Setup
# -----------------------------
st.set_page_config(
    page_title="Enterprise AI Knowledge Agent",
    layout="centered"
)

st.title("🏢 Enterprise AI Knowledge Agent")
st.caption("Upload your documents and ask grounded questions")

# -----------------------------
# Initialize session state
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "llm" not in st.session_state:
    st.session_state.llm = load_llm()

# -----------------------------
# Upload documents
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload documents (TXT only for now)",
    type=["txt"],
    accept_multiple_files=True
)

if uploaded_files:
    prepare_upload_dir()

    for file in uploaded_files:
        with open(os.path.join(UPLOAD_DIR, file.name), "wb") as f:
            f.write(file.getbuffer())

    # Load documents
    loader = DirectoryLoader(
        UPLOAD_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader
    )
    documents = loader.load()

    # ✅ ADD METADATA (THIS IS THE FIX)
    for doc in documents:
        filename = os.path.basename(doc.metadata.get("source", "")).lower()

        if "personal" in filename:
            doc.metadata["department"] = "personal"
            doc.metadata["document_type"] = "profile"
        else:
            doc.metadata["department"] = "general"
            doc.metadata["document_type"] = "knowledge"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = load_embedding_model()
    vector_store = create_vector_store(chunks, embeddings)

    # Save to session
    st.session_state.vector_store = vector_store

    st.success("Documents uploaded and indexed successfully!")

st.divider()

# -----------------------------
# Ask question
# -----------------------------
question = st.text_input("Ask a question about your documents:")

if question:
    if st.session_state.vector_store is None:
        st.warning("Please upload documents first.")
    else:
        with st.spinner("Thinking..."):
            answer = router_agent(
                query=question,
                vector_store=st.session_state.vector_store,
                llm=st.session_state.llm,
                department=None,
                keywords=None
            )

        st.subheader("Answer")
        st.write(answer)