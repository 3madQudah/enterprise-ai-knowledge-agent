🏢 Enterprise AI Knowledge Agent

An enterprise-grade Retrieval-Augmented Generation (RAG) system that enables users to upload their own documents and ask grounded, hallucination-resistant questions through an intelligent multi-agent architecture.

This project demonstrates how to build a production-oriented AI knowledge system using modern LLM tooling, with a strong focus on answer grounding, controllability, and enterprise use cases.

✨ Key Features

📂 Upload & Ask (Dynamic Documents)
Each user can upload their own documents at runtime and immediately query them.

🧠 Multi-Agent Architecture

Router Agent (intent & domain routing)

Search Agent (RAG-based Q&A)

Summary Agent (document summarization)

🧩 Two-Step Answering (Hallucination Control)

Extract only relevant sentences from retrieved context

Generate final answers strictly from extracted content

🎯 Rule-Based Grounding
High-recall rule-based extraction for policies, procedures, and factual documents.

🏢 Domain & Department Awareness
Supports routing across domains such as:

HR / Policies

Sales / Business

Engineering / Technical

Personal / Profile documents

📏 Output Guardrails

Maximum 5 bullet points

Concise answers

No document repetition

No unsupported information

🖥️ Streamlit UI

Simple upload interface

Real-time question answering

Designed for demos and prototypes

🏗️ System Architecture
User Question
      ↓
Router Agent (intent + domain routing)
      ↓
Search Agent
      ↓
Retriever (FAISS + embeddings)
      ↓
Rule-Based Extraction
      ↓
LLM Grounded Answer Generation
      ↓
Final Controlled Answer

📁 Project Structure
enterprise_ai_agent/
│
├── agents/
│   ├── router_agent.py
│   ├── search_agent.py
│   ├── summary_agent.py
│   └── intent_classifier.py
│
├── rag/
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── retriever.py
│   ├── two_step_answering.py
│   ├── context_utils.py
│   └── answer_validator.py
│
├── ui/
│   └── app.py
│
├── main.py
├── requirements.txt
└── README.md

⚙️ Tech Stack

Python

LangChain (LCEL)

HuggingFace Transformers

Sentence-Transformers (Embeddings)

FAISS (Vector Store)

Streamlit (UI)

🚀 Getting Started
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Run the Streamlit App
streamlit run ui/app.py

3️⃣ Usage

Upload one or more .txt documents

Ask questions related only to uploaded content

Receive grounded, concise answers

Example questions:

What is the company code of conduct?

What is the person's name?

Summarize the uploaded document.

🧪 Example Use Cases

Internal company knowledge base

HR policy assistant

Personal document Q&A

Compliance and governance tools

AI-powered document search demos

🔒 Design Principles

No fine-tuning required

User-controlled data (no external knowledge leakage)

Strong hallucination mitigation

Enterprise-friendly modular design

📌 Future Improvements

PDF and DOCX support

User authentication & document isolation

PII masking and permission controls

Streaming responses

Cloud deployment (Docker / AWS / GCP)

🧑‍💼 Author

Emad Qudah
AI / Data / Machine Learning Enthusiast

⭐ Final Note

This project is designed to demonstrate real-world, production-oriented AI system thinking, not just a simple chatbot.
It emphasizes control, grounding, and architecture, which are critical for enterprise AI applications.
