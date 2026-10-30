# 🤖 AI Study Assistant & Mock Interview System

A modular AI-powered assistant built using RAG (Retrieval-Augmented Generation) architecture.

## 🚀 Features

### 📚 Study Assistant
- Ask questions from preloaded knowledge
- Upload your own PDF
- OCR fallback for scanned documents
- Context-aware AI answers

### 📄 Resume Analyzer
- Upload resume (PDF)
- AI evaluates:
  - Resume score
  - Strengths
  - Weak areas
  - Suggested improvements
  - Interview questions

### 🎤 Mock Interview System
- Domain-based (DSA / OOP / SQL)
- Resume-based interview
- Time-limited responses
- AI scoring and feedback

---

## 🏗 Tech Stack

- **Python**
- **Streamlit** (UI)
- **LangChain** (AI orchestration)
- **FAISS** (Vector Database)
- **HuggingFace Embeddings**
- **Ollama (Mistral / LLaMA)** for local LLM
- **Tesseract OCR** for scanned PDFs

---

## 🧠 Architecture

PDF → Chunking → Embeddings → FAISS → Retrieval → LLM → Response

Implements full RAG pipeline with modular design.

---

## ⚙️ Setup Instructions

### 1️⃣ Clone Repository

```bash
git clone <your-repo-link>
cd AI-Study-Assistant


2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows



3️⃣ Install Requirements
pip install -r requirements.txt


4️⃣ Run the App
streamlit run app.py


🖥 Requirements

Python 3.10+

Ollama installed locally

mistral or llama3 model pulled:

ollama pull mistral