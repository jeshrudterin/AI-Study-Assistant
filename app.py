import streamlit as st
import os
import tempfile
import time

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

import pytesseract
from pdf2image import convert_from_path

os.environ["OLLAMA_NO_CUDA"] = "1"

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="AI Assistant", layout="centered")

st.markdown("""
<style>
.block-container { max-width: 760px; padding-top: 2rem; }
h1 { font-size: 1.7rem !important; }
</style>
""", unsafe_allow_html=True)

st.title("🤖 AI Assistant")
st.caption("Study • Resume Analysis • Mock Interviews")

# ---------------- EMBEDDINGS ----------------
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

embeddings = load_embeddings()

# ---------------- SESSION STATE ----------------
defaults = {
    "messages": [],
    "vectorstore": None,
    "resume_text": None,
    "interview_question": None,
    "start_time": None
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------- SIDEBAR ----------------
st.sidebar.header("🧠 Mode")

mode = st.sidebar.radio(
    "Select Mode",
    ["📚 Study Assistant", "📄 Resume Analyzer", "🎤 Mock Interview"]
)

st.sidebar.markdown("---")

if st.sidebar.button("🧹 Clear All"):
    for k in defaults:
        st.session_state[k] = defaults[k]
    st.sidebar.success("Cleared")

# ======================================================
# 📚 STUDY ASSISTANT
# ======================================================
if mode == "📚 Study Assistant":

    uploaded_file = st.sidebar.file_uploader("📄 Upload PDF (optional)", type=["pdf"])

    def load_default_vectorstore():
        return FAISS.load_local(
            "faiss_index",
            embeddings,
            allow_dangerous_deserialization=True
        )

    if st.session_state.vectorstore is None and uploaded_file is None:
        if os.path.exists("faiss_index"):
            st.session_state.vectorstore = load_default_vectorstore()

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            path = tmp.name

        docs = PyPDFLoader(path).load()
        splitter = RecursiveCharacterTextSplitter(1000, 200)
        chunks = splitter.split_documents(docs)

        if not chunks:
            images = convert_from_path(path)
            ocr_text = "".join(pytesseract.image_to_string(img) for img in images)
            chunks = splitter.split_documents([Document(page_content=ocr_text)])

        st.session_state.vectorstore = FAISS.from_documents(chunks, embeddings)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask anything...")

    if question and st.session_state.vectorstore:
        st.session_state.messages.append({"role": "user", "content": question})

        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 1})
        docs = retriever.invoke(question)
        context = "\n\n".join(d.page_content for d in docs)

        prompt = ChatPromptTemplate.from_template(
            """
            Use document context first.
            If not present, answer using general knowledge.

            Context:
            {context}

            Question:
            {question}
            """
        )

        llm = OllamaLLM(model="mistral", temperature=0.3)
        response = (prompt | llm | StrOutputParser()).invoke({
            "context": context,
            "question": question
        })

        with st.chat_message("assistant"):
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

# ======================================================
# 📄 RESUME ANALYZER (FIXED)
# ======================================================
if mode == "📄 Resume Analyzer":

    st.subheader("📄 Resume Analyzer & Interview Coach")
    st.caption("Upload your resume to get role-based feedback")

    role = st.selectbox(
        "🎯 Target Role",
        ["Data Analyst", "Backend Developer", "AI / ML Engineer"]
    )

    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])

    if resume_file:
        with st.spinner("📄 Reading resume..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(resume_file.read())
                path = tmp.name

            docs = PyPDFLoader(path).load()
            resume_text = "\n".join(d.page_content for d in docs)

            if not resume_text.strip():
                images = convert_from_path(path)
                resume_text = "".join(pytesseract.image_to_string(img) for img in images)

            if not resume_text.strip():
                st.error("❌ Unable to extract resume text.")
                st.stop()

            st.session_state.resume_text = resume_text

        st.success("✅ Resume loaded")

        prompt = ChatPromptTemplate.from_template(
            """
            You are an experienced technical recruiter.

            Analyze the resume for the role of: {role}

            Provide:
            1. Resume Score (0–10)
            2. Strengths
            3. Weak areas / missing skills
            4. Actionable improvement suggestions
            5. 5 interview questions

            Resume:
            {resume}
            """
        )

        llm = OllamaLLM(model="mistral", temperature=0.3)
        chain = prompt | llm | StrOutputParser()

        st.markdown("### 🧠 Resume Analysis")
        output = chain.invoke({
            "resume": resume_text,
            "role": role
        })

        st.markdown(output)

# ======================================================
# 🎤 MOCK INTERVIEW SYSTEM
# ======================================================
if mode == "🎤 Mock Interview":

    interview_type = st.selectbox(
        "Interview Type",
        ["Resume-Based Interview", "Domain-Based Interview"]
    )

    TIME_LIMIT = 120
    llm = OllamaLLM(model="mistral", temperature=0.4)

    if interview_type == "Domain-Based Interview":
        domain = st.selectbox("Domain", ["DSA", "OOP", "SQL"])
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"])

        if st.button("🎯 Ask Question"):
            prompt = ChatPromptTemplate.from_template(
                "Ask ONE {difficulty} {domain} interview question. No answer."
            )
            st.session_state.interview_question = (prompt | llm | StrOutputParser()).invoke({
                "domain": domain,
                "difficulty": difficulty
            })
            st.session_state.start_time = time.time()

    else:
        if not st.session_state.resume_text:
            st.warning("Upload resume in Resume Analyzer first.")
            st.stop()

        if st.button("🎯 Ask Resume-Based Question"):
            prompt = ChatPromptTemplate.from_template(
                "Ask ONE interview question based on this resume:\n{resume}"
            )
            st.session_state.interview_question = (prompt | llm | StrOutputParser()).invoke({
                "resume": st.session_state.resume_text
            })
            st.session_state.start_time = time.time()

    if st.session_state.interview_question:
        st.markdown("### ❓ Interview Question")
        st.markdown(st.session_state.interview_question)

        elapsed = time.time() - st.session_state.start_time
        remaining = max(0, TIME_LIMIT - int(elapsed))
        st.warning(f"⏱️ Time Remaining: {remaining} seconds")

        answer = st.text_area("✍️ Your Answer", disabled=remaining == 0)

        if st.button("🧠 Submit") or remaining == 0:
            eval_prompt = ChatPromptTemplate.from_template(
                """
                Evaluate the candidate answer.

                Provide:
                - Score (0–10)
                - Feedback
                - Improvements
                - Ideal answer (brief)

                Question:
                {question}

                Answer:
                {answer}
                """
            )

            result = (eval_prompt | llm | StrOutputParser()).invoke({
                "question": st.session_state.interview_question,
                "answer": answer
            })

            st.markdown("### 📊 Evaluation")
            st.markdown(result)

            st.session_state.interview_question = None
