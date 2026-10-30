from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"



# 1. Load embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load FAISS index
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

# 3. Load LLM
llm = OllamaLLM(
    model="mistral",
    temperature=0.3,
    num_predict=300   # LIMIT output length
)


# 4. Prompt (study mode)
prompt = ChatPromptTemplate.from_template(
    """
    You are a knowledgeable AI assistant, like ChatGPT.

    Instructions:
    1. First, check whether the answer is present in the provided document context.
    2. If the concept IS present in the document:
       - Use the document as the base truth
       - Explain the concept in depth, like a teacher
       - Add clear explanations, examples, and use cases where appropriate
       - You MAY use general knowledge to enrich the explanation,
         but do NOT contradict the document.
    3. If the concept is NOT present in the document:
       - Clearly say:
         "This information is not present in the uploaded document. Answering using general knowledge."
       - Then provide a correct and detailed explanation using general knowledge.
    4. Keep the explanation easy to understand and structured.

    Document Context:
    {context}

    Question:
    {question}
    """
)

print("🤖 PDF Chatbot Ready (with sources). Type 'exit' to quit.\n")

while True:
    question = input("You: ")

    if question.lower() == "exit":
        print("👋 Bye macha!")
        break

    # 5. Retrieve documents (NEW API)
    docs = retriever.invoke(question)


    context_text = "\n\n".join(doc.page_content for doc in docs)

    # 6. Generate answer
    print("\n🤖 Answer:\n", end="", flush=True)
    for chunk in (
        prompt
            | llm
            | StrOutputParser()
    ).stream({
        "context": context_text,
        "question": question
    }):
        print(chunk, end="", flush=True)

    print("\n")


    # # 7. Print answer
    # print("\n🤖 Answer:\n", answer)

    # 8. Print sources
    print("\n📄 Source(s):")
    for doc in docs:
        page = doc.metadata.get("page")
        if isinstance(page, int):
            print(f"- Page {page + 1}")
        else:
            print("- Page Unknown")

    print("\n" + "-"*50 + "\n")
