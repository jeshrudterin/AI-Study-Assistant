import pickle
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# 1. Load chunks created by ingest.py
with open("chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"Total chunks loaded: {len(chunks)}")

if len(chunks) == 0:
    raise ValueError("❌ No chunks found. Run ingest.py first.")

# Debug: show sample metadata
print("Sample chunk metadata:", chunks[0].metadata)

# 2. Local embedding model (FREE)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 3. Store in FAISS
vectorstore = FAISS.from_documents(chunks, embeddings)

# 4. Save vector DB
vectorstore.save_local("faiss_index")

print("✅ FAISS vector store created using LOCAL embeddings")
