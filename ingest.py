from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import pickle

import pytesseract
from pdf2image import convert_from_path


PDF_PATH = "data/sample.pdf"

# 1. Try normal PDF loading (text-based PDFs)
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f"Total pages loaded: {len(documents)}")

# 2. Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

# 3. OCR fallback (for scanned PDFs)
if len(chunks) == 0:
    print("⚠️ No text extracted. Using OCR fallback...")

    images = convert_from_path(PDF_PATH)

    documents = []
    for page_num, img in enumerate(images):
        text = pytesseract.image_to_string(img)

        documents.append(
            Document(
                page_content=text,
                metadata={"page": page_num}
            )
        )

    chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")

# 4. Safety check
if len(chunks) == 0:
    raise ValueError("❌ No text extracted even after OCR.")

# 5. Preview one chunk
print("\nSample chunk:\n")
print(chunks[0].page_content[:500])

# 6. Save chunks
with open("chunks.pkl", "wb") as f:
    pickle.dump(chunks, f)

print("✅ Chunks saved to chunks.pkl")
