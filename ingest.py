
import fitz
import chromadb
import hashlib
from sentence_transformers import SentenceTransformer

# Load embedding model
print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="./vectorstore"
)

collection = client.get_or_create_collection(
    name="document_analysis",
    metadata={"hnsw:space": "cosine"}
)


def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []

    for page_number, page in enumerate(doc):
        text = page.get_text()

        if text.strip():
            pages.append({
                "text": text,
                "page": page_number + 1
            })

    doc.close()
    return pages


def chunk_text(text, chunk_size=1000, overlap=150):
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def ingest_pdf(pdf_path):
    print(f"Processing: {pdf_path}")

    pages = extract_text(pdf_path)

    documents = []
    metadatas = []
    ids = []

    for page in pages:
        chunks = chunk_text(page["text"])

        for chunk_number, chunk in enumerate(chunks):
            chunk_id = hashlib.sha256(
                f"{pdf_path}:{page['page']}:{chunk_number}:{chunk}".encode()
            ).hexdigest()

            documents.append(chunk)

            metadatas.append({
                "source": pdf_path,
                "page": page["page"],
                "chunk": chunk_number
            })

            ids.append(chunk_id)

    if not documents:
        print("No extractable text found.")
        return

    embeddings = model.encode(
        documents,
        normalize_embeddings=True
    ).tolist()

    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Successfully stored {len(documents)} chunks!")


if __name__ == "__main__":
    pdf_path = "documents/sample.pdf"
    ingest_pdf(pdf_path)
