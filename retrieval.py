
import chromadb
from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path="./vectorstore"
)

collection = client.get_collection(
    name="document_analysis"
)


def search_documents(query, top_k=3):

    if collection.count() == 0:
        return []

    # Generate query embedding
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    # Retrieve relevant chunks
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = []

    for i, document in enumerate(results["documents"][0]):

        retrieved_chunks.append({
            "text": document,
            "source": results["metadatas"][0][i]["source"],
            "page": results["metadatas"][0][i]["page"],
            "distance": results["distances"][0][i]
        })

    return retrieved_chunks


if __name__ == "__main__":

    question = input("Ask a question about your PDF: ")

    results = search_documents(question)

    for index, result in enumerate(results, 1):

        print(f"\nResult {index}")
        print("Source:", result["source"])
        print("Page:", result["page"])
        print("Content:", result["text"])
        print("Distance:", result["distance"])
