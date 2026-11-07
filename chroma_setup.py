
import chromadb

client = chromadb.PersistentClient(
    path="./vectorstore"
)

collection = client.get_or_create_collection(
    name="document_analysis",
    metadata={"hnsw:space": "cosine"}
)

print("ChromaDB connected successfully!")
print("Collection:", collection.name)
print("Documents:", collection.count())
