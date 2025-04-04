import chromadb
from chromadb.config import Settings

# Set up ChromaDB client with persistent storage
client = chromadb.PersistentClient(
    path="./chroma_storage",  # Directory where the database will be stored
    settings=Settings(
        anonymized_telemetry=False,  # Disable telemetry if desired
        allow_reset=True  # Allow resetting the database if needed
    )
)

# Create a collection with cosine similarity as the default embedding function
collection = client.create_collection(
    name="example_collection",
    metadata={
        "hnsw:space": "cosine"  # Specify cosine similarity as the distance metric
    }
)

# # Example of adding documents to the collection
# documents = [
#     "The quick brown fox jumps over the lazy dog",
#     "A quick brown dog outpaces a lazy fox",
#     "Machine learning is fascinating and powerful"
# ]

# # Add documents with unique IDs and optional metadata
# collection.add(
#     documents=documents,
#     ids=[f"doc_{i}" for i in range(len(documents))],
#     metadatas=[
#         {"source": "example1"},
#         {"source": "example2"},
#         {"source": "example3"}
#     ]
# )

# # Perform a query
# results = collection.query(
#     query_texts=["quick animals"],
#     n_results=2  # Number of most similar results to return
# )

# # Print the query results
# print("Query Results:")
# for i, result in enumerate(results['documents'][0], 1):
#     print(f"{i}. {result}")

# # Optional: Persist the database (though PersistentClient handles this automatically)
# client.persist()