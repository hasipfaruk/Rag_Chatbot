# Optional: Retrieve and demonstrate similarity search


import os
import shutil
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer



client = chromadb.PersistentClient(path='./chroma_storage')
collection = client.get_collection('technical_docs')

# Example query
query_results = collection.query(
    query_texts=["what is total amount clint pay"],
    n_results=3
)
print("\nQuery Results:")
for doc in query_results['documents'][0]:
    print(doc)

