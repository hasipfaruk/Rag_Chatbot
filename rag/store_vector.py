
# import os
# import chromadb
# import numpy as np
# from sentence_transformers import SentenceTransformer

# # Add support for different document types
# try:
#     import mammoth  # For .docx
# except ImportError:
#     mammoth = None

# try:
#     import PyPDF2  # For PDF support
# except ImportError:
#     PyPDF2 = None

# def extract_text_from_document(file_path):
#     """
#     Extract text from different document types
    
#     Args:
#     - file_path (str): Path to the document file
    
#     Returns:
#     - str: Extracted text from the document
#     """
#     # Get file extension
#     file_extension = os.path.splitext(file_path)[1].lower()
    
#     try:
#         if file_extension == '.docx':
#             # Word document extraction
#             if mammoth is None:
#                 raise ImportError("mammoth library is not installed. Please install it with 'pip install mammoth'")
            
#             with open(file_path, "rb") as docx_file:
#                 result = mammoth.extract_raw_text(docx_file)
#                 return result.value
        
#         elif file_extension == '.pdf':
#             # PDF extraction
#             if PyPDF2 is None:
#                 raise ImportError("PyPDF2 library is not installed. Please install it with 'pip install PyPDF2'")
            
#             text = ""
#             with open(file_path, 'rb') as pdf_file:
#                 pdf_reader = PyPDF2.PdfReader(pdf_file)
#                 for page in pdf_reader.pages:
#                     text += page.extract_text() + "\n"
#             return text
        
#         else:
#             raise ValueError(f"Unsupported file type: {file_extension}")
    
#     except Exception as e:
#         print(f"Error extracting text from {file_path}: {e}")
#         return ""

# def store_document_embeddings(
#     file_path, 
#     db_path='./chroma_storage', 
#     collection_name='document_collection'
# ):
#     """
#     Extract text from a document, generate embeddings, and store in ChromaDB.
    
#     Args:
#     - file_path (str): Path to the document (Word or PDF)
#     - db_path (str): Path for persistent ChromaDB storage
#     - collection_name (str): Name of the ChromaDB collection
    
#     Returns:
#     - dict: Information about stored embeddings
#     """
#     # Extract text from document
#     document_text = extract_text_from_document(file_path)
    
#     if not document_text:
#         raise ValueError("No text could be extracted from the document")
    
#     # Split text into chunks (if too long)
#     def chunk_text(text, chunk_size=500, overlap=50):
#         words = text.split()
#         chunks = []
#         for i in range(0, len(words), chunk_size - overlap):
#             chunk = ' '.join(words[i:i+chunk_size])
#             chunks.append(chunk)
#         return chunks
    
#     # Use a local embedding model
#     embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
#     # Create ChromaDB client
#     client = chromadb.PersistentClient(path=db_path)
    
#     # Create or get collection
#     collection = client.get_or_create_collection(
#         name=collection_name,
#         metadata={"hnsw:space": "cosine"}
#     )
    
#     # Process document chunks
#     text_chunks = chunk_text(document_text)
    
#     # Generate embeddings
#     embeddings = embedding_model.encode(text_chunks)
    
#     # Store in ChromaDB
#     collection.add(
#         documents=text_chunks,
#         embeddings=embeddings.tolist(),
#         ids=[f'chunk_{i}' for i in range(len(text_chunks))]
#     )
    
#     # Return metadata about storage
#     return {
#         "total_chunks": len(text_chunks),
#         "collection_name": collection_name,
#         "embedding_model": "all-MiniLM-L6-v2",
#         "file_type": os.path.splitext(file_path)[1]
#     }

# def main():
#     try:
#         # Example usage with different file types
#         file_paths = [
#             'Documents\Proposal- Embedding Solution Set in a React App with Virtual Customer Service Rep.pdf'  # Word document
#         ]
        
#         for file_path in file_paths:
#             try:
#                 # Store embeddings from the document
#                 result = store_document_embeddings(
#                     file_path=file_path,
#                     db_path='chroma_storage\chroma.sqlite3',
#                     collection_name='example_collection'
#                 )
#                 print(f"Embedding storage successful for {file_path}:", result)
            
#             except Exception as file_error:
#                 print(f"Error processing {file_path}: {file_error}")
        
#         # Optional: Retrieve and demonstrate similarity search
#         # client = chromadb.PersistentClient(path='./my_document_embeddings')
#         # collection = client.get_collection('technical_docs')
        
#         # # Example query
#         # query_results = collection.query(
#         #     query_texts=["important technical concept"],
#         #     n_results=3
#         # )
#         # print("\nQuery Results:")
#         # for doc in query_results['documents'][0]:
#         #     print(doc)
    
#     except Exception as e:
#         print(f"An overall error occurred: {e}")

# # Uncomment to run
# if __name__ == "__main__":
#     main()
















import os
import shutil
import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd  # For Excel support


# Add support for different document types
try:
    import mammoth  # For .docx
except ImportError:
    mammoth = None

try:
    import PyPDF2  # For PDF support
except ImportError:
    PyPDF2 = None

def extract_text_from_document(file_path):
    """
    Extract text from different document types
    
    Args:
    - file_path (str): Path to the document file
    
    Returns:
    - str: Extracted text from the document
    """
    # Get file extension
    file_extension = os.path.splitext(file_path)[1].lower()
    
    try:
        if file_extension == '.docx':
            # Word document extraction
            if mammoth is None:
                raise ImportError("mammoth library is not installed. Please install it with 'pip install mammoth'")
            
            with open(file_path, "rb") as docx_file:
                result = mammoth.extract_raw_text(docx_file)
                return result.value
        
        elif file_extension == '.pdf':
            # PDF extraction
            if PyPDF2 is None:
                raise ImportError("PyPDF2 library is not installed. Please install it with 'pip install PyPDF2'")
            
            text = ""
            with open(file_path, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
            return text
        
        # Add By Haseeb for excel support
        elif file_extension in ['.xlsx', '.xls']:
            try:
                df = pd.read_excel(file_path)
                return "\n".join(df.astype(str).apply(" ".join, axis=1).tolist())
            except Exception as e:
                print(f"Error reading Excel file {file_path}: {e}")
                return ""

        
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

def prepare_db_storage(db_path):
    """
    Prepare ChromaDB storage directory, clearing existing files if needed
    
    Args:
    - db_path (str): Path to ChromaDB storage
    """
    # Ensure the directory exists
    os.makedirs(db_path, exist_ok=True)
    
    # Remove existing ChromaDB files to prevent conflicts
    files_to_remove = [
        'chroma.sqlite3', 
        'chroma-embeddings.parquet', 
        'index'
    ]
    
    for filename in files_to_remove:
        full_path = os.path.join(db_path, filename)
        if os.path.exists(full_path):
            try:
                if os.path.isdir(full_path):
                    shutil.rmtree(full_path)
                else:
                    os.remove(full_path)
                print(f"Removed existing file: {full_path}")
            except Exception as e:
                print(f"Could not remove {full_path}: {e}")

def store_document_embeddings(
    file_path, 
    db_path='./chroma_storage', 
    collection_name='document_collection'
):
    """
    Extract text from a document, generate embeddings, and store in ChromaDB.
    
    Args:
    - file_path (str): Path to the document (Word or PDF)
    - db_path (str): Path for persistent ChromaDB storage
    - collection_name (str): Name of the ChromaDB collection
    
    Returns:
    - dict: Information about stored embeddings
    """
    # Prepare storage directory
    prepare_db_storage(db_path)
    
    # Extract text from document
    document_text = extract_text_from_document(file_path)
    
    if not document_text:
        raise ValueError("No text could be extracted from the document")
    
    # Split text into chunks (if too long)
    def chunk_text(text, chunk_size=500, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), chunk_size - overlap):
            chunk = ' '.join(words[i:i+chunk_size])
            chunks.append(chunk)
        return chunks
    
    # Use a local embedding model
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create ChromaDB client
    client = chromadb.PersistentClient(path=db_path)
    
    # Create or get collection
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )
    
    # Process document chunks
    text_chunks = chunk_text(document_text)
    
    # Generate embeddings
    embeddings = embedding_model.encode(text_chunks)
    
    # Store in ChromaDB
    collection.add(
        documents=text_chunks,
        embeddings=embeddings.tolist(),
        ids=[f'chunk_{i}' for i in range(len(text_chunks))]
    )
    
    # Return metadata about storage
    return {
        "total_chunks": len(text_chunks),
        "collection_name": collection_name,
        "embedding_model": "all-MiniLM-L6-v2",
        "file_type": os.path.splitext(file_path)[1]
    }

def main():
    try:
        # Example usage with different file types
        file_paths = [
            r'Documents\solution_set_list.pdf'
        ]
        
        for file_path in file_paths:
            try:
                # Store embeddings from the document
                result = store_document_embeddings(
                    file_path=file_path,
                    db_path='./chroma_storage',
                    collection_name='technical_docs'
                )
                print(f"Embedding storage successful for {file_path}:", result)
            
            except Exception as file_error:
                print(f"Error processing {file_path}: {file_error}")
        
        # Optional: Retrieve and demonstrate similarity search
        client = chromadb.PersistentClient(path='./chroma_storage')
        collection = client.get_collection('technical_docs')
        
        # Example query
        query_results = collection.query(
            query_texts=["important technical concept"],
            n_results=3
        )
        print("\nQuery Results:")
        for doc in query_results['documents'][0]:
            print(doc)
    
    except Exception as e:
        print(f"An overall error occurred: {e}")

if __name__ == "__main__":
    main()