# import os
# import chromadb
# from groq import Groq
# from typing import List, Dict

# class ContextAwareChatbot:
#     def __init__(
#         self, 
#         groq_api_key: str, 
#         chroma_db_path: str = './chroma_storage',
#         collection_name: str = 'document_collection',
#         max_context_tokens: int = 1000
#     ):
#         """
#         Initialize the chatbot with Groq API and ChromaDB configuration
        
#         Args:
#             groq_api_key (str): Groq API key for language model
#             chroma_db_path (str): Path to ChromaDB persistent storage
#             collection_name (str): Name of the ChromaDB collection
#             max_context_tokens (int): Maximum tokens for context retrieval
#         """
#         # Initialize Groq client
#         self.groq_client = Groq(api_key=groq_api_key)
        
#         # Initialize ChromaDB client
#         self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
#         self.collection = self.chroma_client.get_collection(name=collection_name)
        
#         # Conversation history management
#         self.conversation_history: List[Dict[str, str]] = []
#         self.max_context_tokens = max_context_tokens
        
#         # Embedding model for query processing
#         from sentence_transformers import SentenceTransformer
#         self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#     def _retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
#         """
#         Retrieve relevant context from ChromaDB based on query
        
#         Args:
#             query (str): User's query
#             top_k (int): Number of top relevant contexts to retrieve
        
#         Returns:
#             List of relevant context passages
#         """
#         # Generate embedding for the query
#         query_embedding = self.embedding_model.encode([query])[0].tolist()
        
#         # Retrieve similar passages from ChromaDB
#         results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=top_k
#         )
        
#         return results['documents'][0] if results['documents'] else []

#     def _manage_conversation_history(self, new_message: Dict[str, str]) -> List[Dict[str, str]]:
#         """
#         Manage conversation history to stay within token limit
        
#         Args:
#             new_message (dict): New message to add to history
        
#         Returns:
#             Trimmed conversation history
#         """
#         self.conversation_history.append(new_message)
        
#         # Simple token estimation and trimming
#         while len(str(self.conversation_history)) > self.max_context_tokens:
#             self.conversation_history.pop(0)
        
#         return self.conversation_history

#     def chat(self, user_query: str) -> str:
#         """
#         Main chat method to process user query and generate response
        
#         Args:
#             user_query (str): User's input message
        
#         Returns:
#             AI-generated response
#         """
#         # Retrieve relevant context
#         relevant_contexts = self._retrieve_relevant_context(user_query)
        
#         # Prepare context-augmented prompt
#         context_str = "\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(relevant_contexts)])
        
#         # Prepare conversation history
#         history_str = "\n".join([
#             f"{msg['role'].capitalize()}: {msg['content']}" 
#             for msg in self._manage_conversation_history({'role': 'user', 'content': user_query})
#         ])
        
#         # Construct full prompt
#         full_prompt = f"""
#         Relevant Document Contexts:
#         {context_str}
        
#         Conversation History:
#         {history_str}
        
#         Please provide a helpful and concise response to the latest user query, 
#         leveraging the context from retrieved documents and conversation history.
#         """
        
#         # Generate response using Groq
#         try:
#             chat_completion = self.groq_client.chat.completions.create(
#                 messages=[
#                     {"role": "system", "content": "You are a helpful AI assistant."},
#                     {"role": "user", "content": full_prompt}
#                 ],
#                 model="llama3-8b-8192"  # You can change the model as needed
#             )
            
#             response = chat_completion.choices[0].message.content
            
#             # Add AI response to conversation history
#             self._manage_conversation_history({'role': 'assistant', 'content': response})
            
#             return response
        
#         except Exception as e:
#             return f"An error occurred: {str(e)}"

# # Example Usage
# def main():
     
#     # Replace with your actual Groq API key
#     GROQ_API_KEY = "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"
    
#     try:
#         # Initialize chatbot
#         chatbot = ContextAwareChatbot(
#             groq_api_key=GROQ_API_KEY,
#             chroma_db_path= './chroma_storage',
#             collection_name= 'technical_docs'
#         )
        
#         # Interactive chat loop
#         while True:
#             user_input = input("You: ")
#             if user_input.lower() in ['exit', 'quit', 'bye']:
#                 break
            
#             response = chatbot.chat(user_input)
#             print("AI:", response)
    
#     except Exception as e:
#         print(f"Chatbot initialization error: {e}")

# # Uncomment to run
# if __name__ == "__main__":
#     main()





























# import streamlit as st
# import os
# import chromadb
# from groq import Groq
# from typing import List, Dict
# from sentence_transformers import SentenceTransformer

# class ContextAwareChatbot:
#     def __init__(
#         self, 
#         groq_api_key: str, 
#         chroma_db_path: str = './chroma_storage',
#         collection_name: str = 'document_collection',
#         max_context_tokens: int = 1000
#     ):
#         """
#         Initialize the chatbot with Groq API and ChromaDB configuration
        
#         Args:
#             groq_api_key (str): Groq API key for language model
#             chroma_db_path (str): Path to ChromaDB persistent storage
#             collection_name (str): Name of the ChromaDB collection
#             max_context_tokens (int): Maximum tokens for context retrieval
#         """
#         # Initialize Groq client
#         self.groq_client = Groq(api_key=groq_api_key)
        
#         # Initialize ChromaDB client
#         self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
#         self.collection = self.chroma_client.get_collection(name=collection_name)
        
#         # Conversation history management
#         self.conversation_history: List[Dict[str, str]] = []
#         self.max_context_tokens = max_context_tokens
        
#         # Embedding model for query processing
#         self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

#     def _retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
#         """
#         Retrieve relevant context from ChromaDB based on query
        
#         Args:
#             query (str): User's query
#             top_k (int): Number of top relevant contexts to retrieve
        
#         Returns:
#             List of relevant context passages
#         """
#         # Generate embedding for the query
#         query_embedding = self.embedding_model.encode([query])[0].tolist()
        
#         # Retrieve similar passages from ChromaDB
#         results = self.collection.query(
#             query_embeddings=[query_embedding],
#             n_results=top_k
#         )
        
#         return results['documents'][0] if results['documents'] else []

#     def _manage_conversation_history(self, new_message: Dict[str, str]) -> List[Dict[str, str]]:
#         """
#         Manage conversation history to stay within token limit
        
#         Args:
#             new_message (dict): New message to add to history
        
#         Returns:
#             Trimmed conversation history
#         """
#         self.conversation_history.append(new_message)
        
#         # Simple token estimation and trimming
#         while len(str(self.conversation_history)) > self.max_context_tokens:
#             self.conversation_history.pop(0)
        
#         return self.conversation_history

#     def chat(self, user_query: str) -> str:
#         """
#         Main chat method to process user query and generate response
        
#         Args:
#             user_query (str): User's input message
        
#         Returns:
#             AI-generated response
#         """
#         # Retrieve relevant context
#         relevant_contexts = self._retrieve_relevant_context(user_query)
        
#         # Prepare context-augmented prompt
#         context_str = "\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(relevant_contexts)])
        
#         # Prepare conversation history
#         history_str = "\n".join([
#             f"{msg['role'].capitalize()}: {msg['content']}" 
#             for msg in self._manage_conversation_history({'role': 'user', 'content': user_query})
#         ])
        
#         # Construct full prompt
#         full_prompt = f"""
#         Relevant Document Contexts:
#         {context_str}
        
#         Conversation History:
#         {history_str}
        
#         Please provide a helpful and concise response to the latest user query, 
#         leveraging the context from retrieved documents and conversation history.
#         """
        
#         # Generate response using Groq
#         try:
#             chat_completion = self.groq_client.chat.completions.create(
#                 messages=[
#                     {"role": "system", "content": "You are a helpful AI assistant."},
#                     {"role": "user", "content": full_prompt}
#                 ],
#                 model="llama3-8b-8192"  # You can change the model as needed
#             )
            
#             response = chat_completion.choices[0].message.content
            
#             # Add AI response to conversation history
#             self._manage_conversation_history({'role': 'assistant', 'content': response})
            
#             return response
        
#         except Exception as e:
#             return f"An error occurred: {str(e)}"

# def main():
#     # Streamlit UI setup
#     st.set_page_config(page_title="Context-Aware Chatbot", page_icon="💬")
    
#     # Title and description
#     st.title("🤖 Context-Aware Document Chatbot")
#     st.write("Chat with your documents using advanced context retrieval")

#     # Groq API Key input
#     GROQ_API_KEY = st.sidebar.text_input("Enter Groq API Key", type="password")
    
#     # Chatbot initialization
#     if GROQ_API_KEY:
#         try:
#             # Initialize chatbot
#             chatbot = ContextAwareChatbot(
#                 groq_api_key=GROQ_API_KEY,
#                 chroma_db_path='./chroma_storage',
#                 collection_name='technical_docs'
#             )
            
#             # Initialize chat history in session state
#             if 'messages' not in st.session_state:
#                 st.session_state.messages = []
            
#             # Display chat messages from history on app rerun
#             for message in st.session_state.messages:
#                 with st.chat_message(message["role"]):
#                     st.markdown(message["content"])
            
#             # Chat input
#             if prompt := st.chat_input("What would you like to know?"):
#                 # Add user message to chat history
#                 st.session_state.messages.append({"role": "user", "content": prompt})
                
#                 # Display user message
#                 with st.chat_message("user"):
#                     st.markdown(prompt)
                
#                 # Generate and display assistant response
#                 with st.chat_message("assistant"):
#                     response = chatbot.chat(prompt)
#                     st.markdown(response)
                
#                 # Add assistant response to chat history
#                 st.session_state.messages.append({"role": "assistant", "content": response})
        
#         except Exception as e:
#             st.error(f"Chatbot initialization error: {e}")
#     else:
#         st.warning("Please enter your Groq API Key in the sidebar")

# if __name__ == "__main__":
#     main()










import streamlit as st
import os
import chromadb
from groq import Groq
from typing import List, Dict

# Import the entire existing ContextAwareChatbot class and main function from the original script
# (Copy-paste the entire original code here, which you provided earlier)
import os
import chromadb
from groq import Groq
from typing import List, Dict

class ContextAwareChatbot:
    def __init__(
        self, 
        groq_api_key: str, 
        chroma_db_path: str = './chroma_storage',
        collection_name: str = 'document_collection',
        max_context_tokens: int = 1000
    ):
        """
        Initialize the chatbot with Groq API and ChromaDB configuration
        
        Args:
            groq_api_key (str): Groq API key for language model
            chroma_db_path (str): Path to ChromaDB persistent storage
            collection_name (str): Name of the ChromaDB collection
            max_context_tokens (int): Maximum tokens for context retrieval
        """
        # Initialize Groq client
        self.groq_client = Groq(api_key=groq_api_key)
        
        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        self.collection = self.chroma_client.get_collection(name=collection_name)
        
        # Conversation history management
        self.conversation_history: List[Dict[str, str]] = []
        self.max_context_tokens = max_context_tokens
        
        # Embedding model for query processing
        from sentence_transformers import SentenceTransformer
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def _retrieve_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieve relevant context from ChromaDB based on query
        
        Args:
            query (str): User's query
            top_k (int): Number of top relevant contexts to retrieve
        
        Returns:
            List of relevant context passages
        """
        # Generate embedding for the query
        query_embedding = self.embedding_model.encode([query])[0].tolist()
        
        # Retrieve similar passages from ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        return results['documents'][0] if results['documents'] else []

    def _manage_conversation_history(self, new_message: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Manage conversation history to stay within token limit
        
        Args:
            new_message (dict): New message to add to history
        
        Returns:
            Trimmed conversation history
        """
        self.conversation_history.append(new_message)
        
        # Simple token estimation and trimming
        while len(str(self.conversation_history)) > self.max_context_tokens:
            self.conversation_history.pop(0)
        
        return self.conversation_history

    def chat(self, user_query: str) -> str:
        """
        Main chat method to process user query and generate response
        
        Args:
            user_query (str): User's input message
        
        Returns:
            AI-generated response
        """
        # Retrieve relevant context
        relevant_contexts = self._retrieve_relevant_context(user_query)
        
        # Prepare context-augmented prompt
        context_str = "\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(relevant_contexts)])
        
        # Prepare conversation history
        history_str = "\n".join([
            f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in self._manage_conversation_history({'role': 'user', 'content': user_query})
        ])
        
        # Construct full prompt
        full_prompt = f"""
        Relevant Document Contexts:
        {context_str}
        
        Conversation History:
        {history_str}
        
        Please provide a helpful and concise response to the latest user query, 
        leveraging the context from retrieved documents and conversation history.
        """
        
        # Generate response using Groq
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": full_prompt}
                ],
                model="llama3-8b-8192"  # You can change the model as needed
            )
            
            response = chat_completion.choices[0].message.content
            
            # Add AI response to conversation history
            self._manage_conversation_history({'role': 'assistant', 'content': response})
            
            return response
        
        except Exception as e:
            return f"An error occurred: {str(e)}"

def main():
    # Replace with your actual Groq API key
    GROQ_API_KEY = "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"
    
    try:
        # Initialize chatbot
        chatbot = ContextAwareChatbot(
            groq_api_key=GROQ_API_KEY,
            chroma_db_path= './chroma_storage',
            collection_name= 'technical_docs'
        )
        
        # Interactive chat loop
        while True:
            user_input = input("You: ")
            if user_input.lower() in ['exit', 'quit', 'bye']:
                break
            
            response = chatbot.chat(user_input)
            print("AI:", response)
    
    except Exception as e:
        print(f"Chatbot initialization error: {e}")

# Streamlit UI Wrapper
def streamlit_main():
    st.title("Context-Aware Chatbot")
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Initialize chatbot
    try:
        chatbot = ContextAwareChatbot(
            groq_api_key="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",
            chroma_db_path='./chroma_storage',
            collection_name='technical_docs'
        )
    except Exception as e:
        st.error(f"Chatbot initialization error: {e}")
        return
    
    # Chat input
    if prompt := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get assistant response
        with st.chat_message("assistant"):
            response = chatbot.chat(prompt)
            st.markdown(response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

# Run Streamlit app
if __name__ == "__main__":
    streamlit_main()