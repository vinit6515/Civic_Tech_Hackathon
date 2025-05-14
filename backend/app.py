from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.chains import RetrievalQA
import google.generativeai as genai
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Gemini API key
gemini_api_key = 'AIzaSyDQc9nY4X0TSdv2zS-mL6eetjSl4vERJBI'
if not gemini_api_key:
    logger.error("GEMINI_API_KEY environment variable not set")
    raise ValueError("GEMINI_API_KEY environment variable not set")

# Configure Gemini
genai.configure(api_key=gemini_api_key)

# Initialize embeddings (still using OpenAI for embeddings as Gemini embeddings might not be fully available)
openai_api_key = os.getenv("OPENAI_API_KEY")
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Initialize vector store
vector_store = None

def initialize_vector_store():
    """Initialize the vector store with documents from the data directory."""
    global vector_store
    
    try:
        # Check if we have a pre-built index
        if os.path.exists("backend/vector_store"):
            logger.info("Loading existing vector store...")
            vector_store = FAISS.load_local("backend/vector_store", embeddings)
            return
        
        # Load documents
        logger.info("Building new vector store...")
        loader = DirectoryLoader("backend/data/", glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        # Create vector store
        vector_store = FAISS.from_documents(texts, embeddings)
        
        # Save vector store for future use
        vector_store.save_local("backend/vector_store")
        logger.info(f"Vector store created with {len(texts)} text chunks")
    
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        raise

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process chat requests with RAG using Gemini."""
    try:
        data = request.json
        query = data.get('message', '')
        history = data.get('history', [])
        
        # Ensure vector store is initialized
        if vector_store is None:
            initialize_vector_store()
        
        # Retrieve relevant documents
        docs = vector_store.similarity_search(query, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Format history for Gemini
        formatted_history = []
        for msg in history:
            role = "model" if msg["role"] == "assistant" else "user"
            formatted_history.append({"role": role, "parts": [{"text": msg["content"]}]})
        
        # Create Gemini model
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
            }
        )
        
        # Create chat session
        chat = model.start_chat(history=formatted_history)
        
        # System prompt with context
        system_prompt = f"""You are the Indianapolis Civic Assistant, an AI chatbot designed to help the 870,000 residents of Indianapolis.
        
        Here is relevant information about Indianapolis that may help answer the query:
        {context}
        
        Provide helpful, accurate, and concise information about:
        - City services and departments
        - Public transportation and infrastructure
        - Events and recreational activities
        - Permits, licenses, and regulations
        - Emergency services and public safety
        - Local government and civic participation
        
        Always be polite, professional, and focus on providing factual information about Indianapolis.
        If you don't know something, acknowledge it and suggest contacting the relevant city department."""
        
        # Send system prompt first if this is the first message
        if len(history) <= 1:
            chat.send_message(system_prompt)
        
        # Send user query
        response = chat.send_message(query)
        
        return jsonify({"response": response.text})
    
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize vector store on startup
    initialize_vector_store()
    app.run(debug=True, port=5000)
