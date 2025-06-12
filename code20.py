from langchain_community.document_loaders import TextLoader
from langchain.embeddings import AzureOpenAIEmbeddings
import os
import faiss
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Azure OpenAI Embeddings
embedding_model = AzureOpenAIEmbeddings(
    openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    chunk_size=100,
    deployment="text-embedding-ada-002",
)

# Read the file using LangChain's loader
file_path = "Marwadi_University.txt"
loader = TextLoader(file_path)
documents = loader.load()

# Extract text content from documents
texts = [doc.page_content for doc in documents]

def store_vector(texts):
    # Get embedding vectors
    vectors = embedding_model.embed_documents(texts)
    
    # Ensure vectors are NumPy arrays
    vectors = np.array(vectors)
    
    # Create and store FAISS index
    dimension = vectors.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors)
    faiss.write_index(index, "marwadi_index.index")
    print("Vector stored successfully.")
    return vectors

vector = store_vector(texts)
