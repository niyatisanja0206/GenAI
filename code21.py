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

def search(query):
    index = faiss.read_index("marwadi_index.index")
    query_vector = embedding_model.embed_query(query)
    distances, indices = index.search(query_vector, 1)
    return distances, indices

query = "What are faculties?"
qu_vector = embedding_model.embed_query(query)
distances, indices = search(query)
print(f"Query: {query}")

print(f"Distances: {distances}")
print(f"Indices: {indices}")


