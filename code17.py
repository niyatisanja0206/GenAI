from langchain.embeddings import AzureOpenAIEmbeddings
import os
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

# Read the file
file_path = "Marwadi_University.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text = file.read()

# Embed the whole document as a single vector
embedding = embedding_model.embed_query(text)

print(f"Embedding vector length: {len(embedding)}")
print(f"Embedding vector: {embedding[:10]}...")  # Print first 10 values for brevity
