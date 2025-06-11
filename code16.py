from langchain.embeddings import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

embedding_model = AzureOpenAIEmbeddings(
    openai_api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    chunk_size=100,  # Set chunk size to 1 for single text embedding
    deployment="text-embedding-ada-002",  # Specify the model name
)

text = "Hello World"
embeddings = embedding_model.embed_query(text)

print(f"Embedding vector length: {len(embeddings)}")
print(f"Embedding vector: {embeddings}")