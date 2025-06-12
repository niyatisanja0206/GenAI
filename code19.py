from langchain.embeddings import AzureOpenAIEmbeddings
import os
from dotenv import load_dotenv
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

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

# Read the file and split into individual movie lines
file_path = "movies.txt"
with open(file_path, "r", encoding="utf-8") as file:
    movies = [line.strip() for line in file if line.strip()]  # Remove empty lines and strip whitespace

# Get user input and embed the query
user_input = input("Enter a movie type: ")
query = user_input.lower()
query_vector = embedding_model.embed_query(query)

# Embed individual movie lines
doc_vectors = embedding_model.embed_documents(movies)

# Compute cosine similarity
similarities = cosine_similarity([query_vector], doc_vectors)[0]

# Create DataFrame and sort by similarity
df = pd.DataFrame({
    'movie': movies,
    'similarity': similarities
})

ranked_movies = df.sort_values(by='similarity', ascending=False).reset_index(drop=True)

# Print the top results
print(ranked_movies)
