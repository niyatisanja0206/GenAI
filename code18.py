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

movies = [
    'dhoom 3 : thriller, horror, action, comedy, drama, romance',
    'dilwale dulhania le jayenge : romance, drama, comedy',
    '3 idiots : comedy, drama, romance',
    'zindagi na milegi dobara : comedy, drama, romance',
    'Krrish 3 : action, thriller, drama, romance, science fiction',
    'krrish : action, thriller, drama, romance, science fiction',
    'Chak de! India : drama, sports',
    'barfi : drama, romance, comedy',
    ]

user_input = input("Enter a movie type: ")
query = user_input.lower()
query_vector = embedding_model.embed_query([query])

doc_vectors = embedding_model.embed_documents(movies)

similarities = cosine_similarity(query_vector, doc_vectors)[0]

df = pd.DataFrame({
    'movie': movies,
    'similarity': similarities
})

ranked_movies = df.sort_values(by='similarity', ascending=False). reset_index(drop=True)

print(ranked_movies)