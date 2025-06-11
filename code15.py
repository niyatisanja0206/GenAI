#Sentence trasformation to vectors

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer('BAAI/bge-small-en-v1.5')

str = "Hello World"

embeddings = embedding_model.encode(str, convert_to_tensor=True)
print(embeddings)
print(f"Embedding vector length: {len(embeddings)}")