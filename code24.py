import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

metadata = {
    "eduaction.txt" : { "tags" : ["university","college","education","learning"]},
    "medical.txt" : { "tags" : ["healthcare","medicine","treatment","hospital"]},
    "sports.txt" : { "tags" : ["football","cricket","athletics","sports"]}
}

query = "Which is best university in the state?"

docs_as_text = {filename: " ".join(data["tags"]) for filename, data in metadata.items()}

doc_embeddings = model.encode(list(docs_as_text.values()))
query_embedding = model.encode([query])
similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
sorted_indices = np.argsort(similarities)[::-1]

top_n = 1
top_docs = [(list(docs_as_text.keys())[i], similarities[i]) for i in sorted_indices[:top_n]]
print("Top documents for query:", query)
for doc, score in top_docs:
    print(f"Document: {doc}, Similarity Score: {score:.4f}")
    print(f"Content: {docs_as_text[doc]}\n")

