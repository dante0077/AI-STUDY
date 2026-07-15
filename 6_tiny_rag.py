# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

# pyrefly: ignore [missing-import]
import faiss 

# pyrefly: ignore [missing-import]
import numpy as np


model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Redis is an in-memory database commonly used for caching user sessions",
    "PostgreSQL stores relational data",
    "JWT tokens contain claims"
]

document_embeddings = model.encode(documents)

document_embeddings = np.array(
    document_embeddings,
    dtype="float32"
)


#create index
dimension = 384
index = faiss.IndexFlatL2(dimension)

#add vectors
index.add(document_embeddings)
print(f"Total vectors in index: {index.ntotal}")


# Query
query = "How can I cache user sessions?"

#embedding
query_embedding = model.encode([query])

#convert to digestable for faiss
query_embedding = np.array(
    query_embedding,
    dtype="float32"
)

#search
distances, indices = index.search(
    query_embedding,
    k=2
)

retrieved_docs = [documents[i] for i in indices[0]]
context = "\n".join(retrieved_docs)

prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""

print(prompt)
