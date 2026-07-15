# pyrefly: ignore [missing-import]
import faiss 

# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

# pyrefly: ignore [missing-import]
import numpy as np


model = SentenceTransformer("all-MiniLM-L6-v2")


documents = [    
    "Redis is an in-memory database commonly used for caching user sessions",
    "PostgreSQL stores relational data",
    "JWT tokens contain claims"
    ]


# Create embeddings
document_embeddings = model.encode(documents)

# FAISS expects float32
document_embeddings = np.array(
    document_embeddings,
    dtype = "float32"
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
    k=3
)

for i, idx in enumerate(indices[0]):
    print("Rank:", i+1)
    print("Document:", documents[idx])
    print("Distance:", distances[0][i])
    print("\n")
