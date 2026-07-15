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
    k=1
)



print("Distances:", distances)
print("Indices:", indices)

print("Query result:", documents[indices[0][0]])

best_index = indices[0][0]

print("\n Best Match:")
print(documents[best_index])




#Total vectors in index: 3
#Distances: [[0.86651933]]
#Indices: [[0]]
#Query result: Redis is an in-memory database commonly used for caching user sessions
#
# Best Match:
#Redis is an in-memory database commonly used for caching user sessions