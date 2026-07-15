# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
# pyrefly: ignore [missing-import]
from sentence_transformers.util import cos_sim

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Redis is an in-memory database",
    "Kubernetes schedules workloads using pods",
    "JWT tokens contain claims",
    "PostgreSQL stores relational data"
]

input_query = "How can I cache user sessions?"

input_embedding = model.encode(input_query)
document_embeddings = model.encode(documents)
scors = []
print("Score | Document")
for document_embedding, document in zip(document_embeddings, documents):
    scors.append((cos_sim(input_embedding, document_embedding).item(), document))

scors.sort(key=lambda x: x[0], reverse=True)

for scor, document in scors:
    print(f"{scor:.4f} | {document}")
    print("\n")

#Score | Document
#0.1808 | Redis is an in-memory database
#
#0.1355 | Kubernetes schedules workloads using pods
#
#0.0923 | PostgreSQL stores relational data
#
#0.0529 | JWT tokens contain claims