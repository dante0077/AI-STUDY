# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
# pyrefly: ignore [missing-import]
from sentence_transformers.util import cos_sim

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
    "Redis is an in-memory database commonly used for caching user sessions",
    "Redis supports key value storage",
    "PostgreSQL stores relational data",
    "JWT tokens contain user claims",
    "Kubernetes schedules workloads using pods",
]

input_query = "Where should I store session data?"

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




#
#1. "Redis is an in-memory database commonly used for caching user sessions",
#2. "Redis supports key value storage",
#3. "PostgreSQL stores relational data",
#4. "JWT tokens contain user claims",
#5. "Kubernetes schedules workloads using pods",
#
#
#Score | Document
#0.8579 | Redis is an in-memory database
#
#0.3075 | PostgreSQL stores relational data
#
#0.1497 | Kubernetes schedules workloads using pods
#
#0.0301 | JWT tokens contain claims