# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "Redis is an in-memory database",
    "Cache user sessions in Redis",
    "Dogs are loyal pets"
]

embeddings = model.encode(sentences)

print(embeddings.shape)