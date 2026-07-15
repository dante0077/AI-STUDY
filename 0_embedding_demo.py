
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer

print("Loading model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded!")

text = "Redis is an in-memory database"

embedding = model.encode(text)

print(embedding)


print(type(embedding))
print(len(embedding))
print(embedding[:10])