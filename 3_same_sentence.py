
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
# pyrefly: ignore [missing-import]
from sentence_transformers.util import cos_sim

model = SentenceTransformer("all-MiniLM-L6-v2")

redis_sentence = "Redis is an in-memory database"
redis2_sentence = "Redis is an in-memory database"

redis_embedding = model.encode(redis_sentence)
redis2_embedding = model.encode(redis2_sentence)

redis_cache_similarity = cos_sim(
    redis_embedding,
    redis2_embedding
)

print("Redis ↔ Redis2")
print(redis_cache_similarity)
