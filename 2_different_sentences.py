
# pyrefly: ignore [missing-import]
from sentence_transformers import SentenceTransformer
# pyrefly: ignore [missing-import]
from sentence_transformers.util import cos_sim

model = SentenceTransformer("all-MiniLM-L6-v2")

redis_sentence = "Redis is an in-memory database"
cache_sentence = "Cache user sessions in Redis"
dog_sentence = "Dogs are loyal pets"


redis_embedding = model.encode(redis_sentence)
cache_embedding = model.encode(cache_sentence)
dog_embedding = model.encode(dog_sentence)

redis_cache_similarity = cos_sim(
    redis_embedding,
    cache_embedding
)

redis_dog_similarity = cos_sim(
    redis_embedding,
    dog_embedding
)

print("Redis ↔ Cache")
print(redis_cache_similarity)

print()

print("Redis ↔ Dog")
print(redis_dog_similarity)