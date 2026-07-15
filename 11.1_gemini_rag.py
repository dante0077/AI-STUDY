import os
# pyrefly: ignore [missing-import]
from google import genai

client = genai.Client(
    api_key="XXXXXXX"
)

documents = [
    "Redis is an in-memory database",
    "PostgreSQL stores relational data"
]

query = "How can I cache user sessions?"

context = "\n".join(documents)

prompt = f"""
Context:
{context}

Question:
{query}

Answer:
"""                

print(prompt)