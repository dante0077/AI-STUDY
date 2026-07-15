text = "abcdefghijklmnopqrstuvwxyz"

chunk_size = 10
overlap = 2

step = chunk_size - overlap

chunks = []

for start in range(0, len(text), step):
    end = start + chunk_size
    chunk = text[start:end]

    if len(chunk) < 3:
        continue

    chunks.append(chunk)

print(chunks)