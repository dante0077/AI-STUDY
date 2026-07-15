# pyrefly: ignore [missing-import]
from pypdf import PdfReader

reader = PdfReader("sample.pdf")

full_text = ""

for page in reader.pages:
    full_text += page.extract_text()

chunk_size = 500
overlap = 100

step = chunk_size - overlap

chunks = []

for start in range(0, len(full_text), step):
    end = start + chunk_size

    chunk = full_text[start:end]

    if len(chunk) < 100:
        continue

    chunks.append(chunk)

print(f"Total chunks: {len(chunks)}")

for i, chunk in enumerate(chunks):
    print(f"\n--- Chunk {i} ---")
    print(chunk[:100])