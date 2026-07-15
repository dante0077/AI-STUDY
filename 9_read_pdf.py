# pyrefly: ignore [missing-import]
from pypdf import PdfReader

reader = PdfReader("sample.pdf")

print("Pages:", len(reader.pages))

full_text = ""

for page in reader.pages:
    full_text += page.extract_text()

print(type(full_text))
print(len(full_text))
