

import fitz  # PyMuPDF
import re
import uuid
from pathlib import Path

def read_pdf(path):
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    return pages

def split_into_chunks(pages):
    chunks = []
    current_section = None

    for i, text in enumerate(pages, start=1):
        lines = text.splitlines()
        buffer = []
        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Detect section headings (very rough)
            if re.match(r"^\d+(\.\d+)*\s+[A-Z][\w\s\-]+$", line):
                if buffer:
                    chunks.append({
                        "id": str(uuid.uuid4())[:8],
                        "text": "\n".join(buffer),
                        "section": current_section or "Untitled",
                        "page": i
                    })
                    buffer = []
                current_section = line
            else:
                buffer.append(line)

        if buffer:
            chunks.append({
                "id": str(uuid.uuid4())[:8],
                "text": "\n".join(buffer),
                "section": current_section or "Untitled",
                "page": i
            })

    return chunks

def preview(chunks, n=3):
    for c in chunks[:n]:
        print(f"[{c['section']}] (p{c['page']}) → {len(c['text'])} chars\n")

if __name__ == "__main__":
    PDF_PATH = "Test Manual.pdf"
    pages = read_pdf(PDF_PATH)
    chunks = split_into_chunks(pages)
    preview(chunks)

    # Optional: save to file
    import json
    with open("test_chunks.json", "w") as f:
        json.dump(chunks, f, indent=2)