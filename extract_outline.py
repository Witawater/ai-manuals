#!/usr/bin/env python3
"""
Extracts the outline (top-level section headers) from a PDF
and saves them to the manual_outline table.
"""

import os, re, uuid, json
from pathlib import Path
from typing import List
from sqlalchemy import text
from db import engine
from PyPDF2 import PdfReader
from openai import OpenAI
from dotenv import load_dotenv

# ─── Load API key ───
load_dotenv(".env")
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── Config ───
DOC_ID = "15f371a9f5e540e890a62b2936c31dbb"
PDF_PATH = "JULABO FPW50-HE User Manual.pdf"

# ─── Helper functions ───
def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def get_headings_from_text(text: str) -> List[str]:
    prompt = (
        "From the following user manual content, extract only the top-level numbered section headers "
        "(e.g. '1. Introduction', '2. Setup', '3. Operation'). Ignore tables of contents or preambles. "
        "Return a JSON array of strings like:\n"
        "[\"1. Introduction\", \"2. Installation\", \"3. Operation\"]\n\n"
        f"{text[:7000]}"
    )
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    txt = response.choices[0].message.content.strip()

    # ── Strip Markdown code block if present ──
    if txt.startswith("```json"):
        txt = txt[7:].strip()
    elif txt.startswith("```"):
        txt = txt[3:].strip()
    if txt.endswith("```"):
        txt = txt[:-3].strip()

    try:
        return json.loads(txt)
    except Exception:
        print("🛑 Could not parse GPT output:", txt)
        return []

# ─── Main logic ───
def extract_outline(doc_id: str, pdf_path: str) -> None:
    print(f"📖 Reading {pdf_path} …")
    pdf = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in pdf.pages[:20]]
    full_text = "\n".join(pages)

    print("🤖 Extracting section headers …")
    headers = get_headings_from_text(full_text)
    if not headers:
        print("🛑 No headers extracted.")
        return

    print("💾 Saving to DB …")
    with engine.begin() as conn:
        for order, title in enumerate(headers):
            conn.execute(
                text("""INSERT INTO manual_outline
                        (id, doc_id, title, sort_order)
                        VALUES (:id, :doc, :title, :n)"""),
                {"id": uuid.uuid4().hex, "doc": doc_id,
                 "title": clean(title), "n": order}
            )
    print("✅ Outline saved.")

# ─── Entry point ───
if __name__ == "__main__":
    extract_outline(DOC_ID, PDF_PATH)
