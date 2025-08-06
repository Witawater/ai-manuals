#!/usr/bin/env python3
"""
Extracts section summaries for each top-level header.
Saves to the manual_sections table.
"""

import os, uuid, json, re
from sqlalchemy import text
from PyPDF2 import PdfReader
from openai import OpenAI
from db import engine
from dotenv import load_dotenv

load_dotenv()
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── Config ──
DOC_ID = "15f371a9f5e540e890a62b2936c31dbb"
PDF_PATH = "JULABO FPW50-HE User Manual.pdf"

# ── Helpers ──
def clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def extract_text(pdf_path: str) -> str:
    pdf = PdfReader(pdf_path)
    return "\n".join(page.extract_text() or "" for page in pdf.pages)

def summarize_section(title: str, content: str) -> str:
    prompt = (
        f"Summarize the section titled '{title}' from this user manual content:\n\n"
        f"{content[:7000]}\n\n"
        "Return a short bullet list of key information (max 5 bullets)."
    )
    res = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return res.choices[0].message.content.strip()

def find_section_text(title: str, full_text: str) -> str:
    pattern = re.escape(title[:10])
    match = re.search(rf"{pattern}.*?(?=\n\d+\.)", full_text, re.DOTALL)
    return match.group(0) if match else ""

# ── Main logic ──
def extract_summaries(doc_id: str, pdf_path: str):
    print(f"📖 Reading {pdf_path} …")
    full_text = extract_text(pdf_path)

    print("🧠 Loading outline from DB …")
    with engine.begin() as conn:
        rows = conn.execute(
            text("SELECT id, title FROM manual_outline WHERE doc_id = :doc ORDER BY sort_order"),
            {"doc": doc_id}
        ).fetchall()

    for r in rows:
        title = clean(r.title)
        print(f"✏️  Summarizing: {title}")
        section_text = find_section_text(title, full_text)
        if not section_text:
            print("  ⚠️  Could not find matching content.")
            continue

        summary = summarize_section(title, section_text)
        with engine.begin() as conn:
            conn.execute(
                text("""INSERT INTO manual_sections
                        (id, doc_id, outline_id, summary)
                        VALUES (:id, :doc, :oid, :sum)"""),
                {"id": uuid.uuid4().hex, "doc": doc_id, "oid": r.id, "sum": summary}
            )
    print("✅ All summaries saved.")

# ── Run ──
if __name__ == "__main__":
    extract_summaries(DOC_ID, PDF_PATH)
