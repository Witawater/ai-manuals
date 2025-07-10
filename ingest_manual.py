#!/usr/bin/env python3
"""
ingest_manual.py
────────────────
Chunk ➜ embed ➜ upsert ONE PDF into Pinecone.

Usage (from project root):
    source venv/bin/activate
    python ingest_manual.py CoffeeMaker.pdf --customer demo01
"""

from __future__ import annotations

import os
import sys
import uuid
import time
import pathlib
from typing import List, Tuple

import pdfplumber
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec, CloudProvider

# ────────────────────────── 1.  CONFIG ──────────────────────────
load_dotenv(".env")

INDEX_NAME   = os.getenv("PINECONE_INDEX",  "manuals-small")
EMBED_MODEL  = os.getenv("embedding_model", "text-embedding-3-large")
DIMENSION    = 3072                     # ← text-embedding-3-large
CHUNK_TOKENS = 200                      # ≈ 230–250 words
OVERLAP      = 100                       # keep last 50 tokens as bridge
BATCH_EMBED  = 100                      # OpenAI batch size
BATCH_UPSERT = 100                      # Pinecone batch size

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
enc           = tiktoken.get_encoding("cl100k_base")

# ────────────────────────── 2.  HELPERS ─────────────────────────
def token_len(text: str | List[str]) -> int:
    """Approximate token count for a string OR list-of-strings."""
    if isinstance(text, list):
        text = " ".join(text)
    return len(enc.encode(text))

def pdf_to_chunks(path: str) -> Tuple[List[str], List[dict]]:
    import re

    filename = os.path.basename(path).lower()
    if "coffee" in filename:
        product = "coffee maker"
    elif "printer" in filename:
        product = "printer"
    elif "vacuum" in filename:
        product = "vacuum"
    else:
        product = "machine"

    chunks: List[str] = []
    metas : List[dict] = []

    sections: List[Tuple[str, List[str], int]] = []
    current_head = None
    current_lines: List[str] = []
    current_page: int = 0

    heading_re = re.compile(r"^\d+(\.\d+)*\s+[A-Z]")

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            for line in (page.extract_text() or "").splitlines():
                if heading_re.match(line.strip()):
                    if current_head is not None:
                        sections.append((current_head, current_lines, current_page))
                    current_head = line.strip()
                    current_lines = []
                    current_page = page_num
                current_lines.append(line)
        if current_head is not None:
            sections.append((current_head, current_lines, current_page))
        elif current_lines:
            sections.append(("", current_lines, page_num))

    for title, lines, page_num in sections:
        buffer: List[str] = []
        for line in lines:
            buffer.append(line)
            if token_len(buffer) >= CHUNK_TOKENS:
                chunk_text = "\n".join(buffer)
                if token_len(chunk_text) < 10:
                    continue
                chunks.append(chunk_text)
                metas.append({"title": title, "product": product, "page": page_num})
                encoded = enc.encode(chunk_text)
                overlap_tokens = encoded[-OVERLAP:] if len(encoded) > OVERLAP else encoded
                overlap_text = enc.decode(overlap_tokens)
                buffer = [overlap_text]
        if buffer:
            chunk_text = "\n".join(buffer)
            if token_len(chunk_text) >= 10:
                chunks.append(chunk_text)
                metas.append({"title": title, "product": product, "page": page_num})

    return chunks, metas

def embed_texts(batch: List[str]) -> List[List[float]]:
    """Embed texts in batches."""
    rsp = openai_client.embeddings.create(model=EMBED_MODEL, input=batch)
    return [d.embedding for d in rsp.data]

# ────────────────────────── 3.  PINECONE ────────────────────────
ENV = os.getenv("PINECONE_ENV")      # e.g. aws-us-east-1
if not ENV:
    sys.exit("🟥  PINECONE_ENV missing in .env")

cloud  = CloudProvider.AWS if ENV.startswith("aws") else CloudProvider.GCP
region = "-".join(ENV.split("-")[1:]) if ENV.startswith("aws") else ENV.rsplit("-", 1)[0]

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=ENV)

def ensure_index(dim: int) -> None:
    """Create the Pinecone index if it doesn't exist (idempotent)."""
    if INDEX_NAME in pc.list_indexes().names():
        info = pc.describe_index(INDEX_NAME)
        if info.dimension != dim:
            raise RuntimeError(
                f"Index '{INDEX_NAME}' exists but dim={info.dimension} ≠ {dim}"
            )
        return

    print(f"🛠️  Creating Pinecone index '{INDEX_NAME}' …")
    pc.create_index(
        name      = INDEX_NAME,
        dimension = dim,
        metric    = "cosine",
        spec      = ServerlessSpec(cloud=cloud, region=region),
    )
    # Wait until ready
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print("   …waiting for index to become ready")
        time.sleep(2)
    print("✅  Index ready.")

# ────────────────────────── 4.  MAIN INGEST ─────────────────────
def ingest(path: str, customer: str = "demo01") -> None:
    print("📖  Chunking PDF …")
    chunks, metas = pdf_to_chunks(path)

    print("🔢  Embedding …")
    vectors: List[List[float]] = []
    for i in range(0, len(chunks), BATCH_EMBED):
        vectors.extend(embed_texts(chunks[i : i + BATCH_EMBED]))

    ensure_index(DIMENSION)
    index = pc.Index(INDEX_NAME)

    print("🚚  Upserting into Pinecone …")
    items = [
        (
            f"{customer}-{uuid.uuid4().hex[:8]}",
            vec,
            {
                "customer": customer,
                "chunk":    i,
                "title":    metas[i]["title"],
                "text":     chunks[i].split("\n")[0][:200],   # preview for debugging: first line
                "product":  metas[i]["product"],
            },
        )
        for i, vec in enumerate(vectors)
    ]

    for j in range(0, len(items), BATCH_UPSERT):
        index.upsert(items[j : j + BATCH_UPSERT])

    print(
        f"✅  Ingested {len(chunks)} chunks "
        f"({len(vectors)} vectors) into '{INDEX_NAME}'."
    )

# ────────────────────────── 5.  CLI ENTRY ───────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Embed & upsert a PDF manual into Pinecone."
    )
    parser.add_argument("pdf", help="Path to the PDF manual")
    parser.add_argument("--customer", default="demo01", help="Tenant namespace")
    args = parser.parse_args()

    pdf_path = pathlib.Path(args.pdf)
    if not pdf_path.exists():
        sys.exit(f"🟥  PDF not found: {pdf_path}")

    ingest(str(pdf_path), args.customer)