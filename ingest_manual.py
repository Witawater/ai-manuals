#!/usr/bin/env python3
"""
ingest_manual.py – Chunk → Embed → Upsert ONE PDF into Pinecone
────────────────────────────────────────────────────────────────
Usage:
    python ingest_manual.py CoffeeMaker.pdf \
        --customer demo01 \
        --chunk_tokens 800 \
        --overlap 150

Key points
──────────
• Reads OPENAI + Pinecone keys from .env.
• Embedding dimension inferred from `OPENAI_EMBED_MODEL`.
• Robustly parses PINECONE_ENV (e.g. aws-us-east-1, gcp-us-central1).
• Auto‑creates the index if missing and validates its dimension.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time
import uuid
from typing import List, Tuple

import pdfplumber
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pinecone import ServerlessSpec, Pinecone

# ─────────────── 1. CONFIG ───────────────
load_dotenv(".env")

INDEX_NAME: str = os.getenv("PINECONE_INDEX", "manuals-large")
EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
DIMENSION: int = 3072 if "large" in EMBED_MODEL else 1536

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
enc = tiktoken.get_encoding("cl100k_base")

# ─────────────── 2. HELPERS ───────────────

def token_len(txt: str) -> int:
    """Return token length (tiktoken count) instead of characters."""
    return len(enc.encode(txt))


def retry(fn, *a, **kw):
    """Simple exponential‑backoff retry for OpenAI RateLimitError."""
    for attempt in range(5):
        try:
            return fn(*a, **kw)
        except RateLimitError:
            wait = 2 ** attempt
            print(f"⚠️  Rate‑limited; retrying in {wait}s …")
            time.sleep(wait)
    raise RuntimeError("Too many rate‑limit failures.")


# ───── 3. CHUNKING LOGIC ─────

def pdf_to_chunks(
    path: str,
    chunk_tokens: int,
    overlap: int,
) -> Tuple[List[str], List[dict]]:
    """Return (chunks, metadata) lists for the given PDF."""
    fname = os.path.basename(path).lower()
    product = (
        "coffee maker"
        if "coffee" in fname
        else "printer"
        if "printer" in fname
        else "vacuum"
        if "vacuum" in fname
        else "machine"
    )

    chunks: List[str] = []
    metas: List[dict] = []

    with pdfplumber.open(path) as pdf:
        all_pages = [p.extract_text() or "" for p in pdf.pages]

    paragraphs: List[Tuple[str, int]] = []
    for pg_no, page in enumerate(all_pages, start=1):
        for para in page.splitlines():
            if para.strip():
                paragraphs.append((para.strip(), pg_no))

    def _flush(buffer):
        text = "\n\n".join(p for p, _ in buffer).strip()
        first_page = buffer[0][1]
        chunks.append(text)
        metas.append(
            {
                "title": "",
                "product": product,
                "page": first_page,
                "filename": os.path.basename(path),
            }
        )

    def _overlap_tail(buffer, overlap_tokens):
        if overlap_tokens == 0:
            return [], 0
        rev = list(reversed(buffer))
        keep: List[Tuple[str, int]] = []
        tokens = 0
        for para, pg in rev:
            t = token_len(para)
            if tokens + t > overlap_tokens and keep:
                break
            keep.insert(0, (para, pg))
            tokens += t
        return keep, tokens

    buf: List[Tuple[str, int]] = []
    buf_tokens = 0
    i = 0
    while i < len(paragraphs):
        para, pg_no = paragraphs[i]
        t_para = token_len(para)

        if buf_tokens + t_para <= chunk_tokens or not buf:
            buf.append((para, pg_no))
            buf_tokens += t_para
            i += 1
        else:
            _flush(buf)
            buf, buf_tokens = _overlap_tail(buf, overlap)

    if buf:
        _flush(buf)

    return chunks, metas


# ───── 4. EMBEDDING ─────

def embed_texts(batch: List[str]) -> List[List[float]]:
    rsp = retry(openai_client.embeddings.create, model=EMBED_MODEL, input=batch)
    return [d.embedding for d in rsp.data]


# ───── 5. PINECONE ─────
ENV = os.getenv("PINECONE_ENV")
if not ENV:
    sys.exit("🟥  PINECONE_ENV missing in .env")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def _parse_cloud_and_region(env_str: str):
    """Convert env string → (cloud, region) for ServerlessSpec."""
    parts = env_str.split("-")
    cloud = "aws" if parts[0].lower() == "aws" else "gcp"
    region = "-".join(parts[-2:]) if len(parts) >= 3 else "us-east-1"
    return cloud, region


def ensure_index(dim: int):
    """Create Pinecone index if missing; check dimension otherwise."""
    cloud, region = _parse_cloud_and_region(ENV)

    if INDEX_NAME in pc.list_indexes().names():
        info = pc.describe_index(INDEX_NAME)
        if info.dimension != dim:
            raise RuntimeError(
                f"Index '{INDEX_NAME}' exists but dim={info.dimension} ≠ {dim}"
            )
        return

    print(f"🛠️  Creating Pinecone index '{INDEX_NAME}' …")
    pc.create_index(
        name=INDEX_NAME,
        dimension=dim,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print("   …waiting for index to become ready")
        time.sleep(2)
    print("✅  Index ready.")


# ───── 6. INGEST ─────

def ingest(
    pdf_path: str,
    customer: str,
    chunk_tokens: int,
    overlap: int,
    dry_run: bool,
    batch_embed: int = 100,
    batch_upsert: int = 100,
):
    print("📖  Chunking PDF …")
    chunks, metas = pdf_to_chunks(pdf_path, chunk_tokens, overlap)
    print(f"   → {len(chunks)} chunks")

    if dry_run:
        for i, (ch, meta) in enumerate(zip(chunks, metas)):
            print(f"\n[{i}] page {meta['page']} – {token_len(ch)} tokens\n{ch[:400]}…")
        return

    ensure_index(DIMENSION)
    index = pc.Index(INDEX_NAME)

    print("🔢  Embedding …")
    vectors: List[List[float]] = []
    for i in range(0, len(chunks), batch_embed):
        vectors.extend(embed_texts(chunks[i : i + batch_embed]))
        print(f"   • embedded {min(i + batch_embed, len(chunks))}/{len(chunks)}")

    print("🚚  Upserting …")
    items = [
        (
            f"{customer}-{uuid.uuid4().hex[:8]}",
            vec,
            {
                **meta,
                "customer": customer,
                "chunk": i,
                "text": chunks[i],
                "tokens": token_len(chunks[i]),
            },
        )
        for i, (vec, meta) in enumerate(zip(vectors, metas))
    ]
    print(f"   ➤ total items: {len(items)}")

    for j in range(0, len(items), batch_upsert):
        index.upsert(items[j : j + batch_upsert])
        print(f"   • upserted {min(j + batch_upsert, len(items))}/{len(items)}")

    print(f"✅  Ingested {len(chunks)} chunks into '{INDEX_NAME}'.")


# ───── 7. CLI ─────

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Embed & upsert a PDF manual into Pinecone.")
    ap.add_argument("pdf", help="Path to the PDF")
    ap.add
