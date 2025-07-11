#!/usr/bin/env python3
"""
ingest_manual.py
────────────────
Chunk ➜ embed ➜ upsert ONE PDF into Pinecone.

Usage
─────
    source venv/bin/activate
    python ingest_manual.py CoffeeMaker.pdf \
        --customer demo01 \
        --chunk_tokens 800 \
        --overlap 150
"""

from __future__ import annotations

import os, sys, time, uuid, pathlib, argparse
from typing import List, Tuple

import pdfplumber
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pinecone import Pinecone, ServerlessSpec, CloudProvider

# ────────────────────────── 1. CONFIG ──────────────────────────
load_dotenv(".env")

INDEX_NAME  = os.getenv("PINECONE_INDEX", "manuals-small")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
DIMENSION   = 3072                             # text-embedding-3-large

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
enc           = tiktoken.get_encoding("cl100k_base")

# ────────────────────────── 2. HELPERS ─────────────────────────
def token_len(txt: str) -> int:
    """Return approximate token length of a string."""
    return len(enc.encode(txt))

def retry(fn, *a, **kw):
    """Tiny exponential-back-off for OpenAI calls."""
    for attempt in range(5):
        try:
            return fn(*a, **kw)
        except RateLimitError:
            wait = 2 ** attempt
            print(f"⚠️  Rate-limited; retrying in {wait}s …")
            time.sleep(wait)
    raise RuntimeError("Too many rate-limit failures.")

# ───────────────────── 3. CHUNKING LOGIC ───────────────────────
def pdf_to_chunks(
    path: str,
    chunk_tokens: int,
    overlap: int,
) -> Tuple[List[str], List[dict]]:
    """Convert a PDF into overlapping chunks with metadata."""
    # crude product guess from filename
    fname = os.path.basename(path).lower()
    product = (
        "coffee maker" if "coffee" in fname else
        "printer"      if "printer" in fname else
        "vacuum"       if "vacuum"  in fname else
        "machine"
    )

    chunks: List[str]  = []
    metas : List[dict] = []

    with pdfplumber.open(path) as pdf:
        all_pages = [p.extract_text() or "" for p in pdf.pages]

    # Split on paragraphs first
    paragraphs: List[Tuple[str, int]] = []          # (text, page)
    for pg_no, page in enumerate(all_pages, start=1):
        for para in (page.split("\n\n") or [""]):
            para = para.strip()
            if para:
                paragraphs.append((para, pg_no))

    # ── helpers (must precede first call) ──────────────────────
    def _flush(buffer: List[Tuple[str, int]]):
        """Push current buffer into chunks & metadata lists."""
        text       = "\n\n".join(p for p, _ in buffer).strip()
        first_page = buffer[0][1]
        chunks.append(text)
        metas.append({
            "title":   "",          # can’t reliably detect here
            "product": product,
            "page":    first_page,
        })

    def _overlap_tail(buffer: List[Tuple[str, int]], overlap_tokens: int):
        """Return buffer holding the last N tokens of previous buffer."""
        if overlap_tokens == 0:
            return [], 0
        rev   = list(reversed(buffer))
        keep  = []
        tokens = 0
        for para, pg in rev:
            t = token_len(para)
            if tokens + t > overlap_tokens and keep:
                break
            keep.insert(0, (para, pg))
            tokens += t
        return keep, tokens
    # ───────────────────────────────────────────────────────────

    # Sliding window over paragraphs
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
            # start new buffer with overlap
            buf, buf_tokens = _overlap_tail(buf, overlap)

    if buf:
        _flush(buf)

    return chunks, metas

# ────────────────────────── 4. EMBEDDING ───────────────────────
def embed_texts(batch: List[str]) -> List[List[float]]:
    """Embed a batch of texts with retries."""
    rsp = retry(
        openai_client.embeddings.create,
        model=EMBED_MODEL,
        input=batch,
    )
    return [d.embedding for d in rsp.data]

# ────────────────────────── 5. PINECONE ────────────────────────
ENV = os.getenv("PINECONE_ENV")      # e.g. aws-us-east-1
if not ENV:
    sys.exit("🟥  PINECONE_ENV missing in .env")

cloud = CloudProvider.AWS if ENV.startswith("aws") else CloudProvider.GCP

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=ENV)

def ensure_index(dim: int):
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
        spec      = ServerlessSpec(cloud=cloud, region=ENV.split("-", 1)[-1]),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        print("   …waiting for index to become ready")
        time.sleep(2)
    print("✅  Index ready.")

# ────────────────────────── 6. INGEST ──────────────────────────
def ingest(
    pdf_path: str,
    customer: str = "demo01",
    chunk_tokens: int = 800,   # default matches CLI flag
    overlap: int = 150,
    dry_run: bool = False,
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
        print(f"   • embedded {min(i+batch_embed, len(chunks))}/{len(chunks)}")

    print("🚚  Upserting …")
    items = [
        (
            f"{customer}-{uuid.uuid4().hex[:8]}",
            vec,
            {
                **meta,
                "customer": customer,
                "chunk":    i,
                "text":     chunks[i],
                "tokens":   token_len(chunks[i]),
                "filename": os.path.basename(pdf_path),
            },
        )
        for i, (vec, meta) in enumerate(zip(vectors, metas))
    ]

    for j in range(0, len(items), batch_upsert):
        index.upsert(items[j : j + batch_upsert])
        print(f"   • upserted {min(j+batch_upsert, len(items))}/{len(items)}")

    print(f"✅  Ingested {len(chunks)} chunks into '{INDEX_NAME}'.")

# ────────────────────────── 7. CLI ─────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Embed & upsert a PDF manual into Pinecone."
    )
    ap.add_argument("pdf", help="Path to the PDF")
    ap.add_argument("--customer", default="demo01", help="Tenant name")
    ap.add_argument("--chunk_tokens", type=int, default=800, help="Max tokens / chunk")
    ap.add_argument("--overlap", type=int, default=150, help="Overlap tokens")
    ap.add_argument("--dry", action="store_true", help="Preview chunks only")
    args = ap.parse_args()

    if not pathlib.Path(args.pdf).exists():
        sys.exit(f"🟥  PDF not found: {args.pdf}")

    ingest(
        pdf_path     = args.pdf,
        customer     = args.customer,
        chunk_tokens = args.chunk_tokens,
        overlap      = args.overlap,
        dry_run      = args.dry,
    )
