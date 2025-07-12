#!/usr/bin/env python3
"""
ingest_manual.py  –  Chunk ➜ embed ➜ upsert ONE PDF into Pinecone.
Set FAST_INGEST=1 to skip expensive tiktoken calls (≈20× faster).
"""

from __future__ import annotations
import os, sys, time, uuid, pathlib, argparse
from typing import List, Tuple

import pdfplumber
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pinecone import Pinecone, ServerlessSpec, CloudProvider

# ─── optional deps ────────────────────────────────────────────
FAST_MODE = bool(os.getenv("FAST_INGEST", "0") == "1")
if not FAST_MODE:
    import tiktoken
else:
    tiktoken = None
try:
    from tqdm import tqdm  # progress bar
except ImportError:        # optional
    def tqdm(x, **k): return x

# ────────────────────────── 1. CONFIG ─────────────────────────
load_dotenv(".env")

INDEX_NAME  = os.getenv("PINECONE_INDEX", "manuals-small")
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
DIMENSION   = 3072

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
enc = tiktoken.get_encoding("cl100k_base") if tiktoken else None

# ────────────────────────── 2. HELPERS ────────────────────────
def token_len(txt: str) -> int:
    """Approx token count; fast heuristic when FAST_INGEST=1."""
    if FAST_MODE or enc is None:
        return max(1, len(txt) // 4)     # ≈4 chars / token
    return len(enc.encode(txt))

def retry(fn, *a, **kw):
    for attempt in range(5):
        try:
            return fn(*a, **kw)
        except RateLimitError:
            wait = 2 ** attempt
            print(f"⚠️  Rate-limited; retrying in {wait}s …")
            time.sleep(wait)
    raise RuntimeError("Too many rate-limit failures.")

# ───────────────────── 3. CHUNKING LOGIC ─────────────────────
def pdf_to_chunks(path: str, chunk_tokens: int, overlap: int
) -> Tuple[List[str], List[dict]]:
    fname = os.path.basename(path).lower()
    product = ("coffee maker" if "coffee" in fname else
               "printer"      if "printer" in fname else
               "vacuum"       if "vacuum"  in fname else
               "machine")

    chunks: List[str] = []
    metas : List[dict] = []

    with pdfplumber.open(path) as pdf:
        pages = [p.extract_text() or "" for p in tqdm(pdf.pages,
                 desc="Extract", unit="pg")]

    paragraphs: List[Tuple[str, int]] = []
    for pg_no, page in enumerate(pages, 1):
        for para in (page.split("\n\n") or [""]):
            para = para.strip()
            if para:
                paragraphs.append((para, pg_no))

    # ─ helpers (need before first call) ───────────────────────
    def _flush(buffer: List[Tuple[str, int]]):
        text = "\n\n".join(p for p, _ in buffer).strip()
        chunks.append(text)
        metas.append({"title": "", "product": product, "page": buffer[0][1]})

    def _overlap_tail(buffer: List[Tuple[str, int]], ov_tokens: int):
        if ov_tokens == 0 or not buffer:
            return [], 0
        rev, keep, tokens = list(reversed(buffer)), [], 0
        for para, pg in rev:
            t = token_len(para)
            if tokens + t > ov_tokens and keep:
                break
            keep.insert(0, (para, pg))
            tokens += t
        return keep, tokens
    # ──────────────────────────────────────────────────────────

    buf, buf_tokens, i = [], 0, 0
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

# ───────────────────────── 4. EMBEDDING ───────────────────────
def embed_texts(batch: List[str]) -> List[List[float]]:
    rsp = retry(openai_client.embeddings.create,
                model=EMBED_MODEL, input=batch)
    return [d.embedding for d in rsp.data]

# ───────────────────────── 5. PINECONE ────────────────────────
ENV = os.getenv("PINECONE_ENV")
if not ENV:
    sys.exit("🟥  PINECONE_ENV missing in .env")
cloud = CloudProvider.AWS if ENV.startswith("aws") else CloudProvider.GCP
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=ENV)

def ensure_index(dim: int):
    if INDEX_NAME in pc.list_indexes().names():
        if pc.describe_index(INDEX_NAME).dimension != dim:
            raise RuntimeError(
                f"Index '{INDEX_NAME}' exists but wrong dim.")
        return
    print(f"🛠️  Creating index '{INDEX_NAME}' …")
    pc.create_index(
        name=INDEX_NAME, dimension=dim, metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=ENV.split('-', 1)[-1]),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
    print("✅  Index ready")

# ───────────────────────── 6. INGEST ──────────────────────────
def ingest(pdf_path: str, customer: str = "demo01",
           chunk_tokens: int = 800, overlap: int = 150,
           dry_run: bool = False,
           batch_embed: int = 100, batch_upsert: int = 100):

    print("📖  Chunking PDF …")
    chunks, metas = pdf_to_chunks(pdf_path, chunk_tokens, overlap)
    print(f"   → {len(chunks)} chunks")

    if dry_run:
        for i, (ch, meta) in enumerate(zip(chunks, metas)):
            print(f"\n[{i}] p{meta['page']} – {token_len(ch)} tokens\n"
                  f"{ch[:400]}…")
        return

    ensure_index(DIMENSION)
    index = pc.Index(INDEX_NAME)

    print("🔢  Embedding …")
    vectors: List[List[float]] = []
    for i in tqdm(range(0, len(chunks), batch_embed),
                  desc="Embed", unit="batch"):
        vectors.extend(embed_texts(chunks[i:i+batch_embed]))

    print("🚚  Upserting …")
    items = [
        (f"{customer}-{uuid.uuid4().hex[:8]}", vec,
         {**meta, "customer": customer, "chunk": i,
          "text": chunks[i], "tokens": token_len(chunks[i]),
          "filename": os.path.basename(pdf_path)})
        for i, (vec, meta) in enumerate(zip(vectors, metas))
    ]
    for j in tqdm(range(0, len(items), batch_upsert),
                  desc="Upsert", unit="batch"):
        index.upsert(items[j:j+batch_upsert])

    print(f"✅  Ingested {len(chunks)} chunks into '{INDEX_NAME}'.")

# ───────────────────────── 7. CLI ─────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Embed & upsert a PDF manual into Pinecone.")
    ap.add_argument("pdf", help="Path to the PDF")
    ap.add_argument("--customer", default="demo01", help="Tenant name")
    ap.add_argument("--chunk_tokens", type=int, default=800)
    ap.add_argument("--overlap",     type=int, default=150)
    ap.add_argument("--dry", action="store_true", help="Preview only")
    args = ap.parse_args()

    if not pathlib.Path(args.pdf).exists():
        sys.exit("🟥  PDF not found")

    ingest(args.pdf, args.customer,
           args.chunk_tokens, args.overlap, args.dry)
