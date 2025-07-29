#!/usr/bin/env python3
"""
ingest_manual.py – Chunk ► Embed ► Upsert ONE PDF into Pinecone
────────────────────────────────────────────────────────────────
✓ Uses text-embedding-3-large  (3072-dim)  
✓ Writes into the new “manuals-large” index  
✓ Progress-callback & common-metadata unchanged
"""

from __future__ import annotations
import argparse, os, pathlib, sys, time, uuid
from typing import List, Tuple, Callable, Dict, Any

import pdfplumber, tiktoken
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pinecone import Pinecone, ServerlessSpec

# ───────────── 1. CONFIG ─────────────
load_dotenv(".env")

INDEX_NAME   = os.getenv("PINECONE_INDEX",  "manuals-large")
EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
DIMENSION    = 3072                            # 3-large → 3072 dims

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
enc           = tiktoken.get_encoding("cl100k_base")

# ───────────── 2. HELPERS ─────────────
def token_len(txt: str) -> int:
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

# ───── 3. CHUNKING (unchanged, layout=True keeps tables) ─────
def pdf_to_chunks(path: str, chunk_tokens: int, overlap: int) -> Tuple[List[str], List[dict]]:
    fname = os.path.basename(path).lower()
    product = (
        "coffee maker" if "coffee" in fname else
        "printer"      if "printer" in fname else
        "vacuum"       if "vacuum"  in fname else
        "machine"
    )
    chunks, metas = [], []
    with pdfplumber.open(path) as pdf:
        pages = [p.extract_text(layout=True) or "" for p in pdf.pages]

    paragraphs: List[Tuple[str, int]] = [
        (line.strip(), pg_no) for pg_no, pg in enumerate(pages, 1)
        for line in pg.splitlines() if line.strip()
    ]

    def _flush(buf):
        text       = "\n\n".join(p for p, _ in buf).strip()
        first_page = buf[0][1]
        chunks.append(text)
        metas.append({
            "title": "",
            "product": product,
            "page": first_page,
            "filename": os.path.basename(path),
        })

    buf, buf_tokens, i = [], 0, 0
    while i < len(paragraphs):
        para, pg = paragraphs[i]
        t_para   = token_len(para)
        if buf_tokens + t_para <= chunk_tokens or not buf:
            buf.append((para, pg)); buf_tokens += t_para; i += 1
        else:
            _flush(buf); buf, buf_tokens = [], 0
    if buf: _flush(buf)
    return chunks, metas

# ───── 4. EMBEDDING ─────
def embed_texts(batch: List[str]) -> List[List[float]]:
    resp = retry(openai_client.embeddings.create, model=EMBED_MODEL, input=batch)
    return [d.embedding for d in resp.data]

# ───── 5. PINECONE ─────
ENV = os.getenv("PINECONE_ENV") or sys.exit("🟥  PINECONE_ENV missing")
pc  = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def _cloud_region(env_str: str):
    p = env_str.split("-")
    return ("aws" if p[0].lower() == "aws" else "gcp", "-".join(p[-2:]))

def ensure_index(dim: int):
    cloud, region = _cloud_region(ENV)
    if INDEX_NAME in pc.list_indexes().names():
        if pc.describe_index(INDEX_NAME).dimension != dim:
            raise RuntimeError("Pinecone dimension mismatch")
        return
    print(f"🛠️  Creating '{INDEX_NAME}' …")
    pc.create_index(
        name=INDEX_NAME, dimension=dim, metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region)
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
    print("✅  Index ready")

# ───── 6. INGEST ─────
def ingest(pdf_path: str, customer: str, chunk_tokens: int, overlap: int,
           dry_run: bool, batch_embed: int = 100, batch_upsert: int = 100,
           *, progress_cb: Callable[[int], None]|None = None,
           common_meta: Dict[str, Any]|None = None):
    print("📖  Chunking …")
    chunks, metas = pdf_to_chunks(pdf_path, chunk_tokens, overlap)
    total = len(chunks)
    if dry_run:
        for i,(c,m) in enumerate(zip(chunks, metas)):
            print(f"[{i}] pg {m['page']} {token_len(c)} tokens\n{c[:120]}…")
        return

    ensure_index(DIMENSION)
    index = pc.Index(INDEX_NAME)

    print("🔢  Embedding …")
    vectors = []
    for i in range(0, total, batch_embed):
        vectors.extend(embed_texts(chunks[i:i+batch_embed]))
        print(f"   • {min(i+batch_embed,total)}/{total}")

    print("🚚  Upserting …")
    meta_common = common_meta or {}
    items = [
        (f"{customer}-{uuid.uuid4().hex[:8]}",
         vec,
         {**m, **meta_common, "customer":customer,
          "chunk":i, "text":chunks[i], "tokens":token_len(chunks[i])})
        for i,(vec,m) in enumerate(zip(vectors, metas))
    ]
    for j in range(0,len(items),batch_upsert):
        index.upsert(items[j:j+batch_upsert])
        if progress_cb: progress_cb(min(j+batch_upsert,len(items)))
    if progress_cb: progress_cb(len(items))
    print(f"✅  Ingested {total} chunks into '{INDEX_NAME}'")

# ───── 7. CLI ─────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf"); ap.add_argument("--customer", default="demo01")
    ap.add_argument("--chunk_tokens", type=int, default=400)
    ap.add_argument("--overlap",      type=int, default=80)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--batch_embed", type=int, default=100)
    ap.add_argument("--batch_upsert", type=int, default=100)
    args = ap.parse_args()

    ingest(args.pdf, args.customer, args.chunk_tokens, args.overlap,
           args.dry_run, batch_embed=args.batch_embed,
           batch_upsert=args.batch_upsert)
