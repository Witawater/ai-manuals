#!/usr/bin/env python3
"""
ingest_manual.py – Chunk ► Embed ► Upsert ONE PDF into Pinecone
────────────────────────────────────────────────────────────────
• Model/dimension guard (auto-maps model → dim; verifies index)
• Real token-overlap chunking (sliding window)
• Deterministic vector IDs (doc_id + chunk_no)
• Streamed batching (embed → upsert per batch)
• Trimmed metadata (preview only) + doc/customer/pg info
• Namespace per doc_id (easy isolation/deletion)
"""

from __future__ import annotations
import argparse, os, pathlib, sys, time, uuid, re
from typing import List, Tuple, Callable, Dict, Any, Iterable

import pdfplumber, tiktoken
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
from pinecone import Pinecone, ServerlessSpec

# ───────────── 1) CONFIG ─────────────
load_dotenv(".env")

INDEX_NAME   = os.getenv("PINECONE_INDEX",  "manuals-large")
EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
PINECONE_ENV = os.getenv("PINECONE_ENV") or sys.exit("🟥  PINECONE_ENV missing")
OPENAI_KEY   = os.getenv("OPENAI_API_KEY") or sys.exit("🟥  OPENAI_API_KEY missing")
PINECONE_KEY = os.getenv("PINECONE_API_KEY") or sys.exit("🟥  PINECONE_API_KEY missing")

MODEL_DIM = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}.get(EMBED_MODEL)
if not MODEL_DIM:
    sys.exit(f"🟥 Unknown embedding model: {EMBED_MODEL}")

openai_client = OpenAI(api_key=OPENAI_KEY)
enc = tiktoken.get_encoding("cl100k_base")

# ───────────── 2) HELPERS ─────────────
def token_len(txt: str) -> int:
    return len(enc.encode(txt))

def retry(fn, *a, **kw):
    backoff = [1, 2, 4, 8, 16]
    for i, wait in enumerate(backoff):
        try:
            return fn(*a, **kw)
        except RateLimitError:
            print(f"⚠️  Rate-limited; retry {i+1}/{len(backoff)} in {wait}s …")
            time.sleep(wait)
    # last try without catching to surface the error context
    return fn(*a, **kw)

def clean_text(s: str) -> str:
    s = s.replace("\x00", " ").strip()
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s

def preview(s: str, limit: int = 400) -> str:
    s = s.strip()
    if len(s) <= limit: return s
    return s[:limit] + "…"

# ───────────── 3) CHUNKING (with overlap) ─────────────
def pdf_to_chunks(path: str, chunk_tokens: int, overlap_tokens: int) -> Tuple[List[str], List[dict]]:
    """Naive paragraph packer + token-based overlap between chunks."""
    fname = os.path.basename(path).lower()
    product = (
        "coffee maker" if "coffee" in fname else
        "printer"      if "printer" in fname else
        "vacuum"       if "vacuum"  in fname else
        "machine"
    )

    with pdfplumber.open(path) as pdf:
        pages = [clean_text(p.extract_text(layout=True) or "") for p in pdf.pages]

    paragraphs: List[Tuple[str, int]] = []
    for pg_no, pg in enumerate(pages, 1):
        for line in pg.splitlines():
            line = line.strip()
            if line:
                paragraphs.append((line, pg_no))

    chunks, metas = [], []
    buf: List[Tuple[str, int]] = []
    buf_tokens = 0
    i = 0

    # pack paragraphs up to chunk_tokens
    while i < len(paragraphs):
        para, pg = paragraphs[i]
        t_para = token_len(para)
        if not buf:
            buf.append((para, pg))
            buf_tokens = t_para
            i += 1
            continue

        if buf_tokens + t_para <= chunk_tokens:
            buf.append((para, pg))
            buf_tokens += t_para
            i += 1
        else:
            # flush current buffer
            text = "\n\n".join(p for p, _ in buf).strip()
            first_page = buf[0][1]
            chunks.append(text)
            metas.append({
                "product": product,
                "page": first_page,
                "filename": os.path.basename(path),
                "title": "",
            })

            # build next buffer with tail-overlap
            if overlap_tokens > 0:
                # take tokens from end until we reach overlap_tokens
                tail: List[Tuple[str, int]] = []
                total = 0
                for p, pg_ in reversed(buf):
                    t = token_len(p)
                    if total + t > overlap_tokens and tail:
                        break
                    tail.append((p, pg_))
                    total += t
                tail.reverse()
                buf = tail[:]
                buf_tokens = sum(token_len(p) for p, _ in buf)
            else:
                buf = []
                buf_tokens = 0

    if buf:
        text = "\n\n".join(p for p, _ in buf).strip()
        first_page = buf[0][1] if buf else 1
        chunks.append(text)
        metas.append({
            "product": product,
            "page": first_page,
            "filename": os.path.basename(path),
            "title": "",
        })

    return chunks, metas

# ───────────── 4) EMBEDDING ─────────────
def embed_texts(batch: List[str]) -> List[List[float]]:
    resp = retry(openai_client.embeddings.create, model=EMBED_MODEL, input=batch)
    return [d.embedding for d in resp.data]

# ───────────── 5) PINECONE ─────────────
pc = Pinecone(api_key=PINECONE_KEY)

def _cloud_region(env_str: str):
    """
    Accepts values like 'aws-us-east-1' or 'gcp-us-central1'.
    Returns ('aws'|'gcp', 'us-east-1'|'us-central1'|...).
    """
    parts = env_str.split("-")
    cloud = "aws" if parts[0].lower() == "aws" else "gcp"
    region = "-".join(parts[1:]) if parts[0].lower() in ("aws", "gcp") else "-".join(parts[-2:])
    return cloud, region

def ensure_index(expected_dim: int):
    cloud, region = _cloud_region(PINECONE_ENV)
    names = pc.list_indexes().names()
    if INDEX_NAME in names:
        desc = pc.describe_index(INDEX_NAME)
        dim = getattr(desc, "dimension", None) or getattr(desc, "spec", {}).get("dimension")
        if dim and int(dim) != int(expected_dim):
            raise RuntimeError(f"🟥 Pinecone index '{INDEX_NAME}' dimension {dim} != expected {expected_dim} for {EMBED_MODEL}")
        return
    print(f"🛠️  Creating '{INDEX_NAME}' ({expected_dim}D, {cloud}/{region}) …")
    pc.create_index(
        name=INDEX_NAME,
        dimension=expected_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    # wait ready
    while True:
        if pc.describe_index(INDEX_NAME).status.get("ready"):
            break
        time.sleep(2)
    print("✅  Index ready")

# ───────────── 6) INGEST ─────────────
def ingest(
    pdf_path: str,
    customer: str,
    chunk_tokens: int,
    overlap: int,
    dry_run: bool,
    batch_embed: int = 100,
    batch_upsert: int = 100,
    *,
    progress_cb: Callable[[int], None] | None = None,
    common_meta: Dict[str, Any] | None = None,
    doc_id: str = "",
):
    if not doc_id:
        # make a stable doc_id if not provided, but caller should pass it
        doc_id = uuid.uuid4().hex

    # 1) Chunk
    print("📖  Chunking …")
    chunks, metas = pdf_to_chunks(pdf_path, chunk_tokens, overlap)
    total = len(chunks)
    if total == 0:
        raise RuntimeError(f"No text extracted from {pdf_path}")
    total_tokens = sum(token_len(c) for c in chunks)
    print(f"🧠  {total} chunks, {total_tokens} tokens total (window={chunk_tokens}, overlap={overlap})")

    if dry_run:
        for i,(c,m) in enumerate(zip(chunks, metas)):
            print(f"[{i}] pg {m['page']} {token_len(c)} tokens\n{c[:120]}…")
        return

    # 2) Index guard
    ensure_index(MODEL_DIM)
    index = pc.Index(INDEX_NAME)

    # 3) Stream: embed + upsert per batch (low memory, fast feedback)
    meta_common = (common_meta or {}).copy()
    meta_common["customer"] = customer
    meta_common["doc_id"]   = doc_id

    done = 0
    for start in range(0, total, batch_embed):
        end = min(start + batch_embed, total)
        batch_texts = chunks[start:end]
        batch_metas = metas[start:end]

        # Embed
        vectors = embed_texts(batch_texts)

        # Build vector payloads (deterministic IDs + trimmed metadata)
        vecs = []
        for i, (vec, m, txt) in enumerate(zip(vectors, batch_metas, batch_texts), start):
            vid = f"{doc_id}:{i:05d}"   # deterministic; safe to re-ingest/overwrite
            md = {
                **m,
                **meta_common,
                "chunk": i,
                "tokens": token_len(txt),
                "preview": preview(txt, 500),  # keep Pinecone small
            }
            vecs.append({"id": vid, "values": vec, "metadata": md})

        # Upsert (namespace per doc for easy isolation)
        index.upsert(vectors=vecs, namespace=doc_id)

        done = end
        if progress_cb:
            progress_cb(done)
        print(f"   • Upserted {done}/{total}")

    if progress_cb:
        progress_cb(total)
    print(f"✅  Ingested {total} chunks into '{INDEX_NAME}' (ns={doc_id})")

# ───────────── 7) CLI ─────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf")
    ap.add_argument("--customer", default="demo01")
    ap.add_argument("--chunk_tokens", type=int, default=400)
    ap.add_argument("--overlap",      type=int, default=80)
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--batch_embed", type=int, default=100)
    ap.add_argument("--batch_upsert", type=int, default=100)
    ap.add_argument("--doc_id", default="")
    args = ap.parse_args()

    ingest(
        args.pdf,
        args.customer,
        args.chunk_tokens,
        args.overlap,
        args.dry_run,
        batch_embed=args.batch_embed,
        batch_upsert=args.batch_upsert,
        doc_id=args.doc_id,
    )
