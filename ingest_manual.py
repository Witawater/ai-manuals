#!/usr/bin/env python3
"""
ingest_manual.py  –  Chunk + embed + upsert one PDF into Pinecone
─────────────────────────────────────────────────────────────────
Usage:
    source venv/bin/activate
    python ingest_manual.py CoffeeMaker.pdf --customer demo01
"""

import os, uuid, time, pathlib, sys
from typing import List, Tuple

import pdfplumber, tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec, CloudProvider

# ── config ────────────────────────────────────────────────────────────────
load_dotenv(".env")

INDEX_NAME   = os.getenv("PINECONE_INDEX",  "manuals-small")
EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHUNK_TOKENS = 300         # ≈ 230-250 words
OVERLAP      = 50          # keep last 50 tokens as context bridge
BATCH_EMBED  = 100         # OpenAI embed batch size
BATCH_UPSERT = 100         # Pinecone upsert batch size

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
enc    = tiktoken.get_encoding("cl100k_base")

# ── helpers ───────────────────────────────────────────────────────────────
def token_len(lines: List[str]) -> int:
    """Return #tokens for a list of strings."""
    return len(enc.encode(" ".join(lines)))

def pdf_to_chunks(path: str) -> Tuple[List[str], List[dict]]:
    """
    Split a PDF into ~CHUNK_TOKENS token chunks.
    Returns  (chunks, metas)  where  metas[i]["title"]  holds the last heading seen.
    """
    chunks, metas, buffer = [], [], []
    current_head = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            for line in (page.extract_text() or "").splitlines():
                # crude heading detector
                if line.isupper() or line.startswith(("Using", "Cleaning", "To ")):
                    current_head = line.strip()
                buffer.append(line)

                if token_len(buffer) >= CHUNK_TOKENS:
                    chunks.append("\n".join(buffer))
                    metas.append({"title": current_head})
                    # keep overlap
                    overlap_tokens = enc.encode("\n".join(buffer))[-OVERLAP:]
                    buffer = [enc.decode(overlap_tokens)]
        # flush remainder
        if buffer:
            chunks.append("\n".join(buffer))
            metas.append({"title": current_head})
    return chunks, metas

def embed(batch: List[str]) -> List[List[float]]:
    """Embed a batch of texts and return the vectors."""
    rsp = client.embeddings.create(model=EMBED_MODEL, input=batch)
    return [d.embedding for d in rsp.data]

# ── pinecone client ───────────────────────────────────────────────────────
env = os.getenv("PINECONE_ENV")        # e.g. aws-us-east-1  or  us-central1-gcp
if not env:
    sys.exit("🟥  PINECONE_ENV missing in .env")

parts  = env.split("-")
cloud  = CloudProvider.AWS if parts[0] == "aws" else CloudProvider.GCP
region = "-".join(parts[1:]) if parts[0] == "aws" else "-".join(parts[:-1])

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=env)

# ── ingest one PDF ────────────────────────────────────────────────────────
def ingest(path: str, customer: str):
    print("📖  Chunking PDF …")
    chunks, metas = pdf_to_chunks(path)

    print("🔢  Embedding …")
    vectors: List[List[float]] = []
    for i in range(0, len(chunks), BATCH_EMBED):
        vectors.extend(embed(chunks[i : i + BATCH_EMBED]))

    # (re)create index if missing
    if INDEX_NAME not in pc.list_indexes().names():
        print(f"🛠️  Creating Pinecone index '{INDEX_NAME}' …")
        pc.create_index(
            name      = INDEX_NAME,
            dimension = len(vectors[0]),     # 1536 dims for `text-embedding-3-small`
            metric    = "cosine",
            spec      = ServerlessSpec(cloud=cloud, region=region),
        )
        while pc.describe_index(INDEX_NAME).status["ready"] is not True:
            print("   …waiting for index to become ready")
            time.sleep(2)

    index = pc.Index(INDEX_NAME)

    print("🚚  Upserting into Pinecone …")
    items = []
    for i, vec in enumerate(vectors):
        items.append(
            (
                f"{customer}-{uuid.uuid4().hex[:8]}",
                vec,
                {
                    "customer": customer,
                    "chunk":    i,
                    "title":    metas[i]["title"],
                    "text":     chunks[i][:200],    # preview for debugging
                },
            )
        )
    for j in range(0, len(items), BATCH_UPSERT):
        index.upsert(items[j : j + BATCH_UPSERT])

    print(f"✅  Ingested {len(chunks)} chunks ({len(vectors)} vectors) into '{INDEX_NAME}'.")

# ── CLI wrapper ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Embed & upsert a PDF manual.")
    p.add_argument("pdf", help="Path to the PDF file")
    p.add_argument("--customer", default="demo01", help="Namespace / tenant tag")
    args = p.parse_args()

    pdf_path = pathlib.Path(args.pdf)
    if not pdf_path.exists():
        sys.exit("🟥  PDF not found.")

    ingest(str(pdf_path), args.customer)
