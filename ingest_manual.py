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
CHUNK_TOKENS = 300
OVERLAP      = 50
BATCH_EMBED  = 100
BATCH_UPSERT = 100

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
enc    = tiktoken.get_encoding("cl100k_base")

# ── helpers ───────────────────────────────────────────────────────────────
def token_len(lines: List[str]) -> int:
    return len(enc.encode(" ".join(lines)))

def pdf_to_chunks(path: str) -> Tuple[List[str], List[dict]]:
    """
    Split PDF into ~CHUNK_TOKENS-token chunks.
    Returns (chunks, metas) where metas[i]["title"] holds the last heading seen.
    """
    chunks, metas, buffer = [], [], []
    current_head = ""
    for page in pdfplumber.open(path).pages:
        for line in (page.extract_text() or "").splitlines():
            if line.isupper() or line.startswith(("Using", "Cleaning", "To ")):
                current_head = line.strip()
            buffer.append(line)
            if token_len(buffer) >= CHUNK_TOKENS:
                chunks.append("\n".join(buffer))
                metas.append({"title": current_head})
                # keep overlap
                overlap_tokens = enc.encode("\n".join(buffer))[-OVERLAP:]
                buffer = [enc.decode(overlap_tokens)]
    if buffer:
        chunks.append("\n".join(buffer))
        metas.append({"title": current_head})
    return chunks, metas

def embed(batch: List[str]) -> List[List[float]]:
    rsp = client.embeddings.create(model=EMBED_MODEL, input=batch)
    return [d.embedding for d in rsp.data]

# ── pinecone client ───────────────────────────────────────────────────────
env = os.getenv("PINECONE_ENV")            # aws-us-east-1 or us-central1-gcp
parts = env.split("-")
cloud  = CloudProvider.AWS if parts[0] == "aws" else CloudProvider.GCP
region = "-".join(parts[1:]) if parts[0] == "aws" else "-".join(parts[:-1])

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=env)

# ── ingest one PDF ────────────────────────────────────────────────────────
def ingest(path: str, customer: str):
    chunks, metas = pdf_to_chunks(path)
    vectors = []
    for i in range(0, len(chunks), BATCH_EMBED):
        vectors.extend(embed(chunks[i : i + BATCH_EMBED]))

    # (re)create index if missing
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name       = INDEX_NAME,
            dimension  = len(vectors[0]),         # 1 536 dims for small model
            metric     = "cosine",
            spec       = ServerlessSpec(cloud=cloud, region=region),
        )
        # wait until ready
        while pc.describe_index(INDEX_NAME).status["ready"] is not True:
            time.sleep(2)

    index = pc.Index(INDEX_NAME)

    items = []
    for i, vec in enumerate(vectors):
        items.append(
            (
                f"{customer}-{uuid.uuid4().hex[:8]}",
                vec,
                {
                    "customer": customer,
                    "chunk": i,
                    "title": metas[i]["title"],
                    "text": chunks[i][:200],
                },
            )
        )
    for j in range(0, len(items), BATCH_UPSERT):
        index.upsert(items[j : j + BATCH_UPSERT])

    print(f"✅  Ingested {len(chunks)} chunks into '{INDEX_NAME}'.")

# ── CLI wrapper ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("pdf"), p.add_argument("--customer", default="demo01")
    args = p.parse_args()
    if not pathlib.Path(args.pdf).exists():
        sys.exit("🟥 PDF not found.")
    ingest(args.pdf, args.customer)
