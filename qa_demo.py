#!/usr/bin/env python3
"""
qa_demo.py – Ask questions against the Pinecone index
──────────────────────────────────────────────────────
Quick smoke‑test:

    python -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    python qa_demo.py
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, List

import dotenv
from openai import OpenAI
from pinecone import CloudProvider, Pinecone, ServerlessSpec

# ───── 1. Secrets & clients ─────
dotenv.load_dotenv(".env")

DEBUG = True

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o")
INDEX_NAME = os.getenv("PINECONE_INDEX", "manuals-small")

# Determine embedding dimension from the model name (per OpenAI docs)
DIM = 3072 if "large" in EMBED_MODEL else 1536

ENV = os.getenv("PINECONE_ENV", "")
REGION = (ENV.split("-", 1)[-1] or "us-east1").lower()
cloud = CloudProvider.AWS if "aws" in ENV else CloudProvider.GCP

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=ENV)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ───── 2. Ensure Pinecone index exists ─────
if INDEX_NAME not in pc.list_indexes().names():
    print(f"ℹ️  '{INDEX_NAME}' missing – creating a new one …")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=REGION),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
    print("✅  Index ready.")
else:
    info = pc.describe_index(INDEX_NAME)
    if info.dimension != DIM:
        raise RuntimeError(
            f"Index '{INDEX_NAME}' has dim={info.dimension}, expected {DIM}.\n"
            "Either delete it or change your config."
        )

idx = pc.Index(INDEX_NAME)

# ───── 3. Q‑and‑A helper ─────

def chat(
    question: str,
    customer: str = "demo01",
    top_k: int = 50,
    concise: bool = False,
    fallback: bool = True,
    rerank_keep: int = 4,
    fallback_cut: float = 0.35,
) -> Dict[str, object]:
    """Ask *question* and return an answer dict."""

    # 3‑1) Embed & initial search
    q_vec = (
        client.embeddings.create(model=EMBED_MODEL, input=question).data[0].embedding
    )
    resp = idx.query(
        vector=q_vec,
        top_k=top_k,
        filter={"customer": {"$eq": customer}},
        include_metadata=True,
    )

    product_name = (
        resp.matches[0].metadata.get("product", "machine") if resp.matches else "machine"
    )

    def normalise(q: str, name: str) -> str:
        return re.sub(
            r"\b(the\s)?(machine|unit|device|appliance)\b",
            name,
            q,
            flags=re.IGNORECASE,
        )

    q_norm = normalise(question, product_name)
    q_vec = (
        client.embeddings.create(model=EMBED_MODEL, input=q_norm).data[0].embedding
    )
    resp = idx.query(
        vector=q_vec,
        top_k=top_k,
        filter={"customer": {"$eq": customer}},
        include_metadata=True,
    )

    if DEBUG:
        print(f"\nℹ️ Top {len(resp.matches)} retrieved chunks:")
        for i, m in enumerate(resp.matches[:6]):
            text_snippet = m.metadata.get("text", "")[:100].replace("\n", " ")
            print(f"  [{i}] score={m.score:.4f} → {text_snippet}…")

    # 3‑2) Fallback logic
    have_docs = bool(resp.matches) and (
        sum(m.score for m in resp.matches[:2]) / max(1, len(resp.matches[:2]))
    ) > fallback_cut

    if not have_docs and not fallback:
        return {
            "answer": "I couldn't find anything in the manual.",
            "chunks_used": [],
            "grounded": True,
            "confidence": 0.0,
        }

    # 3‑3) Fallback to GPT general
    if not have_docs:
        sys_prompt = (
            "You are a helpful assistant. The manual doesn't answer this. "
            "If general knowledge helps, answer clearly. Otherwise say so."
        )
        fallback_ans = (
            client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": q_norm},
                ],
                temperature=0.3,
            )
            .choices[0]
            .message.content.strip()
        )

        return {
            "answer": "(General guidance – not in manual) " + fallback_ans,
            "chunks_used": [],
            "grounded": False,
            "confidence": 0.4,
        }

    # 3‑4) Rerank
    rerank_prompt = (
        f"""You are selecting the most relevant chunks for the question.

QUESTION:
{q_norm}

CHUNKS:
"""
        + "\n\n".join(
            f"[{i}] {m.metadata.get('text', '')[:400].strip()}" for i, m in enumerate(resp.matches)
        )
        + f"""

Return the numbers of the {rerank_keep} most relevant chunks, comma‑separated (e.g. 0,2,3,7)."""
    )

    best = (
        client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": rerank_prompt}],
            temperature=0,
        )
        .choices[0]
        .message.content.strip()
    )

    if DEBUG:
        print("ℹ️ Rerank chose:", best)

    try:
        keep = {int(x) for x in re.split(r"[,\s]+", best) if x.strip().isdigit()}
    except Exception:
        keep = set()

    resp.matches = (
        [m for i, m in enumerate(resp.matches) if i in keep][:rerank_keep]
        if keep
        else resp.matches[:rerank_keep]
    )

    if not resp.matches:
        return {
            "answer": "I couldn't find anything relevant.",
            "chunks_used": [],
            "grounded": False,
            "confidence": 0.0,
        }

    # 3‑5) Build context
    context_parts: List[str] = []
    for i, m in enumerate(resp.matches, start=1):
        tag = f"[{i}]"
        text = m.metadata.get("text", "")[:1500].strip()
        context_parts.append(f"{tag} {text}")

    context = "\n\n".join(context_parts)
    if DEBUG:
        context_sample = context[:1000] + ("…" if len(context) > 1000 else "")
        print("ℹ️ Context sample:\n", context_sample)

    # 3‑6) GPT final answer
    sys_prompt = (
        "You are a technical assistant answering strictly from the manual chunks. "
        "Cite sources like [1], [2]. If the excerpts don't answer, say so."
    )
    user_prompt = (
        f"{context}\n\n► QUESTION: {q_norm}\n\n"
        + (
            "Answer in 2–4 sentences and cite sources."
            if concise
            else "Write a precise answer with only relevant citations."
        )
    )

    answer = (
        client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        .choices[0]
        .message.content.strip()
    )

    return {
        "answer": answer,
        "chunks_used": [m.id for m in resp.matches],
        "grounded": True,
        "confidence": min(max(resp.matches[0].score, 0.0), 1.0),
    }

# ───── 4. Smoke test ─────
if __name__ == "__main__":
    for q in [
        "How do I descale the coffee maker?",
        "Can I change the brew strength?",
        "What is 2 + 2?",  # fallback
    ]:
        print(f"\nQ: {q}")
        out = chat(q, concise=True)
        print("Answer:", out["answer"])
        print("Chunks:", out["chunks_used"])
        print("Grounded:", out["grounded"])
