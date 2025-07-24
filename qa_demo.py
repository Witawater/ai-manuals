#!/usr/bin/env python3
"""
qa_demo.py – Ask questions against the Pinecone index
──────────────────────────────────────────────────────
Now supports a *doc_type* boost instead of a hard filter.
"""

from __future__ import annotations

import os, re, time
from typing import Dict, List, Tuple

import dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec          # v3 SDK

# ───── 1. Secrets & clients ─────
dotenv.load_dotenv(".env")

DEBUG       = True
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o")
INDEX_NAME  = os.getenv("PINECONE_INDEX",     "manuals-small")

DIM = 3072 if "large" in EMBED_MODEL else 1536

ENV    = os.getenv("PINECONE_ENV", "")
REGION = (ENV.split("-", 1)[-1] or "us-east1").lower()
CLOUD  = "aws" if "aws" in ENV.lower() else "gcp"

pc      = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=ENV)
openai  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ───── 2. Ensure index exists ─────
if INDEX_NAME not in pc.list_indexes().names():
    print(f"ℹ️  '{INDEX_NAME}' missing – creating …")
    pc.create_index(
        name       = INDEX_NAME,
        dimension  = DIM,
        metric     = "cosine",
        spec       = ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
    print("✅  Index ready.")
else:
    info = pc.describe_index(INDEX_NAME)
    if info.dimension != DIM:
        raise RuntimeError(
            f"Index '{INDEX_NAME}' has dim={info.dimension}, expected {DIM}."
        )

idx = pc.Index(INDEX_NAME)

# ───── 3. Q-and-A helper ─────
def _embed(txt: str) -> List[float]:
    return openai.embeddings.create(model=EMBED_MODEL, input=txt).data[0].embedding


def chat(
    question:   str,
    customer:   str = "demo01",
    doc_type:   str = "",                 # ← boost signal from UI
    top_k:      int = 40,                 # more recall, we'll down-select later
    concise:    bool = False,
    fallback:   bool = True,
    rerank_keep:int  = 8,
    fallback_cut:float = 0.35,            # rolled back to original
) -> Dict[str, object]:
    """Return answer dict: {answer, chunks_used, grounded, confidence}"""

    # 3-1) Initial dense retrieval
    q_vec = _embed(question)
    res   = idx.query(
        vector            = q_vec,
        top_k             = top_k,
        filter            = {"customer": {"$eq": customer}},   # tenant isolation only
        include_metadata  = True,
    )

    if not res.matches:
        return {
            "answer": "Nothing found in the manual.",
            "chunks_used": [],
            "grounded": False,
            "confidence": 0.0,
        }

    # 3-2) Score-boost matching doc_type instead of filtering it out
    boosted: List[Tuple[float, object]] = []
    for m in res.matches:
        score = m.score
        if doc_type and m.metadata.get("doc_type") == doc_type:
            score += 0.10                       # 10-point boost
        boosted.append((score, m))
    boosted.sort(key=lambda t: t[0], reverse=True)
    res.matches = [m for _, m in boosted]

    if DEBUG:
        print(f"\nℹ️ Retrieved {len(res.matches)} chunks (top 6 shown):")
        for i, m in enumerate(res.matches[:6]):
            snippet = m.metadata.get("text", "")[:100].replace("\n", " ")
            print(f"  [{i}] score={m.score:.4f} → {snippet}…")

    # 3-3) Fallback gate
    have_docs = res.matches and (
        sum(m.score for m in res.matches[:2]) / max(1, len(res.matches[:2]))
    ) > fallback_cut

    if not have_docs and not fallback:
        return {"answer": "I couldn't find anything in the manual.",
                "chunks_used": [], "grounded": True, "confidence": 0.0}

    if not have_docs:
        sys_prompt = (
            "You are a helpful assistant. The manual doesn't answer this. "
            "If general knowledge helps, answer clearly; otherwise say so."
        )
        gpt_ans = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": question},
            ],
            temperature=0.3,
        ).choices[0].message.content.strip()

        return {"answer": "(General guidance – not in manual) " + gpt_ans,
                "chunks_used": [], "grounded": False, "confidence": 0.4}

    # 3-4) Rerank (LLM) – keep diversity
    rerank_prompt = (
        "Select the most relevant chunks for answering.\n\nQUESTION:\n"
        f"{question}\n\nCHUNKS:\n" +
        "\n\n".join(f"[{i}] {m.metadata.get('text','')[:400]}" for i, m in enumerate(res.matches)) +
        f"\n\nReturn exactly {rerank_keep} numbers (comma-separated)."
    )
    best_ids = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":rerank_prompt}],
        temperature=0,
    ).choices[0].message.content.strip()

    try:
        keep = {int(x) for x in re.split(r"[,\s]+", best_ids) if x.strip().isdigit()}
    except Exception:
        keep = set()
    selected = [m for i, m in enumerate(res.matches) if i in keep][:rerank_keep]
    if not selected:
        selected = res.matches[:rerank_keep]

    if DEBUG:
        print("ℹ️ Rerank kept:", [res.matches.index(m) for m in selected])

    # 3-5) Build GPT context
    context = "\n\n".join(
        f"[{i+1}] {m.metadata.get('text','')[:1500].strip()}" for i, m in enumerate(selected)
    )

    sys_prompt = (
        "You are a technical assistant who must answer **only** with information "
        "found in the provided manual chunks. Cite each claim like [1]. "
        "Say 'Not found' if the manual doesn't cover it."
    )
    user_prompt = (
        f"{context}\n\n► QUESTION: {question}\n\n" +
        ("Answer in 2–4 sentences and cite sources."
         if concise else "Write a precise answer with citations.")
    )

    answer = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role":"system","content":sys_prompt},
            {"role":"user",  "content":user_prompt},
        ],
        temperature=0,
    ).choices[0].message.content.strip()

    return {
        "answer":       answer,
        "chunks_used":  [m.id for m in selected],
        "grounded":     True,
        "confidence":   min(max(selected[0].score, 0.0), 1.0),
    }

# ───── 4. Smoke test ─────
if __name__ == "__main__":
    for q in [
        "How do I drain the system?",
        "Does it support Modbus?",
        "What is 2 + 2?",
    ]:
        print(f"\nQ: {q}")
        out = chat(q, customer="demo01", doc_type="hardware", concise=True)
        print("Answer:", out["answer"])
        print("Grounded:", out["grounded"])
        print("Chunks:", out["chunks_used"])
