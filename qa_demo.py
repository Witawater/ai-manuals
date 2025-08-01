#!/usr/bin/env python3
"""
qa_demo.py – hybrid Q-and-A for AI-Manuals
──────────────────────────────────────────
• Embedding × BM25 hybrid (α = 0.50)  
• Safety-word guard (WARNING / NOTICE / ALARM …)  
• Max-Marginal-Relevance diversity  
• GPT rerank → clean **Markdown ordered-list** answers
"""

from __future__ import annotations
import os, re, string, time
from typing import Dict, List

import dotenv
import numpy as np
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

# ───────── 1. CONFIG ─────────
dotenv.load_dotenv(".env")

DEBUG        = True
EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL   = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o")
INDEX_NAME   = os.getenv("PINECONE_INDEX",     "manuals-large")

DIM          = 3072 if "large" in EMBED_MODEL else 1536
ENV          = os.getenv("PINECONE_ENV", "")
REGION       = (ENV.split("-", 1)[-1] or "us-east1").lower()
CLOUD        = "aws" if "aws" in ENV.lower() else "gcp"

MMR_KEEP     = 40
RERANK_KEEP  = 24
ALPHA        = 0.50
FALLBACK_CUT = 0.25

SAFETY_WORDS = ("warning", "notice", "important", "alarm", "code ")

pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=ENV)
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ───────── 2. INDEX GUARD ─────────
if INDEX_NAME not in pc.list_indexes().names():
    print(f"ℹ️  '{INDEX_NAME}' missing – creating …")
    pc.create_index(
        name=INDEX_NAME, dimension=DIM, metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
else:
    if pc.describe_index(INDEX_NAME).dimension != DIM:
        raise RuntimeError("Pinecone dimension mismatch")
idx = pc.Index(INDEX_NAME)

_tokenize = re.compile(r"\w+").findall

def _norm(txt: str) -> str:
    txt = txt.strip().lower()
    while txt and txt[-1] in string.punctuation:
        txt = txt[:-1]
    return txt

def _embed(text: str) -> List[float]:
    return openai.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding

def mmr(matches, *, k: int, lam: float = .5):
    if k >= len(matches): return matches
    vecs = [m.values for m in matches]
    sim = cosine_similarity(vecs, vecs)
    q_sim = [m.score for m in matches]
    chosen, rest = [0], list(range(1, len(matches)))
    while len(chosen) < k and rest:
        scores = [lam*q_sim[i] - (1-lam)*sim[i][chosen].max() for i in rest]
        best = rest.pop(scores.index(max(scores)))
        chosen.append(best)
    return [matches[i] for i in chosen]

# ───────── 4. MAIN ─────────
def chat(
    question: str,
    customer: str = "demo01",
    doc_type: str = "",
    doc_id: str = "",
    top_k: int = 60,
    concise: bool = False,
    fallback: bool = True,
) -> Dict[str, object]:

    q_canon = _norm(question)
    q_vec = _embed(q_canon)

    # ✅ Enforce manual-level filtering
    filter_by = {"customer": {"$eq": customer}}
    if doc_id:
        filter_by["doc_id"] = {"$eq": doc_id}

    res = idx.query(
        vector=q_vec, top_k=top_k,
        filter=filter_by,
        include_metadata=True, include_values=True,
    )

    if not res.matches:
        return {"answer": "Nothing found – retry in a few seconds.",
                "chunks_used": [], "grounded": False, "confidence": 0.0}

    docs_tok = [_tokenize((m.metadata.get("text") or "").lower()) for m in res.matches]
    bm25 = BM25Okapi(docs_tok)
    bm25_raw = np.asarray(bm25.get_scores(_tokenize(q_canon)), dtype=float)
    max_bm25 = bm25_raw.max() if bm25_raw.size else 0.0
    bm25_norm = (bm25_raw / max_bm25).tolist() if max_bm25 else bm25_raw.tolist()

    for m, b, e in zip(res.matches, bm25_norm, [m.score for m in res.matches]):
        m.score = float(ALPHA * e + (1 - ALPHA) * b)

    safety, rest = [], []
    for m in res.matches:
        txt = (m.metadata.get("text") or "").lower()
        (safety if any(w in txt for w in SAFETY_WORDS) else rest).append(m)
    safety = safety[:6]
    rest = mmr(rest, k=max(0, MMR_KEEP - len(safety)), lam=.5)
    res.matches = safety + rest

    boosted, q_low = [], q_canon
    for m in res.matches:
        s = m.score
        if doc_type and m.metadata.get("doc_type") == doc_type:
            s += .10
        note = (m.metadata.get("notes") or "").lower()
        if note and note in q_low:
            s += .07
        boosted.append((s, m))
    res.matches = [m for s, m in sorted(boosted, key=lambda t: t[0], reverse=True)]

    if DEBUG:
        print("\nℹ️ After safety/MMR/boost (top 6)")
        for i, m in enumerate(res.matches[:6]):
            print(f"  [{i}] {m.score:.4f} → {(m.metadata.get('text') or '')[:80]}…")

    if np.mean([m.score for m in res.matches[:2]]) < FALLBACK_CUT:
        if not fallback:
            return {"answer": "Manual doesn’t cover this.",
                    "chunks_used": [], "grounded": True, "confidence": 0.0}
        fb = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": question}
            ],
            temperature=.3,
        ).choices[0].message.content.strip()
        return {"answer": "(General guidance) " + fb,
                "chunks_used": [], "grounded": False, "confidence": .4}

    rerank_prompt = (
        "Pick the most relevant chunks.\n\nQUESTION:\n"
        f"{question}\n\nCHUNKS:\n" +
        "\n\n".join(f"[{i}] {m.metadata.get('text','')[:400]}" for i, m in enumerate(res.matches)) +
        f"\n\nReturn exactly {RERANK_KEEP} numbers (comma-separated)."
    )
    keep = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": rerank_prompt}],
        temperature=0,
    ).choices[0].message.content
    idx_keep = {int(x) for x in re.findall(r"\d+", keep)}
    sel = [m for i, m in enumerate(res.matches) if i in idx_keep][:RERANK_KEEP] \
          or res.matches[:RERANK_KEEP]

    context = "\n\n".join(
        f"[{i+1}] {m.metadata.get('text','')[:1500].strip()}"
        for i, m in enumerate(sel)
    )
    sys_prompt = (
        "You are a technical assistant answering **only** from the excerpts below. "
        "Output a **Markdown ordered list**; leave one blank line between items and "
        "**bold the imperative verb** at the start of each step.\n"
        "• Include every numbered step, WARNING / NOTICE block, key sequence and "
        "display confirmation verbatim.\n"
        "• Cite each fact like [2].  Reply “Not found in manual.” if needed."
    )
    user_prompt = (
        f"{context}\n\n► QUESTION: {question}\n\n" +
        ("Answer in 2-4 sentences and cite sources."
         if concise else "Write a precise, complete answer with citations.")
    )
    answer = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=0, max_tokens=450,
    ).choices[0].message.content.strip()

    return {
        "answer": answer,
        "chunks_used": [m.id for m in sel],
        "grounded": True,
        "confidence": float(sel[0].score),
    }

if __name__ == "__main__":
    for q in ["How do I drain the system?",
              "Does it support Modbus?",
              "What is 2 + 2?"]:
        out = chat(q, customer="demo01", doc_id="put-your-doc-id-here", doc_type="hardware", concise=True)
        print("\nQ:", q, "\n", out["answer"])
