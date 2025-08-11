#!/usr/bin/env python3
"""
qa_demo.py – hybrid Q-and-A for AI-Manuals
──────────────────────────────────────────
• Embedding × BM25 hybrid (α = 0.50)
• Safety-word guard (WARNING / NOTICE / ALARM …)
• Max-Marginal-Relevance diversity
• GPT rerank → clean **Markdown ordered-list** answers

Notes:
- Ingest writes vectors to namespace = doc_id and stores text in metadata["preview"].
- This module now requires a doc_id (to match the ingest namespace).
"""

from __future__ import annotations
import os, re, string, time
from typing import Dict, List

import dotenv
import numpy as np
from openai import OpenAI
from pinecone import Pinecone
from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

# ───────── 1. CONFIG ─────────
dotenv.load_dotenv(".env")

DEBUG        = True
EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL   = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o")
INDEX_NAME   = os.getenv("PINECONE_INDEX",     "manuals-large")

MODEL_DIM = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}.get(EMBED_MODEL, 3072)

MMR_KEEP     = 40
RERANK_KEEP  = 24
ALPHA        = 0.50
FALLBACK_CUT = 0.25

SAFETY_WORDS = ("warning", "notice", "important", "alarm", "code ")

pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY") or "")
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "")

# ───────── 2. INDEX GUARD (no silent auto-create here) ─────────
if INDEX_NAME not in pc.list_indexes().names():
    raise RuntimeError(f"Pinecone index '{INDEX_NAME}' not found. Ingest must create it first.")
desc = pc.describe_index(INDEX_NAME)
dim = getattr(desc, "dimension", None) or getattr(desc, "spec", {}).get("dimension")
if dim and int(dim) != int(MODEL_DIM):
    raise RuntimeError(f"Index '{INDEX_NAME}' dim={dim} ≠ model dim={MODEL_DIM} for {EMBED_MODEL}")
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
    # Need values for diversity; if not present, skip MMR safely
    if not all(getattr(m, "values", None) for m in matches):
        return matches[:k]
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
    doc_id: str = "",               # REQUIRED: matches ingest namespace
    top_k: int = 60,
    concise: bool = False,
    fallback: bool = True,
) -> Dict[str, object]:

    if not doc_id:
        # Ingest writes into namespace=doc_id, so querying without it returns nothing
        return {"answer": "No document selected. Please upload a manual first.",
                "chunks_used": [], "grounded": False, "confidence": 0.0}

    q_canon = _norm(question)
    q_vec = _embed(q_canon)

    # Metadata filter still applied (customer, optional doc_type/notes boost later)
    filter_by = {"customer": {"$eq": customer}}

    res = idx.query(
        vector=q_vec,
        top_k=top_k,
        filter=filter_by,
        include_metadata=True,
        include_values=True,   # needed for MMR diversity
        namespace=doc_id,      # ← must match ingest
    )

    if not res.matches:
        print(f"❌ No match – q='{question}' cid='{customer}' doc='{doc_id}'")
        return {"answer": "Nothing found in this manual. Try rephrasing or another section.",
                "chunks_used": [], "grounded": False, "confidence": 0.0}

    # Use preview text stored by ingest (fallback to 'text' for legacy vectors)
    def _txt(m): return (m.metadata.get("preview")
                         or m.metadata.get("text")
                         or "").lower()

    docs_tok = [_tokenize(_txt(m)) for m in res.matches]
    bm25 = BM25Okapi(docs_tok)
    bm25_raw = np.asarray(bm25.get_scores(_tokenize(q_canon)), dtype=float)
    max_bm25 = bm25_raw.max() if bm25_raw.size else 0.0
    bm25_norm = (bm25_raw / max_bm25).tolist() if max_bm25 else bm25_raw.tolist()

    # Hybrid score
    for m, b, e in zip(res.matches, bm25_norm, [m.score for m in res.matches]):
        m.score = float(ALPHA * e + (1 - ALPHA) * b)

    # Safety boost (warnings etc.)
    safety, rest = [], []
    for m in res.matches:
        txt = _txt(m)
        (safety if any(w in txt for w in SAFETY_WORDS) else rest).append(m)
    safety = safety[:6]
    rest = mmr(rest, k=max(0, MMR_KEEP - len(safety)), lam=.5)
    res.matches = safety + rest

    # Heuristic boosts for doc_type/notes signal (metadata from ingest/app)
    boosted = []
    q_low = q_canon
    for m in res.matches:
        s = m.score
        if doc_type and (m.metadata.get("doc_type") or "").lower() == doc_type.lower():
            s += .10
        note = (m.metadata.get("notes") or "").lower()
        if note and note in q_low:
            s += .07
        boosted.append((s, m))
    res.matches = [m for s, m in sorted(boosted, key=lambda t: t[0], reverse=True)]

    if DEBUG:
        print("\nℹ️ After safety/MMR/boost (top 6)")
        for i, m in enumerate(res.matches[:6]):
            pv = _txt(m)[:80].replace("\n", " ")
            print(f"  [{i}] {m.score:.4f} → {pv}…  (pg={m.metadata.get('page')})")

    # Low-signal fallback
    top_scores = [m.score for m in res.matches[:2]] or [0.0]
    if np.mean(top_scores) < FALLBACK_CUT:
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
        print(f"⚠️ Fallback – q='{question}' cid='{customer}' doc='{doc_id}'")
        return {"answer": "(General guidance) " + fb,
                "chunks_used": [], "grounded": False, "confidence": .4}

    # Rerank: ask the model to pick indices of the best chunks (use previews)
    rerank_prompt = (
        "Pick the most relevant chunks.\n\nQUESTION:\n"
        f"{question}\n\nCHUNKS:\n" +
        "\n\n".join(f"[{i}] {_txt(m)[:400]}" for i, m in enumerate(res.matches)) +
        f"\n\nReturn exactly {RERANK_KEEP} numbers (comma-separated). No other text."
    )
    keep = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": rerank_prompt}],
        temperature=0,
    ).choices[0].message.content

    idx_keep = {int(x) for x in re.findall(r"\d+", keep)}
    sel = [m for i, m in enumerate(res.matches) if i in idx_keep][:RERANK_KEEP] \
          or res.matches[:RERANK_KEEP]

    # Build grounded context (still use preview; it’s enough for extraction-style answers)
    context = "\n\n".join(
        f"[{i+1}] {_txt(m)[:1500].strip()}"
        for i, m in enumerate(sel)
    )
    sys_prompt = (
        "You are a technical assistant answering **only** from the excerpts below. "
        "Output a **Markdown ordered list**; leave one blank line between items and "
        "**bold the imperative verb** at the start of each step.\n"
        "• Include every numbered step, WARNING / NOTICE block, key sequence and "
        "display confirmation verbatim.\n"
        "• Cite each fact like [2]. Reply “Not found in manual.” if needed."
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
        "confidence": float(sel[0].score) if sel else 0.0,
    }

if __name__ == "__main__":
    # Quick smoke test: replace doc_id with a real one from your upload.
    for q in ["How do I drain the system?",
              "Does it support Modbus?",
              "What is 2 + 2?"]:
        out = chat(q, customer="demo01", doc_id="PUT_REAL_DOC_ID", doc_type="hardware", concise=True)
        print("\nQ:", q, "\n", out["answer"])
