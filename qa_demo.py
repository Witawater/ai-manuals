#!/usr/bin/env python3
"""
qa_demo.py – hybrid Q-and-A for AI-Manuals
──────────────────────────────────────────
▪ Embedding × BM25 hybrid ranking (α = 0.50)  
▪ Safety-word guard keeps WARNING / NOTICE chunks  
▪ Max-Marginal-Relevance for diversity  
▪ GPT rerank → clean **Markdown ordered list** answer
"""

from __future__ import annotations
import os, re, time
from typing import Dict, List, Tuple

import dotenv
import numpy as np                         # ←  NumPy at top-level
from openai       import OpenAI
from pinecone     import Pinecone, ServerlessSpec
from rank_bm25    import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────── 1. CONFIG ─────────────────────────────
dotenv.load_dotenv(".env")

DEBUG        = True
EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL   = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o")
INDEX_NAME   = os.getenv("PINECONE_INDEX",     "manuals-small")

DIM          = 3072 if "large" in EMBED_MODEL else 1536
ENV          = os.getenv("PINECONE_ENV", "")
REGION       = (ENV.split("-", 1)[-1] or "us-east1").lower()
CLOUD        = "aws" if "aws" in ENV.lower() else "gcp"

MMR_KEEP     = 40          # chunks kept after diversity
RERANK_KEEP  = 24          # chunks shown to GPT
ALPHA        = 0.50        # embed weight in hybrid   (0 ≤ α ≤ 1)
FALLBACK_CUT = 0.25        # below → fallback answer

SAFETY_WORDS = ("warning", "notice", "important", "alarm", "code ")

pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=ENV)
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─────────────────── 2. INDEX GUARD ───────────────────────
if INDEX_NAME not in pc.list_indexes().names():
    print(f"ℹ️  '{INDEX_NAME}' missing – creating …")
    pc.create_index(
        name=INDEX_NAME,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
else:
    info = pc.describe_index(INDEX_NAME)
    if info.dimension != DIM:
        raise RuntimeError(f"Index dim {info.dimension} ≠ expected {DIM}")
idx = pc.Index(INDEX_NAME)

_tokenize = re.compile(r"\w+").findall                 # tiny word-tokeniser

# ─────────────────── 3. HELPERS ───────────────────────────
def _embed(txt: str) -> List[float]:
    return openai.embeddings.create(model=EMBED_MODEL, input=txt).data[0].embedding


def mmr(matches, *, k: int, lam: float = .5):
    """Return *k* Max-Marginal-Relevance matches."""
    if k >= len(matches):
        return matches
    vecs  = [m.values for m in matches]
    sim   = cosine_similarity(vecs, vecs)
    q_sim = [m.score for m in matches]

    chosen, rest = [0], list(range(1, len(matches)))
    while len(chosen) < k and rest:
        mmr_scores = [lam*q_sim[i] - (1-lam)*sim[i][chosen].max() for i in rest]
        best       = rest.pop(mmr_scores.index(max(mmr_scores)))
        chosen.append(best)
    return [matches[i] for i in chosen]

# ─────────────────── 4. MAIN ENTRYPOINT ───────────────────
def chat(
    question:   str,
    customer:   str = "demo01",
    doc_type:   str = "",
    top_k:      int = 60,
    concise:    bool = False,
    fallback:   bool = True,
) -> Dict[str, object]:
    """
    Returns:
        dict = {answer, chunks_used, grounded, confidence}
    """

    # 4-1  dense retrieval (embeddings)
    q_vec = _embed(question)
    res   = idx.query(
        vector=q_vec,
        top_k=top_k,
        filter={"customer": {"$eq": customer}},
        include_metadata=True,
        include_values=True,     # vectors needed for MMR
    )
    if not res.matches:
        return {"answer": "Nothing found – retry in a few seconds.",
                "chunks_used": [], "grounded": False, "confidence": 0.0}

    # 4-2  BM25 keyword scores over the same matches
    docs_tokens = [_tokenize((m.metadata.get("text") or "").lower())
                   for m in res.matches]
    bm25   = BM25Okapi(docs_tokens)
    q_tok  = _tokenize(question.lower())
    bm25_scores = np.asarray(bm25.get_scores(q_tok), dtype=float)
    max_bm25    = bm25_scores.max() if bm25_scores.size else 0.0
    bm25_norm   = (bm25_scores / max_bm25) if max_bm25 else bm25_scores

    # 4-3  hybrid score  (α·embed  + (1-α)·bm25)
    for m, b, e in zip(res.matches, bm25_norm,
                       [m.score for m in res.matches]):
        m.score = ALPHA*e + (1-ALPHA)*b

    # 4-4  safety-word priority  + diversity (MMR)
    safety, rest = [], []
    for m in res.matches:
        txt = (m.metadata.get("text") or "").lower()
        (safety if any(w in txt for w in SAFETY_WORDS) else rest).append(m)
    safety = safety[:6]                                    # cap
    rest   = mmr(rest, k=max(0, MMR_KEEP-len(safety)), lam=.5)
    res.matches = safety + rest

    # 4-5  soft boosts (doc_type + notes keyword)
    boosted = []
    q_low   = question.lower()
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

    # 4-6  fallback gate
    top_avg = np.mean([m.score for m in res.matches[:2]])
    if top_avg < FALLBACK_CUT:
        if not fallback:
            return {"answer": "Manual doesn’t cover this.",
                    "chunks_used": [], "grounded": True, "confidence": 0.0}
        fb = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user",   "content": question}
            ],
            temperature=.3,
        ).choices[0].message.content.strip()
        return {"answer": "(General guidance) "+fb,
                "chunks_used": [], "grounded": False, "confidence": .4}

    # 4-7  GPT rerank → RERANK_KEEP
    rerank_prompt = (
        "Pick the most relevant chunks.\n\nQUESTION:\n"
        f"{question}\n\nCHUNKS:\n" +
        "\n\n".join(f"[{i}] {m.metadata.get('text','')[:400]}"
                    for i, m in enumerate(res.matches)) +
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

    # 4-8  build context + final answer
    context = "\n\n".join(
        f"[{i+1}] {m.metadata.get('text','')[:1500].strip()}"
        for i, m in enumerate(sel)
    )

    sys_prompt = (
        "You are a technical assistant answering **only** from the excerpts below. "
        "Output a **Markdown ordered list** (1., 2., …) and leave one blank line "
        "between items. **Bold the imperative verb** at the start of each step.\n"
        "• Include every numbered step, WARNING / NOTICE block, key sequence and "
        "display confirmation verbatim.\n"
        "• Cite each fact like [2].  Reply “Not found in manual.” if needed."
    )
    user_prompt = (
        f"{context}\n\n► QUESTION: {question}\n\n" +
        ("Answer in 2–4 sentences and cite sources."
         if concise else "Write a precise, complete answer with citations.")
    )

    answer = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "system", "content": sys_prompt},
                  {"role": "user",   "content": user_prompt}],
        temperature=0,
        max_tokens=450,
    ).choices[0].message.content.strip()

    return {
        "answer":      answer,
        "chunks_used": [m.id for m in sel],
        "grounded":    True,
        "confidence":  float(sel[0].score),
    }

# ─────────────────── 5. SMOKE TEST ────────────────────────
if __name__ == "__main__":
    for q in ["How do I drain the system?",
              "Does it support Modbus?",
              "What is 2 + 2?"]:
        res = chat(q, customer="demo01", doc_type="hardware", concise=True)
        print("\nQ:", q, "\n", res["answer"])
