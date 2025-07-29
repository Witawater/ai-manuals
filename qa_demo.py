#!/usr/bin/env python3
"""
qa_demo.py – question-answer helper for AI-Manuals
──────────────────────────────────────────────────
• MMR diversity + soft metadata boosts
• WARNING / NOTICE / IMPORTANT chunks are never dropped
• Deterministic GPT re-rank → Markdown ordered list
• Query normaliser removes trailing “?!.” so “How …” and “How …?” hit the same vectors
"""

from __future__ import annotations
import os, re, time
from typing import Dict, List, Tuple

import dotenv
from openai      import OpenAI
from pinecone    import Pinecone, ServerlessSpec           # v3 SDK
from sklearn.metrics.pairwise import cosine_similarity

# ───────────── 1. CONFIG & CLIENTS ─────────────────────────
dotenv.load_dotenv(".env")

DEBUG        = True
EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL   = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o")
INDEX_NAME   = os.getenv("PINECONE_INDEX",     "manuals-small")

DIM          = 3072 if "large" in EMBED_MODEL else 1536
ENV          = os.getenv("PINECONE_ENV", "")
REGION       = (ENV.split("-", 1)[-1] or "us-east1").lower()
CLOUD        = "aws" if "aws" in ENV.lower() else "gcp"

MMR_KEEP     = 40      # chunks after MMR diversity
RERANK_KEEP  = 24      # chunks passed to GPT re-rank
FALLBACK_CUT = 0.25    # average top-2 score gate

SAFETY_WORDS = ("warning", "notice", "important", "alarm", "code ")

pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=ENV)
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ––– ensure Pinecone index ready ––––––––––––––––––––––––––
if INDEX_NAME not in pc.list_indexes().names():
    print(f"ℹ️  Creating Pinecone index '{INDEX_NAME}' …")
    pc.create_index(
        name=INDEX_NAME, dimension=DIM, metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
elif pc.describe_index(INDEX_NAME).dimension != DIM:
    raise RuntimeError("Pinecone dimension mismatch")
idx = pc.Index(INDEX_NAME)

# ───────────── 2. HELPERS ─────────────────────────────────
def _embed(text: str) -> List[float]:
    """OpenAI embedding helper."""
    return openai.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding


def mmr(matches, *, k: int, lam: float = .5):
    """Return *k* Max-Marginal-Relevance matches."""
    if k >= len(matches):
        return matches
    vecs  = [m.values for m in matches]
    sim   = cosine_similarity(vecs, vecs)
    q_sim = [m.score for m in matches]

    chosen, rest = [0], list(range(1, len(matches)))
    while len(chosen) < k and rest:
        score = [lam*q_sim[i] - (1-lam)*sim[i][chosen].max() for i in rest]
        best  = rest.pop(score.index(max(score)))
        chosen.append(best)
    return [matches[i] for i in chosen]


def _normalise_query(q: str) -> str:
    """Strip trailing punctuation so “ABC” and “ABC?” embed identically."""
    return re.sub(r'[?!.\s]+$', '', q).strip()


# ───────────── 3. MAIN ENTRYPOINT ─────────────────────────
def chat(
    question:   str,
    customer:   str = "demo01",
    doc_type:   str = "",
    top_k:      int = 60,
    concise:    bool = False,
    fallback:   bool = True,
) -> Dict[str, object]:
    """
    Return dict ⇢ {answer, chunks_used, grounded, confidence}
    """

    # 3-1 dense retrieval ---------------------------------------------------
    query_clean = _normalise_query(question)
    q_vec       = _embed(query_clean)
    res         = idx.query(
        vector=q_vec, top_k=top_k,
        filter={"customer": {"$eq": customer}},
        include_metadata=True, include_values=True,
    )
    if not res.matches:
        return {"answer":"Nothing found – retry in a few seconds.",
                "chunks_used":[], "grounded":False, "confidence":0.0}

    # 3-2 safety pass  (keep all WARNING / NOTICE …) ------------------------
    safety, rest = [], []
    for m in res.matches:
        txt = (m.metadata.get("text") or "").lower()
        (safety if any(w in txt for w in SAFETY_WORDS) else rest).append(m)
    safety = safety[:6]                                                # cap
    rest   = mmr(rest, k=max(0, MMR_KEEP-len(safety)), lam=.5)
    res.matches = safety + rest

    # 3-3 soft boosts -------------------------------------------------------
    boosted: List[Tuple[float, object]] = []
    q_lower = query_clean.lower()
    for m in res.matches:
        score = m.score
        if doc_type and m.metadata.get("doc_type") == doc_type:
            score += .10
        note = (m.metadata.get("notes") or "").lower()
        if note and note in q_lower:
            score += .07
        boosted.append((score, m))
    res.matches = [m for s, m in sorted(boosted, key=lambda t: t[0], reverse=True)]

    if DEBUG:
        print("\nℹ️ After safety/MMR/boost (top 6)")
        for i, m in enumerate(res.matches[:6]):
            print(f"  [{i}] {m.score:.4f} → {(m.metadata.get('text') or '')[:80]}…")

    # 3-4 fallback gate -----------------------------------------------------
    best2 = res.matches[:2]
    top_avg = sum(m.score for m in best2) / len(best2)
    if top_avg < FALLBACK_CUT:
        if not fallback:
            return {"answer":"Manual doesn’t cover this.",
                    "chunks_used":[], "grounded":True, "confidence":0.0}
        fb = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role":"system","content":"You are a helpful assistant."},
                {"role":"user",  "content":question}
            ],
            temperature=.3,
        ).choices[0].message.content.strip()
        return {"answer":"(General guidance) "+fb,
                "chunks_used":[], "grounded":False, "confidence":.4}

    # 3-5 GPT rerank --------------------------------------------------------
    rerank_prompt = (
        "Pick the most relevant chunks.\n\nQUESTION:\n"
        f"{question}\n\nCHUNKS:\n" +
        "\n\n".join(f"[{i}] {m.metadata.get('text','')[:400]}"
                    for i, m in enumerate(res.matches)) +
        f"\n\nReturn exactly {RERANK_KEEP} numbers (comma-separated)."
    )
    keep     = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":rerank_prompt}],
        temperature=0,
    ).choices[0].message.content
    indices  = {int(x) for x in re.findall(r"\d+", keep)}
    selected = [m for i, m in enumerate(res.matches) if i in indices][:RERANK_KEEP] \
               or res.matches[:RERANK_KEEP]

    # 3-6 GPT answer --------------------------------------------------------
    context = "\n\n".join(
        f"[{i+1}] {m.metadata.get('text','')[:1500].strip()}"
        for i, m in enumerate(selected)
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
        ("Answer in 2–4 sentences and cite sources."
         if concise else "Write a precise, complete answer with citations.")
    )

    answer = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":sys_prompt},
                  {"role":"user",  "content":user_prompt}],
        temperature=0,
        max_tokens=450,
    ).choices[0].message.content.strip()

    return {
        "answer":      answer,
        "chunks_used": [m.id for m in selected],
        "grounded":    True,
        "confidence":  min(max(selected[0].score, 0.0), 1.0),
    }


# ───────────── 4. SMOKE TEST ───────────────────────────────
if __name__ == "__main__":
    for q in ["How do I drain the system?",
              "Does it support Modbus?",
              "What is 2 + 2?"]:
        res = chat(q, customer="demo01", doc_type="hardware", concise=True)
        print("\nQ:", q)
        print(res["answer"])
