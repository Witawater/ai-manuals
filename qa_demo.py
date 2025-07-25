#!/usr/bin/env python3
"""
qa_demo.py – question-answer helper for AI-Manuals
──────────────────────────────────────────────────
• Max-Marginal-Relevance keeps diverse chunks
• Soft boosts for doc_type and user notes
• Expanded system prompt guarantees every bullet / warning is kept verbatim
"""

from __future__ import annotations

import os, re, time
from typing import Dict, List, Tuple

import dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec               # v3 SDK
from sklearn.metrics.pairwise import cosine_similarity

# ─── 1. Config & clients ─────────────────────────────────
dotenv.load_dotenv(".env")
DEBUG       = True
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o")
INDEX_NAME  = os.getenv("PINECONE_INDEX",     "manuals-small")

DIM    = 3072 if "large" in EMBED_MODEL else 1536
ENV    = os.getenv("PINECONE_ENV", "")
REGION = (ENV.split("-", 1)[-1] or "us-east1").lower()
CLOUD  = "aws" if "aws" in ENV.lower() else "gcp"

pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=ENV)
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ensure index exists
if INDEX_NAME not in pc.list_indexes().names():
    print(f"ℹ️  '{INDEX_NAME}' missing – creating …")
    pc.create_index(
        name=INDEX_NAME, dimension=DIM, metric="cosine",
        spec=ServerlessSpec(cloud=CLOUD, region=REGION),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
else:
    info = pc.describe_index(INDEX_NAME)
    if info.dimension != DIM:
        raise RuntimeError(f"Index dim {info.dimension} ≠ expected {DIM}")
idx = pc.Index(INDEX_NAME)

# ─── 2. helpers ──────────────────────────────────────────
def _embed(text: str) -> List[float]:
    return openai.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding


def mmr(matches, *, k: int = 20, lam: float = 0.5):
    """Return k Max-Marginal-Relevance matches."""
    if k >= len(matches): return matches
    vecs      = [m.values for m in matches]
    simM      = cosine_similarity(vecs, vecs)
    q_sim     = [m.score for m in matches]

    sel, rest = [0], list(range(1, len(matches)))
    while len(sel) < k and rest:
        mmr_scores = [
            lam * q_sim[i] - (1 - lam) * simM[i][sel].max()
            for i in rest
        ]
        idx = rest.pop(mmr_scores.index(max(mmr_scores)))
        sel.append(idx)
    return [matches[i] for i in sel]

# ─── 3. chat() entrypoint ────────────────────────────────
def chat(
    question:    str,
    customer:    str = "demo01",
    doc_type:    str = "",
    top_k:       int = 60,
    concise:     bool = False,
    fallback:    bool = True,
    rerank_keep: int  = 16,
    fallback_cut:float = 0.25,
) -> Dict[str, object]:
    """Return {answer, chunks_used, grounded, confidence}."""

    # 3-1 dense retrieval
    q_vec = _embed(question)
    res   = idx.query(
        vector=q_vec, top_k=top_k,
        filter={"customer": {"$eq": customer}},
        include_metadata=True, include_values=True,
    )
    if not res.matches:
        return {"answer":"Nothing found – try again in a few seconds.",
                "chunks_used":[], "grounded":False, "confidence":0.0}

    # 3-2 diversity then boosts
    res.matches = mmr(res.matches, k=20, lam=0.5)

    boosted: List[Tuple[float, object]] = []
    q_lower = question.lower()
    for m in res.matches:
        score = m.score
        if doc_type and m.metadata.get("doc_type") == doc_type:
            score += 0.10
        note = (m.metadata.get("notes") or "").lower()
        if note and note in q_lower:
            score += 0.07
        boosted.append((score, m))
    boosted.sort(key=lambda t: t[0], reverse=True)
    res.matches = [m for _, m in boosted]

    if DEBUG:
        print(f"\nℹ️ After MMR {len(res.matches)} chunks (top 6):")
        for i, m in enumerate(res.matches[:6]):
            txt = m.metadata.get("text","").splitlines()[0][:90]
            print(f"  [{i}] score={m.score:.4f} → {txt}…")

    # 3-3 fallback gate
    have_docs = res.matches and sum(m.score for m in res.matches[:2]) / 2 > fallback_cut
    if not have_docs:
        if not fallback:
            return {"answer":"Manual doesn’t cover this.",
                    "chunks_used":[], "grounded":True, "confidence":0.0}
        fb_prompt = (
            "You are a helpful assistant. The manual doesn't answer this. "
            "If general knowledge helps, answer clearly; otherwise say so."
        )
        ans = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role":"system","content":fb_prompt},
                      {"role":"user",  "content":question}],
            temperature=0.3,
        ).choices[0].message.content.strip()
        return {"answer":"(General guidance) "+ans,
                "chunks_used":[], "grounded":False, "confidence":0.4}

    # 3-4 LLM rerank
    rerank_prompt = (
        "Select the most relevant chunks for answering.\n\nQUESTION:\n"
        f"{question}\n\nCHUNKS:\n" +
        "\n\n".join(f"[{i}] {m.metadata.get('text','')[:400]}"
                    for i, m in enumerate(res.matches)) +
        f"\n\nReturn exactly {rerank_keep} numbers (comma-separated)."
    )
    best_ids = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":rerank_prompt}],
        temperature=0,
    ).choices[0].message.content.strip()
    keep = {int(x) for x in re.split(r"[,\s]+", best_ids) if x.isdigit()}
    selected = [m for i, m in enumerate(res.matches) if i in keep][:rerank_keep] or res.matches[:rerank_keep]

    # 3-5 build GPT context
    context = "\n\n".join(
        f"[{i+1}] {m.metadata.get('text','')[:1500].strip()}"
        for i, m in enumerate(selected)
    )

    sys_prompt = (
        "You are a technical assistant answering **only** from the manual excerpts "
        "below. Follow these rules:\n"
        "• Include **all numbered steps**, bullet-point warnings, preparation steps, "
        "key sequences, and any display confirmations verbatim.\n"
        "• Do not summarise away instructions; if the manual lists seven steps, "
        "your answer must list seven.\n"
        "• Cite each fact like [2]. If the excerpts don’t answer, reply “Not found in manual.”"
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

# ─── 4. Smoke test ─────────────────────────────────────────
if __name__ == "__main__":
    for q in ["How do I drain the system?",
              "Does it support Modbus?",
              "What is 2 + 2?"]:
        print(f"\nQ: {q}")
        out = chat(q, customer="demo01", doc_type="hardware", concise=True)
        print("Answer:", out["answer"])
        print("Grounded:", out["grounded"])
        print("Chunks:", out["chunks_used"])
