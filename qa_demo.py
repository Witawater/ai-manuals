#!/usr/bin/env python3
"""
qa_demo.py – question-answer helper for AI-Manuals
──────────────────────────────────────────────────
• MMR diversity + soft metadata boosts
• Deterministic LLM-rerank (keeps “NOTICE / IMPORTANT” blocks)
• System prompt forces clean **Markdown ordered-list** layout
"""

from __future__ import annotations
import os, re, time
from typing import Dict, List, Tuple

import dotenv
from openai      import OpenAI
from pinecone    import Pinecone, ServerlessSpec          # v3 SDK
from sklearn.metrics.pairwise import cosine_similarity

# ─────────────────── 1. CONFIG & CLIENTS ───────────────────
dotenv.load_dotenv(".env")

DEBUG        = True
EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL   = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o")
INDEX_NAME   = os.getenv("PINECONE_INDEX",     "manuals-small")

DIM          = 3072 if "large" in EMBED_MODEL else 1536
ENV          = os.getenv("PINECONE_ENV", "")
REGION       = (ENV.split("-", 1)[-1] or "us-east1").lower()
CLOUD        = "aws" if "aws" in ENV.lower() else "gcp"

# tweak-once constants (so behaviour is reproducible)
MMR_KEEP     = 30      # how many chunks survive the diversity pass   (was 20)
RERANK_KEEP  = 24      # how many chunks GPT sees for final answer    (was 16)
FALLBACK_CUT = 0.25    # avg-score gate below which we fall back

pc      = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=ENV)
openai  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ensure index exists / matches dimension
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
        raise RuntimeError("Pinecone index dimension mismatch")
idx = pc.Index(INDEX_NAME)

# ─────────────────── 2. HELPER FUNCS ───────────────────────
def _embed(txt: str) -> List[float]:
    return openai.embeddings.create(model=EMBED_MODEL, input=txt).data[0].embedding


def mmr(matches, *, k: int, lam: float = .5):
    """Max-Marginal-Relevance selection (cosine on stored vectors)."""
    if k >= len(matches):
        return matches
    vecs  = [m.values for m in matches]
    sim   = cosine_similarity(vecs, vecs)
    q_sim = [m.score for m in matches]

    chosen, rest = [0], list(range(1, len(matches)))
    while len(chosen) < k and rest:
        mmr_score = [lam*q_sim[i] - (1-lam)*sim[i][chosen].max() for i in rest]
        best      = rest.pop(mmr_score.index(max(mmr_score)))
        chosen.append(best)
    return [matches[i] for i in chosen]

# ─────────────────── 3. MAIN ENTRYPOINT ────────────────────
def chat(
    question:   str,
    customer:   str = "demo01",
    doc_type:   str = "",
    top_k:      int = 60,
    concise:    bool = False,
    fallback:   bool = True,
) -> Dict[str, object]:
    """Return {answer, chunks_used, grounded, confidence}."""

    # 3-1 dense retrieval ---------------------------------------------------
    q_vec = _embed(question)
    res   = idx.query(
        vector=q_vec, top_k=top_k,
        filter={"customer": {"$eq": customer}},
        include_metadata=True, include_values=True,
    )
    if not res.matches:
        return {"answer":"Nothing found – retry in a few seconds.",
                "chunks_used":[], "grounded":False, "confidence":0.0}

    # 3-2 diversity pass + soft boosts --------------------------------------
    res.matches = mmr(res.matches, k=MMR_KEEP, lam=.5)

    boosted: List[Tuple[float, object]] = []
    q_lower = question.lower()
    for m in res.matches:
        score = m.score
        if doc_type and m.metadata.get("doc_type") == doc_type:
            score += .10
        note = (m.metadata.get("notes") or "").lower()
        if note and note in q_lower:
            score += .07
        boosted.append((score, m))
    res.matches = [m for score, m in sorted(boosted, key=lambda t: t[0], reverse=True)]

    if DEBUG:
        print("\nℹ️ After MMR/boost (top 6)")
        for i, m in enumerate(res.matches[:6]):
            print(f"  [{i}] {m.score:.4f} → {(m.metadata.get('text') or '')[:80]}…")

    # 3-3 fallback gate ------------------------------------------------------
    have_docs = res.matches and (res.matches[0].score + res.matches[1].score)/2 > FALLBACK_CUT
    if not have_docs:
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

    # 3-4 GPT-rerank to RERANK_KEEP -----------------------------------------
    rerank_prompt = (
        "Pick the most relevant chunks.\n\nQUESTION:\n"
        f"{question}\n\nCHUNKS:\n" +
        "\n\n".join(f"[{i}] {m.metadata.get('text','')[:400]}"
                    for i, m in enumerate(res.matches)) +
        f"\n\nReturn exactly {RERANK_KEEP} numbers (comma-separated)."
    )
    keep = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":rerank_prompt}],
        temperature=0,
    ).choices[0].message.content
    indices = {int(x) for x in re.findall(r"\d+", keep)}
    sel     = [m for i, m in enumerate(res.matches) if i in indices][:RERANK_KEEP] \
              or res.matches[:RERANK_KEEP]

    # 3-5 build GPT context & ask -------------------------------------------
    context = "\n\n".join(
        f"[{i+1}] {m.metadata.get('text','')[:1500].strip()}"
        for i, m in enumerate(sel)
    )

    sys_prompt = (
        "You are a technical assistant answering **only** from the excerpts below. "
        "Output a **Markdown ordered list** (1., 2., …); leave one blank line between "
        "items and **bold the imperative verb** at the start of each step.\n"
        "• Include every numbered step, WARNING / NOTICE block, key sequence and "
        "display confirmation verbatim.\n"
        "• Cite each fact like [2].  Reply “Not found in manual.” if needed."
    )
    user_prompt = (
        f"{context}\n\n► QUESTION: {question}\n\n"
        + ("Answer in 2–4 sentences and cite sources."
           if concise else "Write a precise, complete answer with citations.")
    )
    answer = openai.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":sys_prompt},
                  {"role":"user",  "content":user_prompt}],
        temperature=0, max_tokens=450,
    ).choices[0].message.content.strip()

    return {
        "answer":      answer,
        "chunks_used": [m.id for m in sel],
        "grounded":    True,
        "confidence":  min(max(sel[0].score, 0.0), 1.0),
    }

# ─────────────────── 4. SMOKE TEST ─────────────────────────
if __name__ == "__main__":
    for q in ["How do I drain the system?",
              "Does it support Modbus?",
              "What is 2 + 2?"]:
        res = chat(q, customer="demo01", doc_type="hardware", concise=True)
        print("\nQ:", q)
        print(res["answer"])
