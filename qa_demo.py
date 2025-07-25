#!/usr/bin/env python3
"""
qa_demo.py – Ask questions against the Pinecone index
──────────────────────────────────────────────────────
Adds Max-Marginal-Relevance (MMR) so GPT sees diverse chunks instead
of near-duplicates, and keeps the doc_type boost.
"""

from __future__ import annotations

import os, re, time
from typing import Dict, List, Tuple

import dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec      # v3 SDK
from sklearn.metrics.pairwise import cosine_similarity  # ← NEW

# ───── 1. Secrets & clients ─────
dotenv.load_dotenv(".env")

DEBUG       = True
EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o")
INDEX_NAME  = os.getenv("PINECONE_INDEX",     "manuals-small")

DIM = 3072 if "large" in EMBED_MODEL else 1536
ENV = os.getenv("PINECONE_ENV", "")
REGION = (ENV.split("-", 1)[-1] or "us-east1").lower()
CLOUD  = "aws" if "aws" in ENV.lower() else "gcp"

pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY"), environment=ENV)
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ───── 2. Ensure index exists ─────
if INDEX_NAME not in pc.list_indexes().names():
    print(f"ℹ️  '{INDEX_NAME}' missing – creating …")
    pc.create_index(
        name      = INDEX_NAME,
        dimension = DIM,
        metric    = "cosine",
        spec      = ServerlessSpec(cloud=CLOUD, region=REGION),
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

# ───── 3. Helpers ─────
def _embed(txt: str) -> List[float]:
    return openai.embeddings.create(model=EMBED_MODEL, input=txt).data[0].embedding


def mmr(matches, *, k: int = 20, lam: float = 0.5):
    """Max-Marginal-Relevance selection on Pinecone matches."""
    if k >= len(matches):
        return matches

    vecs = [m.values for m in matches]
    sim_mat   = cosine_similarity(vecs, vecs)
    query_sim = [m.score for m in matches]

    selected, rest = [], list(range(len(matches)))
    # first pick = most similar to query
    selected.append(rest.pop(0))

    while len(selected) < k and rest:
        cur_sims = sim_mat[:, selected].max(axis=1)
        mmr_scores = [
            lam * query_sim[i] - (1 - lam) * cur_sims[i] for i in rest
        ]
        idx = rest.pop(mmr_scores.index(max(mmr_scores)))
        selected.append(idx)

    return [matches[i] for i in selected]

# ───── 4. Main chat routine ─────
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
    """Return dict {answer, chunks_used, grounded, confidence}"""

    # 4-1) Dense retrieval
    q_vec = _embed(question)
    res   = idx.query(
        vector           = q_vec,
        top_k            = top_k,
        filter           = {"customer": {"$eq": customer}},
        include_metadata = True,
        include_values   = True,          # ← need vectors for MMR
    )

    if not res.matches:
        return {"answer":"Nothing found in the manual.",
                "chunks_used":[], "grounded":False, "confidence":0.0}

    # 4-2) MMR diversity BEFORE boosting
    res.matches = mmr(res.matches, k=20, lam=0.5)

    # 4-3) Soft doc_type boost
    boosted: List[Tuple[float, object]] = []
    for m in res.matches:
        score = m.score + (0.10 if doc_type and m.metadata.get("doc_type")==doc_type else 0)
        boosted.append((score, m))
    boosted.sort(key=lambda t: t[0], reverse=True)
    res.matches = [m for _, m in boosted]

    if DEBUG:
        print(f"\nℹ️ After MMR {len(res.matches)} chunks (first 6):")
        for i, m in enumerate(res.matches[:6]):
            print(f"  [{i}] score={m.score:.4f} → {m.metadata.get('text','')[:100].replace(chr(10),' ')}…")

    # 4-4) Fallback gate
    have_docs = res.matches and (
        sum(m.score for m in res.matches[:2]) / max(1, len(res.matches[:2]))
    ) > fallback_cut

    if not have_docs:
        if not fallback:
            return {"answer":"I couldn't find anything in the manual.",
                    "chunks_used":[], "grounded":True, "confidence":0.0}
        sys_prompt = (
    "You are a technical assistant answering **only** from the manual excerpts "
    "below. Follow these rules:\n"
    "• Include **all numbered steps verbatim**, key sequences, warnings, and any "
    "display confirmations.\n"
    "• Do not summarise away instructions; if the manual lists seven steps, "
    "your answer must list seven.\n"
    "• After each fact, cite the source like [2].\n"
    "• If the excerpts don’t answer, reply “Not found in manual.”"
    “• Include pre-operation warnings and bullet lists as well as numbered steps. ”
    )
        gpt_ans = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role":"system","content":sys_prompt},
                      {"role":"user","content":question}],
            temperature=0.3,
        ).choices[0].message.content.strip()
        return {"answer":"(General guidance – not in manual) "+gpt_ans,
                "chunks_used":[], "grounded":False, "confidence":0.4}

    # 4-5) LLM rerank
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

    # 4-6) Build GPT context
    context = "\n\n".join(
        f"[{i+1}] {m.metadata.get('text','')[:1500].strip()}" for i, m in enumerate(selected)
    )

    sys_prompt = (
        "You are a technical assistant who must answer **only** with information "
        "found in the provided manual chunks. Cite each claim like [1]. "
        "Do not omit relevant steps. Say 'Not found' if the manual doesn't cover it."
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
        temperature = 0,
        max_tokens  = 450,
    ).choices[0].message.content.strip()

    return {
        "answer":      answer,
        "chunks_used": [m.id for m in selected],
        "grounded":    True,
        "confidence":  min(max(selected[0].score, 0.0), 1.0),
    }

# ───── 5. Smoke test ─────
if __name__ == "__main__":
    for q in ["How do I drain the system?",
              "Does it support Modbus?",
              "What is 2 + 2?"]:
        print(f"\nQ: {q}")
        out = chat(q, customer="demo01", doc_type="hardware", concise=True)
        print("Answer:", out["answer"])
        print("Grounded:", out["grounded"])
        print("Chunks:", out["chunks_used"])
