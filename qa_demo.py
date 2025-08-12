#!/usr/bin/env python3
"""
qa_demo.py – hybrid Q-and-A for AI-Manuals
──────────────────────────────────────────
• Embedding × BM25 hybrid (α = 0.50)
• Safety-word guard (WARNING / NOTICE / ALARM …)
• Max-Marginal-Relevance diversity
• GPT rerank → clean **Markdown ordered-list** answers

Notes:
- Ingest writes to namespace = doc_id (match this when querying).
- We prefer metadata["preview"]; fallback to metadata["text"] for legacy data.
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

# ───────── 1) CONFIG ─────────
dotenv.load_dotenv(".env")

DEBUG         = os.getenv("QA_DEBUG", "true").lower() in {"1", "true", "yes"}
EMBED_MODEL   = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL    = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o")
RERANK_MODEL  = os.getenv("OPENAI_RERANK_MODEL", "gpt-4o-mini")
INDEX_NAME    = os.getenv("PINECONE_INDEX",     "manuals-large")
LOG_PATH      = os.getenv("QA_LOG_PATH", "/mnt/data/manual_eval.log")

MODEL_DIM = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
}.get(EMBED_MODEL, 3072)

MMR_KEEP     = 40
RERANK_KEEP  = 24
ALPHA        = 0.50
FALLBACK_CUT = 0.25

SAFETY_WORDS = ("warning", "notice", "important", "alarm", "code ")
# Small lexical nudges that map to “how do I…?” questions
KEY_BOOST = (
    "drain", "draining", "drain tap", "drain screw", "filling / draining",
    "fill", "empty", "switch on", "switch off", "power", "pump", "cooling machine"
)

pc     = Pinecone(api_key=os.getenv("PINECONE_API_KEY") or "")
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY") or "")

# ───────── 2) INDEX GUARD ─────────
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

def _txt(m) -> str:
    return (m.metadata.get("preview") or m.metadata.get("text") or "").lower()

def _embed(text: str) -> List[float]:
    try:
        return openai.embeddings.create(model=EMBED_MODEL, input=text).data[0].embedding
    except Exception as e:
        if DEBUG: print("embed failed:", e)
        return [0.0] * MODEL_DIM  # safe fallback; yields low scores

def mmr(matches, *, k: int, lam: float = .5):
    if k >= len(matches): return matches
    if not all(getattr(m, "values", None) is not None for m in matches):
        return matches[:k]  # if no vectors available, skip diversity safely
    vecs = [m.values for m in matches]
    sim = cosine_similarity(vecs, vecs)
    q_sim = [m.score for m in matches]
    chosen, rest = [0], list(range(1, len(matches)))
    while len(chosen) < k and rest:
        scores = [lam*q_sim[i] - (1-lam)*sim[i][chosen].max() for i in rest]
        best = rest.pop(scores.index(max(scores)))
        chosen.append(best)
    return [matches[i] for i in chosen]

# ───────── 4) MAIN ─────────
def chat(
    question: str,
    customer: str = "demo01",
    doc_type: str = "",
    doc_id: str = "",
    top_k: int = 60,
    concise: bool = False,
    fallback: bool = True,
) -> Dict[str, object]:

    if not question or not question.strip():
        return {"answer":"Please enter a question.",
                "chunks_used":[], "grounded":False, "confidence":0.0}

    if not doc_id:
        return {"answer": "No document selected. Please upload a manual first.",
                "chunks_used": [], "grounded": False, "confidence": 0.0}

    q_canon = _norm(question)
    q_vec = _embed(q_canon)

    # Filter by tenant; restrict by namespace=doc_id
    filter_by = {"customer": {"$eq": customer}}

    res = idx.query(
        vector=q_vec,
        top_k=top_k,
        filter=filter_by,
        include_metadata=True,
        include_values=True,
        namespace=doc_id,
    )

    if not res.matches:
        if DEBUG: print(f"❌ No match – q='{question}' cid='{customer}' doc='{doc_id}'")
        return {"answer": "Nothing found in this manual. Try rephrasing or another section.",
                "chunks_used": [], "grounded": False, "confidence": 0.0}

    # Tokenize docs for BM25
    docs_tok = [_tokenize(_txt(m)) for m in res.matches]
    bm25 = BM25Okapi(docs_tok)
    bm25_raw = np.asarray(bm25.get_scores(_tokenize(q_canon)), dtype=float)
    max_bm25 = bm25_raw.max() if bm25_raw.size else 0.0
    bm25_norm = (bm25_raw / max_bm25).tolist() if max_bm25 else bm25_raw.tolist()

    # Hybrid score (embed + BM25)
    for m, b, e in zip(res.matches, bm25_norm, [m.score for m in res.matches]):
        m.score = float(ALPHA * e + (1 - ALPHA) * b)

    # Safety-word preselection
    safety, rest = [], []
    for m in res.matches:
        (safety if any(w in _txt(m) for w in SAFETY_WORDS) else rest).append(m)
    safety = safety[:6]
    rest = mmr(rest, k=max(0, MMR_KEEP - len(safety)), lam=.5)
    res.matches = safety + rest

    # Heuristic boosts: doc_type/notes + light keyword contains
    boosted = []
    q_low = q_canon
    q_terms = set(_tokenize(q_low))
    for m in res.matches:
        s = m.score
        md = m.metadata or {}
        if doc_type and (md.get("doc_type") or "").lower() == doc_type.lower():
            s += 0.10
        note = (md.get("notes") or "").lower()
        if note and note in q_low:
            s += 0.07
        text_lc = _txt(m)
        if any(k in text_lc for k in KEY_BOOST):  # small lexical nudge
            s += 0.10
        if any(t in text_lc for t in q_terms):    # tiny nudge for exact tokens
            s += 0.05
        boosted.append((s, m))
    res.matches = [m for s, m in sorted(boosted, key=lambda t: t[0], reverse=True)]

    if DEBUG:
        print("\nℹ️ After safety/MMR/boost (top 6)")
        for i, m in enumerate(res.matches[:6]):
            pv = _txt(m)[:80].replace("\n", " ")
            print(f"  [{i}] {m.score:.4f} → {pv}…  (pg={m.metadata.get('page')})")

    # If scores are very low, optionally do a general-model fallback
    top_scores = [m.score for m in res.matches[:2]] or [0.0]
    if np.mean(top_scores) < FALLBACK_CUT and fallback:
        try:
            fb = openai.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": question}
                ],
                temperature=.3,
            ).choices[0].message.content.strip()
        except Exception as e:
            if DEBUG: print("fallback failed:", e)
            fb = "Sorry—couldn’t generate a fallback just now."
        return {"answer": "(General guidance) " + fb,
                "chunks_used": [], "grounded": False, "confidence": .4}

    # Rerank with a cheaper model; only ask for as many as we have
    want = min(RERANK_KEEP, len(res.matches))
    rerank_prompt = (
        "Pick the most relevant chunks.\n\nQUESTION:\n"
        f"{question}\n\nCHUNKS:\n" +
        "\n\n".join(f"[{i}] {_txt(m)[:400]}" for i, m in enumerate(res.matches)) +
        f"\n\nReturn exactly {want} numbers (comma-separated). No other text."
    )
    try:
        keep = openai.chat.completions.create(
            model=RERANK_MODEL,
            messages=[{"role": "user", "content": rerank_prompt}],
            temperature=0,
        ).choices[0].message.content
        idx_keep = {int(x) for x in re.findall(r"\d+", keep)}
    except Exception as e:
        if DEBUG: print("rerank failed:", e)
        idx_keep = set()

    sel = [m for i, m in enumerate(res.matches) if i in idx_keep][:want] or res.matches[:want]

    # Build grounded context
    context = "\n\n".join(
        f"[{i+1}] {_txt(m)[:1500].strip()}"
        for i, m in enumerate(sel)
    )

    # Generation: be strict about using context; do not force “Not found” unless zero matches
    sys_prompt = (
        "You are a technical assistant answering **only** from the excerpts below. "
        "Output a **Markdown ordered list** of steps; leave one blank line between items and "
        "**bold the imperative verb** at the start of each step.\n"
        "• Include every numbered step, WARNING / NOTICE block, key sequence and "
        "display confirmation verbatim when present.\n"
        "• Cite each fact like [2]."
    )
    user_prompt = (
        f"{context}\n\n► QUESTION: {question}\n\n" +
        ("Answer in 2-4 sentences and cite sources."
         if concise else "Write a precise, complete answer with citations.")
    )

    try:
        answer = openai.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role": "system", "content": sys_prompt},
                      {"role": "user", "content": user_prompt}],
            temperature=0, max_tokens=450,
        ).choices[0].message.content.strip()
    except Exception as e:
        if DEBUG: print("answer gen failed:", e)
        return {"answer":"Temporarily couldn’t generate an answer. Please try again.",
                "chunks_used":[m.id for m in sel], "grounded":True,
                "confidence": float(sel[0].score) if sel else 0.0}

    # Write metrics line for /metrics
    try:
        avg_conf = float(np.mean([m.score for m in sel[:8]])) if sel else 0.0
        with open(LOG_PATH, "a") as f:
            # "<timestamp> <tag> <count> <score>"
            f.write(f"{time.strftime('%Y-%m-%d')} qa {len(sel[:8])} {avg_conf:.3f}\n")
    except Exception as e:
        if DEBUG: print("metrics log failed:", e)

    # Prepare a compact sources list for UI/debugging
    sources = [
        {
            "id": m.id,
            "page": int((m.metadata or {}).get("page") or 0),
            "preview": (_txt(m)[:140] + "…") if len(_txt(m)) > 140 else _txt(m)
        }
        for m in sel[:8]
    ]

    return {
        "answer": answer or "No answer generated.",
        "chunks_used": [m.id for m in sel[:8]],
        "sources": sources,
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
