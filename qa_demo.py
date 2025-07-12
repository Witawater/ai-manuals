#!/usr/bin/env python3
"""
qa_demo.py – Ask questions against the Pinecone index
──────────────────────────────────────────────────────
Set env-vars:

  QA_MIN_SIM   – similarity cut-off (default 0.45)
  QA_RERANK_N  – chunks kept after re-rank (default 6)
  QA_CTX_CHARS – chars per chunk in final prompt (default 1000)
"""

import os, time, re
from typing import Dict, List, Optional

import dotenv
from openai   import OpenAI
from pinecone import Pinecone, ServerlessSpec, CloudProvider

# ─── 0. Env & runtime config ─────────────────────────────────
dotenv.load_dotenv(".env")

DEBUG        = os.getenv("DEBUG_QA", "1") == "1"
EMBED_MODEL  = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL   = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o")
INDEX_NAME   = os.getenv("PINECONE_INDEX",     "manuals-small")
DIM          = 3072

MIN_SIM      = float(os.getenv("QA_MIN_SIM",  "0.45"))
RERANK_N_ENV = int  (os.getenv("QA_RERANK_N", "6"))
CTX_CHARS    = int  (os.getenv("QA_CTX_CHARS", "1000"))

# ─── 1. Clients ──────────────────────────────────────────────
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"),
              environment=os.getenv("PINECONE_ENV") or "")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─── 2. Ensure index exists ─────────────────────────────────
if INDEX_NAME not in pc.list_indexes().names():
    if DEBUG: print(f"ℹ️  '{INDEX_NAME}' missing – creating …")
    cloud  = CloudProvider.AWS if "aws" in os.getenv("PINECONE_ENV", "") \
            else CloudProvider.GCP
    region = os.getenv("PINECONE_ENV", "").split("-", 1)[-1] or "us-east1"
    pc.create_index(INDEX_NAME, dimension=DIM, metric="cosine",
                    spec=ServerlessSpec(cloud=cloud, region=region))
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
else:
    info = pc.describe_index(INDEX_NAME)
    if info.dimension != DIM:
        raise RuntimeError(
            f"Index '{INDEX_NAME}' dim {info.dimension} ≠ {DIM}.")

idx = pc.Index(INDEX_NAME)

# ─── 3. Q-and-A helper ───────────────────────────────────────
def chat(
    question      : str,
    customer      : str = "demo01",
    manual_id     : Optional[str] = None,   # new: filter by manual
    top_k         : int = 50,
    concise       : bool = False,
    fallback      : bool = True,
    rerank_keep   : int | None = None,
    min_sim       : float | None = None,
) -> Dict[str, object]:
    """
    Returns
      {answer, chunks_used, grounded (bool), confidence (0-1)}
    """
    rerank_keep = rerank_keep or RERANK_N_ENV
    min_sim     = min_sim     or MIN_SIM

    # Build Pinecone filter
    flt = {"customer": {"$eq": customer}}
    if manual_id:
        flt["manual_id"] = {"$eq": manual_id}

    # 3-1) embed → initial search (to infer product name)
    q_vec_initial = client.embeddings.create(
        model=EMBED_MODEL, input=question
    ).data[0].embedding

    resp_initial = idx.query(vector=q_vec_initial,
                             top_k=top_k, filter=flt,
                             include_metadata=True)
    product_name = (resp_initial.matches[0].metadata.get("product", "machine")
                    if resp_initial.matches else "machine")

    def normalise(q: str, name: str) -> str:
        return re.sub(r"\b(the\s)?(machine|unit|device|appliance)\b",
                      name, q, flags=re.I)

    q_norm = normalise(question, product_name)
    q_vec  = client.embeddings.create(model=EMBED_MODEL,
                                      input=q_norm).data[0].embedding

    resp = idx.query(vector=q_vec, top_k=top_k,
                     filter=flt, include_metadata=True)

    if DEBUG:
        print(f"ℹ️ Top {len(resp.matches)} retrieved chunks:")
        for i, m in enumerate(resp.matches[:6]):
            snip = m.metadata.get("text", "")[:120].replace("\n", " ")
            print(f"  [{i}] {m.score:.4f} “{snip}…”")

    have_docs = resp.matches and resp.matches[0].score > min_sim

    # 3-2) GPT fallback branch
    if not have_docs:
        if not fallback:
            return {"answer": "No match in manual.",
                    "chunks_used": [], "grounded": True, "confidence": 0.0}
        sys_prompt = (
            "You are a knowledgeable support agent. The manual does not "
            "cover the user's question. Answer from general knowledge "
            "or admit you don't know."
        )
        answer = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[{"role":"system","content":sys_prompt},
                      {"role":"user",  "content":q_norm}],
            temperature=0.3
        ).choices[0].message.content.strip()
        return {"answer": "(General guidance – not in manual) "+answer,
                "chunks_used": [], "grounded": False, "confidence": 0.4}

    # 3-3) GPT re-rank
    rerank_prompt = f"""You must pick the most relevant manual chunks.

QUESTION:
{q_norm}

CHUNKS:
""" + "\n\n".join(f"[{i}] {m.metadata['text'][:400].strip()}"
                  for i, m in enumerate(resp.matches)) + f"""

Return ONLY the numbers of the {rerank_keep} most relevant chunks,
comma-separated. If unsure, return an empty list."""
    best = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":rerank_prompt}],
        temperature=0
    ).choices[0].message.content.strip()

    if DEBUG: print("ℹ️ Rerank chose:", best)

    try:
        keep = {int(x) for x in re.split(r"[,\s]+", best) if x.isdigit()}
    except Exception:
        keep = set()

    resp.matches = ([m for i, m in enumerate(resp.matches) if i in keep][:rerank_keep]
                    if keep else resp.matches[:rerank_keep])

    if not resp.matches:            # safety-net
        resp.matches = resp_initial.matches[:rerank_keep]

    # 3-4) Build context
    ctx_parts = []
    for i, m in enumerate(resp.matches, 1):
        tag = f"[{i}]"
        text = m.metadata.get("text", "")[:CTX_CHARS].strip()
        ctx_parts.append(f"{tag} {text}")
    context = "\n\n".join(ctx_parts)

    # 3-5) Final answer
    system_prompt = ("Answer strictly from the excerpts, citing [n]. "
                     "If not answered, say so.")
    user_prompt = (
        f"{context}\n\n► QUESTION: {q_norm}\n\n" +
        ("Answer in 2-4 sentences and cite sources."
         if concise else
         "Write a full answer and cite only relevant sources.")
    )
    answer = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"system","content":system_prompt},
                  {"role":"user",  "content":user_prompt}],
        temperature=0
    ).choices[0].message.content.strip()

    if DEBUG: print("ℹ️ Answer:", answer)

    confidence = min(max(resp.matches[0].score, 0.0), 1.0)

    return {"answer": answer,
            "chunks_used": [m.id for m in resp.matches],
            "grounded": True,
            "confidence": confidence}

# ─── 4. Smoke-test ────────────────────────────────────────────
if __name__ == "__main__":
    for q in ["How do I descale the coffee maker?",
              "Can I change the brew strength?",
              "What is 2 + 2?"]:
        print("\nQ:", q)
        out = chat(q, concise=True)
        print("Answer:", out["answer"])
        print("Chunks:", out["chunks_used"])
        print("Grounded:", out["grounded"])
