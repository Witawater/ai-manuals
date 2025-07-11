#!/usr/bin/env python3
"""
qa_demo.py – Ask questions against the Pinecone index
──────────────────────────────────────────────────────
Quick smoke-test:

    source venv/bin/activate
    python qa_demo.py
"""

import os, time, re
from typing import Dict, List

import dotenv
from openai   import OpenAI
from pinecone import Pinecone, ServerlessSpec, CloudProvider

# ───────────────────────────────────────────────
# 1. Secrets & clients
# ───────────────────────────────────────────────
dotenv.load_dotenv(".env")

DEBUG = True

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
CHAT_MODEL  = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o")
INDEX_NAME  = os.getenv("PINECONE_INDEX",     "manuals-small")
DIM         = 3072                            # text-embedding-3-large

pc = Pinecone(
    api_key     = os.getenv("PINECONE_API_KEY"),
    environment = os.getenv("PINECONE_ENV") or "",   # e.g. aws-us-east-1
)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ───────────────────────────────────────────────
# 2. Ensure the index exists & dimension matches
# ───────────────────────────────────────────────
if INDEX_NAME not in pc.list_indexes().names():
    if DEBUG: print(f"ℹ️  '{INDEX_NAME}' missing – creating an empty one …")
    cloud  = CloudProvider.AWS if "aws" in os.getenv("PINECONE_ENV", "") \
            else CloudProvider.GCP
    region = os.getenv("PINECONE_ENV", "").split("-", 1)[-1] or "us-east1"
    pc.create_index(
        INDEX_NAME,
        dimension=DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
    if DEBUG: print("✅  Index ready — ingest a PDF before chatting.")
else:
    info = pc.describe_index(INDEX_NAME)
    if info.dimension != DIM:
        raise RuntimeError(
            f"Index '{INDEX_NAME}' dim {info.dimension} ≠ {DIM}.\n"
            "Delete it or point PINECONE_INDEX to a fresh name."
        )

idx = pc.Index(INDEX_NAME)   # host auto-resolved

# ───────────────────────────────────────────────
# 3. Q-and-A helper
# ───────────────────────────────────────────────
def chat(
    question      : str,
    customer      : str = "demo01",
    top_k         : int = 50,
    concise       : bool = False,
    fallback      : bool = True,   # allow GPT fallback
    rerank_keep   : int = 4,       # keep N chunks after re-rank
    fallback_cut  : float = 0.35,  # similarity threshold
) -> Dict[str, object]:
    """
    Returns
      {
        "answer"     : "...",
        "chunks_used": ["vec-id-1", …],   # 0-length if fallback
        "grounded"   : bool,              # True = from manual
        "confidence" : float              # ad-hoc 0-1
      }
    """

    # 3-1)  Embed → initial search (to infer product name)
    q_vec_initial = client.embeddings.create(
        model=EMBED_MODEL, input=question
    ).data[0].embedding

    resp_initial = idx.query(
        vector=q_vec_initial,
        top_k=top_k,
        filter={"customer": {"$eq": customer}},
        include_metadata=True
    )
    product_name = (
        resp_initial.matches[0].metadata.get("product", "machine")
        if resp_initial.matches else "machine"
    )

    # Normalise “the unit/device” ↔ concrete product name
    def normalise(q: str, name: str) -> str:
        return re.sub(
            r"\b(the\s)?(machine|unit|device|appliance)\b",
            name,
            q,
            flags=re.IGNORECASE,
        )

    q_norm  = normalise(question, product_name)
    q_vec   = client.embeddings.create(model=EMBED_MODEL, input=q_norm).data[0].embedding

    resp = idx.query(
        vector=q_vec,
        top_k=top_k,
        filter={"customer": {"$eq": customer}},
        include_metadata=True
    )

    if DEBUG:
        print(f"ℹ️ Top {len(resp.matches)} retrieved chunks:")
        for i, m in enumerate(resp.matches[:6]):  # show first 6
            snippet = m.metadata.get("text", "")[:120].replace("\n", " ")
            print(f"  [{i}] score={m.score:.4f} “{snippet}…”")

    # 3-2) Decide fallback
    have_docs = (
        bool(resp.matches) and
        (sum(m.score for m in resp.matches[:2]) / max(1, len(resp.matches[:2]))) > fallback_cut
    )
    if not have_docs and not fallback:
        return {
            "answer":      "I couldn't find anything relevant in this manual.",
            "chunks_used": [],
            "grounded":    True,
            "confidence":  0.0,
        }

    # 3-3) GPT fallback branch
    if not have_docs:
        sys_prompt = (
            "You are a knowledgeable support agent. "
            "The official manual does not cover the user's question. "
            "If you can answer from general domain knowledge, do so clearly. "
            "Otherwise say you don't have enough information."
        )
        answer = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": q_norm},
            ],
            temperature=0.3,
        ).choices[0].message.content.strip()

        return {
            "answer":      "(General guidance – not in manual) " + answer,
            "chunks_used": [],
            "grounded":    False,
            "confidence":  0.4,
        }

    # 3-4) Re-rank chunks with GPT
    rerank_prompt = f"""You are selecting the most relevant manual chunks to answer the user’s question.

QUESTION:
{q_norm}

CHUNKS:
""" + "\n\n".join(
        f"[{i}] {m.metadata['text'][:400].strip()}"
        for i, m in enumerate(resp.matches)
    ) + f"""

Return ONLY the numbers of the {rerank_keep} most relevant chunks, comma-separated (e.g. 0,2,3,7).
If none are clearly relevant, return an empty list.
"""
    best = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": rerank_prompt}],
        temperature=0,
    ).choices[0].message.content.strip()

    if DEBUG: print(f"ℹ️ Rerank chose: {best}")

    try:
        keep = {int(x) for x in re.split(r"[,\s]+", best) if x.isdigit()}
    except Exception:
        keep = set()

    resp.matches = (
        [m for i, m in enumerate(resp.matches) if i in keep][:rerank_keep]
        if keep else
        resp.matches[:rerank_keep]
    )
    if not resp.matches:
        return {
            "answer":      "I couldn't find anything relevant in this manual.",
            "chunks_used": [],
            "grounded":    False,
            "confidence":  0.0,
        }

    # 3-5) Build context (full chunk, trimmed to ~1 500 chars)
    context_parts: List[str] = []
    for i, m in enumerate(resp.matches, start=1):
        tag  = f"[{i}]"
        text = m.metadata.get("text", "")[:1500].strip()
        context_parts.append(f"{tag} {text}")

    context = "\n\n".join(context_parts)

    if DEBUG:
        print("ℹ️ Context for final answer (truncated to 1 k chars):")
        print(context[:1000] + ("…" if len(context) > 1000 else ""))

    # 3-6) Final answer
    system_prompt = (
        "You are a technical assistant answering strictly from the provided manual excerpts. "
        "Cite sources like [1], [2]. If the excerpts don't answer, say you don't have that info."
    )
    user_prompt = (
        f"{context}\n\n"
        f"► QUESTION: {q_norm}\n\n"
        + (
            "Answer in 2-4 sentences and cite sources."
            if concise else
            "Write a complete, precise answer and cite only relevant sources."
        )
    )
    answer = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0,
    ).choices[0].message.content.strip()

    if DEBUG: print("ℹ️ Answer:", answer)

    # Naïve confidence: similarity of best chunk (0-1) clipped
    confidence = min(max(resp.matches[0].score, 0.0), 1.0)

    return {
        "answer":      answer,
        "chunks_used": [m.id for m in resp.matches],
        "grounded":    True,
        "confidence":  confidence,
    }

# ───────────────────────────────────────────────
# 4. Smoke-test
# ───────────────────────────────────────────────
if __name__ == "__main__":
    for q in [
        "How do I descale the coffee maker?",
        "Can I change the brew strength?",
        "What is 2 + 2?",  # should trigger fallback
    ]:
        print(f"\nQ: {q}")
        out = chat(q, concise=True)
        print("Answer:", out["answer"])
        print("Chunks:", out["chunks_used"])
        print("Grounded:", out["grounded"])
