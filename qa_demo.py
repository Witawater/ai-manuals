#!/usr/bin/env python3
"""
qa_demo.py  –  Ask questions against the Pinecone index
────────────────────────────────────────────────────────
CLI smoke-test:

    source venv/bin/activate
    python qa_demo.py
"""

import os, time
from typing import Dict, List

import dotenv
from openai   import OpenAI
from pinecone import Pinecone, ServerlessSpec, CloudProvider

# ────────────────────────────────────────────────────────────
# 1.  Secrets & clients
# ────────────────────────────────────────────────────────────
dotenv.load_dotenv(".env")

EMBED_MODEL = os.getenv("embedding_model", "text-embedding-3-large")
CHAT_MODEL  = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o")
INDEX_NAME  = os.getenv("PINECONE_INDEX",     "manuals-small")
DIM         = 3072                            # text-embedding-3-large

pc = Pinecone(
    api_key     = os.getenv("PINECONE_API_KEY"),
    environment = os.getenv("PINECONE_ENV") or "",   # e.g. aws-us-east-1
)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ────────────────────────────────────────────────────────────
# 2.  Ensure the index exists & dimension matches
# ────────────────────────────────────────────────────────────
if INDEX_NAME not in pc.list_indexes().names():
    print(f"ℹ️  '{INDEX_NAME}' missing — creating an empty one …")
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
    print("✅  Index ready — ingest a PDF before chatting.")
else:
    info = pc.describe_index(INDEX_NAME)
    if info.dimension != DIM:
        raise RuntimeError(
            f"Index '{INDEX_NAME}' dim {info.dimension} ≠ {DIM}.\n"
            "Delete it or point PINECONE_INDEX to a fresh name."
        )

idx = pc.Index(INDEX_NAME)   # host auto-resolved

# ────────────────────────────────────────────────────────────
# 3.  Q-and-A helper
# ────────────────────────────────────────────────────────────
def chat(
    question : str,
    customer : str = "demo01",
    top_k    : int = 30,
    concise  : bool = False,
    fallback : bool = True,      # allow GPT fallback
    rerank_keep : int = 4        # how many chunks after re-rank
) -> Dict[str, object]:
    """
    Returns
      {
        "answer"     : "...",
        "chunks_used": ["vec-id-1", …],   # 0-length if fallback
        "grounded"   : bool               # True = from manual
      }
    """

    import re

    def normalize_question(q: str, product_name: str) -> str:
        return re.sub(r"\b(the\s)?(machine|unit|device|appliance)\b", product_name, q, flags=re.IGNORECASE)

    # Embed the original question first to get a vector for querying
    q_vec_initial = client.embeddings.create(
        model=EMBED_MODEL, input=[question]
    ).data[0].embedding

    # Initial similarity search to infer product name
    resp_initial = idx.query(
        vector=q_vec_initial,
        top_k=top_k,
        filter={"customer": {"$eq": customer}},
        include_metadata=True
    )

    # Extract product name from top matching chunk's metadata or default
    product_name = "machine"  # default
    if resp_initial.matches:
        product_name = resp_initial.matches[0].metadata.get("product", "machine")

    # Normalize the original question with the product name
    normalized_question = normalize_question(question, product_name)

    # Embed the normalized question
    q_vec = client.embeddings.create(
        model=EMBED_MODEL, input=[normalized_question]
    ).data[0].embedding

    # Query Pinecone once with the normalized question vector
    resp = idx.query(
        vector=q_vec,
        top_k=top_k,
        filter={"customer": {"$eq": customer}},
        include_metadata=True
    )

    print(f"ℹ️ Top {len(resp.matches)} retrieved chunks:")
    for i, m in enumerate(resp.matches):
        snippet = m.metadata.get("text", "")[:75].replace("\n", " ")
        print(f"  [{i}] score={m.score:.4f} preview='{snippet}...'")

    have_docs = bool(resp.matches) and resp.matches[0].score > 0.50
    if not have_docs and not fallback:
        return {"answer": "I couldn't find anything relevant in this manual.",
                "chunks_used": [],
                "grounded": True}

    # 3-A) GPT fallback if no suitable chunks
    if not have_docs:
        sys_prompt = (
            "You are a knowledgeable support agent. "
            "The official manual does not cover the user's question. "
            "If you are confident based on your own general domain knowledge, answer clearly. "
            "If the context is limited or unclear, you may still answer based on best available information, "
            "but always preface with '(General guidance – not in manual)'."
        )
        answer = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": normalized_question},
            ],
            temperature=0.3
        ).choices[0].message.content.strip()

        return {"answer": answer,
                "chunks_used": [],
                "grounded": False}

    # 3-3) GPT-4o re-rank → keep top N chunks
    rerank_prompt = (
        "Which chunks best answer the question?\n\n"
        f"QUESTION:\n{normalized_question}\n\n"
        "CHUNKS:\n" +
        "\n\n".join(f"[{i}] ({m.metadata.get('title', 'No Title')}) {m.metadata['text']}"
                    for i, m in enumerate(resp.matches)) +
        "\n\nNote: If the question refers to tubing, prioritize chunks with keywords like tubing, hose, ID, fittings, M16x1, or temperature rating. Give preference to chunks whose section title is clearly related.\n\n"
        f"Return ONLY the numbers of the {rerank_keep} most relevant chunks, comma-separated (e.g. 0,2,3,7)."
    )
    best = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": rerank_prompt}],
        temperature=0
    ).choices[0].message.content.strip()

    print(f"ℹ️ Rerank selected chunk indices: {best}")

    keep = {int(x) for x in best.split(",") if x.strip().isdigit()}
    resp.matches = (
        [m for i, m in enumerate(resp.matches) if i in keep][:rerank_keep]
        if keep else
        resp.matches[:rerank_keep]
    )

    if not resp.matches:
        return {"answer": "I couldn't find anything relevant in this manual.",
                "chunks_used": [], "grounded": False}

    # 3-4) build context
    context_parts: List[str] = []
    for i, m in enumerate(resp.matches, start=1):
        title = m.metadata.get("title") or ""
        tag   = f"[{i} – {title}]" if title else f"[{i}]"
        snippet = m.metadata["text"][:500]
        context_parts.append(f"{tag} {snippet}")
    context = "\n\n".join(context_parts)

    print(f"ℹ️ Context used for grounding:\n{context}")

    # 3-5) final answer
    prompt = (
        "You are reading a product manual to find an exact answer to a technical question. Focus on the sections below as if you're scanning a PDF. If the context contains technical specifications relevant to the question, summarize them clearly and precisely, regardless of product type.\n\n"
        f"{context}\n\n" +
        ("Answer in 2-4 sentences and cite tags like [1] or [2]."
         if concise else
         "Write a complete answer, using only the context provided. Cite tags like [1] or [2] only if they directly support the answer. Do not cite irrelevant chunks.")
    )
    answer = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    ).choices[0].message.content.strip()

    print(f"ℹ️ Final answer:\n{answer}")

    return {
        "answer":      answer,
        "chunks_used": [m.id for m in resp.matches],
        "grounded":    True,
    }

# ────────────────────────────────────────────────────────────
# 4.  Smoke-test
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for q in [
        "How do I descale the coffee maker?",
        "Can I change the brew strength?",
        "What is 2 + 2?"  # should trigger fallback
    ]:
        print(f"\nQ: {q}")
        out = chat(q)
        print("Answer:", out["answer"])
        print("Chunks:", out["chunks_used"])
        print("Grounded:", out["grounded"])