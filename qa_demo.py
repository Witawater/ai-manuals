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

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-large")
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

    # 3-1) embed the question
    q_vec = client.embeddings.create(
        model=EMBED_MODEL, input=[question]
    ).data[0].embedding

    # 3-2) similarity search
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

    have_docs = bool(resp.matches) and resp.matches[0].score > 0.65
    if not have_docs and not fallback:
        return {"answer": "I couldn't find anything relevant in this manual.",
                "chunks_used": [],
                "grounded": True}

    # 3-A) GPT fallback if no suitable chunks
    if not have_docs:
        sys_prompt = (
            "You are a knowledgeable support agent. "
            "The official manual does not cover the user's question. "
            "Answer from general domain knowledge only if you are confident, "
            "and preface with '(General guidance – not in manual)'."
        )
        answer = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": question},
            ],
            temperature=0.3
        ).choices[0].message.content.strip()

        return {"answer": answer,
                "chunks_used": [],
                "grounded": False}

    # 3-3) GPT-4o re-rank → keep top N chunks
    rerank_prompt = (
        "Which chunks best answer the question?\n\n"
        f"QUESTION:\n{question}\n\n"
        "CHUNKS:\n" +
        "\n\n".join(f"[{i}] {m.metadata['text']}"
                    for i, m in enumerate(resp.matches)) +
        f"\n\nReturn ONLY the numbers of the {rerank_keep} most relevant "
        "chunks, comma-separated (e.g. 0,2,3,7)."
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
        "You are a helpful support agent. ONLY use the context below.\n\n"
        f"{context}\n\n" +
        ("Answer in 2-4 sentences and cite tags like [1] or [2]."
         if concise else
         "Write a complete answer, including any useful details from the "
         "context. Cite the tags you used, e.g. [1] or [2].")
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