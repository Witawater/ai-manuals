#!/usr/bin/env python3
"""
qa_demo.py  –  Ask questions against the Pinecone index
────────────────────────────────────────────────────────
Run inside the project root:

    source venv/bin/activate
    python qa_demo.py
"""

import os, time, dotenv
from openai   import OpenAI
from pinecone import Pinecone, ServerlessSpec, CloudProvider

# ────────────────────────────────────────────────────────────
# 1.  Secrets & clients
# ────────────────────────────────────────────────────────────
dotenv.load_dotenv(".env")

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("OPENAI_CHAT_MODEL",  "gpt-4o")
INDEX_NAME  = os.getenv("PINECONE_INDEX",     "manuals-small")

pc = Pinecone(
    api_key     = os.getenv("PINECONE_API_KEY"),
    environment = os.getenv("PINECONE_ENV"),      # e.g. aws-us-east-1
)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ────────────────────────────────────────────────────────────
# 2.  Ensure the index exists and has correct dimension
# ────────────────────────────────────────────────────────────
if INDEX_NAME not in pc.list_indexes().names():
    print(f"ℹ️  '{INDEX_NAME}' missing — creating an empty one …")
    cloud  = CloudProvider.AWS if "aws" in os.getenv("PINECONE_ENV") else CloudProvider.GCP
    region = os.getenv("PINECONE_ENV").split("-", 1)[-1]
    pc.create_index(
        INDEX_NAME,
        dimension=1536,                       # text-embedding-3-small
        metric="cosine",
        spec=ServerlessSpec(cloud=cloud, region=region),
    )
    while not pc.describe_index(INDEX_NAME).status["ready"]:
        time.sleep(2)
    print("✅  Index ready — ingest a PDF before chatting.")
else:
    info = pc.describe_index(INDEX_NAME)
    if info.dimension != 1536:
        raise RuntimeError(
            f"Index '{INDEX_NAME}' dim {info.dimension} ≠ 1536.\n"
            "Delete it or set PINECONE_INDEX to a fresh name."
        )

idx = pc.Index(INDEX_NAME)        # host auto-resolved

# ────────────────────────────────────────────────────────────
# 3.  Q-and-A helper
# ────────────────────────────────────────────────────────────
def chat(
    question : str,
    customer : str = "demo01",
    top_k    : int = 30,
    concise  : bool = False
) -> dict:
    """
    Returns:
      {
        "answer": "GPT-4o answer …",
        "chunks": ["demo01-a1b2c3d4", "demo01-e5f6…"]   # IDs used
      }
    """

    # 3-1) embed the question
    q_vec = client.embeddings.create(
        model=EMBED_MODEL, input=[question]
    ).data[0].embedding

    # 3-2) initial similarity search
    resp = idx.query(
        vector=q_vec,
        top_k=top_k,
        filter={"customer": {"$eq": customer}},
        include_metadata=True
    )
    if not resp.matches:
        return {"answer": "I couldn't find anything relevant in this manual.",
                "chunks": []}

    # 3-3) GPT-4o re-rank  → keep ≤ 4 chunks
    rerank_prompt = (
        "Which chunks best answer the question?\n\n"
        f"QUESTION:\n{question}\n\n"
        "CHUNKS:\n" +
        "\n\n".join(f"[{i}] {m.metadata['text']}"
                    for i, m in enumerate(resp.matches)) +
        "\n\nReturn the numbers of the 4 most relevant chunks, comma-separated."
    )
    best = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":rerank_prompt}],
        temperature=0
    ).choices[0].message.content

    keep_idx = {int(x) for x in best.split(",") if x.strip().isdigit()}
    resp.matches = [m for i, m in enumerate(resp.matches) if i in keep_idx]
    if not resp.matches:
        return {"answer": "I couldn't find anything relevant in this manual.",
                "chunks": []}

    # 3-4) build context for final answer
    context_parts = []
    for i, m in enumerate(resp.matches, start=1):
        title = m.metadata.get("title","").strip()
        tag   = f"[{i} – {title}]" if title else f"[{i}]"
        context_parts.append(f"{tag} {m.metadata['text']}")
    context = "\n\n".join(context_parts)

    # 3-5) final answer from GPT-4o
    prompt = (
        "You are a helpful support agent. ONLY use the context below.\n\n"
        f"{context}\n\n" +
        (
            "Answer in 2-4 sentences and cite tags like [1] or [2]."
            if concise else
            "Write a complete answer, including any useful details from the context. "
            "Cite the tags you used, e.g. [1] or [2]."
        )
    )
    answer = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{"role":"user","content":prompt}],
        temperature=0
    ).choices[0].message.content.strip()

    # 3-6) collect the *vector IDs* (not the 0-based indexes!)
    chunk_ids = [m.id for m in resp.matches]

    return {"answer": answer, "chunks": chunk_ids}

# ────────────────────────────────────────────────────────────
# 4.  Quick smoke-test
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for q in ["How do I descale the coffee maker?",
              "Can I change the brew strength?"]:
        print(f"\nQ: {q}")
        out = chat(q)
        print("Answer:", out["answer"])
        print("Chunks:", out["chunks"])
