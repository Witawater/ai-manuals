#!/usr/bin/env python3
"""
FastAPI service for AI‑Manuals
──────────────────────────────
• POST /upload            – add a PDF to Pinecone
• POST /chat              – ask a question → {"answer", "chunks_used"}
• POST /feedback          – thumbs‑up / thumbs‑down (+ chunk IDs)
• GET  /feedback/summary  – daily 👍 / 👎 counts
• GET  /feedback/chunks   – hall‑of‑shame per chunk
• static /                – tiny HTML/JS front‑end
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from typing import Dict, List, Optional

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

# ─── local modules ────────────────────────────────────────────
from ingest_manual import ingest
from qa_demo import chat
from auth import require_api_key
from db import engine

# ─── tunables (env‑vars) ──────────────────────────────────────
CHUNK_TOKENS: int = int(os.getenv("CHUNK_TOKENS", "800"))
OVERLAP: int = int(os.getenv("OVERLAP", "150"))

# ─── FastAPI & CORS ───────────────────────────────────────────
app = FastAPI()

# Allow multiple origins so local dev works without edits.
CORS_ORIGINS = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "https://ai-manuals.onrender.com,http://localhost:5173,http://127.0.0.1:5173",
).split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ╭────────────────────────── 1)  PDF upload ─────────────────╮

def _ingest_and_cleanup(path: str, customer: str) -> None:
    """Runs in background: ingest PDF then delete temp file."""
    try:
        ingest(path, customer, CHUNK_TOKENS, OVERLAP, dry_run=False)
        print(f"✅ Ingest complete: {path}")
    except Exception as exc:
        print(f"🛑 Ingest failed: {exc}")
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


@app.post("/upload", dependencies=[Depends(require_api_key)])
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    customer: str = Form("demo01"),
):
    print(f"📥 Upload received: {file.filename} from {customer}")

    tmp_dir = tempfile.gettempdir()
    tmp_name = f"{uuid.uuid4().hex}_{file.filename}"
    tmp_path = os.path.join(tmp_dir, tmp_name)

    with open(tmp_path, "wb") as handle:
        handle.write(await file.read())

    background_tasks.add_task(_ingest_and_cleanup, tmp_path, customer)

    return {"status": "queued", "file": file.filename}


# ╰────────────────────────────────────────────────────────────╯

# ╭────────────────────────── 2)  Chat ────────────────────────╮

@app.post("/chat", dependencies=[Depends(require_api_key)])
async def ask(question: str = Form(...), customer: str = Form("demo01")):
    result = chat(question, customer)
    if result.get("grounded") is False:
        print("⚠️  Fallback to GPT (not grounded)")
    return result


# ╰────────────────────────────────────────────────────────────╯

# ╭────────────────────── 3)  Feedback thumbs ─────────────────╮


class FeedbackIn(BaseModel):
    customer: str
    question: str
    answer: str
    score: int
    chunks_used: Optional[List[str]] = None
    chunks: Optional[List[str]] = None


@app.post("/feedback", dependencies=[Depends(require_api_key)])
def add_feedback(data: FeedbackIn):
    # Incoming validation
    if data.score not in (-1, 1):
        raise HTTPException(status_code=400, detail="score must be +1 or -1")

    chunk_ids: List[str] = data.chunks_used or data.chunks or []
    print(
        f"📝 Feedback: {'👍' if data.score == 1 else '👎'} on {len(chunk_ids)} chunks"
    )

    # NOTE: `chunks_used` is a Postgres **text[]** column. We therefore pass the
    # Python list directly – psycopg / SQLAlchemy will serialise it as an array.
    insert_sql = text(
        """
        INSERT INTO feedback
            (customer, question, answer, score, chunks_used)
        VALUES
            (:customer, :question, :answer, :score, :chunks)
        """
    )

    with engine.begin() as conn:
        conn.execute(
            insert_sql,
            {
                "customer": data.customer,
                "question": data.question,
                "answer": data.answer,
                "score": data.score,
                "chunks": chunk_ids,
            },
        )

    return {"ok": True}


# ╰────────────────────────────────────────────────────────────╯

# ╭────────────────────── 4)  Daily thumbs summary ────────────╮


@app.get("/feedback/summary")
def feedback_summary(days: int = 7) -> List[Dict]:
    sql = text(
        f"""
        SELECT
            date_trunc('day', ts)::date AS day,
            SUM((score =  1)::int)      AS up,
            SUM((score = -1)::int)      AS down
        FROM feedback
        WHERE ts >= now() - INTERVAL '{days} days'
        GROUP BY day
        ORDER BY day;
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(sql)
        return [
            {"day": r.day.isoformat(), "up": int(r.up), "down": int(r.down)} for r in rows
        ]


# ╰────────────────────────────────────────────────────────────╯

# ╭────────────────────── 5)  Hall‑of‑shame chunks ────────────╮


@app.get("/feedback/chunks")
def worst_chunks(days: int = 30, min_votes: int = 1, limit: int = 50) -> List[Dict]:
    sql = text(
        """
        SELECT
            unnest(chunks_used)                        AS chunk_id,
            COUNT(*)                                   AS total,
            SUM((score =  1)::int)                     AS up,
            SUM((score = -1)::int)                     AS down,
            ROUND(100.0 * SUM((score =  1)::int)::numeric / NULLIF(COUNT(*), 0), 1) AS up_pct
        FROM feedback
        WHERE ts >= now() - INTERVAL :days || ' days'
        GROUP BY chunk_id
        HAVING COUNT(*) >= :min_votes
        ORDER BY up_pct ASC, total DESC
        LIMIT :limit;
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(
            sql, {"days": days, "min_votes": min_votes, "limit": limit}
        )
        return [
            {
                "chunk_id": r.chunk_id,
                "total": int(r.total),
                "up": int(r.up),
                "down": int(r.down),
                "up_pct": float(r.up_pct) if r.up_pct is not None else None,
            }
            for r in rows
        ]


# ╰────────────────────────────────────────────────────────────╯

# ╭────────────────── 6)  Serve the small front‑end ───────────╮

app.mount("/", StaticFiles(directory="web", html=True), name="web")

# ╰────────────────────────────────────────────────────────────╯
