#!/usr/bin/env python3
"""
FastAPI service for AI-Manuals
──────────────────────────────
• POST /upload            – add a PDF to Pinecone
• POST /chat              – ask a question → {"answer", "chunks_used"}
• POST /feedback          – thumbs-up / thumbs-down  (+ chunk IDs)
• GET  /feedback/summary  – daily 👍 / 👎 counts
• GET  /feedback/chunks   – hall-of-shame per chunk
• static /                – tiny HTML/JS front-end
"""

from typing import List, Dict, Optional

from fastapi import (
    FastAPI, File, UploadFile, Form,
    HTTPException, Depends
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

# ─── local modules ────────────────────────────────────────────
from ingest_manual import ingest
from qa_demo       import chat                # returns {"answer", "chunks_used"}
from auth          import require_api_key     # header guard: X-API-Key
from db            import engine              # creates/opens Postgres table

# ─── FastAPI  &  CORS ─────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # tighten before prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ╭────────────────────────── 1)  PDF upload ─────────────────╮
@app.post("/upload", dependencies=[Depends(require_api_key)])
async def upload_pdf(
    file: UploadFile = File(...),
    customer: str    = Form("demo01")
):
    tmp = f"/tmp/{file.filename}"
    with open(tmp, "wb") as f:
        f.write(await file.read())
    ingest(tmp, customer)
    return {"status": "ingested", "file": file.filename}
# ╰────────────────────────────────────────────────────────────╯


# ╭────────────────────────── 2)  Chat ────────────────────────╮
@app.post("/chat", dependencies=[Depends(require_api_key)])
async def ask(
    question: str = Form(...),
    customer: str = Form("demo01")
):
    """
    Returns:
      {"answer": "...", "chunks_used": ["cust-id-123", …]}
    """
    return chat(question, customer)
# ╰────────────────────────────────────────────────────────────╯


# ╭────────────────────── 3)  Feedback thumbs ─────────────────╮
class FeedbackIn(BaseModel):
    customer    : str
    question    : str
    answer      : str
    score       : int                      # +1 👍  or  -1 👎
    chunks_used : Optional[List[str]] = None   # preferred (new UI)
    chunks      : Optional[List[str]] = None   # legacy alias

@app.post("/feedback", dependencies=[Depends(require_api_key)])
def add_feedback(data: FeedbackIn):
    if data.score not in (-1, 1):
        raise HTTPException(400, "score must be +1 or -1")

    chunk_ids = data.chunks_used or data.chunks or []

    with engine.begin() as conn:
        conn.execute(
            text("""
              INSERT INTO feedback
                (customer, question, answer, score, chunks_used)
              VALUES
                (:customer, :question, :answer, :score, :chunks)
            """),
            {
                "customer": data.customer,
                "question": data.question,
                "answer":   data.answer,
                "score":    data.score,
                "chunks":   chunk_ids,
            },
        )
    return {"ok": True}
# ╰────────────────────────────────────────────────────────────╯


# ╭────────────────────── 4)  Daily thumbs summary ────────────╮
@app.get("/feedback/summary")
def feedback_summary(days: int = 7) -> List[Dict]:
    days = int(days)                           # extra-safe cast
    sql  = f"""
      SELECT
        date_trunc('day', ts)::date AS day,
        SUM((score =  1)::int)      AS up,
        SUM((score = -1)::int)      AS down
      FROM feedback
      WHERE ts >= now() - INTERVAL '{days} days'
      GROUP BY day
      ORDER BY day;
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql))
        return [
            {"day": r.day.isoformat(), "up": int(r.up), "down": int(r.down)}
            for r in rows
        ]
# ╰────────────────────────────────────────────────────────────╯


# ╭────────────────────── 5)  Hall-of-shame chunks ────────────╮
@app.get("/feedback/chunks")
def worst_chunks(
    days: int      = 30,
    min_votes: int = 1,
    limit: int     = 50
) -> List[Dict]:
    days = int(days)
    sql  = f"""
      SELECT
        unnest(chunks_used)                        AS chunk_id,
        COUNT(*)                                   AS total,
        SUM((score =  1)::int)                     AS up,
        SUM((score = -1)::int)                     AS down,
        ROUND(
          100.0 * SUM((score =  1)::int)::numeric / NULLIF(COUNT(*),0),
          1
        )                                          AS up_pct
      FROM feedback
      WHERE ts >= now() - INTERVAL '{days} days'
      GROUP BY chunk_id
      HAVING COUNT(*) >= :min_votes
      ORDER BY up_pct ASC, total DESC
      LIMIT :limit;
    """
    with engine.begin() as conn:
        rows = conn.execute(
            text(sql), {"min_votes": min_votes, "limit": limit}
        )
        return [
            {
                "chunk_id": r.chunk_id,
                "total"   : int(r.total),
                "up"      : int(r.up),
                "down"    : int(r.down),
                "up_pct"  : float(r.up_pct) if r.up_pct is not None else None,
            }
            for r in rows
        ]
# ╰────────────────────────────────────────────────────────────╯


# ╭────────────────── 6)  Serve the small front-end ───────────╮
app.mount("/", StaticFiles(directory="web", html=True), name="web")
# ╰────────────────────────────────────────────────────────────╯
