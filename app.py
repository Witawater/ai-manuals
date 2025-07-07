#!/usr/bin/env python3
"""
FastAPI service for AI-Manuals
──────────────────────────────
• /upload    – add a PDF to Pinecone
• /chat      – ask a question  → {answer, chunks_used}
• /feedback  – thumbs-up / thumbs-down + chunk IDs
• /feedback/summary  – daily stats
• /feedback/chunks   – hall-of-shame per chunk
• static /           – tiny HTML front-end
"""

from datetime import date, timedelta
from typing import List, Dict, Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

# ── local modules ─────────────────────────────────────────────────────────
from ingest_manual import ingest
from qa_demo       import chat          # returns {"answer":…, "chunks_used":[…]}
from db            import engine        # ensures feedback table exists

# ── app & CORS ────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────── 1)  PDF upload  ───────────────
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    customer: str   = Form("demo01")
):
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    ingest(path, customer)
    return {"status": "ingested", "file": file.filename}

# ─────────────── 2)  Chat  ───────────────
@app.post("/chat")
async def ask(
    question: str = Form(...),
    customer: str = Form("demo01")
):
    return chat(question, customer)     # ⇢ {"answer":…, "chunks_used":[…]}

# ─────────────── 3)  Feedback thumbs  ───────────────
class FeedbackIn(BaseModel):
    customer: str
    question: str
    answer:   str
    score:    int
    # accept *either* field for back-compat
    chunks_used: Optional[List[str]] = None
    chunks:      Optional[List[str]] = None

@app.post("/feedback")
def add_feedback(data: FeedbackIn):
    if data.score not in (-1, 1):
        raise HTTPException(400, "score must be +1 or -1")

    # pick whichever list was sent
    chunks = data.chunks_used or data.chunks or []

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
                "chunks":   chunks
            }
        )
    return {"ok": True}

# ─────────────── 4)  Daily thumbs summary  ───────────────
@app.get("/feedback/summary")
def feedback_summary(days: int = 7) -> List[Dict]:
    sql = f"""
      SELECT
        date_trunc('day', ts)::date               AS day,
        SUM((score =  1)::int)                   AS up,
        SUM((score = -1)::int)                   AS down
      FROM feedback
      WHERE ts >= now() - INTERVAL '{days} days'
      GROUP BY day
      ORDER BY day;
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql))
        return [dict(day=r.day.isoformat(), up=int(r.up), down=int(r.down))
                for r in rows]

# ─────────────── 5)  Hall-of-shame chunks  ───────────────
@app.get("/feedback/chunks")
def worst_chunks(
    days: int      = 30,
    min_votes: int = 3,
    limit: int     = 50
) -> List[Dict]:
    sql = f"""
      SELECT
        unnest(chunks_used)                        AS chunk_id,
        COUNT(*)                                   AS total,
        SUM((score =  1)::int)                     AS up,
        SUM((score = -1)::int)                     AS down,
        ROUND(100.0 * SUM((score = 1)::int)::numeric
                     / NULLIF(COUNT(*),0), 1)      AS up_pct
      FROM feedback
      WHERE ts >= now() - INTERVAL '{days} days'
      GROUP BY chunk_id
      HAVING COUNT(*) >= :min_votes
      ORDER BY up_pct ASC, total DESC
      LIMIT :limit;
    """
    with engine.begin() as conn:
        rows = conn.execute(text(sql),
                            {"min_votes": min_votes, "limit": limit})
        return [dict(r) for r in rows]

# ─────────────── 6)  Serve front-end  ───────────────
app.mount("/", StaticFiles(directory="web", html=True), name="web")
