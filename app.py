from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# local modules
from ingest_manual import ingest
from qa_demo       import chat
from db            import engine            # creates feedback table

# extras for /feedback
from pydantic import BaseModel
from sqlalchemy import text

# ──────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 1) PDF upload & ingest ---------------------------------
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

# ---- 2) Chat ------------------------------------------------
@app.post("/chat")
async def ask(
    question: str = Form(...),
    customer: str = Form("demo01")
):
    resp = chat(question, customer)          # resp is {"answer":…, "chunks":[…]}
    return resp

# ---- 3) Feedback --------------------------------------------
class FeedbackIn(BaseModel):
    customer: str
    question: str
    answer:   str
    score:    int
    chunks:   list[int]  | None = None   # ← allow null for old UI

@app.post("/feedback")
def add_feedback(data: FeedbackIn):
    if data.score not in (-1, 1):
        raise HTTPException(400, "score must be +1 or -1")

    with engine.begin() as conn:
        conn.execute(
            text("""
              INSERT INTO feedback
                (customer, question, answer, score, chunks_used)
              VALUES
                (:customer, :question, :answer, :score, :chunks)
            """),
            data.model_dump()
        )
    return {"ok": True}

from datetime import date, timedelta
from typing import List, Dict

@app.get("/feedback/summary")
def feedback_summary(days: int = 7) -> List[Dict]:
    """
    Return daily counts of 👍 and 👎 for the last `days` days.
    Response:
      [
        {"day": "2025-07-01", "up": 3, "down": 1},
        ...
      ]
    """
    with engine.begin() as conn:
        rows = conn.execute(text(f"""
            SELECT
              date_trunc('day', ts)::date   AS day,
              SUM(CASE WHEN score =  1 THEN 1 ELSE 0 END) AS up,
              SUM(CASE WHEN score = -1 THEN 1 ELSE 0 END) AS down
            FROM feedback
            WHERE ts >= now() - INTERVAL '{days} days'
            GROUP BY day
            ORDER BY day;
        """))

    return [
        {"day": r.day.isoformat(), "up": int(r.up), "down": int(r.down)}
        for r in rows
    ]
# ---- 4) Serve the tiny front-end ----------------------------
app.mount("/", StaticFiles(directory="web", html=True), name="web")

# ── 5) low-score chunks (“hall-of-shame”) ─────────────────────────────
from typing import List, Dict
from sqlalchemy import text          # already imported earlier

@app.get("/feedback/chunks")
def worst_chunks(
        days: int = 30,              # look-back window
        min_votes: int = 3,          # ignore chunks with <3 votes
        limit: int = 50              # how many rows to return
    ) -> List[Dict]:
    """
    Returns a list like:
      [
        {"chunk_id":"demo01-a1b2c3d4", "total":7, "up":2,
         "down":5, "up_pct":28.6},
        ...
      ]
    """
    sql = f"""
      SELECT
        unnest(chunks_used) AS chunk_id,
        COUNT(*)            AS total,
        SUM(CASE WHEN score =  1 THEN 1 ELSE 0 END) AS up,
        SUM(CASE WHEN score = -1 THEN 1 ELSE 0 END) AS down,
        ROUND(100.0 *
              SUM(CASE WHEN score = 1 THEN 1 ELSE 0 END)::numeric
              / NULLIF(COUNT(*),0), 1)               AS up_pct
      FROM   feedback
      WHERE  ts >= now() - INTERVAL '{days} days'
      GROUP  BY chunk_id
      HAVING COUNT(*) >= :min_votes
      ORDER  BY up_pct ASC, total DESC
      LIMIT  :limit;
    """

    with engine.begin() as conn:
        rows = conn.execute(text(sql),
                            {"min_votes": min_votes, "limit": limit})
        return [dict(r) for r in rows]
