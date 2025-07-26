#!/usr/bin/env python3
"""
FastAPI service for AI-Manuals
──────────────────────────────
• POST /upload            – add a PDF to Pinecone (returns doc_id)
• GET  /ingest/status     – progress & ready flag while ingest runs
• POST /upload/metadata   – store user-supplied doc_type / notes
• POST /chat              – ask a question → {"answer", "chunks_used"}
• POST /feedback          – thumbs-up / thumbs-down (+ chunk IDs)
• GET  /feedback/summary  – daily 👍 / 👎 counts
• GET  /feedback/chunks   – hall-of-shame per chunk
• GET  /metrics           – 30-day recall / confidence JSON
• static /                – tiny HTML/JS front-end
"""

from __future__ import annotations

import os, tempfile, uuid, pathlib
from typing import Any, Dict, List, Optional

from fastapi import (
    BackgroundTasks, Depends, FastAPI, File, Form,
    HTTPException, UploadFile
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

# ─── local modules ───────────────────────────────────────────
from ingest_manual import ingest, pdf_to_chunks
from qa_demo      import chat
from auth         import require_api_key
from db           import engine

# ─── tunables ────────────────────────────────────────────────
CHUNK_TOKENS: int = int(os.getenv("CHUNK_TOKENS", "400"))
OVERLAP:      int = int(os.getenv("OVERLAP",      "80"))

LOG_PATH = pathlib.Path("/mnt/data/manual_eval.log")   # shared disk

# in-memory job table
JOBS: Dict[str, Dict[str, Any]] = {}

# ─── FastAPI & CORS ──────────────────────────────────────────
app = FastAPI()
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

# ╭──────────────────── 1) PDF upload & ingest ───────────────╮
def _ingest_and_cleanup(path: str, customer: str, doc_id: str) -> None:
    def _progress(done: int):
        if doc_id in JOBS:
            JOBS[doc_id]["done"] = done

    try:
        ingest(
            path,
            customer,
            CHUNK_TOKENS,
            OVERLAP,
            dry_run=False,
            progress_cb=_progress,
            common_meta=JOBS[doc_id].get("meta", {}),
        )
        JOBS[doc_id]["ready"] = True
        print(f"✅ Ingest complete: {path}")
    except Exception as exc:
        JOBS.setdefault(doc_id, {})["error"] = str(exc)
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
    customer: str    = Form("demo01"),
):
    doc_id = uuid.uuid4().hex
    print(f"📥 Upload received: {file.filename} from {customer} (doc_id={doc_id})")

    tmp_path = os.path.join(tempfile.gettempdir(), f"{doc_id}_{file.filename}")
    with open(tmp_path, "wb") as h:
        h.write(await file.read())

    try:
        chunks, _ = pdf_to_chunks(tmp_path, CHUNK_TOKENS, OVERLAP)
        total = len(chunks)
    except Exception:
        total = 0

    JOBS[doc_id] = {"ready": False, "total": total, "done": 0, "meta": {}}
    background_tasks.add_task(_ingest_and_cleanup, tmp_path, customer, doc_id)
    return {"doc_id": doc_id, "status": "queued", "file": file.filename}


@app.get("/ingest/status", dependencies=[Depends(require_api_key)])
def ingest_status(doc_id: str):
    job = JOBS.get(doc_id)
    if not job:
        raise HTTPException(404, "doc_id not found")
    return job


@app.post("/upload/metadata", dependencies=[Depends(require_api_key)])
def add_metadata(
    doc_id:   str = Form(...),
    doc_type: str = Form(...),
    notes:    str = Form(""),
):
    JOBS.setdefault(doc_id, {}).setdefault("meta", {})
    JOBS[doc_id]["meta"].update({"doc_type": doc_type, "notes": notes[:200]})
    return {"ok": True}
# ╰────────────────────────────────────────────────────────────╯

# ╭──────────────────── 2) Chat route ────────────────────────╮
@app.post("/chat", dependencies=[Depends(require_api_key)])
async def ask(
    question: str  = Form(...),
    customer: str  = Form("demo01"),
    doc_type: str  = Form(""),
):
    result = chat(question, customer, doc_type)
    if result.get("grounded") is False:
        print("⚠️  Fallback to GPT (not grounded)")
    return result
# ╰────────────────────────────────────────────────────────────╯

# ╭────────────────── 3) Feedback handlers ───────────────────╮
class FeedbackIn(BaseModel):
    customer: str
    question: str
    answer:   str
    score:    int
    chunks_used: Optional[List[str]] = None
    chunks:      Optional[List[str]] = None


@app.post("/feedback", dependencies=[Depends(require_api_key)])
def add_feedback(data: FeedbackIn):
    if data.score not in (-1, 1):
        raise HTTPException(400, "score must be +1 or -1")
    chunk_ids = data.chunks_used or data.chunks or []
    print(f"📝 Feedback: {'👍' if data.score==1 else '👎'} on {len(chunk_ids)} chunks")

    insert = text("""
        INSERT INTO feedback (customer, question, answer, score, chunks_used)
        VALUES (:customer, :question, :answer, :score, :chunks)
    """)
    with engine.begin() as conn:
        conn.execute(insert, {
            "customer": data.customer,
            "question": data.question,
            "answer":   data.answer,
            "score":    data.score,
            "chunks":   chunk_ids,
        })
    return {"ok": True}
# ╰────────────────────────────────────────────────────────────╯

# ╭────────────── 4) Feedback summaries & worst chunks ───────╮
@app.get("/feedback/summary")
def feedback_summary(days: int = 7) -> List[Dict]:
    sql = text(f"""
        SELECT date_trunc('day', ts)::date AS day,
               SUM((score=1)::int)  AS up,
               SUM((score=-1)::int) AS down
        FROM feedback
        WHERE ts >= now() - INTERVAL '{days} days'
        GROUP BY day ORDER BY day;
    """)
    with engine.begin() as conn:
        rows = conn.execute(sql)
        return [{"day": r.day.isoformat(), "up": int(r.up), "down": int(r.down)} for r in rows]


@app.get("/feedback/chunks")
def worst_chunks(days: int = 30, min_votes: int = 1, limit: int = 50) -> List[Dict]:
    sql = text("""
        SELECT unnest(chunks_used) AS chunk_id,
               COUNT(*)            AS total,
               SUM((score=1)::int) AS up,
               SUM((score=-1)::int)AS down,
               ROUND(100.0 * SUM((score=1)::int)::numeric / NULLIF(COUNT(*),0),1) AS up_pct
        FROM feedback
        WHERE ts >= now() - INTERVAL :days || ' days'
        GROUP BY chunk_id
        HAVING COUNT(*) >= :min_votes
        ORDER BY up_pct ASC, total DESC
        LIMIT :limit;
    """)
    with engine.begin() as conn:
        rows = conn.execute(sql, {"days": days, "min_votes": min_votes, "limit": limit})
        return [{
            "chunk_id": r.chunk_id,
            "total":    int(r.total),
            "up":       int(r.up),
            "down":     int(r.down),
            "up_pct":   float(r.up_pct) if r.up_pct is not None else None,
        } for r in rows]
# ╰────────────────────────────────────────────────────────────╯

# ╭────────────────── 5) Metrics route (QA scoreboard) ───────╮
def _tail(path: pathlib.Path, n: int = 2000) -> List[str]:
    """Return last *n* lines of a file (efficient tail)."""
    if not path.exists():
        return []
    with path.open("rb") as f:
        f.seek(0, os.SEEK_END)
        end = f.tell()
        size = 0
        block = []
        while len(block) <= n and end > 0:
            step = min(4096, end)
            end -= step
            f.seek(end)
            block[:0] = f.read(step).splitlines()
        return [b.decode("utf-8") for b in block[-n:]]

@app.get("/metrics", dependencies=[Depends(require_api_key)])
def metrics():
    rows = _tail(LOG_PATH, 1500)                      # ≈ last 2-3 months
    day_stats: Dict[str, Dict[str, List[float]]] = {}
    for ln in rows:
        try:
            ts, _tag, chunks, conf = ln.strip().split("\t")
        except ValueError:
            continue
        day = ts[:10]
        d = day_stats.setdefault(day, {"chunks": [], "conf": []})
        d["chunks"].append(int(chunks))
        d["conf"].append(float(conf))
    out = []
    for day in sorted(day_stats)[-30:]:               # last 30 days
        d = day_stats[day]
        out.append({
            "day": day,
            "avg_chunks": sum(d["chunks"]) / len(d["chunks"]),
            "avg_conf":   sum(d["conf"])   / len(d["conf"]),
        })
    return out
# ╰────────────────────────────────────────────────────────────╯

# ╭────────────────── 6) Serve static front-end ───────────────╮
app.mount("/", StaticFiles(directory="web", html=True), name="web")
# ╰────────────────────────────────────────────────────────────╯
