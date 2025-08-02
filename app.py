#!/usr/bin/env python3
"""
FastAPI service for AI-Manuals
(delayed ingest: wait for metadata before embedding)
"""

from __future__ import annotations
import os, tempfile, uuid, pathlib, hashlib
from typing import Any, Dict

from fastapi import (
    BackgroundTasks, Depends, FastAPI, File, Form,
    UploadFile, HTTPException, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text, insert
from sqlalchemy.exc import IntegrityError

# ─── local modules ──────────────────────────────────────────
from ingest_manual import ingest, pdf_to_chunks
from qa_demo       import chat
from auth          import require_api_key
from db            import engine

# ─── config ────────────────────────────────────────────────
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "400"))
OVERLAP      = int(os.getenv("OVERLAP",      "80"))
INDEX_NAME   = os.getenv("PINECONE_INDEX",   "manuals-large")

LOG_PATH = pathlib.Path("/mnt/data/manual_eval.log")
JOBS: Dict[str, Dict[str, Any]] = {}

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "CORS_ALLOW_ORIGINS",
        "https://ai-manuals.onrender.com,http://localhost:5173,http://127.0.0.1:5173",
    ).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Ingestion worker ───────────────────────────────────────
def _ingest_and_cleanup(path: str, customer: str, doc_id: str) -> None:
    def _progress(done: int):
        if doc_id in JOBS:
            JOBS[doc_id]["done"] = done
    try:
        ingest(path, customer, CHUNK_TOKENS, OVERLAP,
               dry_run=False, progress_cb=_progress,
               common_meta=JOBS[doc_id].get("meta", {}),
               doc_id=doc_id)
        JOBS[doc_id]["ready"] = True
        print("✅ ingest complete", path)
    except Exception as exc:
        JOBS.setdefault(doc_id, {})["error"] = str(exc)
        print("🛑 ingest failed", exc)
    finally:
        try: os.remove(path)
        except FileNotFoundError: pass

# ─── 1. Upload PDF ──────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    customer: str = Depends(require_api_key)
):
    tmp = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_{file.filename}")
    sha, size = hashlib.sha256(), 0
    with open(tmp, "wb") as h:
        while chunk := await file.read(8192):
            h.write(chunk)
            sha.update(chunk)
            size += len(chunk)
    file_hash = sha.hexdigest()

    with engine.begin() as conn:
        row = conn.execute(
            text("""SELECT doc_id FROM manual_files
                    WHERE customer=:c AND sha256=:h AND index_name=:i"""),
            {"c": customer, "h": file_hash, "i": INDEX_NAME},
        ).fetchone()

    if row:
        print("🔁 duplicate PDF (same index) – skipping ingest")
        return {"doc_id": row.doc_id, "status": "duplicate", "file": file.filename}

    doc_id = uuid.uuid4().hex
    print(f"📥 upload {file.filename}  ➜  {doc_id}")

    try:
        total = len(pdf_to_chunks(tmp, CHUNK_TOKENS, OVERLAP)[0])
    except Exception:
        total = 0

    JOBS[doc_id] = {
        "ready": False,
        "total": total,
        "done": 0,
        "meta": {},
        "path": tmp,
        "customer": customer
    }

    with engine.begin() as conn:
        try:
            conn.execute(
                text("""INSERT INTO manual_files
                        (sha256, customer, doc_id, filename, bytes, index_name)
                        VALUES (:h,:c,:d,:f,:b,:i)"""),
                {"h": file_hash, "c": customer, "d": doc_id,
                 "f": file.filename, "b": size, "i": INDEX_NAME},
            )
        except IntegrityError:
            print("🔁 duplicate PDF (insert blocked by constraint)")
            return JSONResponse(
                content={"doc_id": row.doc_id if row else "unknown",
                         "status": "duplicate", "file": file.filename},
                status_code=200,
            )

    return {"doc_id": doc_id, "status": "queued", "file": file.filename}

# ─── 2. Save metadata & trigger ingest ──────────────────────
@app.post("/upload/metadata")
def save_meta(
    background_tasks: BackgroundTasks,
    doc_id: str = Form(...),
    doc_type: str = Form(...),
    notes: str = Form(""),
    customer: str = Depends(require_api_key)
):
    job = JOBS.get(doc_id)
    if not job or "path" not in job:
        raise HTTPException(404, "Upload not found or missing file path")

    JOBS[doc_id].setdefault("meta", {}).update(
        {"doc_type": doc_type, "notes": notes[:200]}
    )

    background_tasks.add_task(
        _ingest_and_cleanup,
        path=job["path"],
        customer=job["customer"],
        doc_id=doc_id
    )

    return {"ok": True}

# ─── 3. Ingest status ───────────────────────────────────────
@app.get("/ingest/status")
def ingest_status(doc_id: str, customer: str = Depends(require_api_key)):
    job = JOBS.get(doc_id) or HTTPException(404, "doc_id not found")
    return job

# ─── 4. Chat route ──────────────────────────────────────────
@app.post("/chat")
async def ask(
    question: str = Form(...),
    doc_type: str = Form(""),
    doc_id: str = Form(""),
    customer: str = Depends(require_api_key)
):
    res = chat(question, customer, doc_type=doc_type, doc_id=doc_id)
    if not res.get("grounded"):
        print("⚠️ fallback (ungrounded)")
    return res

# ─── 5. QA Metrics ──────────────────────────────────────────
@app.get("/metrics")
def get_metrics():
    lines = []
    try:
        with open(LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) == 4:
                    timestamp, tag, count, score = parts
                    lines.append({
                        "timestamp": timestamp,
                        "tag": tag,
                        "count": int(count),
                        "avg_conf": float(score),
                    })
    except FileNotFoundError:
        return {"error": "log file not found"}
    return {"records": lines[-30:]}

# ─── 6. Feedback route (new version) ───────────────────────
@app.post("/feedback")
async def log_feedback(
    request: Request,
    customer: str = Depends(require_api_key)
):
    payload = await request.form()

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO feedback (doc_id, question, answer, chunks, good)
                VALUES (:d, :q, :a, :c, :g)
            """),
            {
                "d": payload.get("doc_id"),
                "q": payload.get("question", "")[:500],
                "a": payload.get("answer", "")[:2000],
                "c": payload.get("chunks", "")[:2000],
                "g": payload.get("good") == "true"
            }
        )
    return {"ok": True}

# ─── 7. Feedback summary ────────────────────────────────────
@app.get("/feedback/summary")
def feedback_summary(
    customer: str = Depends(require_api_key),
    doc_id: str = ""
):
    where_clause = "WHERE customer = :c"
    params = {"c": customer}

    if doc_id:
        where_clause += " AND doc_id = :d"
        params["d"] = doc_id

    with engine.begin() as conn:
        rows = conn.execute(text(f"""
            SELECT
              to_char(created, 'YYYY-MM-DD') AS day,
              COUNT(*) AS total,
              SUM(CASE WHEN good THEN 1 ELSE 0 END) AS good,
              SUM(CASE WHEN NOT good THEN 1 ELSE 0 END) AS bad
            FROM feedback
            {where_clause}
            GROUP BY day
            ORDER BY day DESC
            LIMIT 30;
        """), params).fetchall()

    return {"records": [
        {"day": r.day, "total": r.total, "good": r.good, "bad": r.bad}
        for r in rows
    ]}


# ─── Frontend ───────────────────────────────────────────────
app.mount("/", StaticFiles(directory="web", html=True), name="web")
