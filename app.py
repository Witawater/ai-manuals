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
    UploadFile, HTTPException
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy import text
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

# ─── 1. Upload PDF (but do not ingest yet) ──────────────────
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
    return {"records": lines[-30:]}  # Last 30

# ─── Frontend ───────────────────────────────────────────────
app.mount("/", StaticFiles(directory="web", html=True), name="web")
