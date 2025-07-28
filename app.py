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

import os, tempfile, uuid, pathlib, hashlib
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

# in-memory ingest tracker
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
    # ── stream-save + hash in one pass ───────────────────────
    tmp_path   = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_{file.filename}")
    sha        = hashlib.sha256()
    size       = 0
    with open(tmp_path, "wb") as h:
        while chunk := await file.read(8192):
            h.write(chunk)
            sha.update(chunk)
            size += len(chunk)
    file_hash = sha.hexdigest()

    # ── duplicate-PDF guard ──────────────────────────────────
    with engine.begin() as conn:
        dup = conn.execute(
            text("SELECT doc_id FROM manual_files WHERE customer=:c AND sha256=:h"),
            {"c": customer, "h": file_hash},
        ).fetchone()

    if dup:
        print("🔁 Duplicate PDF – skipping ingest")
        return {"doc_id": dup.doc_id, "status": "duplicate", "file": file.filename}

    # ── create new ingest job ────────────────────────────────
    doc_id = uuid.uuid4().hex
    print(f"📥 New upload: {file.filename} from {customer} (doc_id={doc_id})")

    try:
        chunks, _ = pdf_to_chunks(tmp_path, CHUNK_TOKENS, OVERLAP)
        total = len(chunks)
    except Exception:
        total = 0

    JOBS[doc_id] = {"ready": False, "total": total, "done": 0, "meta": {}}
    background_tasks.add_task(_ingest_and_cleanup, tmp_path, customer, doc_id)

    # record in manual_files
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO manual_files(sha256, customer, doc_id, filename, bytes)
                    VALUES (:h,:c,:d,:f,:b)"""),
            {"h": file_hash, "c": customer, "d": doc_id,
             "f": file.filename, "b": size},
        )

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

# (rest of file unchanged: feedback handlers, metrics, static mount)
# ╭────────────────── 6) Serve static front-end ───────────────╮
app.mount("/", StaticFiles(directory="web", html=True), name="web")
# ╰────────────────────────────────────────────────────────────╯
