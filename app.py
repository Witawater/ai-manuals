#!/usr/bin/env python3
"""
FastAPI service for AI-Manuals
(delayed ingest: wait for metadata before embedding)
"""

from __future__ import annotations
import os, tempfile, uuid, pathlib, hashlib
from typing import Any, Dict, Optional, Iterable

from fastapi import (
    BackgroundTasks, Depends, FastAPI, File, Form,
    UploadFile, HTTPException, Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError

# ─── local modules ──────────────────────────────────────────
from ingest_manual import ingest, pdf_to_chunks
from qa_demo       import chat
from auth          import require_api_key
from db            import engine

# Optional: single-section regenerate helper (if present)
try:
    # expected signature: regenerate_section_summary(section_id: str, customer: str) -> None
    from extract_sections import regenerate_section_summary  # type: ignore
except Exception:
    regenerate_section_summary = None  # fallback path will be used

# ─── config ────────────────────────────────────────────────
CHUNK_TOKENS = int(os.getenv("CHUNK_TOKENS", "400"))
OVERLAP      = int(os.getenv("OVERLAP",      "80"))
INDEX_NAME   = os.getenv("PINECONE_INDEX",   "manuals-large")

LOG_PATH = pathlib.Path("/mnt/data/manual_eval.log")
JOBS: Dict[str, Dict[str, Any]] = {}

# ─── FastAPI setup ──────────────────────────────────────────
app = FastAPI()
_allow_origins = os.getenv(
    "CORS_ALLOW_ORIGINS",
    "https://ai-manuals.onrender.com,http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _allow_origins if o.strip()],
    allow_methods=["*"],
    allow_headers=["Content-Type", "X-API-Key"],
    allow_credentials=True,  # needed for cookies/credentials
)

# ─── helpers ────────────────────────────────────────────────
def _chunks_to_csv(val: Any) -> str:
    """Normalize `chunks` to a comma-separated string (DB stores TEXT)."""
    if val is None:
        return ""
    if isinstance(val, str):
        return val[:2000]
    if isinstance(val, Iterable):
        try:
            s = ",".join(str(x) for x in val)
            return s[:2000]
        except Exception:
            return ""
    return str(val)[:2000]

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

    # Ensure your DB has manual_files.index_name
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
    # clear any stale error from prior runs
    JOBS[doc_id].pop("error", None)

    background_tasks.add_task(
        _ingest_and_cleanup,
        path=job["path"],
        customer=job["customer"],
        doc_id=doc_id
    )

    return {"ok": True}

# ─── 3. Ingestion worker ─────────────────────────────────────
def _ingest_and_cleanup(path: str, customer: str, doc_id: str) -> None:
    def _progress(done: int):
        if doc_id in JOBS:
            JOBS[doc_id]["done"] = done
    try:
        # clear any previous error before starting
        JOBS.get(doc_id, {}).pop("error", None)

        ingest(path, customer, CHUNK_TOKENS, OVERLAP,
               dry_run=False, progress_cb=_progress,
               common_meta=JOBS[doc_id].get("meta", {}),
               doc_id=doc_id)

        # mark success; ensure done == total
        JOBS[doc_id]["ready"] = True
        JOBS[doc_id]["done"] = JOBS[doc_id].get("total", JOBS[doc_id].get("done", 0))
        # on success, ensure no error remains
        JOBS[doc_id].pop("error", None)
        print("✅ ingest complete", path)
    except Exception as exc:
        JOBS.setdefault(doc_id, {})["error"] = str(exc)
        print("🛑 ingest failed", exc)
    finally:
        # remove tmp file and hide path from status
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        JOBS.get(doc_id, {}).pop("path", None)

# ─── 4. Ingest status ───────────────────────────────────────
@app.get("/ingest/status")
def ingest_status(doc_id: str, customer: str = Depends(require_api_key)):
    job = JOBS.get(doc_id)
    if job:
        # don’t expose path; drop error once ready
        resp = {k: v for k, v in job.items() if k != "path"}
        if resp.get("ready"):
            resp.pop("error", None)
        return resp

    # Fallback: after restart, check DB to confirm the doc exists for this customer.
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT 1 FROM manual_files
            WHERE doc_id = :d AND customer = :c
            LIMIT 1
        """), {"d": doc_id, "c": customer}).fetchone()

    if row:
        # We don't have live progress after a restart; report ready so UI can proceed.
        return {"ready": True, "total": 0, "done": 0, "meta": {}, "customer": customer}

    raise HTTPException(404, "doc_id not found")

# ─── 5. Ask question ─────────────────────────────────────────
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

# ─── 6. QA metrics file (/metrics) ──────────────────────────
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

# ─── 7. Feedback route ──────────────────────────────────────
@app.post("/feedback")
async def log_feedback(
    request: Request,
    customer: str = Depends(require_api_key)
):
    # Accept JSON or form payloads
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        payload = await request.json()
    else:
        payload = await request.form()

    def _get(key: str, default=""):
        val = payload.get(key, default)
        # starlette FormData returns lists sometimes; normalize
        if isinstance(val, (list, tuple)):
            val = val[0] if val else default
        return val

    good_raw = payload.get("good", False)
    good_val = (
        bool(good_raw) if isinstance(good_raw, bool)
        else str(good_raw).lower() in ("true", "1", "yes", "y")
    )

    chunks_val = payload.get("chunks", "")
    chunks_csv = _chunks_to_csv(chunks_val)

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO feedback (doc_id, question, answer, chunks, good, customer)
                VALUES (:d, :q, :a, :c, :g, :cust)
            """),
            {
                "d": _get("doc_id") or "",
                "q": (_get("question") or "")[:500],
                "a": (_get("answer") or "")[:2000],
                "c": chunks_csv,
                "g": good_val,
                "cust": customer,
            }
        )
    return {"ok": True}

# ─── 8. Feedback summary route ──────────────────────────────
@app.get("/feedback/summary")
def feedback_summary(
    customer: str = Depends(require_api_key),
    doc_id: str = "",
    days: int = 7,
):
    where_parts = ["customer = :c", "created >= NOW() - make_interval(days => :days)"]
    params = {"c": customer, "days": int(days)}
    if doc_id:
        where_parts.append("doc_id = :d")
        params["d"] = doc_id
    where_clause = " WHERE " + " AND ".join(where_parts)

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

# ─── 9. Chunk quality voting data ───────────────────────────
@app.get("/feedback/chunks")
def chunk_quality(
    days: int = 30,
    min_votes: int = 1,
    limit: int = 30,
    doc_id: str = "",
    customer: str = Depends(require_api_key)
):
    with engine.begin() as conn:
        where_parts = ["f.customer = :c", "f.created >= NOW() - make_interval(days => :days)"]
        params = {"c": customer, "days": int(days)}
        if doc_id:
            where_parts.append("f.doc_id = :doc")
            params["doc"] = doc_id
        where_clause = " WHERE " + " AND ".join(where_parts)

        rows = conn.execute(text(f"""
            SELECT
              chunk_id,
              SUM(CASE WHEN good THEN 1 ELSE 0 END) AS up,
              SUM(CASE WHEN NOT good THEN 1 ELSE 0 END) AS down,
              COUNT(*) AS total
            FROM (
              SELECT unnest(string_to_array(chunks, ',')) AS chunk_id, good
              FROM feedback f
              {where_clause}
            ) AS sub
            GROUP BY chunk_id
            HAVING COUNT(*) >= :min
            ORDER BY total DESC
            LIMIT :lim
        """), {**params, "min": int(min_votes), "lim": int(limit)}).fetchall()

    return [{
        "chunk_id": r.chunk_id,
        "up": r.up,
        "down": r.down,
        "total": r.total,
        "up_pct": round(100 * r.up / r.total) if r.total else 0
    } for r in rows]

# ─── 10. Section summaries API (must come before app.mount) ──
@app.get("/manual/{doc_id}/sections")
def get_section_summaries(
    doc_id: str,
    customer: str = Depends(require_api_key)
):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT s.id AS section_id, o.title AS heading, s.summary
            FROM manual_sections s
            JOIN manual_outline o ON s.outline_id = o.id
            WHERE s.doc_id = :doc AND o.customer = :cust
            ORDER BY o.sort_order
        """), {"doc": doc_id, "cust": customer}).fetchall()

    return {"sections": [
        {"section_id": r.section_id, "heading": r.heading, "summary": r.summary} for r in rows
    ]}

# ─── 10b. Outline API (new) ─────────────────────────────────
@app.get("/manual/{doc_id}/outline")
def get_outline(
    doc_id: str,
    customer: str = Depends(require_api_key)
):
    with engine.begin() as conn:
        rows = conn.execute(text("""
            SELECT id, title, sort_order
            FROM manual_outline
            WHERE doc_id = :doc AND customer = :cust
            ORDER BY sort_order
        """), {"doc": doc_id, "cust": customer}).fetchall()

    return {"outline": [
        {"id": r.id, "title": r.title, "sort_order": r.sort_order} for r in rows
    ]}

# ─── 10c. Flag / Unflag a section (new) ─────────────────────
@app.post("/manual/{doc_id}/sections/{section_id}/flag")
async def flag_section(
    doc_id: str,
    section_id: str,
    action: str = Form("flag"),                 # "flag" | "unflag"
    reason: str = Form(""),
    customer: str = Depends(require_api_key)
):
    action = action.lower().strip()
    if action not in ("flag", "unflag"):
        raise HTTPException(400, "action must be 'flag' or 'unflag'")

    with engine.begin() as conn:
        # Ensure section belongs to this doc+customer
        owned = conn.execute(text("""
            SELECT 1
            FROM manual_sections s
            JOIN manual_outline o ON s.outline_id = o.id
            WHERE s.id = :sid AND s.doc_id = :doc AND o.customer = :cust
            LIMIT 1
        """), {"sid": section_id, "doc": doc_id, "cust": customer}).fetchone()
        if not owned:
            raise HTTPException(404, "section not found for this document/customer")

        if action == "flag":
            conn.execute(text("""
                INSERT INTO manual_section_flags (section_id, doc_id, customer, reason)
                VALUES (:sid, :doc, :cust, :r)
                ON CONFLICT (section_id, customer)
                DO UPDATE SET reason = EXCLUDED.reason, updated = NOW()
            """), {"sid": section_id, "doc": doc_id, "cust": customer, "r": reason[:500]})
            conn.execute(text("""
                UPDATE manual_sections SET needs_regen = TRUE WHERE id = :sid
            """), {"sid": section_id})
            return {"ok": True, "status": "flagged"}
        else:
            conn.execute(text("""
                DELETE FROM manual_section_flags WHERE section_id = :sid AND customer = :cust
            """), {"sid": section_id, "cust": customer})
            conn.execute(text("""
                UPDATE manual_sections SET needs_regen = COALESCE(needs_regen, FALSE) AND FALSE WHERE id = :sid
            """), {"sid": section_id})
            return {"ok": True, "status": "unflagged"}

# ─── 10d. Regenerate a section summary (new) ────────────────
@app.post("/manual/{doc_id}/sections/{section_id}/regenerate")
async def regenerate_section(
    background_tasks: BackgroundTasks,
    doc_id: str,
    section_id: str,
    customer: str = Depends(require_api_key)
):
    # Verify access/ownership
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT s.id
            FROM manual_sections s
            JOIN manual_outline o ON s.outline_id = o.id
            WHERE s.id = :sid AND s.doc_id = :doc AND o.customer = :cust
            LIMIT 1
        """), {"sid": section_id, "doc": doc_id, "cust": customer}).fetchone()
        if not row:
            raise HTTPException(404, "section not found for this document/customer")

    def _do_regen(section_id: str, customer: str) -> None:
        try:
            if callable(regenerate_section_summary):
                regenerate_section_summary(section_id, customer)
                print(f"♻️ regenerated summary for section {section_id}")
            else:
                with engine.begin() as conn:
                    conn.execute(text("""
                        UPDATE manual_sections
                        SET needs_regen = TRUE, summary = NULL
                        WHERE id = :sid
                    """), {"sid": section_id})
                print(f"⚠️ regenerate helper not available; marked section {section_id} for batch regen")
        except Exception as e:
            print("🛑 regenerate failed:", e)
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE manual_sections
                    SET needs_regen = TRUE
                    WHERE id = :sid
                """), {"sid": section_id})

    background_tasks.add_task(_do_regen, section_id, customer)
    return {"ok": True, "status": "queued"}

# ─── 11. Serve frontend (keep at bottom) ─────────────────────
app.mount("/", StaticFiles(directory="web", html=True), name="web")
