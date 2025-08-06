#!/usr/bin/env python3
"""
db.py – minimal persistence layer for AI-Manuals
───────────────────────────────────────────────
Creates/updates:

• feedback            – thumbs-up / thumbs-down (+ chunk IDs)
• manual_files        – one row per unique PDF (sha256 dedupe guard)

All DDL is idempotent so the file can run on every cold-start.
"""

from __future__ import annotations

import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()  # ✅ Ensure DATABASE_URL loads from .env even in CLI runs

# ───────────────────────── 1. DB connection ──────────────────────────
DATABASE_URL: str | None = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "🟥 DATABASE_URL missing – set env var, e.g. postgres://user:pass@host/db"
    )

engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=2,
    pool_pre_ping=True,  # keep-alive for serverless DBs
)

# ───────────────────────── 2. Schema migrations ──────────────────────
with engine.begin() as conn:
    # ── feedback ------------------------------------------------------
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id          SERIAL PRIMARY KEY,
                ts          TIMESTAMPTZ DEFAULT now(),
                customer    TEXT,
                question    TEXT,
                answer      TEXT,
                score       SMALLINT CHECK (score IN (-1, 1))
            );
            """
        )
    )
    conn.execute(
        text(
            """ALTER TABLE feedback
               ADD COLUMN IF NOT EXISTS chunks_used text[];"""
        )
    )
    conn.execute(
        text(
            """ALTER TABLE feedback
               ALTER COLUMN chunks_used
               TYPE text[] USING chunks_used::text[];"""
        )
    )
    conn.execute(
        text(
            """CREATE INDEX IF NOT EXISTS feedback_customer_ts_idx
                 ON feedback (customer, ts DESC);"""
        )
    )
    conn.execute(
        text(
            """CREATE INDEX IF NOT EXISTS feedback_chunks_gin
                 ON feedback USING GIN (chunks_used);"""
        )
    )

    # ── manual_files  (duplicate-PDF guard) ---------------------------
    conn.execute(
        text(
            """
            CREATE TABLE IF NOT EXISTS manual_files (
                sha256      CHAR(64) PRIMARY KEY,
                customer    TEXT NOT NULL,
                doc_id      TEXT NOT NULL,
                filename    TEXT,
                pages       INT,
                bytes       BIGINT,
                created_at  TIMESTAMPTZ DEFAULT now()
            );
            """
        )
    )
    conn.execute(
        text(
            """CREATE UNIQUE INDEX IF NOT EXISTS manual_files_cust_sha
                 ON manual_files (customer, sha256);"""
        )
    )

__all__ = ["engine"]
