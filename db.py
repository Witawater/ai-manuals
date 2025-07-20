#!/usr/bin/env python3
"""
db.py – minimal persistence layer for AI‑Manuals
───────────────────────────────────────────────
• Creates the `feedback` table if missing.
• Ensures the `chunks_used` **text[]** column exists.
• Adds two helpful indexes for fast look‑ups:
  – (customer, ts)           for /feedback/summary queries
  – GIN on chunks_used       for /feedback/chunks queries
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine, text

# ---------------------------------------------------------------------------
# 1. Database connection
# ---------------------------------------------------------------------------
DATABASE_URL: str | None = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("🟥 DATABASE_URL missing – set env var, e.g. postgres://user:pass@host/db")

# Slightly bigger pool, enable pre‑ping to avoid broken connections in serverless DBs.
engine = create_engine(
    DATABASE_URL,
    pool_size=10,
    max_overflow=2,
    pool_pre_ping=True,
)

# ---------------------------------------------------------------------------
# 2. Schema migrations (idempotent)
# ---------------------------------------------------------------------------
with engine.begin() as conn:
    # Core table -------------------------------------------------------------
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

    # Chunks column ----------------------------------------------------------
    conn.execute(
        text(
            """
            ALTER TABLE feedback
            ADD COLUMN IF NOT EXISTS chunks_used text[];
            """
        )
    )

    # Force correct type in case a previous JSONB column existed ------------
    conn.execute(
        text(
            """
            ALTER TABLE feedback
            ALTER COLUMN chunks_used TYPE text[] USING chunks_used::text[];
            """
        )
    )

    # Helpful indexes --------------------------------------------------------
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS feedback_customer_ts_idx
                ON feedback (customer, ts DESC);
            """
        )
    )
    conn.execute(
        text(
            """
            CREATE INDEX IF NOT EXISTS feedback_chunks_gin
                ON feedback USING GIN (chunks_used);
            """
        )
    )

__all__ = ["engine"]
