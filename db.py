"""
Very tiny DB helper.
Importing this module anywhere guarantees the `feedback` table exists.
"""

from sqlalchemy import create_engine, text
import os

DB_URL = os.getenv("DATABASE_URL")          # comes from .env or Render
if not DB_URL:
    raise RuntimeError("DATABASE_URL env var is missing")

# Small connection pool → fine for our scale
engine = create_engine(DB_URL, pool_size=5, max_overflow=0)

# ── one-time bootstrap executed on first import ──────────────────────────
with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS feedback (
            id       SERIAL PRIMARY KEY,
            ts       TIMESTAMPTZ DEFAULT now(),
            customer TEXT,
            question TEXT,
            answer   TEXT,
            score    SMALLINT          -- +1 or -1
        );
    """))
