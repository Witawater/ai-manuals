"""
Very tiny DB helper
───────────────────
Importing *any* module that does:

    from db import engine

guarantees the `feedback` table (with the **chunks_used text[]** column)
exists.  All statements are idempotent, so repeated deploys are safe.
"""

import os
from sqlalchemy import create_engine, text

# ── connection string ──────────────────────────────────────────────────
DB_URL = os.getenv("DATABASE_URL")          # .env locally / Render in prod
if not DB_URL:
    raise RuntimeError("DATABASE_URL env var is missing")

# Small pool is fine for our scale
engine = create_engine(DB_URL, pool_size=5, max_overflow=0)

# ── bootstrap: run once on first import ─────────────────────────────────
with engine.begin() as conn:
    # 1) base table
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          SERIAL PRIMARY KEY,
            ts          TIMESTAMPTZ DEFAULT now(),
            customer    TEXT,
            question    TEXT,
            answer      TEXT,
            score       SMALLINT              -- +1 or -1
        );
    """))

    # 2) ensure chunks_used column exists
    conn.execute(text("""
        ALTER TABLE feedback
        ADD COLUMN IF NOT EXISTS chunks_used  text[];
    """))

    # 3) …and is typed as text[] even if an old deploy made it integer[]
    conn.execute(text("""
        ALTER TABLE feedback
        ALTER COLUMN chunks_used
        TYPE text[]
        USING chunks_used::text[];
    """))