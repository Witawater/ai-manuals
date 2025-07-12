import os
from sqlalchemy import create_engine, text

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("🟥 DATABASE_URL missing")

engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=0)

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS feedback (
            id          SERIAL PRIMARY KEY,
            ts          TIMESTAMPTZ DEFAULT now(),
            customer    TEXT,
            question    TEXT,
            answer      TEXT,
            score       SMALLINT CHECK (score IN (-1, 1))
        );
    """))

    conn.execute(text("""
        ALTER TABLE feedback
        ADD COLUMN IF NOT EXISTS chunks_used  text[];
    """))

    conn.execute(text("""
        ALTER TABLE feedback
        ALTER COLUMN chunks_used
        TYPE text[]
        USING chunks_used::text[];
    """))
