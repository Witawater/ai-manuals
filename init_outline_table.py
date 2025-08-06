from sqlalchemy import text
from db import engine

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS manual_outline (
            id         TEXT PRIMARY KEY,
            doc_id     TEXT NOT NULL,
            title      TEXT NOT NULL,
            sort_order INT
        );
    """))
    print("✅ manual_outline table created or patched.")