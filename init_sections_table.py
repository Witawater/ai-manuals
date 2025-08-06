#!/usr/bin/env python3
# init_sections_table.py

from sqlalchemy import text
from db import engine

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS manual_sections (
            id          TEXT PRIMARY KEY,
            doc_id      TEXT NOT NULL,
            outline_id  TEXT NOT NULL,
            summary     TEXT,
            created_at  TIMESTAMPTZ DEFAULT now()
        );
    """))
    print("✅ manual_sections table created.")
