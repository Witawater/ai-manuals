#!/usr/bin/env python3

from sqlalchemy import text
from db import engine

with engine.begin() as conn:
    # 1. Add the column if it doesn't exist
    print("🔧 Adding customer column to manual_outline …")
    conn.execute(text("""
        ALTER TABLE manual_outline
        ADD COLUMN IF NOT EXISTS customer TEXT DEFAULT '';
    """))

    # 2. Backfill the customer from manual_files (based on doc_id)
    print("📦 Backfilling customer values …")
    conn.execute(text("""
        UPDATE manual_outline o
        SET customer = f.customer
        FROM manual_files f
        WHERE o.doc_id = f.doc_id
    """))

print("✅ Patch complete.")
