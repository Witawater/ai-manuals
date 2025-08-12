# list_docs.py (fixed)
from sqlalchemy import text
from db import engine

with engine.begin() as conn:
    rows = conn.execute(text("""
        SELECT doc_id, filename, customer, created_at
        FROM manual_files
        ORDER BY created_at DESC
        LIMIT 10
    """)).fetchall()

for r in rows:
    print(f"{r.doc_id} | {r.customer} | {r.filename} | {r.created_at}")
