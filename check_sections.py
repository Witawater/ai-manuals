#!/usr/bin/env python3
"""
Quick check for section summaries for a given doc_id.
"""

from sqlalchemy import text
from db import engine

DOC_ID = "15f371a9f5e540e890a62b2936c31dbb"

with engine.begin() as conn:
    result = conn.execute(text("""
        SELECT COUNT(*) FROM manual_sections
        WHERE doc_id = :doc
    """), {"doc": DOC_ID}).scalar()

print(f"✅ Found {result} section summaries for doc_id {DOC_ID}")
