from sqlalchemy import text
from db import engine

DOC_ID = "15f371a9f5e540e890a62b2936c31dbb"

with engine.begin() as conn:
    rows = conn.execute(
        text("SELECT * FROM manual_outline WHERE doc_id = :doc"),
        {"doc": DOC_ID}
    ).fetchall()

for row in rows:
    print(row)
