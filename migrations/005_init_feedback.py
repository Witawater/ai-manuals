from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv(".env")
engine = create_engine(os.environ["DATABASE_URL"], echo=True)

with engine.begin() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS feedback (
            id        SERIAL PRIMARY KEY,
            doc_id    TEXT,
            question  TEXT,
            answer    TEXT,
            chunks    TEXT,
            good      BOOLEAN,
            created   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """))
print("✅ feedback table created")
