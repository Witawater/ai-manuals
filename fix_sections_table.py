from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv

load_dotenv(".env")

engine = create_engine(os.getenv("DATABASE_URL"))

with engine.begin() as conn:
    conn.execute(text("""
        ALTER TABLE manual_sections
        ADD COLUMN IF NOT EXISTS outline_id TEXT;
    """))
    print("✅ outline_id column added.")
