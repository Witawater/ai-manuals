# insert_api_key.py
from sqlalchemy import text
from db import engine

with engine.begin() as conn:
    conn.execute(text("""
        INSERT INTO api_keys (key, customer, quota, used)
        VALUES (:k, :c, :q, :u)
    """), {
        "k": "d4d6aa9f2bf75faf76454d8621af7c01",
        "c": "demo01",
        "q": 1000,
        "u": 0
    })
    print("✅ API key inserted")
