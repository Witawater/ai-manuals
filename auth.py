from fastapi import Header, HTTPException, Request
from sqlalchemy import text
from db import engine

# Optional: track usage
TRACK_USAGE = False

def require_api_key(X-API-Key: str = Header(...)):
    with engine.begin() as conn:
        row = conn.execute(
            text("""SELECT customer, quota, used FROM api_keys
                    WHERE key = :k"""),
            {"k": X-API-Key}
        ).fetchone()

        if not row:
            raise HTTPException(status_code=401, detail="Invalid API key")

        customer, quota, used = row

        if TRACK_USAGE:
            if used >= quota:
                raise HTTPException(status_code=429, detail="Quota exceeded")
            conn.execute(
                text("UPDATE api_keys SET used = used + 1 WHERE key = :k"),
                {"k": X-API-Key}
            )

        return customer  # returned to FastAPI dependency injection
