# auth.py
import os
from typing import Optional
from fastapi import Header, HTTPException, Request
from sqlalchemy import text
from db import engine

REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "false").lower() in {"1", "true", "yes"}
TRACK_USAGE     = os.getenv("TRACK_USAGE", "false").lower() in {"1", "true", "yes"}

def require_api_key(
    request: Request,
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> str:
    """
    Returns a 'customer' string for downstream handlers.

    - CORS preflight (OPTIONS) is always allowed.
    - If REQUIRE_API_KEY is False and header is missing → 'public'
    - If header is present → validate against api_keys table
    - If TRACK_USAGE → atomically increment `used`, enforce `quota` (NULL = unlimited)
    """
    # 0) Allow CORS preflight through
    if request.method == "OPTIONS":
        return "public"

    # 1) Allow anonymous if not enforcing keys
    if not x_api_key:
        if REQUIRE_API_KEY:
            raise HTTPException(status_code=401, detail="Missing API key")
        return "public"

    # 2) Validate key
    with engine.begin() as conn:
        row = conn.execute(
            text("""
                SELECT customer, quota, COALESCE(used, 0) AS used
                FROM api_keys
                WHERE "key" = :k
            """),
            {"k": x_api_key}
        ).fetchone()

        if not row:
            raise HTTPException(status_code=401, detail="Invalid API key")

        customer, quota, used = row  # 'used' is 0+ and safe

        if TRACK_USAGE:
            # 3) Enforce quota atomically; allow unlimited if quota IS NULL
            upd = conn.execute(
                text("""
                    UPDATE api_keys
                       SET used = COALESCE(used, 0) + 1
                     WHERE "key" = :k
                       AND (quota IS NULL OR COALESCE(used, 0) < quota)
                 RETURNING quota, COALESCE(used, 0) AS used
                """),
                {"k": x_api_key}
            ).fetchone()

            if not upd:
                # blocked at quota
                raise HTTPException(status_code=429, detail="Quota exceeded")

        return customer
