# auth.py  ─────────────────────────────────────────────
# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=True)

import os
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader

# Retrieve API key from environment
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Dependency to enforce API key auth
def require_api_key(key: str = Depends(api_key_header)):
    """FastAPI dependency that validates X-API-Key header against environment value."""
    if not API_KEY:
        raise RuntimeError("Missing API_KEY in environment. Check your .env file.")
    if key == API_KEY:
        return True
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )
