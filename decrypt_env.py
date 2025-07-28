#!/usr/bin/env python3
"""
decrypt_env.py  – runs inside the container **before** Uvicorn starts.
• Reads VAULT_KEY from Render's env-vars
• Decrypts .env.production.enc (AES-256-CBC, PBKDF2 100 k)
• Writes the plaintext to /app/.env so python-dotenv picks it up
"""

import os, subprocess, sys, pathlib, tempfile

KEY = os.getenv("VAULT_KEY")
if not KEY:
    print("🟥  VAULT_KEY missing in container env"); sys.exit(1)

enc_file = pathlib.Path("/app/.env.production.enc")
if not enc_file.exists():
    print("🟥  /app/.env.production.enc not found"); sys.exit(1)

tmp = tempfile.NamedTemporaryFile(delete=False)

subprocess.check_call(
    [
        "openssl", "enc", "-d",
        "-aes-256-cbc",            # NOTE: CBC (portable)
        "-pbkdf2", "-iter", "100000",
        "-in", str(enc_file),
        "-pass", f"pass:{KEY}"
    ],
    stdout=tmp
)

# Move into place for python-dotenv
os.replace(tmp.name, "/app/.env")
print("🔐  .env decrypted OK")
