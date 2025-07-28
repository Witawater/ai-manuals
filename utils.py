# utils.py
import hashlib, pathlib

def file_sha256(path: str, chunk=8192) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()
