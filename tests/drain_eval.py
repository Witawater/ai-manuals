#!/usr/bin/env python3
"""
Nightly recall monitor for AI-Manuals.
Writes one tab-separated line per canned question to /mnt/data/manual_eval.log:
UTC-time, tag, chunks-retrieved, avg-confidence
"""

import sys, pathlib
# ── ensure repo root is on PYTHONPATH, regardless of working dir ──
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from datetime import datetime
from qa_demo import chat

QUESTIONS = [
    ("drain",  "How do I drain the system?"),
    ("modbus", "Does it support Modbus?"),
]

ts = datetime.utcnow().isoformat(timespec="seconds")
lines = []
for tag, q in QUESTIONS:
    out = chat(q, customer="demo01", doc_type="hardware", concise=True)
    lines.append(f"{ts}\t{tag}\t{len(out['chunks_used'])}\t{out['confidence']:.3f}")

LOG_FILE = "/mnt/data/manual_eval.log"

with open(LOG_FILE, "a", encoding="utf-8") as f:
    for ln in lines:
        print(ln, file=f)
