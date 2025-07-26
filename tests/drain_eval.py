from datetime import datetime
from qa_demo import chat

QUESTIONS = [
    ("drain",  "How do I drain the system?"),
    ("modbus", "Does it support Modbus?")
]

ts = datetime.utcnow().isoformat(timespec="seconds")
lines = []
for tag, q in QUESTIONS:
    out = chat(q, customer="demo01", doc_type="hardware", concise=True)
    lines.append(f"{ts}\t{tag}\t{len(out['chunks_used'])}\t{out['confidence']:.3f}")

with open("/mnt/data/manual_eval.log", "a") as f:
    for ln in lines:
        print(ln, file=f)
