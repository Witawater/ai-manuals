from qa_demo import chat

QUESTIONS = [
    ("drain",  "How do I drain the system?"),
    ("modbus", "Does it support Modbus?")
]

for tag, q in QUESTIONS:
    out = chat(q, customer="demo01", doc_type="hardware", concise=True)
    print(f"{tag}: chunks={len(out['chunks_used'])} conf={out['confidence']:.2f}")
