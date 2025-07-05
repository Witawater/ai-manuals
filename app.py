from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# local modules
from ingest_manual import ingest
from qa_demo       import chat
from db            import engine            # creates feedback table

# extras for /feedback
from pydantic import BaseModel
from sqlalchemy import text

# ──────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- 1) PDF upload & ingest ---------------------------------
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    customer: str   = Form("demo01")
):
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    ingest(path, customer)
    return {"status": "ingested", "file": file.filename}

# ---- 2) Chat ------------------------------------------------
@app.post("/chat")
async def ask(
    question: str = Form(...),
    customer: str = Form("demo01")
):
    return {"answer": chat(question, customer)}

# ---- 3) Feedback --------------------------------------------
class FeedbackIn(BaseModel):
    customer: str
    question: str
    answer:   str
    score:    int            # +1 👍  or  -1 👎

@app.post("/feedback")
def add_feedback(data: FeedbackIn):
    if data.score not in (-1, 1):
        raise HTTPException(400, "score must be +1 or -1")

    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO feedback (customer, question, answer, score)
                VALUES (:customer, :question, :answer, :score)
            """),
            data.model_dump()
        )
    return {"ok": True}

# ---- 4) Serve the tiny front-end ----------------------------
app.mount("/", StaticFiles(directory="web", html=True), name="web")
