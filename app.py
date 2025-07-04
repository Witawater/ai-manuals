from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ―― local modules ――
from ingest_manual import ingest
from qa_demo       import chat          # your Q-and-A helper
from db            import engine        # creates the table on import

# extra deps for feedback endpoint
from pydantic import BaseModel
from sqlalchemy import text

# ──────────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---- 1) PDF upload & ingest ---------------------------------------------
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

# ---- 2) Chat endpoint ----------------------------------------------------
@app.post("/chat")
async def ask(
    question: str = Form(...),
    customer: str = Form("demo01")
):
    return {"answer": chat(question, customer)}

# ---- 3) Feedback endpoint ----------------------------------------------
class FeedbackIn(BaseModel):
    customer: str
    question: str
    answer:   str
    score:    int   # +1 👍  or  -1 👎

@app.post("/feedback")
def add_feedback(data: FeedbackIn):
    if data.sco
