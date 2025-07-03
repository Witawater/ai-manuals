from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ingest_manual import ingest
from qa_demo import chat                                # your working function

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ---- REST endpoints ------------------------------------------------------
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    customer: str = Form("demo01")
):
    path = f"/tmp/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    ingest(path, customer)
    return {"status": "ingested", "file": file.filename}

@app.post("/chat")
async def ask(
    question: str = Form(...),
    customer: str = Form("demo01")
):
    return {"answer": chat(question, customer)}

# ---- serve a super-simple front-end -------------------------------------
app.mount("/", StaticFiles(directory="web", html=True), name="web")
