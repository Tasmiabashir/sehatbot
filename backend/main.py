from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from backend.schemas import AskRequest, SehatBotResponse
from backend.agent import run_agent
import shutil, os

app = FastAPI(title="SehatBot API", version="1.0.0")

app.add_middleware(CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.get("/")
def health():
    return {"status": "SehatBot is running!"}

@app.post("/ask", response_model=SehatBotResponse)
def ask(request: AskRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    try:
        answer = run_agent(request.question)
        return SehatBotResponse(status="success", mode="auto", answer=answer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-report")
def upload_report(file: UploadFile = File(...)):
    if not (file.filename or "").endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")
    path = f"temp_{file.filename or 'report.pdf'}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        answer = run_agent(f"Analyze this lab report file: {path}")
        return {"status": "success", "answer": answer}
    finally:
        os.remove(path)

@app.post("/upload-prescription")
def upload_prescription(file: UploadFile = File(...)):
    allowed = [".jpg", ".jpeg", ".png"]
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail="Only JPG or PNG allowed")
    path = f"temp_{file.filename or 'upload.png'}"
    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    try:
        answer = run_agent(f"Read this prescription image and check medicines: {path}")
        return {"status": "success", "answer": answer}
    finally:
        os.remove(path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)