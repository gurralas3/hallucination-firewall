import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from document_store import build_index
from firewall import chat as firewall_chat

app = FastAPI(title="Hallucination Firewall API")

class ChatRequest(BaseModel):
    question: str
    org_id: str
    firewall_on: bool = True

@app.get("/")
def root():
    return {"message": "Hallucination Firewall is running."}

@app.post("/setup")
async def setup_org(
    org_id: str = Form(...),
    files: list[UploadFile] = File(...)
):
    saved_paths = []
    temp_dir = tempfile.mkdtemp()
    try:
        for file in files:
            path = os.path.join(temp_dir, file.filename)
            with open(path, "wb") as f:
                f.write(await file.read())
            saved_paths.append(path)

        build_index(org_id, saved_paths)
        return {"status": "success", "org_id": org_id, "files_indexed": len(saved_paths)}
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    if request.firewall_on:
        result = firewall_chat(request.question, request.org_id)
    else:
        from firewall import _generate_llm_answer
        raw_answer = _generate_llm_answer(request.question)
        result = {
            "question":        request.question,
            "original_answer": raw_answer,
            "final_answer":    raw_answer,
            "status":          "UNVERIFIED",
            "hallucinations":  0,
            "details":         []
        }
    return result
