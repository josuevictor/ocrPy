import os
import uuid
from fastapi import FastAPI, File, UploadFile, HTTPException
from redis import Redis
from rq import Queue

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_conn = Redis.from_url(REDIS_URL)
q = Queue(connection=redis_conn)

app = FastAPI()

STORAGE_DIR = "/app/storage"
os.makedirs(STORAGE_DIR, exist_ok=True)

def enqueue_job(input_path, output_path):
    # job function name (worker imports a callable)
    return q.enqueue("worker.process_file", input_path, output_path)

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Arquivo inválido")

    # limitar por segurança (ex: 200MB)
    contents = await file.read()
    if len(contents) == 0:
        raise HTTPException(status_code=400, detail="Arquivo vazio")

    uid = str(uuid.uuid4())
    input_filename = f"{uid}_{file.filename}"
    input_path = os.path.join(STORAGE_DIR, input_filename)
    with open(input_path, "wb") as f:
        f.write(contents)

    output_filename = f"{uid}_ocr.pdf"
    output_path = os.path.join(STORAGE_DIR, output_filename)

    job = enqueue_job(input_path, output_path)

    return {"job_id": job.get_id(), "input": input_filename, "output": output_filename}
