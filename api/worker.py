import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
STORAGE_DIR = "/app/storage"

def process_file(input_path, output_path):
    input_full = os.path.join(STORAGE_DIR, os.path.basename(input_path))
    output_full = os.path.join(STORAGE_DIR, os.path.basename(output_path))

    cmd = [
        "ocrmypdf",
        "--force-ocr",
        "--deskew",
        "pgquant",
        input_full,
        output_full
    ]

    logging.info(f"Executando OCR: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logging.info(f"OCR concluído: {output_full}")
        return output_full
    except subprocess.CalledProcessError as e:
        logging.error(f"Erro no OCRmyPDF: {e.stderr.decode()}")
        raise


from redis import Redis
from rq import Worker, Queue

redis_conn = Redis(host="redis", port=6379)
queue = Queue("default", connection=redis_conn)

if __name__ == "__main__":
    worker = Worker([queue], connection=redis_conn)
    worker.work()
