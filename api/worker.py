# ...existing code...
import os
import subprocess
import logging
import mimetypes
from PIL import Image, ImageFilter, ImageOps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

STORAGE_DIR = "/app/storage"


# -------------------------------------------------------------
# 🔧 Pré-processamento da imagem para melhorar OCR
# - autocontrast
# - desnoising (median)
# - sharpen
# - binarização suave
# -------------------------------------------------------------
def preprocess_image_for_ocr(src_path, dst_path, dpi=300):
    try:
        img = Image.open(src_path)

        # Convert to grayscale (works better for many OCR jobs)
        img = img.convert("L")

        # Auto-contrast to expand tonal range
        img = ImageOps.autocontrast(img, cutoff=1)

        # Remove small speckles (despeckle)
        img = img.filter(ImageFilter.MedianFilter(size=3))

        # Slight sharpen to improve character edges
        img = img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))

        # Optional binarize (keep as 8-bit to preserve some greys for OCR)
        # Use a gentle threshold based on average to avoid losing faint strokes
        threshold = int(sum(img.getdata()) / (img.size[0] * img.size[1]) * 0.85)
        img = img.point(lambda p: 255 if p > threshold else 0)

        # Save with target DPI for OCRmyPDF / Tesseract
        img.save(dst_path, format="PNG", dpi=(dpi, dpi))
        logging.info(f"[OK] Imagem pré-processada → {dst_path}")

    except Exception as e:
        logging.error(f"Erro no pré-processamento da imagem: {e}")
        raise


# -------------------------------------------------------------
# 🔧 Converte IMAGEM → PDF com DPI 300 (máxima para OCR)
# -------------------------------------------------------------
def convert_image_to_pdf(image_path, pdf_path):
    try:
        # Preprocess to a temp PNG first (same dir as input)
        preprocessed = image_path + "_preproc.png"
        preprocess_image_for_ocr(image_path, preprocessed, dpi=300)

        img = Image.open(preprocessed)
        img = img.convert("RGB")

        # Sempre salva com DPI 300 — obrigatório para boa leitura
        img.save(pdf_path, "PDF", resolution=300)
        logging.info(f"[OK] Imagem convertida para PDF (300 DPI) → {pdf_path}")

        # Remove temp preprocessed file
        if os.path.exists(preprocessed):
            os.remove(preprocessed)

    except Exception as e:
        logging.error(f"Erro ao converter imagem → PDF: {e}")
        raise


# -------------------------------------------------------------
# 🔧 PROCESSAMENTO OCR COM MÁXIMA QUALIDADE
# -------------------------------------------------------------
def process_file(input_path, output_path, ocr_lang=None):
    input_full = os.path.join(STORAGE_DIR, os.path.basename(input_path))
    output_full = os.path.join(STORAGE_DIR, os.path.basename(output_path))

    if not os.path.exists(input_full):
        raise FileNotFoundError(f"Arquivo não encontrado: {input_full}")

    mime, _ = mimetypes.guess_type(input_full)
    temp_pdf = None

    # ---------------------------------------------------------
    # Se arquivo for imagem → converter primeiro (com preprocess)
    # ---------------------------------------------------------
    if mime and mime.startswith("image/"):
        temp_pdf = input_full + "_temp.pdf"
        convert_image_to_pdf(input_full, temp_pdf)
        input_full = temp_pdf

    # Use variável de ambiente ou argumento para definir idioma
    lang = ocr_lang or os.environ.get("OCR_LANG", "por+eng")

    # ---------------------------------------------------------
    # OCRmyPDF — CONFIGURAÇÃO DE MÁXIMA QUALIDADE (ajustada)
    # ---------------------------------------------------------
    cmd = [
        "ocrmypdf",
        "--force-ocr",
        "--rotate-pages",
        "--deskew",
        "--clean-final",
        "--optimize", "3",
        "--image-dpi", "300",
        "--jobs", "4",
        "--tesseract-timeout", "120",
        "--skip-big", "500",
        "--remove-background",
        "--language", lang,
        "--tesseract-oem", "1",        # usar LSTM (neural) engine
        "--tesseract-pagesegmode", "3",
        input_full,
        output_full
    ]

    logging.info("Executando OCRmyPDF:")
    logging.info(" ".join(cmd))

    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        logging.info(f"[OK] OCR concluído → {output_full}")

    except subprocess.CalledProcessError as e:
        logging.error("❌ ERRO no OCRmyPDF")
        try:
            logging.error(e.stderr.decode())
        except Exception:
            logging.error(str(e))
        raise

    finally:
        if temp_pdf and os.path.exists(temp_pdf):
            os.remove(temp_pdf)

    return output_full
# ...existing code...
from redis import Redis
from rq import Worker, Queue

redis_conn = Redis(host="redis", port=6379)
queue = Queue("default", connection=redis_conn)

if __name__ == "__main__":
    logging.info("🚀 Worker OCR iniciado — aguardando jobs...")
    worker = Worker([queue], connection=redis_conn)
    worker.work()