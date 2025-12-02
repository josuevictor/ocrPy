import os
import subprocess
import logging
import mimetypes
from PIL import Image, ImageFilter, ImageOps
import cv2
import numpy as np

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
        preprocess_image_for_ocr_cv2_advanced(image_path, preprocessed, dpi=300)

        img = Image.open(preprocessed)

        # Se imagem tiver canal alpha (transparência), compor sobre fundo branco
        if img.mode in ("RGBA", "LA") or ("transparency" in img.info):
            try:
                background = Image.new("RGBA", img.size, (255, 255, 255, 255))
                background.paste(img, mask=img.split()[-1])
                img = background.convert("RGB")
            except Exception:
                img = img.convert("RGB")
        else:
            img = img.convert("RGB")

        # ✅ Aceitar qualquer DPI e upscalar para 300 DPI se necessário
        target_dpi = 300
        src_dpi = 72  # default se não tiver metadados
        
        try:
            # Tenta extrair DPI dos metadados da imagem
            if hasattr(img, 'info') and 'dpi' in img.info:
                src_dpi = img.info['dpi'][0]
            logging.info(f"📊 DPI detectado: {src_dpi}")
        except Exception as e:
            logging.warning(f"⚠️ Não foi possível detectar DPI, usando padrão 72. Erro: {e}")
            src_dpi = 72

        # Se DPI for menor que 300, fazer upscaling inteligente
        if src_dpi < target_dpi:
            scale = target_dpi / float(src_dpi)
            w, h = img.size
            new_size = (int(w * scale), int(h * scale))
            logging.info(f"🔍 Upscaling: {w}x{h} ({src_dpi} DPI) → {new_size[0]}x{new_size[1]} ({target_dpi} DPI) | Escala: {scale:.2f}x")
            
            # Usar upscaling de qualidade (LANCZOS é melhor para upscaling)
            img = img.resize(new_size, resample=Image.LANCZOS)
        else:
            logging.info(f"✅ Imagem já em DPI adequado ({src_dpi} DPI)")

        # Sempre salva com DPI 300 — obrigatório para boa leitura
        img.save(pdf_path, "PDF", resolution=target_dpi)
        logging.info(f"[OK] Imagem convertida para PDF ({target_dpi} DPI) → {pdf_path}")

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


# -------------------------------------------------------------
# 🔧 Pré-processamento da imagem usando OpenCV
# - remoção de ruído
# - equalização de histograma
# - binarização adaptativa
# -------------------------------------------------------------
def preprocess_image_for_ocr_cv2(src_path, dst_path, dpi=300):
    try:
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Não foi possível abrir a imagem: {src_path}")
        # Remover ruído (mediana)
        img = cv2.medianBlur(img, 3)
        # Equalização de histograma (contraste)
        img = cv2.equalizeHist(img)
        # Binarização adaptativa
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)
        # Salvar imagem pré-processada
        cv2.imwrite(dst_path, img)
        logging.info(f"[OK][OpenCV] Imagem pré-processada → {dst_path}")
    except Exception as e:
        logging.error(f"[OpenCV] Erro no pré-processamento: {e}")
        raise


# -------------------------------------------------------------
# 🔧 Pré-processamento da imagem usando OpenCV - AVANÇADO
# - remoção de ruído
# - equalização de histograma
# - deskew (correção de inclinação)
# - remoção de linhas horizontais/verticais
# - binarização adaptativa
# -------------------------------------------------------------
def preprocess_image_for_ocr_cv2_advanced(src_path, dst_path, dpi=300):
    try:
        img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Não foi possível abrir a imagem: {src_path}")
        # 1. Remover ruído (mediana)
        img = cv2.medianBlur(img, 3)
        # 2. Equalização de histograma
        img = cv2.equalizeHist(img)
        # 3. Deskew (correção de inclinação)
        coords = np.column_stack(np.where(img > 0))
        angle = 0
        if coords.shape[0] > 0:
            rect = cv2.minAreaRect(coords)
            angle = rect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            (h, w) = img.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        # 4. Remover linhas horizontais/verticais (morfologia)
        # Horizontal
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40,1))
        detect_horizontal = cv2.morphologyEx(img, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
        img = cv2.subtract(img, detect_horizontal)
        # Vertical
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,40))
        detect_vertical = cv2.morphologyEx(img, cv2.MORPH_OPEN, vertical_kernel, iterations=1)
        img = cv2.subtract(img, detect_vertical)
        # 5. Binarização adaptativa
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 31, 10)
        # Salvar imagem pré-processada
        cv2.imwrite(dst_path, img)
        logging.info(f"[OK][OpenCV-ADV] Imagem pré-processada → {dst_path}")
    except Exception as e:
        logging.error(f"[OpenCV-ADV] Erro no pré-processamento: {e}")
        raise


from redis import Redis
from rq import Worker, Queue

redis_conn = Redis(host="redis", port=6379)
queue = Queue("default", connection=redis_conn)

if __name__ == "__main__":
    logging.info("🚀 Worker OCR iniciado — aguardando jobs...")
    worker = Worker([queue], connection=redis_conn)
    worker.work()