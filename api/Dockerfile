FROM python:3.11-bookworm

# ------------------------------------------------------------
# Instala dependências do sistema
# ------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-por \
    ghostscript \
    qpdf \
    poppler-utils \
    unpaper \
    pngquant \
    jbig2enc \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff6 \
    zlib1g \
    libxml2 \
    libxslt1.1 \
    libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Instala Modelo Tesseract BEST (PT / ENG)
# ------------------------------------------------------------
RUN mkdir -p /usr/share/tesseract-ocr/5/tessdata/ \
    && apt-get update \
    && apt-get install -y tesseract-ocr-por-best tesseract-ocr-eng-best \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------
# Instala OCRmyPDF
# ------------------------------------------------------------
RUN pip install --no-cache-dir ocrmypdf

# ------------------------------------------------------------
# Instala OpenCV (necessário para pré-processamento avançado)
# ------------------------------------------------------------
RUN pip install --no-cache-dir opencv-python-headless

# ------------------------------------------------------------
# Configura projeto Python
# ------------------------------------------------------------
WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app

RUN mkdir -p /app/storage

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

