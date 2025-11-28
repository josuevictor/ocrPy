FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-por \
    ghostscript \
    qpdf \
    poppler-utils \
    libjpeg62-turbo \
    zlib1g \
    libpng16-16 \
    libtiff6 \
    libxml2 \
    unpaper \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY . /app
RUN mkdir -p /app/storage

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
