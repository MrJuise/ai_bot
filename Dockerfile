FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DEFAULT_TIMEOUT=100

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        libffi-dev \
        libjpeg62-turbo-dev \
        libpng-dev \
        libfreetype6-dev \
        libgomp1 \
        libssl-dev \
        libxml2-dev \
        libxslt1-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir -r requirements.txt

COPY . .

RUN rm -rf venv \
    && mkdir -p vectorstore/chroma_db

RUN useradd --create-home appuser \
    && chown -R appuser:appuser /app

USER appuser

VOLUME ["/app/chat_memory.db", "/app/vectorstore"]

CMD ["python", "bot.py"]