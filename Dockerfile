# --- 1. БАЗОВЫЙ ОБРАЗ ---
    FROM python:3.11-slim

    # --- 2. ОКРУЖЕНИЕ PYTHON ---
    ENV PYTHONDONTWRITEBYTECODE=1 \
        PYTHONUNBUFFERED=1 \
        PIP_DISABLE_PIP_VERSION_CHECK=1 \
        PIP_NO_CACHE_DIR=1 \
        PIP_DEFAULT_TIMEOUT=100
    
    # --- 3. РАБОЧАЯ ПАПКА ---
    WORKDIR /app
    
    # --- 4. СИСТЕМНЫЕ ЗАВИСИМОСТИ ---
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
        ca-certificates \
        && rm -rf /var/lib/apt/lists/*
    
    # --- 5. КОПИРУЕМ requirements ---
    COPY requirements.txt ./
    
    # --- 6. УСТАНОВКА ВСЕХ PYTHON ЗАВИСИМОСТЕЙ ---
    # Включая: langchain, langchain-community, langchain-ollama, chromadb, aiogram, ollama и др.
    RUN pip install --no-cache-dir -r requirements.txt
    
    # --- 7. КОПИРУЕМ ВЕСЬ ПРОЕКТ (делаем это ПОСЛЕ, чтобы не пересобирать кэш при каждом изменении кода) ---
    COPY . .
    
    # --- 8. СОЗДАЁМ ПАПКИ ДЛЯ CHROMA DB ---
    RUN mkdir -p /app/data/vectorstore/chroma_db && \
        mkdir -p /app/data/huggingface_cache
    
    # --- 9. ПОЛЬЗОВАТЕЛЬ ---
    RUN useradd --create-home appuser \
        && chown -R appuser:appuser /app
    USER appuser
    
    # --- 10. VOLUME ---
    # Сохраняем данные между перезапусками контейнера
    VOLUME ["/app/chat_memory.db", "/app/vectorstore"]
    
    # --- 11. ЗАПУСК ---
    # ВАЖНО: Ollama должен быть запущен отдельно (через docker-compose или локально)
    # Модель nomic-embed-text должна быть загружена в Ollama перед запуском бота
    # Команда для загрузки: ollama pull nomic-embed-text
    CMD ["python", "bot.py"]