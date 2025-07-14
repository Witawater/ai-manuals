FROM python:3.11-slim

# ─── Install required OS packages ───────────────
RUN apt-get update && apt-get install -y \
    build-essential \
    libjpeg-dev \
    poppler-utils \
    gcc \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# ─── App setup ───────────────────────────────────
WORKDIR /app
COPY . /app

# ─── Install Python deps ─────────────────────────
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Port and start ──────────────────────────────
ENV PORT=8000
EXPOSE ${PORT}
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
