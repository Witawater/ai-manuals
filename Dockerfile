FROM python:3.11-slim

# ─── Install required OS packages ───────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    poppler-utils \
    gcc \
    curl \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ─── App setup ───────────────────────────────────
WORKDIR /app
COPY requirements.txt .

# ─── Python env tweaks ───────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ─── Install Python deps ─────────────────────────
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Secrets vault files (encrypted env + decrypt helper) ──
COPY .env.production.enc decrypt_env.py /app/

# ─── Copy the rest of the codebase ───────────────
COPY . .

# ─── Port and start ──────────────────────────────
ENV PORT=8000
EXPOSE ${PORT}

# Run decrypt_env.py first, then launch Uvicorn
CMD ["bash", "-c", "python decrypt_env.py && uvicorn app:app --host 0.0.0.0 --port ${PORT}"]
