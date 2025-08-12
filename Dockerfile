FROM python:3.11-slim

# ─── System deps ─────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    poppler-utils \
    gcc \
    curl \
    libpq-dev \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

# ─── App setup ───────────────────────────────────────────────
WORKDIR /app

# Python runtime flags
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Python deps first (better layer caching)
COPY requirements.txt .
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Render will inject $PORT at runtime; default to 8000 for local runs
EXPOSE 8000
CMD ["bash", "-lc", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
