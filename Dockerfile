FROM python:3.11-slim

# deps for pdfplumber and general build tools
RUN apt-get update && apt-get install -y \
        build-essential libjpeg-dev poppler-utils && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# upgrade pip before install
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

ENV PORT=8000
EXPOSE ${PORT}
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
