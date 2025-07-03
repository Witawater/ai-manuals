FROM python:3.11-slim

# deps for pdfplumber
RUN apt-get update && apt-get install -y \
        build-essential libjpeg-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

ENV PORT=8000
EXPOSE ${PORT}
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
