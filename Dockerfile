# Use a highly optimized, small Python image
FROM python:3.11-slim

# RAM optimization environment variables
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install ONLY the bare minimum system tools for FAISS
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# THE KEY FIX: Force CPU-only installation to stay under 512MB RAM
# We install Torch and Sentence-Transformers specifically with the +cpu tag
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1+cpu \
    sentence-transformers==3.0.1 \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# CRITICAL: Build the database INSIDE the container to avoid corrupted file errors
RUN python ingest.py

# DYNAMIC PORT FIX: Render assigns a random port; we must use the $PORT variable
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
