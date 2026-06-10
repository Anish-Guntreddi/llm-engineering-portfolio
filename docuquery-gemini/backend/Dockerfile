# Backend image for Render/Fly/Railway. Local embedding model is downloaded at build time.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

COPY pyproject.toml ./
COPY src ./src
# Install with CPU torch (smaller) + local embeddings + gemini support.
RUN pip install --upgrade pip && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install -e ".[local,gemini]"

EXPOSE 8000
CMD ["uvicorn", "docuquery.api:app", "--host", "0.0.0.0", "--port", "8000"]
