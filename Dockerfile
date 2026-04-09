# ──────────────────────────────────────────────────────────────────────────────
# FactoryGPT Cloud Broker — production container
# Small, fast, no ML deps. Deploy to Fly.io, Railway, Cloud Run, or ECS.
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Prevent .pyc files and buffer flushing
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first (better layer caching)
COPY requirements-cloud.txt .
RUN pip install --no-cache-dir -r requirements-cloud.txt

# Copy only what the cloud broker needs
COPY cloud_backend.py .

# Non-root user
RUN useradd --create-home --shell /bin/bash factory && chown -R factory:factory /app
USER factory

EXPOSE 8000

# Healthcheck hits the /health endpoint we built into cloud_backend.py
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health').read()" || exit 1

CMD ["sh", "-c", "uvicorn cloud_backend:app --host 0.0.0.0 --port ${PORT:-8000}"]
