FROM python:3.11-slim AS base

# ─── Metadata ─────────────────────────────────────────────────────────────
LABEL maintainer="Healthcare ML Team"
LABEL description="Hospital Readmission Risk Predictor — ML Pipeline"
LABEL version="1.0.0"

# ─── System dependencies ─────────────────────────────────────────────────
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ─── Non-root user ────────────────────────────────────────────────────────
RUN groupadd -r mluser && useradd -r -g mluser -m -s /bin/bash mluser

# ─── Application directory ───────────────────────────────────────────────
WORKDIR /app

# ─── Install dependencies first (layer caching) ──────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ─── Copy application code ───────────────────────────────────────────────
COPY src/ ./src/
COPY notebooks/ ./notebooks/
COPY tests/ ./tests/
COPY .env.example ./.env.example

# ─── Create data / output directories ────────────────────────────────────
RUN mkdir -p data/raw data/processed outputs && \
    chown -R mluser:mluser /app

# ─── Switch to non-root ──────────────────────────────────────────────────
USER mluser

# ─── Expose Jupyter port ─────────────────────────────────────────────────
EXPOSE 8888

# ─── Default: start JupyterLab ───────────────────────────────────────────
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--NotebookApp.token=''"]
