# ================================================================
# Adversarial Memory Forensics — Pipeline Container
# ================================================================
# Multi-stage build for the complete analysis pipeline.
# Includes:
#   - Volatility 3 + VolMemLyzer for feature extraction
#   - ML dependencies (scikit-learn, xgboost, lightgbm, shap)
#   - LiME kernel module (for acquisition VMs only)
#
# Usage:
#   docker build -t amf-pipeline .
#   docker run --rm -v $(pwd)/data:/app/data amf-pipeline \
#       python -m pipeline.ensemble_model --train
# ================================================================

FROM python:3.11-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application Stage ─────────────────────────────────────────────

FROM base AS app

COPY setup.py .
COPY config/ config/
COPY perturbation_primitives/ perturbation_primitives/
COPY pipeline/ pipeline/
COPY analysis/ analysis/
COPY validation/ validation/
COPY infrastructure/scripts/ infrastructure/scripts/

# Install package in editable mode
RUN pip install -e .

# Create data directories
RUN mkdir -p \
    data/raw_dumps \
    data/features \
    data/metadata \
    data/timing_logs \
    results/models \
    results/phase0 \
    results/truncation \
    results/adversarial \
    results/shap \
    results/figures

# Non-root user for security
RUN useradd -m -s /bin/bash researcher
USER researcher

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "from pipeline import EnsembleClassifier; print('OK')"

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python"]
CMD ["--help"]

# ── LiME Acquisition Stage (for acquisition VMs) ─────────────────

FROM ubuntu:22.04 AS acquisition

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    linux-headers-$(uname -r || echo "generic") \
    git \
    bc \
    jq \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt

# Clone and build LiME
RUN git clone https://github.com/504ensicsLabs/LiME.git && \
    cd LiME/src && \
    make || echo "LiME build deferred to runtime (needs matching kernel headers)"

COPY infrastructure/scripts/truncation_acquirer.sh /usr/local/bin/truncation_acquirer.sh
RUN chmod +x /usr/local/bin/truncation_acquirer.sh

ENTRYPOINT ["/usr/local/bin/truncation_acquirer.sh"]
