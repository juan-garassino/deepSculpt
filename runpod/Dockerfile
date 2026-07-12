FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# ---------------------------------------------------------------------------
# System deps
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3-pip \
    curl git make ca-certificates gnupg lsb-release apt-transport-https \
    tini \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Node.js 20 + Claude Code CLI
# ---------------------------------------------------------------------------
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

# ---------------------------------------------------------------------------
# Google Cloud SDK (gsutil/gcloud)
# ---------------------------------------------------------------------------
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" \
        > /etc/apt/sources.list.d/google-cloud-sdk.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg \
    && apt-get update && apt-get install -y --no-install-recommends google-cloud-cli \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# uv (Rust Python package manager)
# ---------------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# ---------------------------------------------------------------------------
# Python deps (PyTorch CUDA wheels) — cached layer
#
# --index-url (not --extra-index-url): with the extra index, pip resolved the
# newest torch from PyPI — 2.12+cu130 — whose CUDA 13 runtime cannot init on
# Vertex/RunPod hosts running 12.x drivers (T4 fleet reports 12.2), so
# training silently fell back to CPU. CUDA 12.6 wheels run on any 12.x
# driver >= 525 and still cover sm_75 (T4) through sm_90.
#
# TORCH_INDEX build-arg: cu126 (default) for the T4/L4 fleet; cu128 for the
# Cloud Run RTX PRO 6000 (Blackwell sm_120 — cu126 wheels ship no kernels for
# it: cudaErrorNoKernelImageForDevice). Two image variants, never merged.
# ---------------------------------------------------------------------------
ARG TORCH_INDEX=cu126
COPY requirements.txt requirements-colab.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/${TORCH_INDEX} \
        torch torchvision torchaudio \
    && pip install --no-cache-dir -r requirements.txt typer click \
    && pip install --no-cache-dir google-cloud-storage

# ---------------------------------------------------------------------------
# Project source
# ---------------------------------------------------------------------------
COPY . /app
RUN pip install --no-cache-dir --no-deps -e . \
    && chmod +x /app/runpod/entrypoint.sh /app/runpod/scripts/run_research.sh \
    && find /app/runpod/scripts -name "*.py" -exec chmod +x {} \;

# ---------------------------------------------------------------------------
# Non-root user
# ---------------------------------------------------------------------------
RUN useradd -m -s /bin/bash researcher \
    && chown -R researcher:researcher /app \
    && mkdir -p /workspace \
    && chown -R researcher:researcher /workspace

USER researcher
WORKDIR /app

RUN git config --global user.email "research@deepsculpt.runpod" \
    && git config --global user.name "deepsculpt-research"

ENV PYTHONPATH="/app:/app/deepsculpt"
ENV WORKSPACE_DIR=/workspace
ENV DEEPSCULPT_BACKEND=gcs
ENV MODE=train

ENTRYPOINT ["/usr/bin/tini", "--", "/app/runpod/entrypoint.sh"]
