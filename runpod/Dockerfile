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
# Python deps (PyTorch with CUDA 12.8) — cached layer
# ---------------------------------------------------------------------------
COPY requirements.txt requirements-colab.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cu128 \
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
