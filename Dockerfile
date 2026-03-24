# Stage 1: Builder
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock README.md ./

# Install dependencies with server extras and build wheel
RUN uv sync --frozen --extra server --no-dev

# Copy source code
COPY gigaam/ ./gigaam/
COPY gigaam_server/ ./gigaam_server/

# Create non-root user
RUN useradd -m -u 1000 gigaam && \
    mkdir -p /app && \
    chown -R gigaam:gigaam /app

USER gigaam
WORKDIR /app

# Environment variables
ENV PATH="/app/.venv/bin:$PATH" \
    GIGAAM_HOST=0.0.0.0 \
    GIGAAM_PORT=8000 \
    GIGAAM_DEFAULT_MODEL=v3_e2e_rnnt \
    GIGAAM_DEVICE=cuda \
    GIGAAM_FP16_ENCODER=true

# Cache directory for models
VOLUME ["/root/.cache/gigaam"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["uvicorn", "gigaam_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
