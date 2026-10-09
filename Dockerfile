# CPU runtime for the installed MCP stdio package.
FROM python:3.11-slim-bookworm AS builder
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN python -m pip install --no-cache-dir --upgrade pip
# Select the official CPU distribution before resolving project dependencies.
# Runtime/dependency metadata remains owned by pyproject.toml.
RUN python -m pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install --no-cache-dir --upgrade setuptools wheel
WORKDIR /build
COPY pyproject.toml README.md ./
# Derive runtime dependencies from the sole declaration before copying changing sources.
RUN python -c 'import pathlib, tomllib; spec = tomllib.loads(pathlib.Path("pyproject.toml").read_text()); pathlib.Path("/tmp/runtime-requirements.txt").write_text("\n".join(spec["project"]["dependencies"]))'
RUN python -m pip install --no-cache-dir -r /tmp/runtime-requirements.txt
COPY . .
RUN python -m pip install --no-cache-dir --no-deps . && python -m pip check

FROM python:3.11-slim-bookworm AS production
RUN python -m pip install --no-cache-dir --upgrade pip 'setuptools>=83' wheel
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:$PATH"
ENV DATA_PATH=/app/data HAYSTACK_TELEMETRY_ENABLED=False
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libmagic1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /bin/bash appuser
WORKDIR /app
COPY --chown=appuser:appgroup config/config.json ./config/config.json
COPY --chown=appuser:appgroup config/processor_mappings.json ./config/processor_mappings.json
RUN mkdir -p /app/data /app/output /app/logs /app/temp \
    && chown -R appuser:appgroup /app
USER appuser
# This process serves MCP over stdin/stdout; it has no HTTP health endpoint.
# Optional PDF rendering requires a Chromium installation, as on other hosts.
HEALTHCHECK NONE
ENTRYPOINT ["mcp-academic-rag-server"]

LABEL maintainer="Academic RAG Team"
LABEL version="1.0.0"
LABEL description="Academic document processing and RAG server with MCP interface"
LABEL org.opencontainers.image.source="https://github.com/Jackela/mcp-academic-rag-server"
LABEL org.opencontainers.image.documentation="https://github.com/Jackela/mcp-academic-rag-server/blob/main/README.md"
LABEL org.opencontainers.image.title="MCP Academic RAG Server"
LABEL org.opencontainers.image.description="Containerized academic document processing pipeline with RAG capabilities"
