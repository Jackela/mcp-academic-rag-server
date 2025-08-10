# Multi-stage build for Python academic RAG server
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    pkg-config \
    libhdf5-dev \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    python3-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt /tmp/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libhdf5-103 \
    libopenblas0 \
    liblapack3 \
    libjpeg62-turbo \
    libpng16-16 \
    libtiff5 \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    libv4l-0 \
    libxvidcore4 \
    libx264-160 \
    libgtk-3-0 \
    libatlas3-base \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Create application user for security
RUN groupadd -r appgroup && useradd -r -g appgroup -d /app -s /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appgroup . .

# Create necessary directories
RUN mkdir -p /app/data /app/output /app/logs /app/temp \
    && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Create data directories with proper permissions
RUN mkdir -p /app/data/documents /app/data/embeddings /app/data/cache \
    && mkdir -p /app/output/processed /app/output/results

# Set Python path
ENV PYTHONPATH="/app:$PYTHONPATH"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; sys.path.append('/app'); from health_check import check_system_health; exit(0 if check_system_health() else 1)"

# Expose port for MCP server
EXPOSE 8000

# Default command - run MCP server
CMD ["python", "mcp_server.py"]

# Labels for metadata
LABEL maintainer="Academic RAG Team"
LABEL version="1.0.0"
LABEL description="Academic document processing and RAG server with MCP interface"
LABEL org.opencontainers.image.source="https://github.com/academic-rag/mcp-server"
LABEL org.opencontainers.image.documentation="https://github.com/academic-rag/mcp-server/README.md"
LABEL org.opencontainers.image.title="MCP Academic RAG Server"
LABEL org.opencontainers.image.description="Containerized academic document processing pipeline with RAG capabilities"