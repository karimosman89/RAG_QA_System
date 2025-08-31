# ===============================
# Enterprise RAG Platform Dockerfile
# Multi-stage build for production optimization
# ===============================

# Build stage
FROM python:3.11-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    APP_HOME="/app"

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Create app user for security
RUN groupadd -r app && useradd -r -g app app

# Create application directory
RUN mkdir -p $APP_HOME && chown app:app $APP_HOME
WORKDIR $APP_HOME

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=app:app . .

# Create necessary directories
RUN mkdir -p logs data chroma_db static/uploads && \
    chown -R app:app logs data chroma_db static

# Create supervisor configuration
RUN echo '[supervisord]' > /etc/supervisor/conf.d/supervisord.conf && \
    echo 'nodaemon=true' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'user=root' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'logfile=/app/logs/supervisord.log' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'pidfile=/app/logs/supervisord.pid' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo '' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo '[program:rag_platform]' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'command=/opt/venv/bin/python -m uvicorn src.api.server:app --host 0.0.0.0 --port 8000' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'directory=/app' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'user=app' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'autostart=true' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'autorestart=true' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stdout_logfile=/app/logs/rag_platform.log' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stderr_logfile=/app/logs/rag_platform_error.log' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stdout_logfile_maxbytes=50MB' >> /etc/supervisor/conf.d/supervisord.conf && \
    echo 'stdout_logfile_backups=10' >> /etc/supervisor/conf.d/supervisord.conf

# Create environment file template
RUN echo '# Enterprise RAG Platform Configuration' > .env.example && \
    echo '# Copy this file to .env and configure your values' >> .env.example && \
    echo '' >> .env.example && \
    echo '# Application Settings' >> .env.example && \
    echo 'APP_NAME="Enterprise RAG Platform"' >> .env.example && \
    echo 'APP_VERSION="2.0.0"' >> .env.example && \
    echo 'ENVIRONMENT="production"' >> .env.example && \
    echo 'DEBUG=false' >> .env.example && \
    echo '' >> .env.example && \
    echo '# Server Configuration' >> .env.example && \
    echo 'HOST="0.0.0.0"' >> .env.example && \
    echo 'PORT=8000' >> .env.example && \
    echo 'WORKERS=4' >> .env.example && \
    echo '' >> .env.example && \
    echo '# Security Configuration' >> .env.example && \
    echo 'JWT_SECRET_KEY="your-super-secret-key-change-this"' >> .env.example && \
    echo 'JWT_ALGORITHM="HS256"' >> .env.example && \
    echo 'ACCESS_TOKEN_EXPIRE_MINUTES=1440' >> .env.example && \
    echo '' >> .env.example && \
    echo '# AI Provider API Keys' >> .env.example && \
    echo 'OPENAI_API_KEY="your-openai-api-key"' >> .env.example && \
    echo 'ANTHROPIC_API_KEY="your-anthropic-api-key"' >> .env.example && \
    echo 'GOOGLE_API_KEY="your-google-api-key"' >> .env.example && \
    echo '' >> .env.example && \
    echo '# Database Configuration' >> .env.example && \
    echo 'DATABASE_URL="sqlite+aiosqlite:///./data/rag_platform.db"' >> .env.example && \
    echo '' >> .env.example && \
    echo '# Redis Configuration' >> .env.example && \
    echo 'REDIS_URL="redis://redis:6379/0"' >> .env.example && \
    echo '' >> .env.example && \
    echo '# Vector Store Configuration' >> .env.example && \
    echo 'VECTOR_STORE_TYPE="chroma"' >> .env.example && \
    echo 'CHROMA_PERSIST_DIRECTORY="./chroma_db"' >> .env.example

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Expose port
EXPOSE 8000

# Switch to app user
USER app

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]