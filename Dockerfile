# Production-ready Docker image for AgenticAI Framework
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy framework
COPY agentic_framework/ ./agentic_framework/
COPY examples/ ./examples/
COPY docs/ ./docs/

# Create directory for user data
RUN mkdir -p /app/data

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV LLM_PROVIDER=ollama
ENV EMBEDDING_PROVIDER=sentence-transformers

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from agentic_framework import AgenticAI; AgenticAI()" || exit 1

# Default command
CMD ["python", "-m", "agentic_framework"]
