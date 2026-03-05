# ---------- STAGE 1: Builder ----------
    FROM python:3.11-slim AS builder

    WORKDIR /app
    
    # Install build dependencies
    RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        python3-dev \
        libpq-dev \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy lean requirements
    COPY requirements-prod.txt .
    
    # Install Python dependencies to user directory
    RUN pip install --upgrade pip && \
        pip install --user --no-cache-dir -r requirements-prod.txt
    
    # Aggressive cleanup of standard library bloat
    RUN find /root/.local/lib/python3.11/site-packages -name "tests" -type d -exec rm -rf {} + && \
        find /root/.local/lib/python3.11/site-packages -name "__pycache__" -type d -exec rm -rf {} + && \
        find /root/.local/lib/python3.11/site-packages -name "*.pyc" -delete
    
    # ---------- STAGE 2: Runtime ----------
    FROM python:3.11-slim
    
    WORKDIR /app
    
    # Runtime system dependencies only (libgomp1 is still needed for ONNX)
    RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        libpq5 \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy installed packages from builder
    COPY --from=builder /root/.local /root/.local
    
    # Environment variables
    ENV PATH=/root/.local/bin:$PATH
    ENV PYTHONPATH=/app
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1
    
    # Copy only required project files
    COPY src/ ./src/
    COPY models/user_tower_production.onnx ./models/
    # FIX: Copy the new ONNX model instead of the .pkl
    COPY models/reranker.onnx ./models/ 
    COPY models/mappings.pkl ./models/
    COPY deployments/start.sh ./deployments/
    COPY alembic/ ./alembic/
    COPY alembic.ini .
    
    # Create necessary directories
    RUN mkdir -p /app/logs /app/models
    
    # Make startup script executable
    RUN chmod +x /app/deployments/start.sh
    
    EXPOSE 8000
    
    # Start the application
    CMD ["/bin/bash", "/app/deployments/start.sh"]