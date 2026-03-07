# ---------- STAGE 1: Builder ----------
    FROM python:3.11-slim AS builder
    WORKDIR /app
    
    RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc python3-dev libpq-dev && rm -rf /var/lib/apt/lists/*
    
    COPY requirements-prod.txt .
    RUN pip install --upgrade pip && \
        pip install --user --no-cache-dir -r requirements-prod.txt
    
    # Aggressive cleanup of site-packages to keep image size <500MB
    RUN find /root/.local/lib/python3.11/site-packages -name "tests" -type d -exec rm -rf {} + && \
        find /root/.local/lib/python3.11/site-packages -name "__pycache__" -type d -exec rm -rf {} + && \
        find /root/.local/lib/python3.11/site-packages -name "*.pyc" -delete
    
    # ---------- STAGE 2: Runtime ----------
    FROM python:3.11-slim
    WORKDIR /app
    
    # Runtime dependencies (libgomp1 is critical for ONNX performance)
    RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libpq5 && rm -rf /var/lib/apt/lists/*
    
    COPY --from=builder /root/.local /root/.local
    
    # Environment variables for Python and Pathing
    ENV PATH=/root/.local/bin:$PATH
    ENV PYTHONPATH=/app
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1
    
    # --- COPY ALL PROJECT DIRECTORIES ---
    COPY src/ ./src/
    COPY mlops/ ./mlops/
    COPY db/ ./db/
    COPY alembic/ ./alembic/
    COPY alembic.ini .
    
    # Models: Copy only what's necessary to save space
    COPY models/user_tower_production.onnx ./models/
    COPY models/reranker.onnx ./models/ 
    COPY models/mappings.pkl ./models/
    COPY models/metadata.json ./models/ 
    
    
    COPY deployments/start.sh ./deployments/
    RUN mkdir -p /app/logs /app/models && chmod +x /app/deployments/start.sh
    
    EXPOSE 8000
    CMD ["/bin/bash", "/app/deployments/start.sh"]