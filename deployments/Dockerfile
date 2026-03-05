# Use a slim Python 3.11 image for a smaller footprint
FROM python:3.11-slim

# Set environment variables for better Python logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies for ONNX, Postgres (psycopg2), and compilation
# libgomp1 is required for ONNX/XGBoost inference
# libpq-dev, gcc, and python3-dev are required for building psycopg2-binary in some environments
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
# IMPORTANT: Ensure your .dockerignore does NOT exclude the 'models' folder 
# so that user_tower_production.onnx, reranker_xgb.pkl, and mappings.pkl are copied.
COPY . .

# Ensure models and logs directories exist
# We also create a symlink if necessary to ensure the API always finds 
# 'user_tower_production.onnx' regardless of the training script's naming.
RUN mkdir -p /app/models /app/logs && \
    if [ -f /app/models/user_tower_quantized.onnx ] && [ ! -f /app/models/user_tower_production.onnx ]; then \
        ln -s /app/models/user_tower_quantized.onnx /app/models/user_tower_production.onnx; \
    fi

# Make the startup script executable
RUN chmod +x /app/deployments/start.sh

# Expose the API port (Render uses $PORT, but we expose 8000 as a default)
EXPOSE 8000

# Execute through the startup script
# The start.sh should be updated to use --workers 1 to avoid OOM on Render.
CMD ["/bin/bash", "/app/deployments/start.sh"]