# Use a slim Python 3.11 image for a smaller footprint
FROM python:3.11-slim

# Set environment variables for better Python logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies for ONNX, Postgres, and compilation
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libpq-dev \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Ensure models and logs directories exist
RUN mkdir -p /app/models /app/logs

# Make the startup script executable
RUN chmod +x /app/deployments/start.sh

# Expose the API port
EXPOSE 8000

# Execute through the startup script
CMD ["/bin/bash", "/app/deployments/start.sh"]