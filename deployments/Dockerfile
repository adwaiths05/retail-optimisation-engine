# Use a slim Python 3.11 image
FROM python:3.11-slim

# Set working directory to /app
WORKDIR /app

# Install system dependencies for ONNX and Postgres
RUN apt-get update && apt-get install -y \
    libgomp1 \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from root (Docker context will be root)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into the container
COPY . .

# Ensure the models directory exists for the ONNX file
RUN mkdir -p models

# Set Python path to find the 'src' and 'api' modules
ENV PYTHONPATH=/app

# Expose FastAPI port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]