#!/bin/bash
set -e

echo "Running database migrations..."
alembic upgrade head

# ADD THIS LINE:
echo "Ensuring pgvector extension and HNSW index exist..."
python db/init_db.py

echo "Starting FastAPI server..."
uvicorn src.api.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 4