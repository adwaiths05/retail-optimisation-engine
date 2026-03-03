from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI

def setup_monitoring(app: FastAPI):
    # Instruments FastAPI to export standard metrics (latency, request count)
    # Accessible at http://localhost:8000/metrics
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")