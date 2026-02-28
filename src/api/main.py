from fastapi import FastAPI
from api.routes import recommendations, events, pricing, models, experiments, metrics
from api.middleware.logging import LatencyLoggingMiddleware
from api.middleware.errors import GlobalExceptionHandlerMiddleware  # New
from api.middleware.security import add_security_headers            # New
from src.core.config import settings

app = FastAPI(
    title="Retail Optimisation Engine",
    description="Business-aware, margin-optimized recommendation API",
    version="1.0.0"
)

# 1. Security Layer (CORS & Headers)
# This allows your frontend to talk to the backend
add_security_headers(app)

# 2. Error Handling Layer
# Prevents raw Python crashes from reaching the user
app.add_middleware(GlobalExceptionHandlerMiddleware)

# 3. Benchmarking Middleware (Step 15)
app.add_middleware(LatencyLoggingMiddleware)

# 4. Include Modular Routes (Step 9)
app.include_router(recommendations.router, prefix="/api/v1")
app.include_router(events.router, prefix="/api/v1")
app.include_router(pricing.router, prefix="/api/v1/pricing")
app.include_router(models.router, prefix="/api/v1/models")
app.include_router(experiments.router, prefix="/api/v1/experiments")
app.include_router(metrics.router, prefix="/api/v1/metrics")

@app.get("/health")
async def health_check():
    """Endpoint for deployment platform health checks."""
    return {
        "status": "healthy",
        "onnx_model": settings.MODEL_PATH,
        "database": "connected"
    }

if __name__ == "__main__":
    import uvicorn
    # 0.0.0.0 is used so it can be reached inside a Docker container
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)