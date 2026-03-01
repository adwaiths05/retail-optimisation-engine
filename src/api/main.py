from fastapi import FastAPI, Depends
from src.api.routes import (
    recommendations, events, pricing, models, 
    experiments, metrics, auth_routes  # Added auth_routes
)
from src.api.middleware.logging import LatencyLoggingMiddleware
from src.api.middleware.errors import GlobalExceptionHandlerMiddleware
from src.api.middleware.security import add_security_headers, rate_limiter
from src.api.middleware.auth import get_current_user, role_required # Added JWT dependency
from src.core.config import settings

app = FastAPI(
    title="Retail Optimisation Engine",
    description="Business-aware, margin-optimized recommendation API",
    version="1.0.0"
)

# 1. Security Layer (CORS & Headers)
add_security_headers(app)

# 2. Error Handling Layer
app.add_middleware(GlobalExceptionHandlerMiddleware)

# 3. Benchmarking Middleware
app.add_middleware(LatencyLoggingMiddleware)

# --- 4. PUBLIC ROUTES (No Auth Required) ---
# These are used by your Landing Page and Login Page
app.include_router(auth_routes.router, prefix="/api/v1/auth", tags=["auth"])

@app.get("/health",dependencies=[Depends(rate_limiter)], tags=["system"])
async def health_check():
    """Public health check for deployment & landing page."""
    return {
        "status": "healthy",
        "onnx_model": settings.MODEL_PATH,
        "database": "connected"
    }

viewer_deps = [Depends(get_current_user), Depends(rate_limiter)]

# Admin Guard: Must be an admin AND stay under the rate limit
admin_deps = [Depends(role_required("admin")), Depends(rate_limiter)]

# 2. Apply to Routers
# General Access
app.include_router(recommendations.router, prefix="/api/v1", dependencies=viewer_deps)
app.include_router(experiments.router, prefix="/api/v1/experiments", dependencies=viewer_deps)
app.include_router(events.router, prefix="/api/v1", dependencies=viewer_deps)

# Restricted Access
app.include_router(pricing.router, prefix="/api/v1/pricing", dependencies=admin_deps)
app.include_router(models.router, prefix="/api/v1/models", dependencies=admin_deps)
app.include_router(metrics.router, prefix="/api/v1/metrics", dependencies=admin_deps)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)