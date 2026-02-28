import time
import logging
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

# Configure structured logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api-monitor")

class LatencyLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Process the request
        response = await call_next(request)
        
        # Calculate latency in milliseconds
        process_time_ms = (time.time() - start_time) * 1000
        
        # Attach to response headers for benchmarking (Step 12)
        response.headers["X-Latency-MS"] = str(round(process_time_ms, 2))
        
        # Log to console for real-time monitoring
        logger.info(
            f"Method: {request.method} | Path: {request.url.path} | "
            f"Status: {response.status_code} | Latency: {process_time_ms:.2f}ms"
        )
        
        return response