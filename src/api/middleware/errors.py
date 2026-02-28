from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import logging

logger = logging.getLogger("api-errors")

class GlobalExceptionHandlerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            logger.error(f"🔥 Global Error: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=500,
                content={
                    "error": "InternalServerException",
                    "message": "The recommendation engine encountered an unexpected error.",
                    "details": str(e) if "DATABASE" in str(e).upper() else "Contact support."
                }
            )