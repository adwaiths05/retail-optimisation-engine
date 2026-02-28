from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, HTTPException
import time

def add_security_headers(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with your specific domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

class SimpleRateLimiter:
    """Basic in-memory rate limiter to prevent bot spamming."""
    def __init__(self, requests_limit: int = 100, window_seconds: int = 60):
        self.limit = requests_limit
        self.window = window_seconds
        self.client_history = {}

    async def check_limit(self, client_ip: str):
        now = time.time()
        if client_ip not in self.client_history:
            self.client_history[client_ip] = [now]
            return
        
        # Filter timestamps within the current window
        self.client_history[client_ip] = [t for t in self.client_history[client_ip] if now - t < self.window]
        
        if len(self.client_history[client_ip]) >= self.limit:
            raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")
        
        self.client_history[client_ip].append(now)