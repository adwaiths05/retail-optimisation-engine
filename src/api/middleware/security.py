# src/api/middleware/security.py
from fastapi import Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import time

def add_security_headers(app):
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"], 
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

class SimpleRateLimiter:
    def __init__(self, requests_limit: int = 50, window_seconds: int = 60):
        self.limit = requests_limit
        self.window = window_seconds
        self.client_history = {}

    # This __call__ makes the class usable as a FastAPI dependency
    async def __call__(self, request: Request):
        client_ip = request.client.host
        now = time.time()
        
        if client_ip not in self.client_history:
            self.client_history[client_ip] = []
        
        # Cleanup old timestamps
        self.client_history[client_ip] = [t for t in self.client_history[client_ip] if now - t < self.window]
        
        if len(self.client_history[client_ip]) >= self.limit:
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Slow down, partner.")
        
        self.client_history[client_ip].append(now)

# Create one instance to be used across the app
rate_limiter = SimpleRateLimiter(requests_limit=50, window_seconds=60)