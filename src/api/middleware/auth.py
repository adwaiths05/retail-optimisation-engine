from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Security, status, Depends
from jose import jwt, JWTError
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from src.core.config import settings

# This allows Swagger UI to show the 'Authorize' lock icon
security = HTTPBearer()

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Generates a JWT token for the user."""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

async def get_current_user(auth: HTTPAuthorizationCredentials = Security(security)):
    """
    The Global Guard: Validates the token.
    If valid, returns the user info. If not, raises 401.
    """
    try:
        payload = jwt.decode(auth.credentials, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload  # Return the full payload (includes 'sub' and 'role')
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def role_required(required_role: str):
    """Dependency factory to restrict routes by role."""
    def role_checker(payload: dict = Depends(get_current_user)):
        user_role = payload.get("role")
        if user_role != required_role and user_role != "admin":
            raise HTTPException(
                status_code=403, 
                detail=f"Operation restricted to {required_role} only"
            )
        return payload
    return role_checker