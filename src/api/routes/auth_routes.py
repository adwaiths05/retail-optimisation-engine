from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from src.api.middleware.auth import create_access_token

router = APIRouter()

# Schema for the login request (Swagger uses this to build the input boxes)
class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/token")
async def login(request: LoginRequest):
    """
    Public Endpoint: Authenticates the user.
    Demo credentials: admin / retail-opt-2026
    """
    # In a real backend role, you'd check a database here.
    # Hardcoding is fine for a portfolio demo to keep it 'self-contained'.
    if request.username == "admin" and request.password == "retail-opt-2026":
        access_token = create_access_token(data={"sub": request.username, "role": "admin"})
    elif request.username == "viewer" and request.password == "retail-view":
        access_token = create_access_token(data={"sub": request.username, "role": "viewer"})
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    return {"access_token": access_token, "token_type": "bearer"}
    