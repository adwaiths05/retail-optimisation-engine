from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
import os
from pathlib import Path

# Get the absolute path of the project root
BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # --- Database ---
    DATABASE_URL: str = Field(..., alias="DATABASE_URL")
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # --- Model Config ---
    MODEL_PATH: str = str(BASE_DIR / "models" / "user_tower_quantized.onnx")
    RERANKER_PATH: str = str(BASE_DIR / "models" / "reranker_xgb.pkl")
    MAPPINGS_PATH: str = str(BASE_DIR / "models" / "mappings.pkl")
    EMBEDDING_DIM: int = 64
    
    # --- JWT Authentication Config ---
    # These must exist for auth.py to work
    SECRET_KEY: str = Field("default_secret_for_dev_only", alias="SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    
    # --- Business Logic Defaults ---
    DEFAULT_TOP_K: int = 20
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Global instance to be imported by other files
settings = Settings()