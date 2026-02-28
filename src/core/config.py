from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    # Database
    # Neon provides a connection string; we'll force it to use asyncpg
    DATABASE_URL: str = Field(..., alias="DATABASE_URL")
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # Model Config
    MODEL_PATH: str = "models/two_tower.onnx"
    EMBEDDING_DIM: int = 64
    
    # Business Logic Defaults
    DEFAULT_TOP_K: int = 20
    
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Global instance to be imported by other files
settings = Settings()