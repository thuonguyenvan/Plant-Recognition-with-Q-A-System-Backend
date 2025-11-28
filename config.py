"""
Configuration management for Plant Medicine RAG Backend
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Supabase
    supabase_url: str
    supabase_anon_key: str
    supabase_db_uri: str = ""  # Optional: for direct DB connection in scripts
    
    # MegLLM
    megllm_api_key: str = ""
    
    # Embedding Model
    embedding_model_name: str = "AITeamVN/Vietnamese_Embedding"
    embedding_dimension: int = 1024  # Actual dimension from model
    
    # CV API
    cv_api_url: str = "https://thuonguyenvan-plantsclassify.hf.space"
    cv_api_timeout: int = 60
    
    # Data paths
    data_dir: str = "data"
    photos_dir: str = "inat_representative_photos"
    
    # RAG settings
    chunk_max_tokens: int = 250
    retrieval_top_k: int = 20
    
    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
