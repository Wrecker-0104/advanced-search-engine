"""
Configuration settings for the Advanced Search Engine Backend
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Advanced Search Engine"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Configuration
    API_V1_STR: str = "/api"
    
    # Security
    SECRET_KEY: str = "your-super-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000"
    ]
    ALLOWED_HOSTS: List[str] = ["*"]
    
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/searchengine"
    DATABASE_HOST: str = "localhost"
    DATABASE_PORT: int = 5432
    DATABASE_USER: str = "searchengine_user"
    DATABASE_PASSWORD: str = "searchengine_password"
    DATABASE_NAME: str = "searchengine_db"
    
    # Redis Cache
    REDIS_URL: str = "redis://localhost:6379"
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    CACHE_EXPIRE_SECONDS: int = 3600
    
    # Elasticsearch
    ELASTICSEARCH_URL: str = "http://localhost:9200"
    ELASTICSEARCH_INDEX: str = "search_documents"
    ELASTICSEARCH_USER: Optional[str] = None
    ELASTICSEARCH_PASSWORD: Optional[str] = None
    
    # Search Engine Settings
    MAX_SEARCH_RESULTS: int = 1000
    DEFAULT_RESULTS_PER_PAGE: int = 10
    MAX_RESULTS_PER_PAGE: int = 100
    
    # Crawler Settings
    CRAWL_DELAY: float = 1.0
    MAX_CRAWL_DEPTH: int = 5
    RESPECT_ROBOTS_TXT: bool = True
    USER_AGENT: str = "AdvancedSearchBot/1.0"
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: str = "100/minute"
    RATE_LIMIT_SEARCH: str = "50/minute"
    RATE_LIMIT_AUTH: str = "10/minute"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "search_engine.log"
    
    # File Upload
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_FILE_TYPES: List[str] = [".pdf", ".doc", ".docx", ".txt", ".html"]
    
    # NLP and ML Settings
    SENTENCE_TRANSFORMER_MODEL: str = "all-MiniLM-L6-v2"
    SPACY_MODEL: str = "en_core_web_sm"
    
    # Monitoring
    ENABLE_PROMETHEUS: bool = True
    PROMETHEUS_PORT: int = 8001
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()
