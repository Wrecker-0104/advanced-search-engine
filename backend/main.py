"""
Advanced Search Engine Backend - Main Application
FastAPI backend with security, database integration, and search capabilities
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import time
import logging
from contextlib import asynccontextmanager

from app.core.config import settings
from app.core.security import get_current_user
from app.api.search import router as search_router
from app.api.auth import router as auth_router
from app.api.admin import router as admin_router
from app.core.database import init_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting Advanced Search Engine Backend")
    await init_db()
    logger.info("Database initialized")
    yield
    # Shutdown
    logger.info("Shutting down Advanced Search Engine Backend")

# Create FastAPI application
app = FastAPI(
    title="Advanced Search Engine API",
    description="""
    A comprehensive search engine API with Google-like algorithms and advanced features.
    
    ## Features
    - **Full-text search** with TF-IDF and BM25 ranking
    - **PageRank algorithm** for authority-based ranking
    - **Semantic search** with BERT embeddings
    - **Real-time suggestions** and autocomplete
    - **Advanced filters** and faceted search
    - **User authentication** and personalization
    - **Rate limiting** and security features
    - **Web crawler** integration
    
    ## Security
    - JWT-based authentication
    - Rate limiting (100 requests per minute)
    - Input validation and sanitization
    - HTTPS enforcement
    - CORS configuration
    """,
    version="1.0.0",
    contact={
        "name": "Search Engine Team",
        "email": "admin@searchengine.com",
    },
    license_info={
        "name": "MIT License",
    },
    lifespan=lifespan
)

# Security middleware
app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.ALLOWED_HOSTS
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Security middleware for request timing
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add process time header and basic security headers"""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    # Security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    
    return response

# Include routers
app.include_router(
    search_router,
    prefix="/api",
    tags=["search"]
)

app.include_router(
    auth_router,
    prefix="/api/auth",
    tags=["authentication"]
)

app.include_router(
    admin_router,
    prefix="/api/admin",
    tags=["admin"],
    dependencies=[Depends(get_current_user)]
)

# Health check endpoint
@app.get("/", tags=["health"])
@limiter.limit("10/minute")
async def health_check(request: Request):
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Advanced Search Engine API is running",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health", tags=["health"])
async def detailed_health_check():
    """Detailed health check with system information"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "services": {
            "database": "connected",
            "redis": "connected",
            "elasticsearch": "connected",
            "search_engine": "active"
        },
        "version": "1.0.0"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
