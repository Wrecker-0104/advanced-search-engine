"""
Database configuration and session management
"""

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
import redis
from typing import Generator

from app.core.config import settings

# SQLAlchemy setup
engine = create_engine(
    settings.DATABASE_URL,
    poolclass=StaticPool if "sqlite" in settings.DATABASE_URL else None,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
    pool_pre_ping=True,
    pool_recycle=300,
    echo=settings.DEBUG
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for SQLAlchemy models
Base = declarative_base()

# Redis setup
try:
    redis_client = redis.Redis.from_url(
        settings.REDIS_URL,
        decode_responses=True,
        health_check_interval=30
    )
    # Test connection
    redis_client.ping()
except Exception as e:
    print(f"Warning: Redis connection failed: {e}")
    redis_client = None


def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_redis() -> redis.Redis:
    """Get Redis client"""
    if redis_client is None:
        raise Exception("Redis client not available")
    return redis_client


async def init_db():
    """Initialize database tables"""
    # Import models to register them
    from app.models import user, search_history, document, search_analytics
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    
    # Create default admin user if not exists
    db = SessionLocal()
    try:
        from app.models.user import User
        from app.core.security import get_password_hash
        
        admin_user = db.query(User).filter(User.username == "admin").first()
        if not admin_user:
            admin_user = User(
                username="admin",
                email="admin@searchengine.com",
                hashed_password=get_password_hash("admin123"),
                is_admin=True,
                is_active=True
            )
            db.add(admin_user)
            db.commit()
            print("Created default admin user (admin/admin123)")
    
    except Exception as e:
        print(f"Error creating admin user: {e}")
    finally:
        db.close()


# Database utilities
class DatabaseManager:
    """Database management utilities"""
    
    @staticmethod
    def create_tables():
        """Create all database tables"""
        Base.metadata.create_all(bind=engine)
    
    @staticmethod
    def drop_tables():
        """Drop all database tables"""
        Base.metadata.drop_all(bind=engine)
    
    @staticmethod
    def get_session():
        """Get a new database session"""
        return SessionLocal()


# Cache utilities
class CacheManager:
    """Cache management utilities"""
    
    def __init__(self):
        self.redis = get_redis() if redis_client else None
    
    async def get(self, key: str):
        """Get value from cache"""
        if not self.redis:
            return None
        try:
            return self.redis.get(key)
        except Exception:
            return None
    
    async def set(self, key: str, value: str, expire: int = None):
        """Set value in cache"""
        if not self.redis:
            return False
        try:
            if expire:
                return self.redis.setex(key, expire, value)
            else:
                return self.redis.set(key, value)
        except Exception:
            return False
    
    async def delete(self, key: str):
        """Delete key from cache"""
        if not self.redis:
            return False
        try:
            return self.redis.delete(key)
        except Exception:
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.redis:
            return False
        try:
            return self.redis.exists(key)
        except Exception:
            return False


# Global cache manager instance
cache_manager = CacheManager()
