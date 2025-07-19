"""
API Key Authentication Module
Handles API key validation and management
"""

import os
import hashlib
import secrets
from typing import List, Optional, Dict, Any
from fastapi import HTTPException, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime, timedelta
import json

class APIKeyManager:
    """Manages API key authentication and validation"""
    
    def __init__(self):
        self.api_key_required = os.getenv("API_KEY_REQUIRED", "false").lower() == "true"
        self.api_key_header = os.getenv("API_KEY_HEADER", "X-API-Key")
        
        # Load valid API keys from environment
        valid_keys_str = os.getenv("VALID_API_KEYS", "[]")
        try:
            self.valid_api_keys = json.loads(valid_keys_str.replace("'", '"'))
        except json.JSONDecodeError:
            self.valid_api_keys = ["sk-search-engine-2025-demo-key-123456"]
        
        # API key usage tracking
        self.key_usage = {}
        
    def generate_api_key(self, prefix: str = "sk-search") -> str:
        """Generate a new API key with the specified prefix"""
        timestamp = datetime.now().strftime("%Y%m%d")
        random_part = secrets.token_hex(16)
        api_key = f"{prefix}-{timestamp}-{random_part}"
        return api_key
    
    def hash_api_key(self, api_key: str) -> str:
        """Create a secure hash of the API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def is_valid_api_key(self, api_key: str) -> bool:
        """Check if the provided API key is valid"""
        if not self.api_key_required:
            return True
            
        return api_key in self.valid_api_keys
    
    def track_api_key_usage(self, api_key: str, endpoint: str, request_data: Dict[str, Any] = None):
        """Track API key usage for monitoring and analytics"""
        if api_key not in self.key_usage:
            self.key_usage[api_key] = {
                "first_used": datetime.now(),
                "total_requests": 0,
                "endpoints": {},
                "last_used": None
            }
        
        usage = self.key_usage[api_key]
        usage["total_requests"] += 1
        usage["last_used"] = datetime.now()
        
        if endpoint not in usage["endpoints"]:
            usage["endpoints"][endpoint] = 0
        usage["endpoints"][endpoint] += 1
    
    def get_api_key_from_request(self, request: Request) -> Optional[str]:
        """Extract API key from request headers or query parameters"""
        # Check header
        api_key = request.headers.get(self.api_key_header)
        if api_key:
            return api_key
        
        # Check query parameter as fallback
        api_key = request.query_params.get("api_key")
        if api_key:
            return api_key
        
        return None
    
    def get_api_key_info(self, api_key: str) -> Dict[str, Any]:
        """Get information about an API key"""
        if api_key not in self.key_usage:
            return {
                "exists": api_key in self.valid_api_keys,
                "usage": None
            }
        
        usage = self.key_usage[api_key]
        return {
            "exists": True,
            "first_used": usage["first_used"].isoformat(),
            "last_used": usage["last_used"].isoformat() if usage["last_used"] else None,
            "total_requests": usage["total_requests"],
            "endpoints_used": list(usage["endpoints"].keys()),
            "most_used_endpoint": max(usage["endpoints"].items(), key=lambda x: x[1])[0] if usage["endpoints"] else None
        }

# Global API key manager instance
api_key_manager = APIKeyManager()

def verify_api_key(request: Request) -> str:
    """Dependency function to verify API key in requests"""
    api_key = api_key_manager.get_api_key_from_request(request)
    
    if api_key_manager.api_key_required:
        if not api_key:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "API key required",
                    "message": "Please provide a valid API key in the X-API-Key header or api_key query parameter",
                    "demo_key": "sk-search-engine-2025-demo-key-123456"
                }
            )
        
        if not api_key_manager.is_valid_api_key(api_key):
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "Invalid API key",
                    "message": "The provided API key is not valid",
                    "demo_key": "sk-search-engine-2025-demo-key-123456"
                }
            )
    
    return api_key or "no-key-required"

def track_api_usage(api_key: str, endpoint: str, request_data: Dict[str, Any] = None):
    """Track API key usage"""
    api_key_manager.track_api_key_usage(api_key, endpoint, request_data)

# Rate limiting by API key
class APIKeyRateLimit:
    """Rate limiting based on API key"""
    
    def __init__(self):
        self.request_counts = {}
        self.reset_time = datetime.now() + timedelta(minutes=1)
    
    def is_rate_limited(self, api_key: str, limit: int = 100) -> bool:
        """Check if API key has exceeded rate limit"""
        now = datetime.now()
        
        # Reset counters every minute
        if now > self.reset_time:
            self.request_counts = {}
            self.reset_time = now + timedelta(minutes=1)
        
        if api_key not in self.request_counts:
            self.request_counts[api_key] = 0
        
        if self.request_counts[api_key] >= limit:
            return True
        
        self.request_counts[api_key] += 1
        return False

# Global rate limiter
api_rate_limiter = APIKeyRateLimit()

def check_rate_limit(api_key: str = Depends(verify_api_key)) -> str:
    """Dependency to check rate limiting"""
    if api_rate_limiter.is_rate_limited(api_key, limit=100):
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "message": "Too many requests. Please try again later.",
                "limit": "100 requests per minute",
                "reset_time": api_rate_limiter.reset_time.isoformat()
            }
        )
    return api_key
