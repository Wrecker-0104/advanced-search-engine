"""
Search schemas for request/response validation
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class SearchRequest(BaseModel):
    """Search request schema"""
    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    page: int = Field(1, ge=1, description="Page number")
    limit: int = Field(10, ge=1, le=100, description="Results per page")
    category: Optional[str] = Field(None, description="Category filter")
    date_range: Optional[str] = Field(None, description="Date range filter")
    domain: Optional[str] = Field(None, description="Domain filter")
    sort_by: str = Field("relevance", description="Sort by: relevance, date, popularity")
    safe_search: bool = Field(True, description="Enable safe search")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "machine learning algorithms",
                "page": 1,
                "limit": 10,
                "category": "technology",
                "sort_by": "relevance",
                "safe_search": True
            }
        }


class SearchResult(BaseModel):
    """Individual search result"""
    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Document title")
    url: str = Field(..., description="Document URL")
    description: str = Field(..., description="Document description")
    snippet: str = Field(..., description="Text snippet")
    score: float = Field(..., description="Relevance score")
    category: str = Field(..., description="Document category")
    timestamp: str = Field(..., description="Document timestamp")


class SearchResponse(BaseModel):
    """Search response schema"""
    results: List[SearchResult] = Field(..., description="Search results")
    totalResults: int = Field(..., description="Total number of results")
    searchTime: float = Field(..., description="Search time in seconds")
    suggestions: List[str] = Field(default=[], description="Query suggestions")
    facets: Dict[str, Dict[str, int]] = Field(default={}, description="Search facets")
    
    class Config:
        schema_extra = {
            "example": {
                "results": [
                    {
                        "id": "1",
                        "title": "Introduction to Machine Learning",
                        "url": "https://example.com/ml-intro",
                        "description": "A comprehensive guide to machine learning",
                        "snippet": "Machine learning is a subset of artificial intelligence...",
                        "score": 0.95,
                        "category": "technology",
                        "timestamp": "2024-01-15T10:30:00"
                    }
                ],
                "totalResults": 150,
                "searchTime": 0.045,
                "suggestions": ["machine learning tutorial", "machine learning guide"],
                "facets": {
                    "categories": {"technology": 120, "academic": 30},
                    "domains": {"example.com": 50, "tech.org": 30}
                }
            }
        }


class SearchSuggestion(BaseModel):
    """Search suggestion schema"""
    query: str = Field(..., description="Suggested query")
    count: int = Field(..., description="Popularity count")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "machine learning tutorial",
                "count": 1250
            }
        }


class SearchAnalytics(BaseModel):
    """Search analytics schema"""
    total_searches: int = Field(..., description="Total number of searches")
    unique_queries: int = Field(..., description="Number of unique queries")
    avg_results_clicked: float = Field(..., description="Average results clicked per search")
    most_searched_categories: List[Dict[str, Any]] = Field(..., description="Popular categories")
    search_trends: List[Dict[str, Any]] = Field(..., description="Search trends over time")


class FeedbackRequest(BaseModel):
    """Feedback request schema"""
    query: str = Field(..., description="Original search query")
    result_id: str = Field(..., description="Result document ID")
    feedback_type: str = Field(..., description="Feedback type")
    
    class Config:
        schema_extra = {
            "example": {
                "query": "machine learning",
                "result_id": "doc123",
                "feedback_type": "relevant"
            }
        }
