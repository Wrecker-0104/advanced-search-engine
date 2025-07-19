"""
Search API endpoints
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Request, File, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
import json
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.core.database import get_db, cache_manager
from app.core.security import get_current_active_user, sanitize_input, validate_search_query
from app.core.config import settings
from app.models.user import User
from app.services.search_service import SearchService
from app.schemas.search import SearchRequest, SearchResponse, SearchSuggestion

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)

# Initialize search service
search_service = SearchService()


@router.get("/search")
@limiter.limit(settings.RATE_LIMIT_SEARCH)
async def search_documents(
    request: Request,
    q: str = Query(..., description="Search query", min_length=1, max_length=500),
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=settings.MAX_RESULTS_PER_PAGE, description="Results per page"),
    category: str = Query("all", description="Category filter"),
    date_range: str = Query("anytime", description="Date range filter"),
    domain: str = Query("", description="Domain filter"),
    sort_by: str = Query("relevance", description="Sort by: relevance, date, popularity"),
    safe_search: bool = Query(True, description="Enable safe search"),
    db: Session = Depends(get_db),
    current_user: Optional[User] = None
):
    """
    Search documents with advanced filtering and ranking
    
    - **q**: Search query (required)
    - **page**: Page number (default: 1)
    - **limit**: Results per page (default: 10, max: 100)
    - **category**: Filter by category (all, news, academic, commercial, etc.)
    - **date_range**: Filter by date (anytime, hour, day, week, month, year)
    - **domain**: Filter by domain (e.g., wikipedia.org)
    - **sort_by**: Sort results by relevance, date, or popularity
    - **safe_search**: Enable safe search filtering
    """
    
    # Validate and sanitize query
    if not validate_search_query(q):
        raise HTTPException(status_code=400, detail="Invalid search query")
    
    sanitized_query = sanitize_input(q)
    
    try:
        # Create cache key
        cache_key = f"search:{hash(f'{sanitized_query}:{page}:{limit}:{category}:{date_range}:{domain}:{sort_by}:{safe_search}')}"
        
        # Try to get from cache first
        cached_result = await cache_manager.get(cache_key)
        if cached_result:
            return json.loads(cached_result)
        
        # Perform search
        search_params = {
            "query": sanitized_query,
            "page": page,
            "limit": limit,
            "category": category if category != "all" else None,
            "date_range": date_range if date_range != "anytime" else None,
            "domain": domain if domain else None,
            "sort_by": sort_by,
            "safe_search": safe_search,
            "user_id": current_user.id if current_user else None
        }
        
        search_results = await search_service.search(**search_params)
        
        # Cache results for 5 minutes
        await cache_manager.set(cache_key, json.dumps(search_results, default=str), expire=300)
        
        # Log search analytics (if user is authenticated)
        if current_user:
            await search_service.log_search_analytics(
                user_id=current_user.id,
                query=sanitized_query,
                results_count=len(search_results.get("results", [])),
                page=page
            )
        
        return search_results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/suggestions")
@limiter.limit("50/minute")
async def get_search_suggestions(
    request: Request,
    q: str = Query(..., description="Partial query for suggestions", min_length=2),
    limit: int = Query(5, ge=1, le=10, description="Number of suggestions")
):
    """
    Get search suggestions based on partial query
    
    - **q**: Partial search query (minimum 2 characters)
    - **limit**: Number of suggestions to return (max: 10)
    """
    
    # Validate and sanitize query
    if len(q.strip()) < 2:
        return {"suggestions": []}
    
    sanitized_query = sanitize_input(q)
    
    try:
        # Create cache key for suggestions
        cache_key = f"suggestions:{hash(sanitized_query)}:{limit}"
        
        # Try cache first
        cached_suggestions = await cache_manager.get(cache_key)
        if cached_suggestions:
            return json.loads(cached_suggestions)
        
        # Get suggestions from search service
        suggestions = await search_service.get_suggestions(sanitized_query, limit)
        
        result = {"suggestions": suggestions}
        
        # Cache for 1 hour
        await cache_manager.set(cache_key, json.dumps(result), expire=3600)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get suggestions: {str(e)}")


@router.get("/popular")
@limiter.limit("20/minute")
async def get_popular_searches(
    request: Request,
    limit: int = Query(10, ge=1, le=20, description="Number of popular searches"),
    timeframe: str = Query("day", description="Timeframe: hour, day, week, month")
):
    """
    Get popular search queries
    
    - **limit**: Number of popular searches to return
    - **timeframe**: Time period for popularity calculation
    """
    
    try:
        cache_key = f"popular:{timeframe}:{limit}"
        
        # Try cache first
        cached_popular = await cache_manager.get(cache_key)
        if cached_popular:
            return json.loads(cached_popular)
        
        # Get popular searches
        popular_searches = await search_service.get_popular_searches(limit, timeframe)
        
        result = {"popular_searches": popular_searches}
        
        # Cache for 30 minutes
        await cache_manager.set(cache_key, json.dumps(result), expire=1800)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get popular searches: {str(e)}")


@router.get("/categories")
async def get_search_categories():
    """Get available search categories"""
    
    categories = [
        {"id": "all", "name": "All Categories", "description": "Search across all content"},
        {"id": "news", "name": "News", "description": "Latest news articles"},
        {"id": "academic", "name": "Academic", "description": "Research papers and academic content"},
        {"id": "commercial", "name": "Commercial", "description": "Business and commercial websites"},
        {"id": "entertainment", "name": "Entertainment", "description": "Movies, music, games, and entertainment"},
        {"id": "technology", "name": "Technology", "description": "Tech news, tutorials, and resources"},
        {"id": "science", "name": "Science", "description": "Scientific articles and research"},
        {"id": "health", "name": "Health", "description": "Medical and health information"},
        {"id": "sports", "name": "Sports", "description": "Sports news and information"},
    ]
    
    return {"categories": categories}


@router.post("/feedback")
@limiter.limit("10/minute")
async def submit_search_feedback(
    request: Request,
    query: str,
    result_id: str,
    feedback_type: str,  # "relevant", "irrelevant", "spam", "broken_link"
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
):
    """
    Submit feedback on search results to improve ranking
    
    - **query**: Original search query
    - **result_id**: ID of the search result
    - **feedback_type**: Type of feedback (relevant, irrelevant, spam, broken_link)
    """
    
    valid_feedback_types = ["relevant", "irrelevant", "spam", "broken_link"]
    if feedback_type not in valid_feedback_types:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid feedback type. Must be one of: {valid_feedback_types}"
        )
    
    try:
        # Log feedback for machine learning improvements
        await search_service.log_feedback(
            user_id=current_user.id,
            query=sanitize_input(query),
            result_id=result_id,
            feedback_type=feedback_type
        )
        
        return {"message": "Feedback submitted successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to submit feedback: {str(e)}")


@router.get("/analytics")
async def get_search_analytics(
    current_user: User = Depends(get_current_active_user),
    days: int = Query(30, ge=1, le=365, description="Number of days to analyze")
):
    """
    Get search analytics for the current user
    
    - **days**: Number of days to include in analytics
    """
    
    try:
        analytics = await search_service.get_user_analytics(current_user.id, days)
        return analytics
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get analytics: {str(e)}")


# Advanced search endpoint for complex queries
@router.post("/advanced")
@limiter.limit("30/minute")
async def advanced_search(
    request: Request,
    search_request: SearchRequest,
    db: Session = Depends(get_db),
    current_user: Optional[User] = None
):
    """
    Advanced search with complex query parameters
    
    Supports:
    - Boolean operators (AND, OR, NOT)
    - Phrase matching with quotes
    - Field-specific searches
    - Date range filtering
    - Multiple category filtering
    """
    
    try:
        # Validate complex query
        if not validate_search_query(search_request.query):
            raise HTTPException(status_code=400, detail="Invalid search query")
        
        # Perform advanced search
        results = await search_service.advanced_search(
            search_request=search_request,
            user_id=current_user.id if current_user else None
        )
        
        return results
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Advanced search failed: {str(e)}")


@router.post("/search/image")
@limiter.limit("10/minute")
async def search_by_image(
    request: Request,
    image: UploadFile = File(..., description="Image file for reverse image search"),
    current_user: Optional[User] = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    """
    Reverse image search endpoint
    
    Upload an image to find similar or related content
    """
    
    try:
        # Validate image file
        if not image.content_type or not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Check file size (max 10MB)
        if image.size and image.size > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
        
        # Read image content
        image_content = await image.read()
        
        # For demo purposes, we'll simulate image search results
        # In a real implementation, you would:
        # 1. Extract features from the image using computer vision
        # 2. Compare with indexed images
        # 3. Return similar images or related content
        
        # Simulated results based on image filename or random content
        mock_results = {
            "results": [
                {
                    "id": "img_search_1",
                    "title": f"Similar to {image.filename}",
                    "url": f"https://example.com/similar-image-1",
                    "description": f"Content related to uploaded image: {image.filename}",
                    "score": 0.95,
                    "category": "images",
                    "timestamp": "2024-01-20T10:00:00Z",
                    "snippet": "This content was found using reverse image search technology.",
                    "image_url": "https://example.com/thumbnail1.jpg"
                },
                {
                    "id": "img_search_2", 
                    "title": "Related Visual Content",
                    "url": "https://example.com/related-content",
                    "description": "Visually similar content found in our database",
                    "score": 0.87,
                    "category": "images",
                    "timestamp": "2024-01-19T15:30:00Z",
                    "snippet": "Advanced image recognition found this related content.",
                    "image_url": "https://example.com/thumbnail2.jpg"
                },
                {
                    "id": "img_search_3",
                    "title": "Image Context Analysis",
                    "url": "https://example.com/context-analysis", 
                    "description": "Content based on image context and objects detected",
                    "score": 0.76,
                    "category": "technology",
                    "timestamp": "2024-01-18T09:15:00Z",
                    "snippet": "Our AI detected objects and context in your image to find relevant information.",
                    "image_url": "https://example.com/thumbnail3.jpg"
                }
            ],
            "totalResults": 3,
            "searchTime": 0.85,
            "suggestions": [
                "reverse image search",
                "similar images", 
                "visual search",
                "image recognition"
            ],
            "facets": {
                "categories": {"images": 2, "technology": 1},
                "domains": {"example.com": 3}
            },
            "searchType": "image",
            "originalImageName": image.filename
        }
        
        return mock_results
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image search failed: {str(e)}")


@router.get("/search/visual")
@limiter.limit("20/minute") 
async def visual_search_suggestions(
    request: Request,
    q: str = Query(..., description="Query for visual content suggestions", min_length=2),
    limit: int = Query(8, ge=1, le=20, description="Number of visual suggestions")
):
    """
    Get visual search suggestions and related images
    """
    
    try:
        sanitized_query = sanitize_input(q)
        
        # Mock visual suggestions
        visual_suggestions = {
            "suggestions": [
                {"text": f"{sanitized_query} images", "type": "image"},
                {"text": f"{sanitized_query} photos", "type": "photo"},
                {"text": f"{sanitized_query} pictures", "type": "picture"},
                {"text": f"{sanitized_query} gallery", "type": "gallery"},
                {"text": f"{sanitized_query} visual", "type": "visual"},
                {"text": f"high quality {sanitized_query}", "type": "quality"},
                {"text": f"{sanitized_query} HD", "type": "hd"},
                {"text": f"{sanitized_query} wallpaper", "type": "wallpaper"}
            ][:limit],
            "relatedImages": [
                {
                    "url": f"https://example.com/img1_{sanitized_query.replace(' ', '_')}.jpg",
                    "title": f"Sample {sanitized_query} #1",
                    "source": "example.com"
                },
                {
                    "url": f"https://example.com/img2_{sanitized_query.replace(' ', '_')}.jpg", 
                    "title": f"Sample {sanitized_query} #2",
                    "source": "example.com"
                }
            ]
        }
        
        return visual_suggestions
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Visual search suggestions failed: {str(e)}")
