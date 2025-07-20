# Simplified Backend for Railway
# Simple FastAPI backend that works with Railway's Python environment

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import json
import random

app = FastAPI(title="Search Engine API")

# CORS for all origins (Railway deployment)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SearchResult(BaseModel):
    id: str
    title: str
    url: str
    description: str
    score: float
    category: str
    timestamp: str
    snippet: str

class SearchResponse(BaseModel):
    query: str
    total_results: int
    page: int
    limit: int
    results: List[SearchResult]
    search_time: float

@app.get("/")
async def root():
    return {"message": "Search Engine API is running!"}

@app.get("/health")
async def health():
    return {"status": "healthy", "message": "API is working"}

@app.get("/api/search")
async def search(
    q: str = Query(..., description="Search query"),
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100)
):
    """Simple search endpoint that returns mock results"""
    
    results = []
    for i in range(1, limit + 1):
        result = SearchResult(
            id=f"result_{page}_{i}",
            title=f"Search result for '{q}' - Result {i}",
            url=f"https://example.com/result-{i}",
            description=f"This is a relevant result for your search query: {q}. This demonstrates the search functionality.",
            score=round(random.uniform(0.5, 1.0), 2),
            category="general",
            timestamp="2025-07-21T00:00:00Z",
            snippet=f"Snippet for {q} result {i}"
        )
        results.append(result)
    
    return SearchResponse(
        query=q,
        total_results=random.randint(1000, 50000),
        page=page,
        limit=limit,
        results=results,
        search_time=round(random.uniform(0.01, 0.1), 3)
    )

@app.get("/api/suggestions")
async def get_suggestions(q: str = Query(..., description="Query for suggestions")):
    """Get search suggestions"""
    suggestions = [
        f"{q} tutorial",
        f"{q} guide",
        f"{q} examples",
        f"{q} documentation",
        f"what is {q}"
    ]
    return {"suggestions": suggestions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
