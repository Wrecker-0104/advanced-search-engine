"""
Search service for handling search operations
"""

import time
from typing import List, Dict, Any, Optional
import json

class SearchService:
    """Search service with Google-like algorithms"""
    
    def __init__(self):
        self.documents = self._load_sample_data()
    
    def _load_sample_data(self):
        """Load sample search data for demonstration"""
        return [
            {
                "id": "1",
                "title": "Introduction to Machine Learning",
                "url": "https://example.com/ml-intro",
                "description": "A comprehensive guide to machine learning fundamentals and algorithms.",
                "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms...",
                "category": "technology",
                "timestamp": "2024-01-15T10:30:00",
                "domain": "example.com",
                "score": 0.95
            },
            {
                "id": "2", 
                "title": "Advanced Search Algorithms",
                "url": "https://search-tech.com/algorithms",
                "description": "Deep dive into modern search algorithms including TF-IDF and BM25.",
                "content": "Search algorithms have evolved significantly. TF-IDF and BM25 are fundamental...",
                "category": "academic",
                "timestamp": "2024-01-10T14:20:00",
                "domain": "search-tech.com",
                "score": 0.89
            },
            {
                "id": "3",
                "title": "Web Crawling Best Practices",
                "url": "https://webdev.com/crawling",
                "description": "Learn how to build efficient web crawlers that respect robots.txt.",
                "content": "Web crawling is the process of systematically browsing the web to index content...",
                "category": "technology",
                "timestamp": "2024-01-08T09:15:00",
                "domain": "webdev.com",
                "score": 0.87
            }
        ]
    
    async def search(self, **params) -> Dict[str, Any]:
        """Perform search with various parameters"""
        query = params.get("query", "")
        page = params.get("page", 1)
        limit = params.get("limit", 10)
        category = params.get("category")
        sort_by = params.get("sort_by", "relevance")
        
        # Simulate search processing time
        start_time = time.time()
        
        # Filter documents based on category
        filtered_docs = self.documents
        if category:
            filtered_docs = [doc for doc in filtered_docs if doc["category"] == category]
        
        # Simple text matching (in production, this would use Elasticsearch)
        matching_docs = []
        query_lower = query.lower()
        
        for doc in filtered_docs:
            # Calculate relevance score based on title and content matching
            title_match = query_lower in doc["title"].lower()
            content_match = query_lower in doc["content"].lower()
            
            if title_match or content_match:
                # Boost score for title matches
                if title_match:
                    doc["score"] = min(1.0, doc["score"] * 1.2)
                matching_docs.append(doc)
        
        # Sort results
        if sort_by == "relevance":
            matching_docs.sort(key=lambda x: x["score"], reverse=True)
        elif sort_by == "date":
            matching_docs.sort(key=lambda x: x["timestamp"], reverse=True)
        
        # Pagination
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        paginated_results = matching_docs[start_idx:end_idx]
        
        # Calculate search time
        search_time = time.time() - start_time
        
        # Format results
        results = []
        for doc in paginated_results:
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "url": doc["url"],
                "description": doc["description"],
                "snippet": doc["description"][:200] + "...",
                "score": doc["score"],
                "category": doc["category"],
                "timestamp": doc["timestamp"]
            })
        
        return {
            "results": results,
            "totalResults": len(matching_docs),
            "searchTime": round(search_time, 3),
            "suggestions": [],
            "facets": {
                "categories": {"technology": 2, "academic": 1},
                "domains": {"example.com": 1, "search-tech.com": 1, "webdev.com": 1}
            }
        }
    
    async def get_suggestions(self, query: str, limit: int = 5) -> List[str]:
        """Get search suggestions"""
        # Simple suggestion logic based on existing documents
        suggestions = []
        query_lower = query.lower()
        
        # Extract words from document titles for suggestions
        for doc in self.documents:
            title_words = doc["title"].lower().split()
            for word in title_words:
                if word.startswith(query_lower) and word not in suggestions:
                    suggestions.append(word)
                    if len(suggestions) >= limit:
                        break
            if len(suggestions) >= limit:
                break
        
        # Add some common search suggestions
        common_suggestions = [
            f"{query} tutorial",
            f"{query} guide",
            f"{query} examples",
            f"what is {query}",
            f"{query} best practices"
        ]
        
        for suggestion in common_suggestions:
            if len(suggestions) < limit:
                suggestions.append(suggestion)
        
        return suggestions[:limit]
    
    async def get_popular_searches(self, limit: int = 10, timeframe: str = "day") -> List[Dict]:
        """Get popular search queries"""
        # Mock popular searches
        popular = [
            {"query": "machine learning", "count": 1250},
            {"query": "python programming", "count": 980},
            {"query": "web development", "count": 756},
            {"query": "data science", "count": 642},
            {"query": "artificial intelligence", "count": 589},
            {"query": "javascript tutorial", "count": 445},
            {"query": "react framework", "count": 398},
            {"query": "database design", "count": 321},
            {"query": "api development", "count": 287},
            {"query": "cloud computing", "count": 256}
        ]
        
        return popular[:limit]
    
    async def log_search_analytics(self, user_id: int, query: str, results_count: int, page: int):
        """Log search analytics"""
        # In production, this would store in database
        print(f"Analytics: User {user_id} searched '{query}', got {results_count} results on page {page}")
    
    async def log_feedback(self, user_id: int, query: str, result_id: str, feedback_type: str):
        """Log user feedback"""
        # In production, this would store in database for ML training
        print(f"Feedback: User {user_id} gave '{feedback_type}' feedback for result {result_id} on query '{query}'")
    
    async def get_user_analytics(self, user_id: int, days: int = 30) -> Dict[str, Any]:
        """Get user search analytics"""
        # Mock analytics data
        return {
            "total_searches": 147,
            "unique_queries": 89,
            "avg_results_clicked": 2.3,
            "most_searched_categories": [
                {"category": "technology", "count": 56},
                {"category": "academic", "count": 34},
                {"category": "news", "count": 28}
            ],
            "search_trends": [
                {"date": "2024-01-15", "searches": 12},
                {"date": "2024-01-16", "searches": 8},
                {"date": "2024-01-17", "searches": 15}
            ]
        }
    
    async def advanced_search(self, search_request, user_id: Optional[int] = None) -> Dict[str, Any]:
        """Perform advanced search with complex parameters"""
        # In production, this would handle Boolean queries, field searches, etc.
        return await self.search(
            query=search_request.query,
            page=search_request.page,
            limit=search_request.limit
        )
