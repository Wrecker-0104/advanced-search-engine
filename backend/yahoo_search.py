"""
Yahoo Search API Integration via SearchAPI.io
Integrates real web search results from Yahoo Search
"""

import os
import asyncio
import aiohttp
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class YahooSearchAPI:
    """Yahoo Search API integration using SearchAPI.io"""
    
    def __init__(self):
        self.api_url = os.getenv("YAHOO_SEARCH_API_URL", "https://www.searchapi.io/api/v1/search")
        self.api_key = os.getenv("YAHOO_SEARCH_API_KEY", "your-searchapi-io-key-here")
        self.use_real_search = os.getenv("USE_REAL_SEARCH", "true").lower() == "true"
        self.timeout = 10  # seconds
        
    async def search_web(self, query: str, page: int = 1, limit: int = 10, **filters) -> Dict[str, Any]:
        """
        Search web using Yahoo Search API via SearchAPI.io
        
        Args:
            query: Search query string
            page: Page number (1-based)
            limit: Number of results per page
            **filters: Additional search filters
            
        Returns:
            Dict containing search results and metadata
        """
        if not self.use_real_search or self.api_key == "your-searchapi-io-key-here":
            logger.warning("Real search disabled or API key not configured, using mock data")
            return self._get_mock_results(query, page, limit)
        
        try:
            # Calculate offset for pagination
            offset = (page - 1) * limit
            
            # Prepare search parameters
            params = {
                "engine": "yahoo",
                "q": query,
                "num": min(limit, 20),  # Yahoo API limit
                "start": offset,
                "api_key": self.api_key
            }
            
            # Add filters if provided
            if filters.get("category") and filters["category"] != "all":
                params["tbm"] = self._map_category_to_yahoo(filters["category"])
            
            if filters.get("domain"):
                params["q"] = f'{query} site:{filters["domain"]}'
            
            if filters.get("dateRange") and filters["dateRange"] != "anytime":
                params["tbs"] = self._map_date_range(filters["dateRange"])
            
            start_time = datetime.now()
            
            # Make API request
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout)) as session:
                async with session.get(self.api_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        search_time = (datetime.now() - start_time).total_seconds()
                        return self._process_yahoo_results(data, query, page, limit, search_time)
                    else:
                        logger.error(f"Yahoo API request failed: {response.status}")
                        error_text = await response.text()
                        logger.error(f"Error details: {error_text}")
                        return self._get_mock_results(query, page, limit, error_message=f"API Error: {response.status}")
                        
        except asyncio.TimeoutError:
            logger.error("Yahoo API request timed out")
            return self._get_mock_results(query, page, limit, error_message="Search timeout")
        except Exception as e:
            logger.error(f"Yahoo API request failed: {str(e)}")
            return self._get_mock_results(query, page, limit, error_message=f"API Error: {str(e)}")
    
    def _process_yahoo_results(self, data: Dict, query: str, page: int, limit: int, search_time: float) -> Dict[str, Any]:
        """Process Yahoo API response into our search result format"""
        results = []
        
        # Extract organic results
        organic_results = data.get("organic_results", [])
        
        for idx, result in enumerate(organic_results[:limit]):
            processed_result = {
                "id": f"yahoo_{page}_{idx}",
                "title": result.get("title", ""),
                "url": result.get("link", ""),
                "description": result.get("snippet", ""),
                "score": 1.0 - (idx * 0.05),  # Decreasing relevance score
                "category": self._detect_category(result),
                "timestamp": datetime.now().isoformat(),
                "snippet": result.get("snippet", "")[:200] + "..." if result.get("snippet", "") else "",
                "domain": self._extract_domain(result.get("link", "")),
                "source": "yahoo_search"
            }
            
            # Add rich snippets if available
            if result.get("rich_snippet"):
                processed_result["rich_snippet"] = result["rich_snippet"]
            
            results.append(processed_result)
        
        # Calculate total results
        search_info = data.get("search_information", {})
        total_results = search_info.get("total_results", len(results))
        if isinstance(total_results, str):
            # Parse string like "About 1,234,567 results"
            import re
            numbers = re.findall(r'[\d,]+', total_results)
            if numbers:
                total_results = int(numbers[0].replace(',', ''))
            else:
                total_results = len(results)
        
        # Generate suggestions based on related searches
        suggestions = []
        related_searches = data.get("related_searches", [])
        for related in related_searches[:5]:
            if isinstance(related, dict):
                suggestions.append(related.get("query", ""))
            elif isinstance(related, str):
                suggestions.append(related)
        
        # Add some generic suggestions if none found
        if not suggestions:
            suggestions = [
                f"{query} tutorial",
                f"{query} guide",
                f"{query} examples",
                f"how to {query}",
                f"{query} tips"
            ]
        
        return {
            "query": query,
            "total_results": total_results,
            "page": page,
            "limit": limit,
            "total_pages": max(1, (total_results + limit - 1) // limit),
            "results": results,
            "search_time": search_time,
            "algorithm": "yahoo_web_search",
            "suggestions": suggestions,
            "filters_applied": {
                "category": "all",
                "domain": "",
                "sort_by": "relevance"
            },
            "facets": self._generate_facets(results),
            "source": "Yahoo Search via SearchAPI.io"
        }
    
    def _get_mock_results(self, query: str, page: int, limit: int, error_message: str = None) -> Dict[str, Any]:
        """Generate mock results when API is unavailable"""
        results = []
        
        for i in range(1, min(limit + 1, 6)):
            result = {
                "id": f"mock_{page}_{i}",
                "title": f"Real search result for '{query}' - Result {i}",
                "url": f"https://example.com/search-result-{i}",
                "description": f"This would be a real search result from Yahoo for '{query}'. The API integration is ready but requires a valid SearchAPI.io key to fetch live results.",
                "score": 1.0 - (i * 0.1),
                "category": "general",
                "timestamp": datetime.now().isoformat(),
                "snippet": f"Mock search result demonstrating Yahoo API integration for query: {query}",
                "domain": "example.com",
                "source": "mock_data"
            }
            results.append(result)
        
        return {
            "query": query,
            "total_results": 50000,
            "page": page,
            "limit": limit,
            "total_pages": 5000,
            "results": results,
            "search_time": 0.123,
            "algorithm": "mock_yahoo_search",
            "suggestions": [f"{query} examples", f"{query} tutorial", f"learn {query}"],
            "filters_applied": {"category": "all", "domain": "", "sort_by": "relevance"},
            "facets": {"categories": {"general": 50000}},
            "source": "Mock Data (Yahoo API ready)",
            "notice": error_message or "Yahoo API integration ready - add your SearchAPI.io key to .env file"
        }
    
    def _map_category_to_yahoo(self, category: str) -> str:
        """Map internal category to Yahoo search type"""
        mapping = {
            "images": "isch",
            "videos": "vid", 
            "news": "nws",
            "shopping": "shop"
        }
        return mapping.get(category, "")
    
    def _map_date_range(self, date_range: str) -> str:
        """Map date range to Yahoo search parameter"""
        mapping = {
            "day": "qdr:d",
            "week": "qdr:w", 
            "month": "qdr:m",
            "year": "qdr:y"
        }
        return mapping.get(date_range, "")
    
    def _detect_category(self, result: Dict) -> str:
        """Detect category based on result content"""
        title = result.get("title", "").lower()
        snippet = result.get("snippet", "").lower()
        url = result.get("link", "").lower()
        
        if any(word in title + snippet for word in ["news", "breaking", "report", "today"]):
            return "news"
        elif any(word in url for word in ["youtube.com", "vimeo.com", "video"]):
            return "videos"
        elif any(word in title + snippet for word in ["buy", "price", "shop", "store", "$"]):
            return "shopping"
        elif any(word in title + snippet for word in ["technology", "tech", "software", "programming"]):
            return "technology"
        else:
            return "general"
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc.replace("www.", "")
        except:
            return "unknown"
    
    def _generate_facets(self, results: List[Dict]) -> Dict[str, Dict[str, int]]:
        """Generate facets from search results"""
        categories = {}
        domains = {}
        
        for result in results:
            # Category facets
            category = result.get("category", "general")
            categories[category] = categories.get(category, 0) + 1
            
            # Domain facets
            domain = result.get("domain", "unknown")
            domains[domain] = domains.get(domain, 0) + 1
        
        return {
            "categories": categories,
            "domains": domains
        }

# Global Yahoo search API instance
yahoo_search_api = YahooSearchAPI()

async def search_web_yahoo(query: str, page: int = 1, limit: int = 10, **filters) -> Dict[str, Any]:
    """Search web using Yahoo API"""
    return await yahoo_search_api.search_web(query, page, limit, **filters)
