"""
Advanced Search Engine Integration Layer
Combines all components into a unified search engine with:
- Multi-algorithm ranking
- Real-time indexing 
- Advanced query processing
- Performance monitoring
- A/B testing integration
- Quality evaluation
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import json

# Import our search engine components
from crawler.web_crawler import WebCrawler
from algorithms.ranking_algorithms import AdvancedSearchEngine
from nlp.query_processor import NLPProcessor
from analytics.search_analytics import SearchAnalytics, ABTestFramework, QualityEvaluator
from analytics.search_analytics import SearchEvent, EventType, PerformanceMetrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class SearchRequest:
    """Search request with all parameters"""
    query: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    filters: Optional[Dict] = None
    page: int = 1
    page_size: int = 10
    algorithm: str = "advanced"  # advanced, tfidf, bm25, pagerank, semantic
    personalized: bool = False
    include_suggestions: bool = True
    include_analytics: bool = True


@dataclass
class SearchResult:
    """Individual search result"""
    url: str
    title: str
    snippet: str
    score: float
    position: int
    metadata: Optional[Dict] = None
    highlighted_terms: Optional[List[str]] = None


@dataclass
class SearchResponse:
    """Complete search response"""
    query: str
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    page: int
    page_size: int
    suggestions: Optional[List[str]] = None
    query_analysis: Optional[Dict] = None
    experiment_variant: Optional[str] = None
    metadata: Optional[Dict] = None


class SearchEngineCore:
    """
    Main search engine class integrating all components
    """
    
    def __init__(self, config: Dict = None):
        """Initialize search engine with configuration"""
        self.config = config or self._default_config()
        
        # Initialize components
        self.crawler = WebCrawler()
        self.ranking_engine = AdvancedSearchEngine()
        self.nlp_processor = NLPProcessor()
        self.analytics = SearchAnalytics()
        self.ab_framework = ABTestFramework()
        self.quality_evaluator = QualityEvaluator()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Cache for frequent queries
        self.query_cache: Dict[str, SearchResponse] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # Performance tracking
        self.performance_stats = {
            'total_searches': 0,
            'avg_response_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        
        logger.info("Search engine core initialized")
    
    def _default_config(self) -> Dict:
        """Default search engine configuration"""
        return {
            'cache_enabled': True,
            'analytics_enabled': True,
            'ab_testing_enabled': True,
            'max_results_per_page': 50,
            'query_timeout_seconds': 10,
            'enable_personalization': True,
            'enable_autocomplete': True,
            'ranking_algorithms': {
                'advanced': {
                    'tfidf_weight': 0.3,
                    'bm25_weight': 0.4,
                    'pagerank_weight': 0.2,
                    'semantic_weight': 0.1
                }
            }
        }
    
    async def search(self, request: SearchRequest) -> SearchResponse:
        """
        Main search method
        """
        start_time = time.time()
        
        try:
            # Update performance stats
            self.performance_stats['total_searches'] += 1
            
            # Log search event
            if self.config['analytics_enabled'] and request.session_id:
                search_event = SearchEvent(
                    event_id=f"search_{int(time.time() * 1000)}",
                    session_id=request.session_id,
                    user_id=request.user_id,
                    event_type=EventType.SEARCH,
                    timestamp=datetime.now(),
                    query=request.query
                )
                self.analytics.log_event(search_event)
            
            # A/B testing assignment
            experiment_variant = None
            if self.config['ab_testing_enabled'] and request.user_id:
                # Example: test different ranking algorithms
                try:
                    experiment_variant = self.ab_framework.assign_user_to_variant(
                        request.user_id, 'ranking_algorithm_test'
                    )
                    # Override algorithm based on A/B test
                    if experiment_variant == 'algorithm_b':
                        request.algorithm = 'bm25'
                except:
                    pass  # Experiment might not exist yet
            
            # Check cache
            cache_key = self._generate_cache_key(request)
            cached_response = self._get_cached_response(cache_key)
            if cached_response:
                self.performance_stats['cache_hits'] += 1
                cached_response.experiment_variant = experiment_variant
                return cached_response
            
            self.performance_stats['cache_misses'] += 1
            
            # Process query with NLP
            query_analysis = None
            processed_query = request.query
            suggestions = []
            
            if request.include_analytics:
                query_analysis_obj = self.nlp_processor.analyze_query(request.query)
                query_analysis = {
                    'intent': query_analysis_obj.intent.value,
                    'entities': query_analysis_obj.entities,
                    'keywords': query_analysis_obj.keywords,
                    'sentiment': query_analysis_obj.sentiment,
                    'query_type': query_analysis_obj.query_type
                }
                
                processed_query = query_analysis_obj.cleaned_query
                
                if request.include_suggestions:
                    suggestions = query_analysis_obj.suggestions[:5]
            
            # Execute search based on selected algorithm
            search_results = await self._execute_search(
                processed_query, request.algorithm, request.filters
            )
            
            # Pagination
            start_idx = (request.page - 1) * request.page_size
            end_idx = start_idx + request.page_size
            paginated_results = search_results[start_idx:end_idx]
            
            # Create search results
            results = []
            for i, (url, title, snippet, score) in enumerate(paginated_results):
                result = SearchResult(
                    url=url,
                    title=title,
                    snippet=snippet,
                    score=score,
                    position=start_idx + i + 1,
                    highlighted_terms=self._extract_highlighted_terms(
                        request.query, snippet
                    )
                )
                results.append(result)
            
            # Calculate response time
            response_time_ms = (time.time() - start_time) * 1000
            
            # Update performance stats
            self._update_performance_stats(response_time_ms)
            
            # Create response
            response = SearchResponse(
                query=request.query,
                results=results,
                total_results=len(search_results),
                search_time_ms=response_time_ms,
                page=request.page,
                page_size=request.page_size,
                suggestions=suggestions if request.include_suggestions else None,
                query_analysis=query_analysis,
                experiment_variant=experiment_variant,
                metadata={
                    'algorithm_used': request.algorithm,
                    'cache_hit': False,
                    'total_indexed_docs': len(self.ranking_engine.documents)
                }
            )
            
            # Cache response
            if self.config['cache_enabled']:
                self._cache_response(cache_key, response)
            
            # Log performance metrics
            if self.config['analytics_enabled']:
                perf_metrics = PerformanceMetrics(
                    response_time_ms=response_time_ms,
                    index_size=len(self.ranking_engine.documents),
                    query_throughput=1000 / response_time_ms if response_time_ms > 0 else 0,
                    memory_usage_mb=0.0,  # Would implement actual memory tracking
                    cpu_usage_percent=0.0,  # Would implement actual CPU tracking
                    cache_hit_rate=self.performance_stats['cache_hits'] / 
                                   (self.performance_stats['cache_hits'] + 
                                    self.performance_stats['cache_misses']),
                    error_rate=0.0
                )
                self.analytics.log_performance_metrics(perf_metrics)
            
            return response
            
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            # Return error response
            return SearchResponse(
                query=request.query,
                results=[],
                total_results=0,
                search_time_ms=(time.time() - start_time) * 1000,
                page=request.page,
                page_size=request.page_size,
                metadata={'error': str(e)}
            )
    
    async def _execute_search(self, query: str, algorithm: str, 
                            filters: Optional[Dict] = None) -> List[Tuple[str, str, str, float]]:
        """Execute search using specified algorithm"""
        
        if algorithm == "advanced":
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, self.ranking_engine.search, query, 20
            )
        elif algorithm == "tfidf":
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, self.ranking_engine.tfidf_ranker.search, query, 20
            )
        elif algorithm == "bm25":
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, self.ranking_engine.bm25_ranker.search, query, 20
            )
        elif algorithm == "pagerank":
            # For PageRank, we need to combine with text search
            text_results = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.ranking_engine.tfidf_ranker.search, query, 50
            )
            # Apply PageRank scoring (simplified)
            return text_results[:20]  # Return top 20 for now
        elif algorithm == "semantic":
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, self.ranking_engine.semantic_engine.search, query, 20
            )
        else:
            # Default to advanced algorithm
            return await asyncio.get_event_loop().run_in_executor(
                self.executor, self.ranking_engine.search, query, 20
            )
    
    def _generate_cache_key(self, request: SearchRequest) -> str:
        """Generate cache key for request"""
        key_components = [
            request.query.lower().strip(),
            str(request.page),
            str(request.page_size),
            request.algorithm,
            json.dumps(request.filters, sort_keys=True) if request.filters else "none"
        ]
        return "|".join(key_components)
    
    def _get_cached_response(self, cache_key: str) -> Optional[SearchResponse]:
        """Get cached response if valid"""
        if not self.config['cache_enabled']:
            return None
        
        if cache_key in self.query_cache:
            # Check TTL
            cache_time = self.cache_timestamps.get(cache_key)
            if cache_time and (datetime.now() - cache_time).seconds < self.cache_ttl:
                response = self.query_cache[cache_key]
                # Update metadata to indicate cache hit
                if response.metadata:
                    response.metadata['cache_hit'] = True
                return response
            else:
                # Remove expired cache entry
                del self.query_cache[cache_key]
                if cache_key in self.cache_timestamps:
                    del self.cache_timestamps[cache_key]
        
        return None
    
    def _cache_response(self, cache_key: str, response: SearchResponse):
        """Cache search response"""
        if not self.config['cache_enabled']:
            return
        
        # Limit cache size (simple LRU-like behavior)
        if len(self.query_cache) > 1000:
            # Remove oldest 100 entries
            oldest_keys = sorted(
                self.cache_timestamps.keys(),
                key=lambda k: self.cache_timestamps[k]
            )[:100]
            
            for key in oldest_keys:
                self.query_cache.pop(key, None)
                self.cache_timestamps.pop(key, None)
        
        self.query_cache[cache_key] = response
        self.cache_timestamps[cache_key] = datetime.now()
    
    def _extract_highlighted_terms(self, query: str, text: str) -> List[str]:
        """Extract terms to highlight in search results"""
        query_terms = query.lower().split()
        text_lower = text.lower()
        
        highlighted = []
        for term in query_terms:
            if term in text_lower:
                highlighted.append(term)
        
        return highlighted
    
    def _update_performance_stats(self, response_time_ms: float):
        """Update performance statistics"""
        total_searches = self.performance_stats['total_searches']
        current_avg = self.performance_stats['avg_response_time']
        
        # Update running average
        new_avg = ((current_avg * (total_searches - 1)) + response_time_ms) / total_searches
        self.performance_stats['avg_response_time'] = new_avg
    
    def log_user_interaction(self, session_id: str, user_id: Optional[str],
                           event_type: EventType, query: str,
                           result_url: Optional[str] = None,
                           result_position: Optional[int] = None,
                           dwell_time: Optional[float] = None):
        """Log user interaction for analytics"""
        if not self.config['analytics_enabled']:
            return
        
        event = SearchEvent(
            event_id=f"event_{int(time.time() * 1000)}",
            session_id=session_id,
            user_id=user_id,
            event_type=event_type,
            timestamp=datetime.now(),
            query=query,
            result_url=result_url,
            result_position=result_position,
            dwell_time=dwell_time
        )
        
        self.analytics.log_event(event)
        
        # Log A/B test event if applicable
        if user_id and self.config['ab_testing_enabled']:
            try:
                self.ab_framework.log_experiment_event(
                    user_id, 'ranking_algorithm_test', event_type
                )
            except:
                pass  # Experiment might not exist
    
    def get_analytics_summary(self) -> Dict:
        """Get search analytics summary"""
        return {
            'performance_stats': self.performance_stats,
            'top_queries': [
                {
                    'query': metrics.query,
                    'searches': metrics.total_searches,
                    'ctr': metrics.ctr,
                    'satisfaction': metrics.satisfaction_score
                }
                for metrics in self.analytics.get_top_queries(10)
            ],
            'performance_summary': self.analytics.get_performance_summary()
        }
    
    def add_documents_to_index(self, documents: List[Dict]):
        """Add documents to search index"""
        for doc in documents:
            self.ranking_engine.add_document(
                doc.get('url', ''),
                doc.get('title', ''),
                doc.get('content', '')
            )
        
        logger.info(f"Added {len(documents)} documents to index")
    
    def setup_ab_experiment(self, experiment_id: str, variants: List[str],
                          traffic_split: List[float], description: str = ""):
        """Setup A/B test experiment"""
        self.ab_framework.create_experiment(
            experiment_id, variants, traffic_split, description
        )
        logger.info(f"Created A/B experiment: {experiment_id}")
    
    def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get search suggestions for autocomplete"""
        if len(partial_query) < 2:
            return []
        
        # Simple prefix matching from query history
        suggestions = []
        for query in self.analytics.query_stats.keys():
            if query.lower().startswith(partial_query.lower()):
                suggestions.append(query)
        
        # Sort by search frequency
        suggestions.sort(
            key=lambda q: self.analytics.query_stats[q]['searches'],
            reverse=True
        )
        
        return suggestions[:limit]


def demo_integrated_search_engine():
    """Demonstrate the integrated search engine"""
    
    # Initialize search engine
    search_engine = SearchEngineCore()
    
    # Add sample documents
    sample_docs = [
        {
            'url': 'https://example.com/python-tutorial',
            'title': 'Complete Python Tutorial for Beginners',
            'content': 'Learn Python programming from basics to advanced concepts. This comprehensive tutorial covers variables, functions, classes, and more.'
        },
        {
            'url': 'https://example.com/machine-learning',
            'title': 'Machine Learning with Python',
            'content': 'Discover machine learning algorithms and how to implement them in Python. Topics include supervised learning, neural networks, and data science.'
        },
        {
            'url': 'https://example.com/web-development',
            'title': 'Modern Web Development Guide',
            'content': 'Build modern web applications using JavaScript, React, Node.js, and other cutting-edge technologies.'
        },
        {
            'url': 'https://example.com/data-science',
            'title': 'Data Science Fundamentals',
            'content': 'Master data science with Python, pandas, numpy, and visualization libraries. Learn statistical analysis and data processing.'
        }
    ]
    
    search_engine.add_documents_to_index(sample_docs)
    
    # Setup A/B test
    search_engine.setup_ab_experiment(
        'ranking_algorithm_test',
        ['algorithm_a', 'algorithm_b'],
        [0.5, 0.5],
        'Testing different ranking algorithms'
    )
    
    print("Integrated Search Engine Demo")
    print("=" * 50)
    
    async def run_demo():
        # Test searches
        test_queries = [
            "python tutorial",
            "machine learning algorithms", 
            "web development javascript",
            "data science python"
        ]
        
        for i, query in enumerate(test_queries):
            print(f"\nSearch Query: '{query}'")
            
            request = SearchRequest(
                query=query,
                user_id=f"user_{i % 3}",
                session_id=f"session_{i}",
                page=1,
                page_size=5,
                algorithm="advanced",
                include_suggestions=True,
                include_analytics=True
            )
            
            response = await search_engine.search(request)
            
            print(f"Results: {len(response.results)} of {response.total_results}")
            print(f"Search Time: {response.search_time_ms:.2f}ms")
            print(f"Algorithm: {response.metadata.get('algorithm_used', 'unknown')}")
            if response.experiment_variant:
                print(f"A/B Variant: {response.experiment_variant}")
            
            # Show top results
            for result in response.results[:3]:
                print(f"  [{result.position}] {result.title}")
                print(f"      Score: {result.score:.3f}")
                print(f"      URL: {result.url}")
            
            # Log user interaction (simulate click)
            if response.results:
                search_engine.log_user_interaction(
                    request.session_id,
                    request.user_id,
                    EventType.CLICK,
                    query,
                    response.results[0].url,
                    1,
                    dwell_time=45.0
                )
            
            print("-" * 30)
        
        # Show analytics summary
        print("\nAnalytics Summary:")
        summary = search_engine.get_analytics_summary()
        print(f"Total Searches: {summary['performance_stats']['total_searches']}")
        print(f"Avg Response Time: {summary['performance_stats']['avg_response_time']:.2f}ms")
        print(f"Cache Hit Rate: {summary['performance_stats']['cache_hits'] / (summary['performance_stats']['cache_hits'] + summary['performance_stats']['cache_misses']):.3f}")
        
        print("\nTop Queries:")
        for query_data in summary['top_queries'][:3]:
            print(f"  '{query_data['query']}': {query_data['searches']} searches, CTR: {query_data['ctr']:.3f}")
    
    # Run the demo
    asyncio.run(run_demo())


if __name__ == "__main__":
    demo_integrated_search_engine()
