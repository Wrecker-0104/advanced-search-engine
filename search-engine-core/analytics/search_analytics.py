"""
Search Quality Metrics and Analytics
Implements comprehensive search analytics including:
- Click-through rates (CTR)
- Position-based metrics
- User engagement tracking
- A/B testing framework
- Search result quality evaluation
- Performance monitoring
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
from enum import Enum
import statistics
import math


class MetricType(Enum):
    """Types of search metrics"""
    CTR = "click_through_rate"
    POSITION = "average_position"
    DWELL_TIME = "dwell_time"
    BOUNCE_RATE = "bounce_rate"
    CONVERSION = "conversion_rate"
    SATISFACTION = "satisfaction_score"


class EventType(Enum):
    """Types of user interaction events"""
    SEARCH = "search"
    CLICK = "click"
    VIEW = "view"
    SCROLL = "scroll"
    BOOKMARK = "bookmark"
    SHARE = "share"
    BACK = "back"
    CONVERSION = "conversion"


@dataclass
class SearchEvent:
    """Individual search interaction event"""
    event_id: str
    session_id: str
    user_id: Optional[str]
    event_type: EventType
    timestamp: datetime
    query: str
    result_url: Optional[str] = None
    result_position: Optional[int] = None
    result_title: Optional[str] = None
    dwell_time: Optional[float] = None  # seconds
    metadata: Optional[Dict] = None


@dataclass
class SearchSession:
    """Complete search session data"""
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    queries: List[str]
    events: List[SearchEvent]
    total_searches: int
    total_clicks: int
    successful: bool = False  # Did user find what they were looking for
    
    
@dataclass
class QueryMetrics:
    """Metrics for a specific query"""
    query: str
    total_searches: int
    total_clicks: int
    ctr: float
    average_position: float
    bounce_rate: float
    average_dwell_time: float
    conversion_rate: float
    satisfaction_score: float


@dataclass
class PerformanceMetrics:
    """System performance metrics"""
    response_time_ms: float
    index_size: int
    query_throughput: float  # queries per second
    memory_usage_mb: float
    cpu_usage_percent: float
    cache_hit_rate: float
    error_rate: float


class SearchAnalytics:
    """Main search analytics engine"""
    
    def __init__(self):
        self.events: List[SearchEvent] = []
        self.sessions: Dict[str, SearchSession] = {}
        self.query_stats: Dict[str, Dict] = defaultdict(lambda: {
            'searches': 0,
            'clicks': 0,
            'click_positions': [],
            'dwell_times': [],
            'bounces': 0,
            'conversions': 0
        })
        self.performance_history: List[PerformanceMetrics] = []
        
    def log_event(self, event: SearchEvent):
        """Log a search interaction event"""
        self.events.append(event)
        
        # Update session
        if event.session_id not in self.sessions:
            self.sessions[event.session_id] = SearchSession(
                session_id=event.session_id,
                user_id=event.user_id,
                start_time=event.timestamp,
                end_time=None,
                queries=[],
                events=[],
                total_searches=0,
                total_clicks=0
            )
        
        session = self.sessions[event.session_id]
        session.events.append(event)
        session.end_time = event.timestamp
        
        # Update query statistics
        if event.event_type == EventType.SEARCH:
            session.total_searches += 1
            session.queries.append(event.query)
            self.query_stats[event.query]['searches'] += 1
            
        elif event.event_type == EventType.CLICK:
            session.total_clicks += 1
            self.query_stats[event.query]['clicks'] += 1
            if event.result_position:
                self.query_stats[event.query]['click_positions'].append(event.result_position)
            if event.dwell_time:
                self.query_stats[event.query]['dwell_times'].append(event.dwell_time)
                
        elif event.event_type == EventType.BACK:
            # Potential bounce
            self.query_stats[event.query]['bounces'] += 1
            
        elif event.event_type == EventType.CONVERSION:
            self.query_stats[event.query]['conversions'] += 1
            session.successful = True
    
    def calculate_ctr(self, query: str, time_period: Optional[Tuple[datetime, datetime]] = None) -> float:
        """Calculate click-through rate for a query"""
        stats = self.query_stats[query]
        if stats['searches'] == 0:
            return 0.0
        return stats['clicks'] / stats['searches']
    
    def calculate_average_position(self, query: str) -> float:
        """Calculate average click position for a query"""
        positions = self.query_stats[query]['click_positions']
        if not positions:
            return 0.0
        return statistics.mean(positions)
    
    def calculate_bounce_rate(self, query: str) -> float:
        """Calculate bounce rate for a query"""
        stats = self.query_stats[query]
        if stats['searches'] == 0:
            return 0.0
        return stats['bounces'] / stats['searches']
    
    def calculate_average_dwell_time(self, query: str) -> float:
        """Calculate average time spent on results"""
        dwell_times = self.query_stats[query]['dwell_times']
        if not dwell_times:
            return 0.0
        return statistics.mean(dwell_times)
    
    def calculate_conversion_rate(self, query: str) -> float:
        """Calculate conversion rate for a query"""
        stats = self.query_stats[query]
        if stats['searches'] == 0:
            return 0.0
        return stats['conversions'] / stats['searches']
    
    def calculate_satisfaction_score(self, query: str) -> float:
        """Calculate user satisfaction score (0-1)"""
        # Composite score based on multiple factors
        ctr = self.calculate_ctr(query)
        bounce_rate = self.calculate_bounce_rate(query)
        avg_dwell_time = self.calculate_average_dwell_time(query)
        conversion_rate = self.calculate_conversion_rate(query)
        
        # Normalize dwell time (assume 30 seconds is good)
        normalized_dwell = min(avg_dwell_time / 30.0, 1.0)
        
        # Calculate weighted satisfaction score
        satisfaction = (
            ctr * 0.3 +  # 30% weight to CTR
            (1 - bounce_rate) * 0.2 +  # 20% weight to engagement
            normalized_dwell * 0.3 +  # 30% weight to dwell time
            conversion_rate * 0.2  # 20% weight to conversion
        )
        
        return min(satisfaction, 1.0)
    
    def get_query_metrics(self, query: str) -> QueryMetrics:
        """Get comprehensive metrics for a query"""
        return QueryMetrics(
            query=query,
            total_searches=self.query_stats[query]['searches'],
            total_clicks=self.query_stats[query]['clicks'],
            ctr=self.calculate_ctr(query),
            average_position=self.calculate_average_position(query),
            bounce_rate=self.calculate_bounce_rate(query),
            average_dwell_time=self.calculate_average_dwell_time(query),
            conversion_rate=self.calculate_conversion_rate(query),
            satisfaction_score=self.calculate_satisfaction_score(query)
        )
    
    def get_top_queries(self, limit: int = 10, metric: MetricType = MetricType.CTR) -> List[QueryMetrics]:
        """Get top performing queries by specified metric"""
        query_metrics = []
        
        for query in self.query_stats.keys():
            if self.query_stats[query]['searches'] >= 5:  # Minimum threshold
                metrics = self.get_query_metrics(query)
                query_metrics.append(metrics)
        
        # Sort by specified metric
        if metric == MetricType.CTR:
            query_metrics.sort(key=lambda x: x.ctr, reverse=True)
        elif metric == MetricType.SATISFACTION:
            query_metrics.sort(key=lambda x: x.satisfaction_score, reverse=True)
        elif metric == MetricType.CONVERSION:
            query_metrics.sort(key=lambda x: x.conversion_rate, reverse=True)
        
        return query_metrics[:limit]
    
    def log_performance_metrics(self, metrics: PerformanceMetrics):
        """Log system performance metrics"""
        self.performance_history.append(metrics)
        
        # Keep only last 24 hours of performance data
        cutoff_time = datetime.now() - timedelta(hours=24)
        # Note: In real implementation, metrics would have timestamps
    
    def get_performance_summary(self) -> Dict:
        """Get performance metrics summary"""
        if not self.performance_history:
            return {}
        
        recent_metrics = self.performance_history[-100:]  # Last 100 measurements
        
        return {
            'avg_response_time_ms': statistics.mean(m.response_time_ms for m in recent_metrics),
            'max_response_time_ms': max(m.response_time_ms for m in recent_metrics),
            'avg_cpu_usage': statistics.mean(m.cpu_usage_percent for m in recent_metrics),
            'avg_memory_usage_mb': statistics.mean(m.memory_usage_mb for m in recent_metrics),
            'avg_cache_hit_rate': statistics.mean(m.cache_hit_rate for m in recent_metrics),
            'error_rate': statistics.mean(m.error_rate for m in recent_metrics),
            'throughput_qps': statistics.mean(m.query_throughput for m in recent_metrics)
        }


class ABTestFramework:
    """A/B testing framework for search algorithms"""
    
    def __init__(self):
        self.experiments: Dict[str, Dict] = {}
        self.user_assignments: Dict[str, Dict] = {}  # user_id -> {experiment_id: variant}
    
    def create_experiment(self, experiment_id: str, variants: List[str], 
                         traffic_split: List[float], description: str = ""):
        """Create a new A/B test experiment"""
        if len(variants) != len(traffic_split):
            raise ValueError("Variants and traffic split must have same length")
        
        if abs(sum(traffic_split) - 1.0) > 0.001:
            raise ValueError("Traffic split must sum to 1.0")
        
        self.experiments[experiment_id] = {
            'variants': variants,
            'traffic_split': traffic_split,
            'description': description,
            'created_at': datetime.now(),
            'active': True,
            'results': defaultdict(lambda: {
                'users': 0,
                'searches': 0,
                'clicks': 0,
                'conversions': 0,
                'satisfaction_scores': []
            })
        }
    
    def assign_user_to_variant(self, user_id: str, experiment_id: str) -> str:
        """Assign user to experiment variant"""
        if experiment_id not in self.experiments:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}
        
        if experiment_id not in self.user_assignments[user_id]:
            # Hash user ID to get consistent assignment
            hash_value = hash(user_id + experiment_id) % 1000 / 1000.0
            
            experiment = self.experiments[experiment_id]
            cumulative = 0
            
            for i, split in enumerate(experiment['traffic_split']):
                cumulative += split
                if hash_value <= cumulative:
                    variant = experiment['variants'][i]
                    self.user_assignments[user_id][experiment_id] = variant
                    experiment['results'][variant]['users'] += 1
                    break
        
        return self.user_assignments[user_id][experiment_id]
    
    def log_experiment_event(self, user_id: str, experiment_id: str, 
                           event_type: EventType, metadata: Dict = None):
        """Log event for A/B test analysis"""
        if experiment_id not in self.experiments:
            return
        
        variant = self.assign_user_to_variant(user_id, experiment_id)
        results = self.experiments[experiment_id]['results'][variant]
        
        if event_type == EventType.SEARCH:
            results['searches'] += 1
        elif event_type == EventType.CLICK:
            results['clicks'] += 1
        elif event_type == EventType.CONVERSION:
            results['conversions'] += 1
        
        if metadata and 'satisfaction_score' in metadata:
            results['satisfaction_scores'].append(metadata['satisfaction_score'])
    
    def get_experiment_results(self, experiment_id: str) -> Dict:
        """Get A/B test results with statistical significance"""
        if experiment_id not in self.experiments:
            return {}
        
        experiment = self.experiments[experiment_id]
        results = {}
        
        for variant in experiment['variants']:
            variant_data = experiment['results'][variant]
            
            # Calculate metrics
            ctr = variant_data['clicks'] / variant_data['searches'] if variant_data['searches'] > 0 else 0
            conversion_rate = variant_data['conversions'] / variant_data['searches'] if variant_data['searches'] > 0 else 0
            avg_satisfaction = statistics.mean(variant_data['satisfaction_scores']) if variant_data['satisfaction_scores'] else 0
            
            results[variant] = {
                'users': variant_data['users'],
                'searches': variant_data['searches'], 
                'clicks': variant_data['clicks'],
                'conversions': variant_data['conversions'],
                'ctr': ctr,
                'conversion_rate': conversion_rate,
                'avg_satisfaction': avg_satisfaction,
                'sample_size': variant_data['searches']
            }
        
        # Calculate statistical significance (simplified)
        if len(experiment['variants']) == 2:
            control, treatment = experiment['variants']
            control_data = results[control]
            treatment_data = results[treatment]
            
            # Simple z-test for CTR difference
            if control_data['sample_size'] > 30 and treatment_data['sample_size'] > 30:
                p1 = control_data['ctr']
                p2 = treatment_data['ctr']
                n1 = control_data['sample_size']
                n2 = treatment_data['sample_size']
                
                p_pool = (control_data['clicks'] + treatment_data['clicks']) / (n1 + n2)
                se = math.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
                
                if se > 0:
                    z_score = abs(p2 - p1) / se
                    significant = z_score > 1.96  # 95% confidence
                    
                    results['statistical_analysis'] = {
                        'z_score': z_score,
                        'significant': significant,
                        'confidence_level': 0.95 if significant else None,
                        'lift': ((p2 - p1) / p1 * 100) if p1 > 0 else None
                    }
        
        return results


class QualityEvaluator:
    """Evaluate search result quality"""
    
    def __init__(self):
        self.relevance_judgments: Dict[str, Dict[str, float]] = {}  # query -> {url: relevance_score}
    
    def add_relevance_judgment(self, query: str, url: str, relevance_score: float):
        """Add manual relevance judgment (0-1 scale)"""
        if query not in self.relevance_judgments:
            self.relevance_judgments[query] = {}
        self.relevance_judgments[query][url] = relevance_score
    
    def calculate_dcg(self, query: str, search_results: List[str], k: int = 10) -> float:
        """Calculate Discounted Cumulative Gain"""
        if query not in self.relevance_judgments:
            return 0.0
        
        dcg = 0.0
        judgments = self.relevance_judgments[query]
        
        for i, url in enumerate(search_results[:k]):
            if url in judgments:
                relevance = judgments[url]
                dcg += (2**relevance - 1) / math.log2(i + 2)
        
        return dcg
    
    def calculate_ndcg(self, query: str, search_results: List[str], k: int = 10) -> float:
        """Calculate Normalized Discounted Cumulative Gain"""
        dcg = self.calculate_dcg(query, search_results, k)
        
        # Calculate IDCG (Ideal DCG)
        if query not in self.relevance_judgments:
            return 0.0
        
        judgments = self.relevance_judgments[query]
        ideal_results = sorted(judgments.keys(), key=lambda x: judgments[x], reverse=True)
        idcg = self.calculate_dcg(query, ideal_results, k)
        
        return dcg / idcg if idcg > 0 else 0.0
    
    def calculate_precision_at_k(self, query: str, search_results: List[str], k: int = 10, 
                               threshold: float = 0.5) -> float:
        """Calculate Precision@K"""
        if query not in self.relevance_judgments:
            return 0.0
        
        judgments = self.relevance_judgments[query]
        relevant_count = 0
        
        for url in search_results[:k]:
            if url in judgments and judgments[url] >= threshold:
                relevant_count += 1
        
        return relevant_count / min(k, len(search_results))
    
    def evaluate_search_quality(self, query: str, search_results: List[str]) -> Dict[str, float]:
        """Comprehensive search quality evaluation"""
        return {
            'ndcg_10': self.calculate_ndcg(query, search_results, 10),
            'ndcg_5': self.calculate_ndcg(query, search_results, 5),
            'precision_at_1': self.calculate_precision_at_k(query, search_results, 1),
            'precision_at_5': self.calculate_precision_at_k(query, search_results, 5),
            'precision_at_10': self.calculate_precision_at_k(query, search_results, 10)
        }


def demo_search_analytics():
    """Demonstrate search analytics capabilities"""
    analytics = SearchAnalytics()
    ab_test = ABTestFramework()
    quality_eval = QualityEvaluator()
    
    # Create A/B test
    ab_test.create_experiment(
        'algorithm_test_1',
        ['algorithm_a', 'algorithm_b'],
        [0.5, 0.5],
        'Testing new ranking algorithm'
    )
    
    # Simulate search events
    import uuid
    
    queries = [
        "python tutorial",
        "machine learning",
        "web development",
        "data science",
        "javascript guide"
    ]
    
    print("Search Analytics Demo")
    print("=" * 50)
    
    # Generate sample events
    for i in range(100):
        user_id = f"user_{i % 20}"  # 20 different users
        session_id = f"session_{i % 30}"  # 30 different sessions
        query = queries[i % len(queries)]
        
        # Search event
        search_event = SearchEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            user_id=user_id,
            event_type=EventType.SEARCH,
            timestamp=datetime.now(),
            query=query
        )
        analytics.log_event(search_event)
        
        # A/B test assignment
        variant = ab_test.assign_user_to_variant(user_id, 'algorithm_test_1')
        ab_test.log_experiment_event(user_id, 'algorithm_test_1', EventType.SEARCH)
        
        # Some clicks (60% CTR)
        if i % 10 < 6:
            click_event = SearchEvent(
                event_id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                event_type=EventType.CLICK,
                timestamp=datetime.now(),
                query=query,
                result_url=f"https://example.com/result_{i}",
                result_position=(i % 5) + 1,
                dwell_time=30 + (i % 60)
            )
            analytics.log_event(click_event)
            ab_test.log_experiment_event(user_id, 'algorithm_test_1', EventType.CLICK)
        
        # Some conversions (10% conversion rate)
        if i % 10 == 0:
            conversion_event = SearchEvent(
                event_id=str(uuid.uuid4()),
                session_id=session_id,
                user_id=user_id,
                event_type=EventType.CONVERSION,
                timestamp=datetime.now(),
                query=query
            )
            analytics.log_event(conversion_event)
            ab_test.log_experiment_event(user_id, 'algorithm_test_1', EventType.CONVERSION, 
                                       {'satisfaction_score': 0.8})
    
    # Display analytics
    print("\nTop Queries by CTR:")
    top_queries = analytics.get_top_queries(5, MetricType.CTR)
    for metrics in top_queries:
        print(f"'{metrics.query}': CTR={metrics.ctr:.3f}, Satisfaction={metrics.satisfaction_score:.3f}")
    
    print(f"\nA/B Test Results:")
    ab_results = ab_test.get_experiment_results('algorithm_test_1')
    for variant, data in ab_results.items():
        if variant != 'statistical_analysis':
            print(f"{variant}: CTR={data['ctr']:.3f}, Searches={data['searches']}")
    
    if 'statistical_analysis' in ab_results:
        stat_analysis = ab_results['statistical_analysis']
        print(f"Statistical Significance: {stat_analysis['significant']}")
        if stat_analysis['lift']:
            print(f"Lift: {stat_analysis['lift']:.2f}%")
    
    # Performance metrics
    perf_metrics = PerformanceMetrics(
        response_time_ms=45.2,
        index_size=1000000,
        query_throughput=150.0,
        memory_usage_mb=512.0,
        cpu_usage_percent=25.0,
        cache_hit_rate=0.85,
        error_rate=0.001
    )
    analytics.log_performance_metrics(perf_metrics)
    
    perf_summary = analytics.get_performance_summary()
    if perf_summary:
        print(f"\nPerformance Summary:")
        print(f"Avg Response Time: {perf_summary['avg_response_time_ms']:.1f}ms")
        print(f"Cache Hit Rate: {perf_summary['avg_cache_hit_rate']:.3f}")
        print(f"Error Rate: {perf_summary['error_rate']:.4f}")


if __name__ == "__main__":
    demo_search_analytics()
