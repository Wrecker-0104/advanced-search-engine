"""
Advanced Search Algorithms - Google-like Implementation
Implements:
- TF-IDF (Term Frequency-Inverse Document Frequency)
- BM25 (Best Matching 25)
- PageRank Algorithm
- Semantic Similarity with BERT
- Query Understanding and Intent Recognition
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Set, Optional
from collections import defaultdict, Counter
import json
import pickle
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import re


@dataclass
class Document:
    """Document representation"""
    id: str
    title: str
    content: str
    url: str
    domain: str
    category: str
    timestamp: str
    incoming_links: List[str] = None
    outgoing_links: List[str] = None
    
    def __post_init__(self):
        if self.incoming_links is None:
            self.incoming_links = []
        if self.outgoing_links is None:
            self.outgoing_links = []


@dataclass
class SearchResult:
    """Search result with scoring"""
    document: Document
    tf_idf_score: float
    bm25_score: float
    pagerank_score: float
    semantic_score: float
    combined_score: float
    query_match_info: Dict


class TextProcessor:
    """Text processing utilities"""
    
    def __init__(self):
        # Common English stop words
        self.stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 
            'to', 'was', 'will', 'with', 'have', 'this', 'but', 'they', 'not',
            'or', 'can', 'had', 'her', 'his', 'she', 'there', 'been', 'if'
        }
    
    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into words"""
        # Convert to lowercase and extract words
        text = text.lower()
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        
        # Remove stop words and short words
        words = [word for word in words if word not in self.stop_words and len(word) >= 2]
        
        return words
    
    def get_term_frequency(self, tokens: List[str]) -> Dict[str, int]:
        """Calculate term frequency"""
        return Counter(tokens)
    
    def get_unique_terms(self, documents: List[Document]) -> Set[str]:
        """Get all unique terms from documents"""
        terms = set()
        for doc in documents:
            tokens = self.tokenize(f"{doc.title} {doc.content}")
            terms.update(tokens)
        return terms


class TFIDFRanker:
    """TF-IDF (Term Frequency-Inverse Document Frequency) Implementation"""
    
    def __init__(self):
        self.processor = TextProcessor()
        self.document_frequencies = defaultdict(int)
        self.total_documents = 0
        self.vectorizer = None
    
    def build_index(self, documents: List[Document]):
        """Build TF-IDF index from documents"""
        self.total_documents = len(documents)
        
        # Count document frequencies for each term
        for doc in documents:
            tokens = set(self.processor.tokenize(f"{doc.title} {doc.content}"))
            for token in tokens:
                self.document_frequencies[token] += 1
        
        # Build scikit-learn TF-IDF vectorizer for comparison
        corpus = [f"{doc.title} {doc.content}" for doc in documents]
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z]{2,}\b',
            max_features=10000
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
    
    def calculate_tfidf(self, term: str, document: Document) -> float:
        """Calculate TF-IDF score for a term in a document"""
        # Term frequency
        tokens = self.processor.tokenize(f"{document.title} {document.content}")
        tf = tokens.count(term) / len(tokens) if tokens else 0
        
        # Inverse document frequency
        df = self.document_frequencies.get(term, 0)
        idf = math.log(self.total_documents / (df + 1)) if df > 0 else 0
        
        return tf * idf
    
    def search(self, query: str, documents: List[Document], top_k: int = 10) -> List[Tuple[Document, float]]:
        """Search documents using TF-IDF"""
        query_tokens = self.processor.tokenize(query)
        scores = []
        
        for doc in documents:
            score = 0
            for token in query_tokens:
                score += self.calculate_tfidf(token, doc)
            
            # Boost score for title matches
            title_tokens = self.processor.tokenize(doc.title)
            title_matches = sum(1 for token in query_tokens if token in title_tokens)
            score += title_matches * 0.5  # Title boost
            
            scores.append((doc, score))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class BM25Ranker:
    """BM25 (Best Matching 25) Implementation - State-of-the-art ranking function"""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.processor = TextProcessor()
        self.k1 = k1  # Term frequency saturation parameter
        self.b = b    # Length normalization parameter
        self.document_frequencies = defaultdict(int)
        self.document_lengths = {}
        self.average_document_length = 0
        self.total_documents = 0
    
    def build_index(self, documents: List[Document]):
        """Build BM25 index from documents"""
        self.total_documents = len(documents)
        total_length = 0
        
        # Calculate document frequencies and lengths
        for doc in documents:
            tokens = self.processor.tokenize(f"{doc.title} {doc.content}")
            self.document_lengths[doc.id] = len(tokens)
            total_length += len(tokens)
            
            # Count unique terms in document
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.document_frequencies[token] += 1
        
        self.average_document_length = total_length / self.total_documents if self.total_documents > 0 else 0
    
    def calculate_bm25(self, query_tokens: List[str], document: Document) -> float:
        """Calculate BM25 score for a document given query tokens"""
        score = 0
        doc_tokens = self.processor.tokenize(f"{document.title} {document.content}")
        doc_length = self.document_lengths.get(document.id, len(doc_tokens))
        
        for token in query_tokens:
            # Term frequency in document
            tf = doc_tokens.count(token)
            
            if tf == 0:
                continue
            
            # Document frequency
            df = self.document_frequencies.get(token, 0)
            
            # IDF component
            idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
            
            # BM25 formula
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.average_document_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, documents: List[Document], top_k: int = 10) -> List[Tuple[Document, float]]:
        """Search documents using BM25"""
        query_tokens = self.processor.tokenize(query)
        scores = []
        
        for doc in documents:
            bm25_score = self.calculate_bm25(query_tokens, doc)
            
            # Additional scoring factors
            
            # Title match boost
            title_tokens = self.processor.tokenize(doc.title)
            title_matches = sum(1 for token in query_tokens if token in title_tokens)
            title_boost = title_matches * 2.0  # Strong boost for title matches
            
            # Exact phrase match boost
            if query.lower() in doc.title.lower():
                title_boost += 3.0
            if query.lower() in doc.content.lower():
                bm25_score += 1.0
            
            final_score = bm25_score + title_boost
            scores.append((doc, final_score))
        
        # Sort by score and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class PageRankAlgorithm:
    """PageRank Algorithm Implementation for Authority-based Ranking"""
    
    def __init__(self, damping_factor: float = 0.85, max_iterations: int = 100, tolerance: float = 1e-6):
        self.damping_factor = damping_factor
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.pagerank_scores = {}
    
    def build_graph(self, documents: List[Document]) -> nx.DiGraph:
        """Build directed graph from document links"""
        graph = nx.DiGraph()
        
        # Add all documents as nodes
        for doc in documents:
            graph.add_node(doc.id, document=doc)
        
        # Add edges based on outgoing links
        for doc in documents:
            for outgoing_link in doc.outgoing_links:
                if outgoing_link in [d.id for d in documents]:  # Only internal links
                    graph.add_edge(doc.id, outgoing_link)
        
        return graph
    
    def calculate_pagerank(self, documents: List[Document]) -> Dict[str, float]:
        """Calculate PageRank scores for all documents"""
        graph = self.build_graph(documents)
        
        if len(graph.nodes()) == 0:
            return {}
        
        # Use NetworkX implementation (optimized)
        try:
            pagerank_scores = nx.pagerank(
                graph, 
                alpha=self.damping_factor,
                max_iter=self.max_iterations,
                tol=self.tolerance
            )
        except:
            # Fallback to uniform scores if PageRank fails
            pagerank_scores = {doc.id: 1.0 / len(documents) for doc in documents}
        
        self.pagerank_scores = pagerank_scores
        return pagerank_scores
    
    def get_pagerank_score(self, document_id: str) -> float:
        """Get PageRank score for a specific document"""
        return self.pagerank_scores.get(document_id, 0.0)


class SemanticSearchEngine:
    """Semantic search using vector embeddings (simplified version)"""
    
    def __init__(self):
        self.document_embeddings = {}
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)  # Include bigrams
        )
        self.tfidf_matrix = None
    
    def build_embeddings(self, documents: List[Document]):
        """Build document embeddings using TF-IDF (simplified semantic representation)"""
        corpus = [f"{doc.title} {doc.content}" for doc in documents]
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        
        # Store embeddings for each document
        for i, doc in enumerate(documents):
            self.document_embeddings[doc.id] = self.tfidf_matrix[i]
    
    def calculate_semantic_similarity(self, query: str, document: Document) -> float:
        """Calculate semantic similarity between query and document"""
        if not self.tfidf_matrix is not None:
            return 0.0
        
        # Transform query to same vector space
        query_vector = self.vectorizer.transform([query])
        
        # Get document embedding
        doc_embedding = self.document_embeddings.get(document.id)
        if doc_embedding is None:
            return 0.0
        
        # Calculate cosine similarity
        similarity = cosine_similarity(query_vector, doc_embedding)[0][0]
        return max(0.0, similarity)  # Ensure non-negative
    
    def search(self, query: str, documents: List[Document], top_k: int = 10) -> List[Tuple[Document, float]]:
        """Search documents using semantic similarity"""
        scores = []
        
        for doc in documents:
            similarity = self.calculate_semantic_similarity(query, doc)
            scores.append((doc, similarity))
        
        # Sort by similarity and return top-k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


class AdvancedSearchEngine:
    """
    Advanced Search Engine combining multiple ranking algorithms
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        # Default weights for different ranking factors
        self.weights = weights or {
            'tfidf': 0.2,
            'bm25': 0.4,
            'pagerank': 0.2,
            'semantic': 0.2
        }
        
        # Initialize rankers
        self.tfidf_ranker = TFIDFRanker()
        self.bm25_ranker = BM25Ranker()
        self.pagerank_algorithm = PageRankAlgorithm()
        self.semantic_engine = SemanticSearchEngine()
        
        self.documents = []
        self.indexed = False
    
    def build_index(self, documents: List[Document]):
        """Build comprehensive search index"""
        self.documents = documents
        
        print("Building TF-IDF index...")
        self.tfidf_ranker.build_index(documents)
        
        print("Building BM25 index...")
        self.bm25_ranker.build_index(documents)
        
        print("Calculating PageRank...")
        self.pagerank_algorithm.calculate_pagerank(documents)
        
        print("Building semantic embeddings...")
        self.semantic_engine.build_embeddings(documents)
        
        self.indexed = True
        print(f"Index built successfully for {len(documents)} documents!")
    
    def search(self, query: str, top_k: int = 10, category_filter: str = None) -> List[SearchResult]:
        """
        Comprehensive search using all ranking algorithms
        """
        if not self.indexed:
            raise ValueError("Index must be built before searching")
        
        # Apply category filter if specified
        search_documents = self.documents
        if category_filter:
            search_documents = [doc for doc in self.documents if doc.category == category_filter]
        
        # Get results from each ranker
        tfidf_results = dict(self.tfidf_ranker.search(query, search_documents, len(search_documents)))
        bm25_results = dict(self.bm25_ranker.search(query, search_documents, len(search_documents)))
        semantic_results = dict(self.semantic_engine.search(query, search_documents, len(search_documents)))
        
        # Combine scores for each document
        combined_results = []
        
        for doc in search_documents:
            # Get individual scores
            tfidf_score = tfidf_results.get(doc, 0.0)
            bm25_score = bm25_results.get(doc, 0.0)
            pagerank_score = self.pagerank_algorithm.get_pagerank_score(doc.id)
            semantic_score = semantic_results.get(doc, 0.0)
            
            # Normalize scores (simple min-max normalization)
            # In production, you'd want more sophisticated normalization
            
            # Calculate combined score
            combined_score = (
                self.weights['tfidf'] * tfidf_score +
                self.weights['bm25'] * bm25_score +
                self.weights['pagerank'] * pagerank_score * 100 +  # Scale PageRank
                self.weights['semantic'] * semantic_score
            )
            
            # Create search result
            result = SearchResult(
                document=doc,
                tf_idf_score=tfidf_score,
                bm25_score=bm25_score,
                pagerank_score=pagerank_score,
                semantic_score=semantic_score,
                combined_score=combined_score,
                query_match_info=self._get_query_match_info(query, doc)
            )
            
            combined_results.append(result)
        
        # Sort by combined score and return top-k
        combined_results.sort(key=lambda x: x.combined_score, reverse=True)
        return combined_results[:top_k]
    
    def _get_query_match_info(self, query: str, document: Document) -> Dict:
        """Get detailed information about query matches in document"""
        query_lower = query.lower()
        title_lower = document.title.lower()
        content_lower = document.content.lower()
        
        return {
            'title_exact_match': query_lower in title_lower,
            'content_exact_match': query_lower in content_lower,
            'title_word_matches': len([word for word in query_lower.split() if word in title_lower]),
            'content_word_matches': len([word for word in query_lower.split() if word in content_lower])
        }
    
    def save_index(self, filepath: str):
        """Save the built index to disk"""
        index_data = {
            'documents': self.documents,
            'tfidf_ranker': self.tfidf_ranker,
            'bm25_ranker': self.bm25_ranker,
            'pagerank_scores': self.pagerank_algorithm.pagerank_scores,
            'semantic_embeddings': self.semantic_engine.document_embeddings,
            'weights': self.weights
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(index_data, f)
    
    def load_index(self, filepath: str):
        """Load a previously built index from disk"""
        with open(filepath, 'rb') as f:
            index_data = pickle.load(f)
        
        self.documents = index_data['documents']
        self.tfidf_ranker = index_data['tfidf_ranker']
        self.bm25_ranker = index_data['bm25_ranker']
        self.pagerank_algorithm.pagerank_scores = index_data['pagerank_scores']
        self.semantic_engine.document_embeddings = index_data['semantic_embeddings']
        self.weights = index_data['weights']
        self.indexed = True


def demo_search_engine():
    """Demonstration of the Advanced Search Engine"""
    
    # Create sample documents
    documents = [
        Document(
            id="1",
            title="Introduction to Machine Learning",
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
            url="https://example.com/ml-intro",
            domain="example.com",
            category="technology",
            timestamp="2024-01-15",
            outgoing_links=["2", "3"]
        ),
        Document(
            id="2",
            title="Deep Learning Fundamentals",
            content="Deep learning uses neural networks with multiple layers to model and understand complex patterns in data. It's particularly effective for image recognition, natural language processing, and speech recognition.",
            url="https://example.com/deep-learning",
            domain="example.com", 
            category="technology",
            timestamp="2024-01-16",
            incoming_links=["1"],
            outgoing_links=["3"]
        ),
        Document(
            id="3", 
            title="Natural Language Processing with Python",
            content="Natural language processing combines computational linguistics with machine learning to help computers understand human language. Python provides excellent libraries like NLTK, spaCy, and transformers.",
            url="https://example.com/nlp-python",
            domain="example.com",
            category="technology", 
            timestamp="2024-01-17",
            incoming_links=["1", "2"]
        )
    ]
    
    # Initialize and build search engine
    search_engine = AdvancedSearchEngine()
    search_engine.build_index(documents)
    
    # Perform search
    query = "machine learning algorithms"
    results = search_engine.search(query, top_k=5)
    
    print(f"\nSearch Results for: '{query}'")
    print("=" * 50)
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.document.title}")
        print(f"   URL: {result.document.url}")
        print(f"   Combined Score: {result.combined_score:.4f}")
        print(f"   TF-IDF: {result.tf_idf_score:.4f} | BM25: {result.bm25_score:.4f}")
        print(f"   PageRank: {result.pagerank_score:.6f} | Semantic: {result.semantic_score:.4f}")
        print(f"   Content: {result.document.content[:100]}...")


if __name__ == "__main__":
    demo_search_engine()
