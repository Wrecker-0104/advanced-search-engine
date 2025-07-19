"""
Advanced Search Engine with Machine Learning Training
Implements TF-IDF, BM25, and semantic similarity search algorithms
"""

import numpy as np
import json
from typing import List, Dict, Any, Tuple
from collections import defaultdict, Counter
import math
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class AdvancedSearchEngine:
    """Advanced search engine with machine learning algorithms"""
    
    def __init__(self):
        self.documents = []
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        self.tfidf_matrix = None
        self.document_frequencies = defaultdict(int)
        self.total_documents = 0
        self.trained = False
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better matching"""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text.lower())
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def calculate_tf_idf_score(self, query_terms: List[str], document: Dict) -> float:
        """Calculate TF-IDF score for a document given query terms"""
        if not self.trained:
            return 0.0
        
        # Combine document text fields
        doc_text = f"{document.get('title', '')} {document.get('snippet', '')} {document.get('content', '')}"
        doc_text = self.preprocess_text(doc_text)
        doc_words = doc_text.split()
        doc_word_count = len(doc_words)
        
        if doc_word_count == 0:
            return 0.0
        
        score = 0.0
        for term in query_terms:
            # Term frequency
            tf = doc_words.count(term) / doc_word_count if doc_word_count > 0 else 0
            
            # Inverse document frequency
            df = self.document_frequencies.get(term, 0)
            if df > 0:
                idf = math.log(self.total_documents / df)
            else:
                idf = 0
            
            # TF-IDF score
            score += tf * idf
        
        return score
    
    def calculate_bm25_score(self, query_terms: List[str], document: Dict, k1: float = 1.2, b: float = 0.75) -> float:
        """Calculate BM25 score for a document"""
        if not self.trained:
            return 0.0
        
        # Combine document text fields
        doc_text = f"{document.get('title', '')} {document.get('snippet', '')} {document.get('content', '')}"
        doc_text = self.preprocess_text(doc_text)
        doc_words = doc_text.split()
        doc_length = len(doc_words)
        
        if doc_length == 0:
            return 0.0
        
        # Average document length
        avg_doc_length = sum(len(self.preprocess_text(f"{d.get('title', '')} {d.get('snippet', '')} {d.get('content', '')}").split()) 
                           for d in self.documents) / len(self.documents)
        
        score = 0.0
        for term in query_terms:
            # Term frequency in document
            tf = doc_words.count(term)
            
            # Document frequency
            df = self.document_frequencies.get(term, 0)
            if df > 0:
                idf = math.log((self.total_documents - df + 0.5) / (df + 0.5))
            else:
                continue
            
            # BM25 score calculation
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (doc_length / avg_doc_length))
            score += idf * (numerator / denominator)
        
        return max(0, score)  # Ensure non-negative score
    
    def train_on_dataset(self, documents: List[Dict]):
        """Train the search engine on a dataset of documents"""
        self.documents = documents
        self.total_documents = len(documents)
        
        # Build document frequency index
        self.document_frequencies = defaultdict(int)
        doc_texts = []
        
        for doc in documents:
            # Combine all text fields
            text = f"{doc.get('title', '')} {doc.get('snippet', '')} {doc.get('content', '')}"
            text = self.preprocess_text(text)
            doc_texts.append(text)
            
            # Count unique terms per document
            unique_terms = set(text.split())
            for term in unique_terms:
                self.document_frequencies[term] += 1
        
        # Train TF-IDF vectorizer
        try:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(doc_texts)
            self.trained = True
            print(f"Search engine trained on {len(documents)} documents")
        except Exception as e:
            print(f"Training error: {e}")
            self.trained = False
    
    def semantic_search(self, query: str, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """Perform semantic search using TF-IDF cosine similarity"""
        if not self.trained:
            return []
        
        try:
            # Transform query using trained vectorizer
            query_text = self.preprocess_text(query)
            query_vector = self.tfidf_vectorizer.transform([query_text])
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = similarities.argsort()[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.01:  # Filter low similarity
                    doc = self.documents[idx].copy()
                    doc['semantic_score'] = round(float(similarities[idx]), 4)
                    results.append((doc, float(similarities[idx])))
            
            return results
        except Exception as e:
            print(f"Semantic search error: {e}")
            return []
    
    def hybrid_search(self, query: str, top_k: int = 10, 
                     tfidf_weight: float = 0.3, bm25_weight: float = 0.4, 
                     semantic_weight: float = 0.3) -> List[Dict]:
        """Combine multiple search algorithms for better results"""
        if not self.trained:
            return []
        
        query_terms = self.preprocess_text(query).split()
        results = []
        
        for doc in self.documents:
            # Calculate different scores
            tfidf_score = self.calculate_tf_idf_score(query_terms, doc)
            bm25_score = self.calculate_bm25_score(query_terms, doc)
            
            # Normalize scores (simple min-max normalization)
            tfidf_normalized = min(1.0, tfidf_score / 10.0) if tfidf_score > 0 else 0
            bm25_normalized = min(1.0, bm25_score / 20.0) if bm25_score > 0 else 0
            
            # Combined score
            combined_score = (tfidf_weight * tfidf_normalized + 
                            bm25_weight * bm25_normalized)
            
            if combined_score > 0.05:  # Filter low scores
                result_doc = doc.copy()
                result_doc.update({
                    'tfidf_score': round(tfidf_score, 4),
                    'bm25_score': round(bm25_score, 4),
                    'combined_score': round(combined_score, 4),
                    'relevance_score': round(combined_score, 3)
                })
                results.append(result_doc)
        
        # Add semantic search results
        semantic_results = self.semantic_search(query, top_k * 2)
        for doc, sem_score in semantic_results:
            # Find if document already exists in results
            existing_idx = None
            for i, result in enumerate(results):
                if result['id'] == doc['id']:
                    existing_idx = i
                    break
            
            if existing_idx is not None:
                # Update existing result with semantic score
                results[existing_idx]['semantic_score'] = sem_score
                results[existing_idx]['combined_score'] += semantic_weight * sem_score
                results[existing_idx]['relevance_score'] = round(results[existing_idx]['combined_score'], 3)
            else:
                # Add new semantic result
                doc['semantic_score'] = sem_score
                doc['combined_score'] = semantic_weight * sem_score
                doc['relevance_score'] = round(semantic_weight * sem_score, 3)
                if doc['relevance_score'] > 0.05:
                    results.append(doc)
        
        # Sort by combined score and return top results
        results.sort(key=lambda x: x['combined_score'], reverse=True)
        return results[:top_k]
    
    def save_model(self, filepath: str):
        """Save trained model to file"""
        if not self.trained:
            print("Model not trained yet!")
            return
        
        model_data = {
            'documents': self.documents,
            'document_frequencies': dict(self.document_frequencies),
            'total_documents': self.total_documents,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'tfidf_matrix': self.tfidf_matrix
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model from file"""
        if not os.path.exists(filepath):
            print(f"Model file {filepath} not found!")
            return False
        
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.documents = model_data['documents']
            self.document_frequencies = defaultdict(int, model_data['document_frequencies'])
            self.total_documents = model_data['total_documents']
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            self.tfidf_matrix = model_data['tfidf_matrix']
            self.trained = True
            
            print(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

# Global search engine instance
search_engine = AdvancedSearchEngine()

def initialize_search_engine(documents: List[Dict]):
    """Initialize and train the search engine"""
    global search_engine
    
    # Try to load existing model
    model_path = "search_model.pkl"
    if not search_engine.load_model(model_path):
        # Train new model if loading fails
        search_engine.train_on_dataset(documents)
        search_engine.save_model(model_path)
    
    return search_engine

def advanced_search(query: str, documents: List[Dict], limit: int = 10) -> List[Dict]:
    """Perform advanced search with machine learning algorithms"""
    global search_engine
    
    if not search_engine.trained:
        initialize_search_engine(documents)
    
    return search_engine.hybrid_search(query, top_k=limit)

if __name__ == "__main__":
    # Test the search engine
    from dataset import COMPREHENSIVE_DATASET
    
    engine = AdvancedSearchEngine()
    engine.train_on_dataset(COMPREHENSIVE_DATASET)
    
    # Test searches
    test_queries = [
        "python machine learning",
        "web development react",
        "database optimization",
        "artificial intelligence"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = engine.hybrid_search(query, top_k=3)
        for i, result in enumerate(results, 1):
            print(f"{i}. {result['title']} (Score: {result['relevance_score']})")
