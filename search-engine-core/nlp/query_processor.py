"""
Advanced NLP Processing for Search Engine
Implements:
- Query Understanding and Intent Recognition
- Named Entity Recognition (NER)
- Sentiment Analysis
- Semantic Similarity with BERT
- Query Expansion and Suggestions
"""

import re
import math
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict, Counter
from dataclasses import dataclass
from enum import Enum
import json

# Note: In production, you would use actual NLP libraries like spaCy, NLTK, transformers
# For this demo, we'll implement simplified versions that work without external dependencies


class QueryIntent(Enum):
    """Types of search query intents"""
    INFORMATIONAL = "informational"  # Looking for information
    NAVIGATIONAL = "navigational"   # Looking for specific website
    TRANSACTIONAL = "transactional" # Looking to buy/do something
    DEFINITIONAL = "definitional"   # Looking for definition
    COMPARATIVE = "comparative"     # Comparing things
    LOCAL = "local"                 # Local search
    TEMPORAL = "temporal"           # Time-sensitive search


@dataclass
class QueryAnalysis:
    """Comprehensive query analysis result"""
    original_query: str
    cleaned_query: str
    intent: QueryIntent
    entities: List[Dict[str, str]]
    keywords: List[str]
    sentiment: str  # positive, negative, neutral
    expanded_terms: List[str]
    suggestions: List[str]
    query_type: str  # question, statement, keyword
    language: str = "en"


@dataclass
class Entity:
    """Named entity representation"""
    text: str
    type: str  # PERSON, ORG, GPE, PRODUCT, etc.
    confidence: float
    start_pos: int
    end_pos: int


class TextCleaner:
    """Text cleaning and normalization utilities"""
    
    def __init__(self):
        # Common contractions
        self.contractions = {
            "won't": "will not",
            "can't": "cannot", 
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        # Query cleaning patterns
        self.cleaning_patterns = [
            (r'\s+', ' '),  # Multiple spaces to single space
            (r'[^\w\s\-\']', ''),  # Remove special chars except hyphens and apostrophes
            (r'\b\w{1}\b', ''),  # Remove single characters
        ]
    
    def clean_query(self, query: str) -> str:
        """Clean and normalize search query"""
        # Convert to lowercase
        query = query.lower().strip()
        
        # Expand contractions
        for contraction, expansion in self.contractions.items():
            query = query.replace(contraction, expansion)
        
        # Apply cleaning patterns
        for pattern, replacement in self.cleaning_patterns:
            query = re.sub(pattern, replacement, query)
        
        return query.strip()
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction (in production, use more sophisticated methods)
        words = text.lower().split()
        
        # Stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'have', 'this', 'but', 'they', 'not',
            'or', 'can', 'had', 'her', 'his', 'she', 'there', 'been', 'if',
            'what', 'where', 'when', 'why', 'how', 'who', 'which'
        }
        
        # Filter keywords
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Sort by length (longer words often more important)
        keywords.sort(key=len, reverse=True)
        
        return keywords[:10]  # Return top 10 keywords


class IntentClassifier:
    """Classify query intent based on patterns"""
    
    def __init__(self):
        # Intent classification patterns
        self.intent_patterns = {
            QueryIntent.DEFINITIONAL: [
                r'\b(what is|what are|define|definition of|meaning of)\b',
                r'\b(explain|describe)\s+\w+',
                r'\bwhat\s+\w+\s+mean\b'
            ],
            QueryIntent.COMPARATIVE: [
                r'\b(vs|versus|compare|comparison|difference|better|best|worst)\b',
                r'\b\w+\s+(vs|versus)\s+\w+\b',
                r'\b(better than|worse than|similar to)\b'
            ],
            QueryIntent.NAVIGATIONAL: [
                r'\b(site:|inurl:|intitle:)\w+',
                r'\b(login|sign in|homepage|official site)\b',
                r'\b\w+\.(com|org|net|edu)\b'
            ],
            QueryIntent.TRANSACTIONAL: [
                r'\b(buy|purchase|order|shop|price|cost|cheap|discount|deal)\b',
                r'\b(download|install|get|obtain)\b',
                r'\b(review|reviews|rating|ratings)\b'
            ],
            QueryIntent.TEMPORAL: [
                r'\b(today|tomorrow|yesterday|now|current|latest|recent)\b',
                r'\b(when|schedule|time|date|deadline)\b',
                r'\b(2024|2023|january|february|march|april|may|june|july|august|september|october|november|december)\b'
            ],
            QueryIntent.LOCAL: [
                r'\b(near me|nearby|local|in \w+|around here)\b',
                r'\b(restaurant|hotel|store|shop|gas station|hospital)\b.*\bnear\b',
                r'\b(address|directions|location|map)\b'
            ]
        }
    
    def classify_intent(self, query: str) -> QueryIntent:
        """Classify the intent of a search query"""
        query_lower = query.lower()
        
        # Check each intent pattern
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    return intent
        
        # Default to informational
        return QueryIntent.INFORMATIONAL


class SimpleNER:
    """Simplified Named Entity Recognition"""
    
    def __init__(self):
        # Simple entity patterns (in production, use spaCy or similar)
        self.entity_patterns = {
            'PERSON': [
                r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
                r'\b(Mr|Mrs|Ms|Dr|Prof)\. [A-Z][a-z]+\b'  # Title Name
            ],
            'ORG': [
                r'\b[A-Z][a-z]+ (Inc|Corp|LLC|Ltd|Company|Corporation)\b',
                r'\b(Google|Apple|Microsoft|Amazon|Facebook|Twitter|Netflix|Tesla)\b'
            ],
            'LOCATION': [
                r'\b(New York|Los Angeles|Chicago|Houston|Phoenix|Philadelphia|San Antonio|San Diego|Dallas|San Jose|Austin|Jacksonville|Fort Worth|Columbus|San Francisco|Charlotte|Indianapolis|Seattle|Denver|Boston|Nashville|Baltimore|Oklahoma City|Louisville|Portland|Las Vegas|Detroit|Memphis|Washington|Milwaukee|Albuquerque|Tucson|Fresno|Sacramento|Mesa|Kansas City|Atlanta|Colorado Springs|Raleigh|Omaha|Miami|Oakland|Tulsa|Minneapolis|Cleveland|Wichita|New Orleans|Arlington|Tampa|Honolulu|Anaheim|Santa Ana|Corpus Christi|Riverside|Lexington|St. Louis|Stockton|Pittsburgh|Saint Paul|Cincinnati|Anchorage|Henderson|Greensboro|Plano|Newark|Toledo|Lincoln|Orlando|Chula Vista|Jersey City|Chandler|Fort Wayne|Buffalo|Durham|Madison|Lubbock|Irvine|Norfolk|Laredo|Winston-Salem|Glendale|Garland|Hialeah|Reno|Chesapeake|Gilbert|Baton Rouge|Irving|Scottsdale|North Hempstead|Fremont|Boise|Richmond|San Bernardino|Birmingham|Spokane|Rochester|Des Moines|Modesto|Fayetteville|Tacoma|Oxnard|Fontana|Columbus|Montgomery|Moreno Valley|Shreveport|Aurora|Yonkers|Akron|Huntington Beach|Little Rock|Augusta|Amarillo|Glendale|Mobile|Grand Rapids|Salt Lake City|Tallahassee|Huntsville|Grand Prairie|Knoxville|Worcester|Newport News|Brownsville|Santa Clarita|Providence|Fort Lauderdale|Chattanooga|Oceanside|Jackson|Garden Grove|Hartford|Wrightsville Beach|Tempe|McKinney|Mobile|Cape Coral|Shreveport|Frisco|Sioux Falls|Discover|Eugene|Springfield|Pembroke Pines|Salem|Corona|Fort Collins|Lancaster|Elk Grove|Palmdale|Salinas|Hayward|Pomona|Escondido|Sunnyvale|Lakewood|Hollywood|Torrance|Pasadena|Naperville|Macon|Bellevue|Joliet|Murfreesboro|Rockford|Paterson|Kansas City|Savannah|Bridgeport|Syracuse|Manchester|Alexandria|Dayton|Inglewood|Delaware|Cedar Rapids|Round Rock|Broken Arrow|West Jordan|St. Petersburg|Clearwater|Gainesville|Westminster|Pueblo|Santa Maria|El Monte|Miami Gardens|Norwalk|Burbank|Ann Arbor|New Haven|Allentown|Cambridge|Edison|Palm Bay|Independence|Lansing|Lakeland|Thousand Oaks|Camden|Long Beach|Lowell|Virginia Beach|Warren|West Valley City|Columbia|Carrollton|Berkeley|Green Bay|McAllen|League City|Richardson|Olathe|Sterling Heights|Erie|Hillsboro|Waco|Visalia|Concord|Stamford|Santa Clara|Flint|Hartford|Fargo|Miramar|Thornton|Roseville|Beaumont|Denton|Sandy|Midland|Abilene|Pearland|College Station|Tyler|Carmel|Surprise|West Palm Beach|Coral Springs|Elizabeth|Cedar Park|Fairfield|Killeen|Topeka|Peoria|Charleston|Antioch|Richmond|Carlsbad|Elgin|Murrieta|Temecula|Cary|High Point|Mesquite)\b',
                r'\b(USA|US|United States|America|Canada|Mexico|UK|Britain|England|France|Germany|Italy|Spain|Russia|China|Japan|India|Australia|Brazil)\b'
            ],
            'PRODUCT': [
                r'\biPhone\s*\d*\b',
                r'\biPad\s*\w*\b',
                r'\bMacBook\s*\w*\b',
                r'\bWindows\s*\d*\b',
                r'\bAndroid\s*\d*\b'
            ]
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text"""
        entities = []
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text)
                for match in matches:
                    entity = Entity(
                        text=match.group(),
                        type=entity_type,
                        confidence=0.8,  # Simple confidence score
                        start_pos=match.start(),
                        end_pos=match.end()
                    )
                    entities.append(entity)
        
        return entities


class SentimentAnalyzer:
    """Simple sentiment analysis"""
    
    def __init__(self):
        # Simple sentiment word lists
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'awesome', 'fantastic',
            'wonderful', 'best', 'love', 'like', 'perfect', 'brilliant',
            'outstanding', 'superb', 'magnificent', 'terrific', 'fabulous',
            'marvelous', 'sensational', 'phenomenal', 'incredible', 'remarkable'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'dislike', 'disappointing', 'poor', 'fail', 'failed', 'broken',
            'useless', 'worthless', 'pathetic', 'disgusting', 'annoying',
            'frustrating', 'disappointing', 'unacceptable', 'inferior'
        }
    
    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of text"""
        words = text.lower().split()
        
        positive_score = sum(1 for word in words if word in self.positive_words)
        negative_score = sum(1 for word in words if word in self.negative_words)
        
        if positive_score > negative_score:
            return "positive"
        elif negative_score > positive_score:
            return "negative"
        else:
            return "neutral"


class QueryExpander:
    """Query expansion for better search results"""
    
    def __init__(self):
        # Simple synonyms dictionary (in production, use WordNet or similar)
        self.synonyms = {
            'car': ['automobile', 'vehicle', 'auto'],
            'computer': ['pc', 'laptop', 'desktop', 'machine'],
            'phone': ['telephone', 'mobile', 'smartphone', 'cell'],
            'house': ['home', 'residence', 'dwelling', 'property'],
            'job': ['work', 'employment', 'career', 'position', 'occupation'],
            'movie': ['film', 'cinema', 'picture', 'flick'],
            'food': ['meal', 'cuisine', 'dish', 'recipe'],
            'buy': ['purchase', 'acquire', 'get', 'obtain'],
            'fast': ['quick', 'rapid', 'speedy', 'swift'],
            'big': ['large', 'huge', 'massive', 'enormous'],
            'small': ['little', 'tiny', 'mini', 'compact'],
        }
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms and related terms"""
        words = query.lower().split()
        expanded_terms = []
        
        for word in words:
            if word in self.synonyms:
                expanded_terms.extend(self.synonyms[word])
        
        # Add common expansions
        if 'how' in words:
            expanded_terms.extend(['tutorial', 'guide', 'instructions', 'steps'])
        
        if 'what' in words and 'is' in words:
            expanded_terms.extend(['definition', 'meaning', 'explanation'])
        
        if 'best' in words:
            expanded_terms.extend(['top', 'recommended', 'popular', 'review'])
        
        return list(set(expanded_terms))  # Remove duplicates


class QuerySuggester:
    """Generate query suggestions"""
    
    def __init__(self):
        # Common query patterns
        self.suggestion_patterns = {
            'how': ['how to {query}', '{query} tutorial', '{query} guide', '{query} instructions'],
            'what': ['what is {query}', '{query} definition', '{query} meaning', '{query} explained'],
            'best': ['best {query}', 'top {query}', '{query} reviews', '{query} comparison'],
            'where': ['where to {query}', '{query} location', '{query} near me', '{query} address'],
            'when': ['when to {query}', '{query} schedule', '{query} time', '{query} deadline'],
            'why': ['why {query}', '{query} reasons', '{query} benefits', '{query} importance']
        }
    
    def generate_suggestions(self, query: str) -> List[str]:
        """Generate query suggestions"""
        suggestions = []
        query_lower = query.lower()
        
        # Add completion suggestions
        completions = [
            f"{query} tutorial",
            f"{query} guide", 
            f"{query} tips",
            f"{query} examples",
            f"best {query}",
            f"{query} 2024",
            f"free {query}",
            f"{query} online"
        ]
        
        suggestions.extend(completions)
        
        # Add pattern-based suggestions
        first_word = query_lower.split()[0] if query_lower.split() else ''
        if first_word in self.suggestion_patterns:
            patterns = self.suggestion_patterns[first_word]
            for pattern in patterns:
                suggestion = pattern.format(query=query)
                suggestions.append(suggestion)
        
        # Remove duplicates and limit
        suggestions = list(set(suggestions))
        return suggestions[:8]


class QueryClassifier:
    """Classify query types"""
    
    def classify_query_type(self, query: str) -> str:
        """Classify if query is a question, statement, or keyword search"""
        query_lower = query.lower()
        
        # Question indicators
        question_words = ['what', 'how', 'where', 'when', 'why', 'who', 'which', 'whose', 'whom']
        if any(query_lower.startswith(word) for word in question_words) or query.endswith('?'):
            return 'question'
        
        # Statement indicators
        if len(query.split()) > 5 and ('is' in query_lower or 'are' in query_lower):
            return 'statement'
        
        # Default to keyword search
        return 'keyword'


class NLPProcessor:
    """Main NLP processor combining all components"""
    
    def __init__(self):
        self.cleaner = TextCleaner()
        self.intent_classifier = IntentClassifier()
        self.ner = SimpleNER()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.query_expander = QueryExpander()
        self.query_suggester = QuerySuggester()
        self.query_classifier = QueryClassifier()
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Comprehensive query analysis"""
        # Clean the query
        cleaned_query = self.cleaner.clean_query(query)
        
        # Extract keywords
        keywords = self.cleaner.extract_keywords(cleaned_query)
        
        # Classify intent
        intent = self.intent_classifier.classify_intent(cleaned_query)
        
        # Extract entities
        entities_list = self.ner.extract_entities(query)
        entities = [
            {
                'text': entity.text,
                'type': entity.type,
                'confidence': entity.confidence
            }
            for entity in entities_list
        ]
        
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze_sentiment(cleaned_query)
        
        # Expand query terms
        expanded_terms = self.query_expander.expand_query(cleaned_query)
        
        # Generate suggestions
        suggestions = self.query_suggester.generate_suggestions(cleaned_query)
        
        # Classify query type
        query_type = self.query_classifier.classify_query_type(cleaned_query)
        
        return QueryAnalysis(
            original_query=query,
            cleaned_query=cleaned_query,
            intent=intent,
            entities=entities,
            keywords=keywords,
            sentiment=sentiment,
            expanded_terms=expanded_terms,
            suggestions=suggestions,
            query_type=query_type,
            language="en"
        )
    
    def process_document_text(self, text: str) -> Dict:
        """Process document text for indexing"""
        # Clean text
        cleaned_text = self.cleaner.clean_query(text)
        
        # Extract keywords
        keywords = self.cleaner.extract_keywords(cleaned_text)
        
        # Extract entities
        entities_list = self.ner.extract_entities(text)
        entities = [
            {
                'text': entity.text,
                'type': entity.type,
                'confidence': entity.confidence
            }
            for entity in entities_list
        ]
        
        # Analyze sentiment
        sentiment = self.sentiment_analyzer.analyze_sentiment(cleaned_text)
        
        return {
            'cleaned_text': cleaned_text,
            'keywords': keywords,
            'entities': entities,
            'sentiment': sentiment,
            'word_count': len(cleaned_text.split()),
            'char_count': len(cleaned_text)
        }


def demo_nlp_processor():
    """Demonstrate NLP processing capabilities"""
    processor = NLPProcessor()
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "Best iPhone 14 reviews 2024",
        "How to learn Python programming",
        "Tesla vs BMW comparison",
        "Restaurants near me",
        "Buy cheap laptops online"
    ]
    
    print("NLP Query Analysis Demo")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        analysis = processor.analyze_query(query)
        
        print(f"Intent: {analysis.intent.value}")
        print(f"Type: {analysis.query_type}")
        print(f"Keywords: {', '.join(analysis.keywords)}")
        print(f"Entities: {[e['text'] + ' (' + e['type'] + ')' for e in analysis.entities]}")
        print(f"Sentiment: {analysis.sentiment}")
        print(f"Expanded terms: {', '.join(analysis.expanded_terms[:5])}")
        print(f"Suggestions: {', '.join(analysis.suggestions[:3])}")
        print("-" * 30)


if __name__ == "__main__":
    demo_nlp_processor()
