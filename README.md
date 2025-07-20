# Advanced Search Engine Project

[![Deploy to Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/Wrecker-0104/advanced-search-engine)
[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/fastapi)
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/Wrecker-0104/advanced-search-engine)

## 🚀 Project Overview

This is a comprehensive, production-ready search engine built with modern technologies and algorithms similar to those used by Google. The project now includes **real web search integration** via Yahoo Search API.

### 1. **Advanced Frontend** (React + TypeScript)
- Modern, responsive search interface with real-time suggestions
- Voice search capabilities and advanced filtering
- API key management and configuration UI
- Bootstrap 5 responsive design for all devices
- Search result highlighting and bookmarking
- Multi-modal search interface with autocomplete

### 2. **Advanced Backend** (FastAPI + PostgreSQL + Yahoo API)
- High-performance REST API with comprehensive security
- **Yahoo Search API integration** for real web search results
- API key authentication with secure token management
- PostgreSQL database with SQLAlchemy ORM
- Redis caching for improved performance
- Rate limiting, CORS, and DDoS protection
- Input sanitization and SQL injection prevention
- OpenAPI documentation with Swagger

### 3. **Google-like Search Algorithms** (Python Core Engine)
- **TF-IDF (Term Frequency-Inverse Document Frequency)** - Relevance scoring
- **BM25 (Best Matching 25)** - Advanced probabilistic ranking
- **PageRank Algorithm** - Link-based authority scoring  
- **Semantic Search** - Vector-based similarity matching
- **Web Crawler** - Automated content discovery with Scrapy
- **NLP Query Processing** - Intent recognition and entity extraction
- **Search Analytics** - Performance monitoring and A/B testing

## 🚀 Quick Start

### Prerequisites
- Node.js 18+ and npm
- Python 3.9+
- PostgreSQL 13+
- Redis 6+
- Elasticsearch 8+

### Installation

1. **Clone and setup the project**:
```bash
cd "search engine...2"
```

2. **Setup Frontend**:
```bash
cd frontend
npm install
npm start
```

3. **Setup Backend**:
```bash
cd backend
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
uvicorn main:app --reload
```

4. **Setup Search Engine Core**:
```bash
cd search-engine-core
pip install -r requirements.txt
python crawler/main.py
```

## 🔧 Configuration

### Environment Variables
Create `.env` files in each directory:

**Backend `.env`**:
```env
DATABASE_URL=postgresql://user:password@localhost/searchengine
REDIS_URL=redis://localhost:6379
JWT_SECRET_KEY=your-secret-key
ELASTICSEARCH_URL=http://localhost:9200

# API Key Authentication
API_KEY_REQUIRED=true
VALID_API_KEYS=["sk-search-engine-2025-demo-key-123456"]

# Yahoo Search API (Real Web Search)
YAHOO_SEARCH_API_URL=https://www.searchapi.io/api/v1/search
YAHOO_SEARCH_API_KEY=your-searchapi-io-key-here
USE_REAL_SEARCH=true
```

**Search Core `.env`**:
```env
ELASTICSEARCH_URL=http://localhost:9200
CRAWL_DELAY=1
MAX_DEPTH=5
RESPECT_ROBOTS_TXT=true
```

### 🔍 Yahoo Search API Setup
1. **Get API Key**: Sign up at [SearchAPI.io](https://www.searchapi.io/)
2. **Configure**: Add your key to `YAHOO_SEARCH_API_KEY` in `.env`
3. **Test**: Run a search to get real web results!

```bash
# Test with real Yahoo search
curl -H "X-API-Key: sk-search-engine-2025-demo-key-123456" \
     "http://localhost:8002/api/search?q=artificial+intelligence"
```

## 📊 Features

### Search Capabilities
- **Full-text Search**: Advanced text matching with stemming and synonyms
- **Boolean Search**: Support for AND, OR, NOT operators
- **Phrase Search**: Exact phrase matching with proximity scoring
- **Faceted Search**: Filter by categories, dates, content types
- **Auto-complete**: Real-time search suggestions
- **Spell Correction**: Query correction and suggestion
- **Multi-language**: Support for multiple languages
- **Semantic Search**: Context-aware search using NLP models

### Performance & Scalability
- **Caching**: Multi-level caching with Redis
- **Load Balancing**: Horizontal scaling support
- **CDN Integration**: Static asset optimization
- **Database Optimization**: Connection pooling and query optimization
- **Monitoring**: Prometheus metrics and Grafana dashboards

### Security Features
- **Authentication**: JWT-based user authentication
- **Authorization**: Role-based access control
- **Input Validation**: Comprehensive input sanitization
- **Rate Limiting**: API rate limiting and abuse prevention
- **HTTPS**: SSL/TLS encryption enforcement
- **CORS**: Cross-origin resource sharing configuration
- **Firewall**: Network-level security rules

## 🧠 Algorithms Implemented

### Ranking Algorithms
1. **TF-IDF**: Classic term frequency scoring
2. **BM25**: Probabilistic ranking function
3. **PageRank**: Link-based authority scoring
4. **Cosine Similarity**: Vector space model similarity
5. **Semantic Similarity**: BERT-based semantic understanding

### Machine Learning Features
- **Query Classification**: Intent recognition
- **Result Clustering**: Grouping similar results
- **Personalization**: User behavior-based ranking
- **A/B Testing**: Search result optimization

## 📈 Monitoring & Analytics

- **Search Analytics**: Query analysis and performance metrics
- **User Behavior**: Click-through rates and search patterns
- **System Health**: Application performance monitoring
- **Error Tracking**: Comprehensive error logging and alerting

## 🧪 Testing

```bash
# Frontend tests
cd frontend && npm test

# Backend tests
cd backend && python -m pytest

# Core engine tests
cd search-engine-core && python -m pytest
```

## 📚 API Documentation

Once the backend is running, visit:
- **API Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🔗 Links

- [Elasticsearch Documentation](https://www.elastic.co/guide/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs/)
- [Scrapy Documentation](https://docs.scrapy.org/)

---

**Built with ❤️ using modern web technologies and advanced search algorithms**
