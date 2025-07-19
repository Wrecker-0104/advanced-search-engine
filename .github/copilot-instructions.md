<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# Advanced Search Engine Project - Copilot Instructions

This is an advanced search engine project with three main components:

## Project Structure
- `frontend/` - React-based modern search interface
- `backend/` - FastAPI backend with security and database integration
- `search-engine-core/` - Core search algorithms and web crawler

## Key Technologies
- **Frontend**: React, TypeScript, Bootstrap, Elastic Search UI
- **Backend**: FastAPI, PostgreSQL, Redis, JWT Authentication
- **Search Core**: Python, Elasticsearch, Scrapy, spaCy, BERT, NetworkX

## Search Algorithms Implemented
- TF-IDF (Term Frequency-Inverse Document Frequency)
- BM25 (Best Matching 25) 
- PageRank for authority scoring
- Semantic similarity with BERT embeddings
- Cosine similarity for vector space models

## Code Guidelines
1. Use TypeScript for frontend components with proper type definitions
2. Implement comprehensive error handling and input validation
3. Follow REST API best practices with proper HTTP status codes
4. Use async/await patterns for asynchronous operations
5. Implement proper logging and monitoring
6. Add comprehensive unit tests and integration tests
7. Follow security best practices (HTTPS, input sanitization, rate limiting)
8. Use proper database indexing and query optimization
9. Implement caching strategies with Redis
10. Follow Google-style docstrings for Python code

## Security Considerations
- Always validate and sanitize user inputs
- Use parameterized queries to prevent SQL injection
- Implement proper authentication and authorization
- Use HTTPS for all communications
- Add rate limiting to prevent abuse
- Log security events for monitoring

## Performance Optimization
- Implement efficient indexing strategies
- Use database connection pooling
- Add proper caching layers
- Optimize search algorithms for speed
- Use lazy loading for large datasets
- Implement pagination for search results

## Testing Requirements
- Write unit tests for all core functions
- Add integration tests for API endpoints
- Test search algorithms with various datasets
- Performance testing for high load scenarios
- Security testing for vulnerabilities
