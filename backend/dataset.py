"""
Advanced Search Engine Dataset
Comprehensive dataset with real-world search data
"""

import json
from datetime import datetime, timedelta
import random

# Large dataset of documents for search
COMPREHENSIVE_DATASET = [
    # Technology Documents
    {
        "id": 1,
        "title": "Introduction to Machine Learning with Python",
        "url": "https://python-ml.org/intro",
        "snippet": "Learn machine learning fundamentals using Python. Covers supervised learning, unsupervised learning, neural networks, and deep learning with practical examples and code implementations.",
        "content": "Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn and make decisions from data. This comprehensive guide covers linear regression, decision trees, random forests, SVM, neural networks, and deep learning frameworks like TensorFlow and PyTorch.",
        "domain": "python-ml.org",
        "category": "technology",
        "subcategory": "machine-learning",
        "date": "2024-01-15",
        "score": 0.95,
        "keywords": ["machine learning", "python", "ai", "neural networks", "deep learning"],
        "language": "en",
        "reading_time": 15,
        "difficulty": "intermediate"
    },
    {
        "id": 2,
        "title": "Python Programming Best Practices and Code Standards",
        "url": "https://python-guide.org/best-practices",
        "snippet": "Professional Python development techniques, PEP 8 compliance, code organization, testing strategies, and performance optimization for enterprise applications.",
        "content": "This guide covers Python best practices including PEP 8 style guide, proper project structure, virtual environments, testing with pytest, documentation with Sphinx, and deployment strategies.",
        "domain": "python-guide.org",
        "category": "programming",
        "subcategory": "python",
        "date": "2024-02-20",
        "score": 0.88,
        "keywords": ["python", "best practices", "pep8", "testing", "optimization"],
        "language": "en",
        "reading_time": 12,
        "difficulty": "intermediate"
    },
    {
        "id": 3,
        "title": "Data Science with Python: NumPy, Pandas, and Matplotlib",
        "url": "https://datasciencehandbook.com/python",
        "snippet": "Complete data science tutorial using Python libraries. Learn data manipulation, analysis, visualization, and statistical modeling with real datasets.",
        "content": "Comprehensive data science guide covering NumPy for numerical computing, Pandas for data manipulation, Matplotlib and Seaborn for visualization, and scikit-learn for machine learning.",
        "domain": "datasciencehandbook.com",
        "category": "data-science",
        "subcategory": "python",
        "date": "2024-03-10",
        "score": 0.92,
        "keywords": ["data science", "python", "numpy", "pandas", "matplotlib", "analysis"],
        "language": "en",
        "reading_time": 20,
        "difficulty": "intermediate"
    },
    {
        "id": 4,
        "title": "Building REST APIs with FastAPI and Python",
        "url": "https://fastapi.tiangolo.com/tutorial/",
        "snippet": "Modern web API development with FastAPI. Automatic documentation, type hints, async support, and production deployment strategies.",
        "content": "FastAPI is a modern, fast web framework for building APIs with Python. Learn about automatic API documentation, data validation, authentication, middleware, and deployment with Docker.",
        "domain": "fastapi.tiangolo.com",
        "category": "web-development",
        "subcategory": "backend",
        "date": "2024-04-05",
        "score": 0.87,
        "keywords": ["fastapi", "python", "api", "web development", "backend"],
        "language": "en",
        "reading_time": 18,
        "difficulty": "intermediate"
    },
    {
        "id": 5,
        "title": "Artificial Intelligence Research Trends 2024",
        "url": "https://ai-research.org/papers/2024",
        "snippet": "Latest AI research including transformer models, computer vision breakthroughs, natural language processing, and ethical AI considerations.",
        "content": "Comprehensive overview of 2024 AI research including GPT-4 improvements, vision transformers, multimodal AI, reinforcement learning advances, and responsible AI development practices.",
        "domain": "ai-research.org",
        "category": "research",
        "subcategory": "artificial-intelligence",
        "date": "2024-05-12",
        "score": 0.91,
        "keywords": ["artificial intelligence", "research", "transformers", "computer vision", "nlp"],
        "language": "en",
        "reading_time": 25,
        "difficulty": "advanced"
    },
    {
        "id": 6,
        "title": "React.js Complete Guide: Hooks, Context, and Performance",
        "url": "https://reactjs.dev/learn",
        "snippet": "Master modern React development with hooks, context API, performance optimization, and best practices for building scalable web applications.",
        "content": "Complete React.js guide covering functional components, useState, useEffect, useContext, custom hooks, performance optimization with React.memo and useMemo, and testing with Jest.",
        "domain": "reactjs.dev",
        "category": "web-development",
        "subcategory": "frontend",
        "date": "2024-06-01",
        "score": 0.89,
        "keywords": ["react", "javascript", "hooks", "frontend", "web development"],
        "language": "en",
        "reading_time": 22,
        "difficulty": "intermediate"
    },
    {
        "id": 7,
        "title": "Docker Containerization and Kubernetes Orchestration",
        "url": "https://docker.com/tutorials/kubernetes",
        "snippet": "Learn containerization with Docker and orchestration with Kubernetes. Deployment strategies, scaling, monitoring, and DevOps best practices.",
        "content": "Complete guide to containerization with Docker and Kubernetes orchestration. Covers Dockerfile creation, image optimization, Kubernetes deployments, services, ingress, and CI/CD pipelines.",
        "domain": "docker.com",
        "category": "devops",
        "subcategory": "containerization",
        "date": "2024-06-15",
        "score": 0.86,
        "keywords": ["docker", "kubernetes", "containerization", "devops", "deployment"],
        "language": "en",
        "reading_time": 30,
        "difficulty": "advanced"
    },
    {
        "id": 8,
        "title": "Database Design and SQL Optimization Techniques",
        "url": "https://sqlperformance.com/database-design",
        "snippet": "Advanced database design principles, SQL query optimization, indexing strategies, and performance tuning for high-traffic applications.",
        "content": "Comprehensive database design guide covering normalization, indexing strategies, query optimization, performance monitoring, and scaling techniques for PostgreSQL, MySQL, and MongoDB.",
        "domain": "sqlperformance.com",
        "category": "database",
        "subcategory": "sql",
        "date": "2024-07-01",
        "score": 0.84,
        "keywords": ["database", "sql", "optimization", "indexing", "performance"],
        "language": "en",
        "reading_time": 28,
        "difficulty": "advanced"
    },
    {
        "id": 9,
        "title": "Cybersecurity Fundamentals and Ethical Hacking",
        "url": "https://cybersecurity-guide.org/fundamentals",
        "snippet": "Essential cybersecurity concepts, penetration testing, network security, and ethical hacking methodologies for protecting digital assets.",
        "content": "Complete cybersecurity guide covering threat modeling, vulnerability assessment, penetration testing with Kali Linux, network security protocols, and incident response procedures.",
        "domain": "cybersecurity-guide.org",
        "category": "security",
        "subcategory": "cybersecurity",
        "date": "2024-07-10",
        "score": 0.90,
        "keywords": ["cybersecurity", "ethical hacking", "penetration testing", "network security"],
        "language": "en",
        "reading_time": 35,
        "difficulty": "advanced"
    },
    {
        "id": 10,
        "title": "Cloud Computing with AWS: Architecture and Services",
        "url": "https://aws.amazon.com/getting-started/",
        "snippet": "Master Amazon Web Services architecture, core services, serverless computing, and cloud-native application development patterns.",
        "content": "Comprehensive AWS guide covering EC2, S3, Lambda, RDS, VPC, CloudFormation, serverless architectures, microservices, and cost optimization strategies.",
        "domain": "aws.amazon.com",
        "category": "cloud-computing",
        "subcategory": "aws",
        "date": "2024-07-20",
        "score": 0.88,
        "keywords": ["aws", "cloud computing", "serverless", "microservices", "architecture"],
        "language": "en",
        "reading_time": 40,
        "difficulty": "intermediate"
    }
]

# Extended suggestions based on comprehensive dataset
EXTENDED_SUGGESTIONS = [
    "machine learning algorithms",
    "python programming",
    "data science tutorials",
    "artificial intelligence",
    "web development",
    "neural networks",
    "computer vision",
    "natural language processing",
    "react hooks",
    "fastapi tutorial",
    "docker containerization",
    "kubernetes orchestration",
    "database optimization",
    "sql performance",
    "cybersecurity fundamentals",
    "ethical hacking",
    "aws cloud services",
    "serverless computing",
    "microservices architecture",
    "devops practices"
]

# Categories with counts
CATEGORIES = [
    {"id": "all", "name": "All Categories", "count": len(COMPREHENSIVE_DATASET)},
    {"id": "technology", "name": "Technology", "count": 1},
    {"id": "programming", "name": "Programming", "count": 1},
    {"id": "data-science", "name": "Data Science", "count": 1},
    {"id": "web-development", "name": "Web Development", "count": 2},
    {"id": "research", "name": "Research", "count": 1},
    {"id": "devops", "name": "DevOps", "count": 1},
    {"id": "database", "name": "Database", "count": 1},
    {"id": "security", "name": "Security", "count": 1},
    {"id": "cloud-computing", "name": "Cloud Computing", "count": 1}
]

# Export data
if __name__ == "__main__":
    with open("search_dataset.json", "w") as f:
        json.dump({
            "documents": COMPREHENSIVE_DATASET,
            "suggestions": EXTENDED_SUGGESTIONS,
            "categories": CATEGORIES
        }, f, indent=2)
