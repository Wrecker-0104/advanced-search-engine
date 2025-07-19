"""
Advanced Web Crawler with Google-like capabilities
Implements:
- Politeness (robots.txt compliance)
- Rate limiting
- Content extraction
- Link discovery
- Duplicate detection
"""

import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.http import Request
from urllib.robotparser import RobotFileParser
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import time
import hashlib
import json
import logging
from typing import Set, Dict, List, Optional
import requests
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedWebCrawler(CrawlSpider):
    """
    Advanced web crawler with Google-like features
    """
    name = 'advanced_crawler'
    
    # Crawler settings
    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'DOWNLOAD_DELAY': 1,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'CONCURRENT_REQUESTS': 16,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 8,
        'DEPTH_LIMIT': 5,
        'CLOSESPIDER_PAGECOUNT': 10000,
        'USER_AGENT': 'AdvancedSearchBot/1.0 (+http://your-domain.com/bot)',
        'HTTPCACHE_ENABLED': True,
        'HTTPCACHE_EXPIRATION_SECS': 3600,
        'AUTOTHROTTLE_ENABLED': True,
        'AUTOTHROTTLE_START_DELAY': 1,
        'AUTOTHROTTLE_MAX_DELAY': 10,
        'AUTOTHROTTLE_TARGET_CONCURRENCY': 2.0,
    }
    
    # Link extraction rules
    rules = (
        Rule(
            LinkExtractor(
                deny_extensions=['pdf', 'jpg', 'jpeg', 'png', 'gif', 'mp4', 'mp3', 'zip'],
                allow_domains=[],  # Will be set dynamically
                deny=r'(login|admin|private|api|ajax)',
            ),
            callback='parse_page',
            follow=True
        ),
    )
    
    def __init__(self, start_urls=None, allowed_domains=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Set start URLs and allowed domains
        self.start_urls = start_urls or [
            'https://en.wikipedia.org/',
            'https://news.ycombinator.com/',
            'https://stackoverflow.com/',
        ]
        
        self.allowed_domains = allowed_domains or [
            'wikipedia.org',
            'news.ycombinator.com', 
            'stackoverflow.com'
        ]
        
        # Initialize crawler state
        self.visited_urls: Set[str] = set()
        self.url_hashes: Set[str] = set()
        self.documents: List[Dict] = []
        self.robots_cache: Dict[str, RobotFileParser] = {}
        
        # Content extraction patterns
        self.content_selectors = {
            'title': ['title', 'h1', '.title', '#title'],
            'content': ['article', '.content', '.main', 'main', '.post'],
            'description': ['meta[name="description"]', '.description', '.summary'],
            'keywords': ['meta[name="keywords"]', '.tags', '.keywords'],
        }
    
    def start_requests(self):
        """Generate initial requests"""
        for url in self.start_urls:
            if self.can_crawl_url(url):
                yield Request(
                    url,
                    callback=self.parse_page,
                    meta={
                        'depth': 0,
                        'timestamp': datetime.now().isoformat()
                    }
                )
    
    def can_crawl_url(self, url: str) -> bool:
        """Check if URL can be crawled according to robots.txt"""
        try:
            parsed_url = urlparse(url)
            domain = parsed_url.netloc
            
            # Get robots.txt for domain
            if domain not in self.robots_cache:
                robots_url = f"{parsed_url.scheme}://{domain}/robots.txt"
                try:
                    rp = RobotFileParser()
                    rp.set_url(robots_url)
                    rp.read()
                    self.robots_cache[domain] = rp
                except Exception as e:
                    logger.warning(f"Could not read robots.txt for {domain}: {e}")
                    # Allow crawling if robots.txt is not accessible
                    return True
            
            # Check if URL is allowed
            rp = self.robots_cache[domain]
            return rp.can_fetch(self.custom_settings['USER_AGENT'], url)
            
        except Exception as e:
            logger.error(f"Error checking robots.txt for {url}: {e}")
            return True
    
    def parse_page(self, response):
        """Parse a web page and extract content"""
        try:
            # Skip if already visited (duplicate detection)
            url_hash = hashlib.md5(response.url.encode()).hexdigest()
            if url_hash in self.url_hashes:
                return
            
            self.url_hashes.add(url_hash)
            self.visited_urls.add(response.url)
            
            # Extract content using BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer"]):
                script.decompose()
            
            # Extract structured data
            document = self.extract_document_data(soup, response)
            
            if self.is_valid_document(document):
                self.documents.append(document)
                
                # Save document to file (in production, save to database)
                self.save_document(document)
                
                logger.info(f"Crawled: {response.url} - Title: {document.get('title', 'N/A')}")
            
            # Extract and follow links (handled by rules)
            yield from self.extract_links(response, soup)
            
        except Exception as e:
            logger.error(f"Error parsing {response.url}: {e}")
    
    def extract_document_data(self, soup: BeautifulSoup, response) -> Dict:
        """Extract structured data from HTML"""
        document = {
            'url': response.url,
            'title': '',
            'content': '',
            'description': '',
            'keywords': [],
            'headings': [],
            'links': [],
            'images': [],
            'timestamp': datetime.now().isoformat(),
            'domain': urlparse(response.url).netloc,
            'language': 'en',  # Could be detected automatically
            'category': 'general',  # Could be classified automatically
        }
        
        # Extract title
        title_element = soup.find('title')
        if title_element:
            document['title'] = title_element.get_text().strip()
        
        # Extract meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'})
        if meta_desc and meta_desc.get('content'):
            document['description'] = meta_desc['content'].strip()
        
        # Extract meta keywords
        meta_keywords = soup.find('meta', attrs={'name': 'keywords'})
        if meta_keywords and meta_keywords.get('content'):
            document['keywords'] = [kw.strip() for kw in meta_keywords['content'].split(',')]
        
        # Extract main content
        content_areas = soup.find_all(['article', 'main', '.content', '.post'])
        if not content_areas:
            # Fallback to body content
            content_areas = [soup.find('body')]
        
        content_text = ""
        for area in content_areas:
            if area:
                content_text += area.get_text(separator=' ', strip=True)
        
        document['content'] = self.clean_text(content_text)
        
        # Extract headings
        headings = []
        for heading in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            headings.append({
                'level': int(heading.name[1]),
                'text': heading.get_text().strip()
            })
        document['headings'] = headings
        
        # Extract internal links
        links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.startswith('http') or href.startswith('/'):
                absolute_url = urljoin(response.url, href)
                links.append({
                    'url': absolute_url,
                    'text': link.get_text().strip(),
                    'title': link.get('title', '')
                })
        document['links'] = links[:50]  # Limit to first 50 links
        
        # Extract images
        images = []
        for img in soup.find_all('img', src=True):
            images.append({
                'src': urljoin(response.url, img['src']),
                'alt': img.get('alt', ''),
                'title': img.get('title', '')
            })
        document['images'] = images[:10]  # Limit to first 10 images
        
        return document
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        import re
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:()\[\]{}"\'-]', '', text)
        
        return text.strip()
    
    def is_valid_document(self, document: Dict) -> bool:
        """Check if document is valid for indexing"""
        # Must have title and content
        if not document.get('title') or not document.get('content'):
            return False
        
        # Content must be substantial (at least 100 characters)
        if len(document['content']) < 100:
            return False
        
        # Filter out non-English content (simple heuristic)
        english_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = set(document['content'].lower().split())
        english_ratio = len(words.intersection(english_words)) / max(len(words), 1)
        
        if english_ratio < 0.05:  # Less than 5% English words
            return False
        
        return True
    
    def extract_links(self, response, soup):
        """Extract and filter links for crawling"""
        for link in soup.find_all('a', href=True):
            href = link['href']
            absolute_url = urljoin(response.url, href)
            
            # Check domain restrictions
            parsed_url = urlparse(absolute_url)
            if parsed_url.netloc not in self.allowed_domains:
                continue
            
            # Check robots.txt
            if not self.can_crawl_url(absolute_url):
                continue
            
            # Check if already visited
            if absolute_url in self.visited_urls:
                continue
            
            yield Request(
                absolute_url,
                callback=self.parse_page,
                meta={
                    'depth': response.meta.get('depth', 0) + 1,
                    'timestamp': datetime.now().isoformat()
                }
            )
    
    def save_document(self, document: Dict):
        """Save document to storage"""
        # Create data directory if it doesn't exist
        os.makedirs('crawled_data', exist_ok=True)
        
        # Save individual document
        filename = f"crawled_data/doc_{hashlib.md5(document['url'].encode()).hexdigest()[:8]}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(document, f, ensure_ascii=False, indent=2)
    
    def closed(self, reason):
        """Called when spider closes"""
        logger.info(f"Crawler finished. Reason: {reason}")
        logger.info(f"Total documents crawled: {len(self.documents)}")
        logger.info(f"Total URLs visited: {len(self.visited_urls)}")
        
        # Save crawl summary
        summary = {
            'total_documents': len(self.documents),
            'total_urls_visited': len(self.visited_urls),
            'crawl_time': datetime.now().isoformat(),
            'reason': reason,
            'domains': list(self.allowed_domains)
        }
        
        with open('crawled_data/crawl_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


class CrawlerManager:
    """Manage crawler execution and configuration"""
    
    def __init__(self, config_file: str = None):
        self.config = self.load_config(config_file)
    
    def load_config(self, config_file: str) -> Dict:
        """Load crawler configuration"""
        default_config = {
            'start_urls': [
                'https://en.wikipedia.org/',
                'https://news.ycombinator.com/',
                'https://stackoverflow.com/',
            ],
            'allowed_domains': [
                'wikipedia.org',
                'news.ycombinator.com',
                'stackoverflow.com'
            ],
            'max_pages': 1000,
            'delay': 1,
            'depth_limit': 5,
        }
        
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def run_crawler(self):
        """Run the web crawler"""
        process = CrawlerProcess({
            'USER_AGENT': 'AdvancedSearchBot/1.0',
            'ROBOTSTXT_OBEY': True,
            'DOWNLOAD_DELAY': self.config['delay'],
            'DEPTH_LIMIT': self.config['depth_limit'],
            'CLOSESPIDER_PAGECOUNT': self.config['max_pages'],
        })
        
        process.crawl(
            AdvancedWebCrawler,
            start_urls=self.config['start_urls'],
            allowed_domains=self.config['allowed_domains']
        )
        
        process.start()


if __name__ == '__main__':
    # Run the crawler
    crawler_manager = CrawlerManager()
    crawler_manager.run_crawler()
