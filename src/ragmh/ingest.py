"""Data ingestion pipeline for mental health datasets"""
import os
import json
import csv
import requests
from pathlib import Path
from typing import List, Dict, Optional
import logging
from datasets import load_dataset
import time
import random
from bs4 import BeautifulSoup
from .chunk import chunk_pubmed_data, chunk_who_data
from .embed import embed_pubmed_and_who

logger = logging.getLogger(__name__)

# Data directories - use absolute path to project root
# Go up from src/ragmh to project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
COUNSELCHAT_DIR = DATA_DIR / "counselchat"
MIND_DIR = DATA_DIR / "mind"
REDDIT_DIR = DATA_DIR / "reddit"

print(f"Data will be saved to: {DATA_DIR.absolute()}")

# Create directories
for dir in [COUNSELCHAT_DIR, MIND_DIR, REDDIT_DIR]:
    dir.mkdir(parents=True, exist_ok=True)

# --- CounselChat Dataset Functions ---
def download_counselchat() -> Dict[str, List[Dict]]:
    """Download CounselChat dataset from HuggingFace"""
    logger.info("Downloading CounselChat dataset...")
    
    try:
        # Load dataset from HuggingFace
        dataset = load_dataset("nbertagnolli/counsel-chat")
        
        # Convert to dict format
        data = {
            "train": [item for item in dataset["train"]],
            "test": [item for item in dataset["test"]] if "test" in dataset else []
        }
        
        # Save to JSON
        output_path = COUNSELCHAT_DIR / "counselchat_full.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(data['train'])} training examples to {output_path}")
        return data
        
    except Exception as e:
        logger.error(f"Failed to download CounselChat: {e}")
        return {"train": [], "test": []}

def process_counselchat_qa(data: Dict[str, List[Dict]]) -> List[Dict]:
    """Extract Q&A pairs from CounselChat"""
    qa_pairs = []
    
    for split in ['train', 'test']:
        for item in data.get(split, []):
            # Extract relevant fields
            qa_pair = {
                'question': item.get('questionText', ''),
                'answer': item.get('answerText', ''),
                'topic': item.get('topic', 'general'),
                'upvotes': item.get('upvotes', 0),
                'views': item.get('views', 0),
                'source': 'counselchat'
            }
            qa_pairs.append(qa_pair)
    
    # Save processed data
    output_path = COUNSELCHAT_DIR / "counselchat_qa.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(qa_pairs, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Processed {len(qa_pairs)} Q&A pairs")
    return qa_pairs

# --- Mind.org.uk Scraping Functions ---
def get_mind_topics() -> List[str]:
    """Get list of mental health topics from Mind.org.uk"""
    # Common mental health topics covered by Mind
    topics = [
        "anxiety", "depression", "stress", "panic-attacks",
        "phobias", "obsessive-compulsive-disorder", "ptsd",
        "eating-problems", "sleep-problems", "self-harm",
        "suicidal-feelings", "bipolar-disorder", "personality-disorders",
        "psychosis", "schizophrenia", "anger", "loneliness",
        "self-esteem", "grief", "trauma"
    ]
    return topics

def scrape_mind_page(topic: str) -> Dict:
    """Scrape a single Mind.org.uk page with improved anti-bot evasion and parsing"""
    url = f"https://www.mind.org.uk/information-support/types-of-mental-health-problems/{topic}/"
    session = requests.Session()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/'
    }
    try:
        response = session.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "lxml")
            text = soup.get_text(separator=' ', strip=True)
            content = {
                'topic': topic,
                'url': url,
                'text': text[:2000],  # Store first 2000 chars
                'status': 'success',
                'source': 'mind.org.uk'
            }
            logger.info(f"Scraped {topic} successfully")
        else:
            content = {
                'topic': topic,
                'url': url,
                'status': f'error_{response.status_code}',
                'source': 'mind.org.uk'
            }
            logger.warning(f"Failed to scrape {topic}: {response.status_code}")
        # Random delay to mimic human browsing
        time.sleep(random.uniform(2, 5))
        return content
    except Exception as e:
        logger.error(f"Error scraping {topic}: {e}")
        return {
            'topic': topic,
            'url': url,
            'status': 'error',
            'error': str(e),
            'source': 'mind.org.uk'
        }

def scrape_mind_content(topics: Optional[List[str]] = None) -> List[Dict]:
    """Scrape content from Mind.org.uk"""
    if topics is None:
        topics = get_mind_topics()
    
    scraped_data = []
    
    for topic in topics:
        logger.info(f"Scraping Mind.org.uk: {topic}")
        data = scrape_mind_page(topic)
        scraped_data.append(data)
        
        # Be respectful - add delay between requests
        time.sleep(2)
    
    # Save raw data
    output_path = MIND_DIR / "mind_raw.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(scraped_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Scraped {len(scraped_data)} Mind.org.uk pages")
    return scraped_data

# --- Reddit Mental Health Functions ---
def fetch_reddit_posts(subreddit: str = "mentalhealth", 
                      limit: int = 100) -> List[Dict]:
    """Fetch posts from Reddit (using public JSON API)"""
    posts = []
    
    # Reddit's public JSON API
    url = f"https://www.reddit.com/r/{subreddit}/top.json"
    headers = {'User-Agent': 'Educational Research Project 1.0'}
    params = {'limit': limit, 't': 'month'}  # Top posts from last month
    
    try:
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            data = response.json()
            
            for post in data['data']['children']:
                post_data = post['data']
                
                # Extract relevant fields
                processed_post = {
                    'title': post_data.get('title', ''),
                    'text': post_data.get('selftext', ''),
                    'score': post_data.get('score', 0),
                    'num_comments': post_data.get('num_comments', 0),
                    'created_utc': post_data.get('created_utc', 0),
                    'id': post_data.get('id', ''),
                    'subreddit': subreddit,
                    'source': 'reddit'
                }
                
                # Only include posts with substantial content
                if len(processed_post['text']) > 50:
                    posts.append(processed_post)
        
        logger.info(f"Fetched {len(posts)} posts from r/{subreddit}")
        
    except Exception as e:
        logger.error(f"Error fetching Reddit posts: {e}")
    
    # Save posts
    output_path = REDDIT_DIR / f"reddit_{subreddit}_posts.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(posts, f, indent=2, ensure_ascii=False)
    
    return posts

# --- PubMed Ingestion Functions ---
def fetch_pubmed_abstracts(query: str, max_results: int = 50) -> list:
    """Fetch PubMed abstracts for a given query using Entrez API"""
    import xml.etree.ElementTree as ET
    import urllib.parse
    PUBMED_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    PUBMED_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {
        'db': 'pubmed',
        'term': query,
        'retmax': max_results,
        'retmode': 'json'
    }
    # Step 1: Search for IDs
    resp = requests.get(PUBMED_ESEARCH, params=params)
    idlist = resp.json().get('esearchresult', {}).get('idlist', [])
    if not idlist:
        logger.warning(f"No PubMed results for query: {query}")
        return []
    # Step 2: Fetch abstracts
    fetch_params = {
        'db': 'pubmed',
        'id': ','.join(idlist),
        'retmode': 'xml'
    }
    fetch_resp = requests.get(PUBMED_EFETCH, params=fetch_params)
    root = ET.fromstring(fetch_resp.content)
    results = []
    for article in root.findall('.//PubmedArticle'):
        title = article.findtext('.//ArticleTitle', default='')
        abstract = article.findtext('.//Abstract/AbstractText', default='')
        pmid = article.findtext('.//PMID', default='')
        results.append({
            'title': title,
            'abstract': abstract,
            'pmid': pmid,
            'source': 'pubmed',
            'query': query
        })
    logger.info(f"Fetched {len(results)} PubMed abstracts for '{query}'")
    # Save to file
    PUBMED_DIR = DATA_DIR / "pubmed"
    PUBMED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PUBMED_DIR / f"pubmed_{query.replace(' ', '_')}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return results

# --- WHO Ingestion Functions ---
def fetch_who_topic_summary(topic: str) -> dict:
    """Download or scrape summary from WHO mental health topic page"""
    WHO_BASE = "https://www.who.int/news-room/fact-sheets/detail/"
    url = WHO_BASE + topic
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, 'lxml')
            # Try to extract the summary (first paragraph or intro)
            summary = ''
            main = soup.find('div', {'id': 'PageContent_T001'}) or soup.find('main')
            if main:
                p = main.find('p')
                if p:
                    summary = p.get_text(strip=True)
            if not summary:
                # fallback: first <p> in page
                p = soup.find('p')
                if p:
                    summary = p.get_text(strip=True)
            result = {
                'topic': topic,
                'url': url,
                'summary': summary,
                'status': 'success',
                'source': 'who'
            }
            logger.info(f"Fetched WHO summary for {topic}")
        else:
            result = {
                'topic': topic,
                'url': url,
                'status': f'error_{resp.status_code}',
                'source': 'who'
            }
            logger.warning(f"Failed to fetch WHO topic {topic}: {resp.status_code}")
        WHO_DIR = DATA_DIR / "who"
        WHO_DIR.mkdir(parents=True, exist_ok=True)
        output_path = WHO_DIR / f"who_{topic.replace(' ', '_')}.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        return result
    except Exception as e:
        logger.error(f"Error fetching WHO topic {topic}: {e}")
        return {
            'topic': topic,
            'url': url,
            'status': 'error',
            'error': str(e),
            'source': 'who'
        }

# --- Main ingestion function ---
def ingest_all_data():
    """Run complete data ingestion pipeline"""
    logger.info("Starting data ingestion pipeline...")
    logger.info(f"Project root: {PROJECT_ROOT.absolute()}")
    logger.info(f"Data directory: {DATA_DIR.absolute()}")
    # 1. Download CounselChat
    counselchat_data = download_counselchat()
    if counselchat_data['train']:
        process_counselchat_qa(counselchat_data)
    # 2. Scrape Mind.org.uk (limited to 3 topics for demo)
    mind_topics = get_mind_topics()[:3]  # Limit for demo
    scrape_mind_content(mind_topics)
    # 3. Fetch Reddit posts
    fetch_reddit_posts("mentalhealth", limit=50)
    fetch_reddit_posts("Anxiety", limit=30)
    fetch_reddit_posts("depression", limit=30)
    # 4. Fetch PubMed abstracts for key topics
    for topic in ["anxiety", "depression", "stress"]:
        fetch_pubmed_abstracts(topic, max_results=30)
    # 5. Fetch WHO summaries for key topics
    for topic in ["mental-health", "depression", "anxiety-disorders"]:
        fetch_who_topic_summary(topic)
    # 6. Chunk PubMed and WHO data (now in chunk.py)
    chunk_pubmed_data()
    chunk_who_data()
    # 7. Embed PubMed and WHO chunks (now in embed.py)
    embed_pubmed_and_who()
    logger.info("Data ingestion complete!")
    # Print summary
    print("\n=== Data Ingestion Summary ===")
    print(f"CounselChat: {COUNSELCHAT_DIR.absolute()}")
    print(f"Mind.org.uk: {MIND_DIR.absolute()}")
    print(f"Reddit: {REDDIT_DIR.absolute()}")
    print(f"PubMed: {(DATA_DIR / 'pubmed').absolute()}")
    print(f"WHO: {(DATA_DIR / 'who').absolute()}")
    print(f"Chunks: {(DATA_DIR / 'chunks').absolute()}")
    print(f"Embeddings: {(DATA_DIR / 'embeddings').absolute()}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Run specific ingestion or all
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "counselchat":
            data = download_counselchat()
            process_counselchat_qa(data)
        elif sys.argv[1] == "mind":
            scrape_mind_content()
        elif sys.argv[1] == "reddit":
            fetch_reddit_posts()
        elif sys.argv[1] == "pubmed":
            fetch_pubmed_abstracts("mental health")
        elif sys.argv[1] == "who":
            fetch_who_topic_summary("depression")
    else:
        ingest_all_data()