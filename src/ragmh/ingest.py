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
    """Scrape a single Mind.org.uk page (simplified version)"""
    url = f"https://www.mind.org.uk/information-support/types-of-mental-health-problems/{topic}/"
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Educational Research Project)'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Note: In production, use BeautifulSoup for proper HTML parsing
            # This is a simplified example
            content = {
                'topic': topic,
                'url': url,
                'raw_html': response.text[:1000],  # Store first 1000 chars as sample
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
    
    logger.info("Data ingestion complete!")
    
    # Print summary
    print("\n=== Data Ingestion Summary ===")
    print(f"CounselChat: {COUNSELCHAT_DIR.absolute()}")
    print(f"Mind.org.uk: {MIND_DIR.absolute()}")
    print(f"Reddit: {REDDIT_DIR.absolute()}")


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
    else:
        ingest_all_data()