"""Text chunking pipeline for mental health documents"""
import json
from pathlib import Path
from typing import List, Dict, Tuple
import re
import logging

logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

# Chunking parameters
DEFAULT_CHUNK_SIZE = 512  # characters
DEFAULT_OVERLAP = 50      # character overlap between chunks
MIN_CHUNK_SIZE = 100      # minimum chunk size to keep

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\'\"]', '', text)
    # Strip leading/trailing whitespace
    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    # Simple sentence splitter (could use nltk for better results)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def create_chunks_with_overlap(text: str, 
                             chunk_size: int = DEFAULT_CHUNK_SIZE,
                             overlap: int = DEFAULT_OVERLAP) -> List[str]:
    """Create overlapping chunks from text"""
    chunks = []
    sentences = split_into_sentences(text)
    
    current_chunk = ""
    
    for sentence in sentences:
        # If adding sentence exceeds chunk size, save current chunk
        if current_chunk and len(current_chunk) + len(sentence) > chunk_size:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from previous
            if overlap > 0 and current_chunk:
                words = current_chunk.split()
                overlap_words = words[-(overlap//10):]  # Rough word-based overlap
                current_chunk = ' '.join(overlap_words) + ' '
            else:
                current_chunk = ""
        
        current_chunk += sentence + " "
    
    # Add final chunk
    if current_chunk.strip() and len(current_chunk.strip()) >= MIN_CHUNK_SIZE:
        chunks.append(current_chunk.strip())
    
    return chunks

def chunk_counselchat_data() -> List[Dict]:
    """Process CounselChat Q&A pairs into chunks"""
    input_file = DATA_DIR / "counselchat" / "counselchat_qa.json"
    
    if not input_file.exists():
        logger.error(f"CounselChat data not found at {input_file}")
        return []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        qa_pairs = json.load(f)
    
    chunks = []
    
    for qa in qa_pairs:
        # For Q&A, keep question and answer together as one chunk
        combined_text = f"Question: {qa['question']}\n\nAnswer: {qa['answer']}"
        cleaned_text = clean_text(combined_text)
        
        # If too long, chunk the answer part
        if len(cleaned_text) > DEFAULT_CHUNK_SIZE:
            question_part = f"Question: {qa['question']}\n\n"
            answer_chunks = create_chunks_with_overlap(qa['answer'])
            
            for i, chunk in enumerate(answer_chunks):
                chunk_data = {
                    'text': question_part + f"Answer: {chunk}" if i == 0 else chunk,
                    'source': 'counselchat',
                    'metadata': {
                        'topic': qa.get('topic', 'general'),
                        'upvotes': qa.get('upvotes', 0),
                        'chunk_index': i,
                        'total_chunks': len(answer_chunks)
                    }
                }
                chunks.append(chunk_data)
        else:
            # Single chunk for short Q&A
            chunk_data = {
                'text': cleaned_text,
                'source': 'counselchat',
                'metadata': {
                    'topic': qa.get('topic', 'general'),
                    'upvotes': qa.get('upvotes', 0),
                    'chunk_index': 0,
                    'total_chunks': 1
                }
            }
            chunks.append(chunk_data)
    
    logger.info(f"Created {len(chunks)} chunks from CounselChat")
    return chunks

def chunk_reddit_data() -> List[Dict]:
    """Process Reddit posts into chunks"""
    chunks = []
    reddit_files = list((DATA_DIR / "reddit").glob("*.json"))
    
    for file_path in reddit_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            posts = json.load(f)
        
        for post in posts:
            # Combine title and text
            full_text = f"Title: {post['title']}\n\n{post['text']}"
            cleaned_text = clean_text(full_text)
            
            # Only process posts with substantial content
            if len(cleaned_text) < MIN_CHUNK_SIZE:
                continue
            
            # Create chunks
            text_chunks = create_chunks_with_overlap(cleaned_text)
            
            for i, chunk in enumerate(text_chunks):
                chunk_data = {
                    'text': chunk,
                    'source': 'reddit',
                    'metadata': {
                        'subreddit': post['subreddit'],
                        'score': post.get('score', 0),
                        'num_comments': post.get('num_comments', 0),
                        'post_id': post.get('id', ''),
                        'chunk_index': i,
                        'total_chunks': len(text_chunks)
                    }
                }
                chunks.append(chunk_data)
    
    logger.info(f"Created {len(chunks)} chunks from Reddit")
    return chunks

def chunk_mind_data() -> List[Dict]:
    """Process Mind.org.uk data into chunks"""
    input_file = DATA_DIR / "mind" / "mind_raw.json"
    
    if not input_file.exists():
        logger.warning(f"Mind data not found at {input_file}")
        return []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        mind_data = json.load(f)
    
    chunks = []
    
    for page in mind_data:
        if page.get('status') != 'success':
            continue
        
        # Note: In real implementation, parse HTML properly
        # For now, just use the raw HTML sample
        raw_html = page.get('raw_html', '')
        
        # Very basic text extraction (should use BeautifulSoup)
        text = re.sub(r'<[^>]+>', ' ', raw_html)  # Remove HTML tags
        cleaned_text = clean_text(text)
        
        if len(cleaned_text) < MIN_CHUNK_SIZE:
            continue
        
        text_chunks = create_chunks_with_overlap(cleaned_text)
        
        for i, chunk in enumerate(text_chunks):
            chunk_data = {
                'text': chunk,
                'source': 'mind.org.uk',
                'metadata': {
                    'topic': page['topic'],
                    'url': page['url'],
                    'chunk_index': i,
                    'total_chunks': len(text_chunks)
                }
            }
            chunks.append(chunk_data)
    
    logger.info(f"Created {len(chunks)} chunks from Mind.org.uk")
    return chunks

def save_chunks(chunks: List[Dict], filename: str):
    """Save chunks to JSON file"""
    output_path = CHUNKS_DIR / filename
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(chunks)} chunks to {output_path}")

def chunk_all_data():
    """Run complete chunking pipeline"""
    logger.info("Starting chunking pipeline...")
    
    all_chunks = []
    
    # Process each data source
    counselchat_chunks = chunk_counselchat_data()
    all_chunks.extend(counselchat_chunks)
    save_chunks(counselchat_chunks, "counselchat_chunks.json")
    
    reddit_chunks = chunk_reddit_data()
    all_chunks.extend(reddit_chunks)
    save_chunks(reddit_chunks, "reddit_chunks.json")
    
    mind_chunks = chunk_mind_data()
    all_chunks.extend(mind_chunks)
    save_chunks(mind_chunks, "mind_chunks.json")
    
    # Save all chunks combined
    save_chunks(all_chunks, "all_chunks.json")
    
    # Print summary
    print("\n=== Chunking Summary ===")
    print(f"Total chunks created: {len(all_chunks)}")
    print(f"- CounselChat: {len(counselchat_chunks)} chunks")
    print(f"- Reddit: {len(reddit_chunks)} chunks")
    print(f"- Mind.org.uk: {len(mind_chunks)} chunks")
    print(f"\nChunks saved to: {CHUNKS_DIR.absolute()}")
    
    # Show sample chunk
    if all_chunks:
        print("\n=== Sample Chunk ===")
        sample = all_chunks[0]
        print(f"Source: {sample['source']}")
        print(f"Text: {sample['text'][:200]}...")
        print(f"Metadata: {sample['metadata']}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "counselchat":
            chunks = chunk_counselchat_data()
            save_chunks(chunks, "counselchat_chunks.json")
        elif sys.argv[1] == "reddit":
            chunks = chunk_reddit_data()
            save_chunks(chunks, "reddit_chunks.json")
        elif sys.argv[1] == "mind":
            chunks = chunk_mind_data()
            save_chunks(chunks, "mind_chunks.json")
    else:
        chunk_all_data()