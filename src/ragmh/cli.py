"""Command line interface for RAG Mental Health system"""
import argparse
import logging
from pathlib import Path

# Import our modules
from .ingest import ingest_all_data, download_counselchat, scrape_mind_content, fetch_reddit_posts
from .chunk import chunk_all_data
from .embed import embed_all_sources
from .vectordb import build_vectordb, test_search
from .chains import run_rag_pipeline, compare_with_vanilla_llm

def setup_logging(verbose: bool = False):
    """Configure logging"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="RAG Mental Health Support System",
        epilog="Example: python -m ragmh setup"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Setup command - run entire pipeline
    setup_parser = subparsers.add_parser('setup', help='Run complete setup pipeline')
    setup_parser.add_argument('--source', choices=['all', 'counselchat', 'reddit', 'mind'], 
                            default='counselchat', help='Data source to setup')
    
    # Individual pipeline steps
    ingest_parser = subparsers.add_parser('ingest', help='Ingest data from sources')
    ingest_parser.add_argument('source', choices=['all', 'counselchat', 'reddit', 'mind'],
                             help='Data source to ingest')
    
    chunk_parser = subparsers.add_parser('chunk', help='Chunk documents')
    chunk_parser.add_argument('--source', default='all', help='Source to chunk')
    
    embed_parser = subparsers.add_parser('embed', help='Generate embeddings')
    embed_parser.add_argument('--source', default='all', help='Source to embed')
    
    vectordb_parser = subparsers.add_parser('vectordb', help='Build vector database')
    vectordb_parser.add_argument('--rebuild', action='store_true', help='Rebuild from scratch')
    
    # Query commands
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('question', nargs='+', help='Your mental health question')
    query_parser.add_argument('--source', help='Filter by source')
    
    compare_parser = subparsers.add_parser('compare', help='Compare RAG vs vanilla LLM')
    compare_parser.add_argument('question', nargs='+', help='Question to compare')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the system')
    test_parser.add_argument('--query', default='How to deal with anxiety?', help='Test query')
    
    # Chat mode
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')
    
    # Add verbose flag to all subparsers
    for subparser in [setup_parser, ingest_parser, chunk_parser, embed_parser, 
                     vectordb_parser, query_parser, compare_parser, test_parser, chat_parser]:
        subparser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose if hasattr(args, 'verbose') else False)
    
    # Execute commands
    if args.command == 'setup':
        print("🚀 Running complete setup pipeline...")
        print(f"   Data source: {args.source}")
        
        # Step 1: Ingest
        print("\n1️⃣  Ingesting data...")
        if args.source == 'all':
            ingest_all_data()
        elif args.source == 'counselchat':
            data = download_counselchat()
            from .ingest import process_counselchat_qa
            process_counselchat_qa(data)
        elif args.source == 'reddit':
            fetch_reddit_posts()
        elif args.source == 'mind':
            scrape_mind_content()
        
        # Step 2: Chunk
        print("\n2️⃣  Chunking documents...")
        chunk_all_data()
        
        # Step 3: Embed
        print("\n3️⃣  Generating embeddings...")
        embed_all_sources()
        
        # Step 4: Vector DB
        print("\n4️⃣  Building vector database...")
        build_vectordb()
        
        print("\n✅ Setup complete! You can now use 'python -m ragmh query <your question>'")
        
    elif args.command == 'ingest':
        if args.source == 'all':
            ingest_all_data()
        elif args.source == 'counselchat':
            data = download_counselchat()
            from .ingest import process_counselchat_qa
            process_counselchat_qa(data)
        elif args.source == 'reddit':
            fetch_reddit_posts()
        elif args.source == 'mind':
            scrape_mind_content()
    
    elif args.command == 'chunk':
        chunk_all_data()
    
    elif args.command == 'embed':
        embed_all_sources()
    
    elif args.command == 'vectordb':
        build_vectordb(rebuild=args.rebuild)
    
    elif args.command == 'query':
        query = ' '.join(args.question)
        run_rag_pipeline(query, source_filter=args.source)
    
    elif args.command == 'compare':
        query = ' '.join(args.question)
        compare_with_vanilla_llm(query)
    
    elif args.command == 'test':
        # Test vector search
        test_search(args.query)
        
        # Test RAG pipeline
        print("\n" + "="*60)
        print("Testing RAG Pipeline")
        print("="*60)
        run_rag_pipeline(args.query)
    
    elif args.command == 'chat':
        print("\n🧠 Mental Health RAG Chat")
        print("Commands: 'quit' to exit, 'compare <query>' for comparison")
        print("-"*60)
        
        while True:
            query = input("\n❓ You: ").strip()
            
            if query.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif query.lower().startswith('compare '):
                test_query = query[8:]  # Remove 'compare '
                compare_with_vanilla_llm(test_query)
            elif query:
                run_rag_pipeline(query)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()