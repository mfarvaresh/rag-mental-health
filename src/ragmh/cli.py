"""Command line interface for RAG Mental Health system"""
import argparse
import logging
from pathlib import Path

# Import our modules
from .ingest import ingest_all_data, download_counselchat, scrape_mind_content, fetch_reddit_posts, fetch_pubmed_abstracts, fetch_who_topic_summary
from .chunk import chunk_all_data
from .embed import embed_all_sources
from .vectordb import build_vectordb, test_search, hybrid_search_vectordb
from .chains import run_rag_pipeline
from .vanilla_llm import compare_responses_cli

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
    query_parser.add_argument('--llm', choices=['ollama', 'gpt-3.5-turbo-0125', 'gemini-1.5-flash-8b'], default='ollama', help='LLM to use (ollama, gpt-3.5-turbo-0125, gemini-1.5-flash-8b)')

    compare_parser = subparsers.add_parser('compare', help='Compare RAG vs vanilla LLM')
    compare_parser.add_argument('question', nargs='+', help='Question to compare')
    compare_parser.add_argument('--llm', choices=['ollama', 'gpt-3.5-turbo-0125', 'gemini-1.5-flash-8b'], default='ollama', help='LLM to use (ollama, gpt-3.5-turbo-0125, gemini-1.5-flash-8b)')
    compare_parser.add_argument('--no-save-logs', action='store_true', help='Do not save comparison logs')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test the system')
    test_parser.add_argument('--query', default='How to deal with anxiety?', help='Test query')
    
    # Chat mode
    chat_parser = subparsers.add_parser('chat', help='Interactive chat mode')
    
    # PubMed ingestion
    pubmed_parser = subparsers.add_parser('pubmed', help='Ingest PubMed abstracts')
    pubmed_parser.add_argument('query', help='Query for PubMed search')
    pubmed_parser.add_argument('--max', type=int, default=30, help='Max results')
    # WHO ingestion
    who_parser = subparsers.add_parser('who', help='Ingest WHO topic summary')
    who_parser.add_argument('topic', help='WHO topic slug (e.g. mental-health)')
    
    # Add verbose flag to all subparsers
    for subparser in [setup_parser, ingest_parser, chunk_parser, embed_parser, 
                     vectordb_parser, query_parser, compare_parser, test_parser, chat_parser, pubmed_parser, who_parser]:
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
        # Use advanced hybrid search for retrieval
        from .vectordb import init_chromadb
        client = init_chromadb()
        collection = client.get_collection("mental_health_rag")
        results = hybrid_search_vectordb(query, collection, n_results=10, filter_dict={'source': args.source} if args.source else None)
        print("\n=== Hybrid Search Results ===")
        for i, r in enumerate(results, 1):
            print(f"{i}. Source: {r['metadata'].get('source', '')}")
            print(f"   Text: {r['text'][:200]}...")
            print(f"   Metadata: {r['metadata']}")
            print()
        # Optionally, still run the full RAG pipeline for answer generation
        run_rag_pipeline(query, source_filter=args.source, llm=args.llm)
    
    elif args.command == 'compare':
        query = ' '.join(args.question)
        print("\n🔬 Comparing RAG vs Vanilla LLM")
        print("="*60)
        save_logs = not args.no_save_logs
        comparison = compare_responses_cli(query, llm=args.llm)
        if comparison and save_logs:
            from pathlib import Path
            import json
            from datetime import datetime
            PROJECT_ROOT = Path(__file__).parent.parent.parent
            LOGS_DIR = PROJECT_ROOT / "data" / "logs"
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            log_file = LOGS_DIR / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2)
            print(f"\n💾 Comparison saved to: {log_file}")
    
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
        llm = 'ollama'
        try:
            import readline  # for better input UX if available
        except ImportError:
            pass
        while True:
            query = input("\n❓ You: ").strip()
            if query.lower() == 'quit':
                print("👋 Goodbye!")
                break
            elif query.lower().startswith('llm '):
                llm_choice = query[4:].strip()
                if llm_choice in ['ollama', 'gpt-3.5-turbo-0125', 'gemini-1.5-flash-8b']:
                    llm = llm_choice
                    print(f"[LLM switched to: {llm}]")
                else:
                    print("Invalid LLM. Choose from: ollama, gpt-3.5-turbo-0125, gemini-1.5-flash-8b.")
            elif query.lower().startswith('compare '):
                print("[compare] This feature is currently unavailable. Please update your CLI or re-implement compare_with_vanilla_llm if needed.")
            elif query:
                run_rag_pipeline(query, llm=llm)
    elif args.command == 'pubmed':
        fetch_pubmed_abstracts(args.query, max_results=args.max)
    elif args.command == 'who':
        fetch_who_topic_summary(args.topic)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()