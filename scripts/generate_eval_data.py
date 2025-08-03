import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from tqdm import tqdm
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent / "src"))
from ragmh.chains import run_rag_pipeline as ragmh_run_rag_pipeline

# Set up logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)  # Create logs directory if it doesn't exist

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / 'eval_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGEvaluationGenerator:
    """Generate evaluation data by running RAG pipeline on test questions."""
    
    def __init__(self, llm: str = "ollama", source_filter: Optional[str] = None):
        self.llm = llm
        self.source_filter = source_filter
        self.stats = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'avg_response_time': 0,
            'total_time': 0
        }
    
    def run_rag_pipeline(self, question: str) -> Tuple[str, List[str], float, Optional[str]]:
        """
        Calls the RAG pipeline and returns (answer, contexts, duration, error).
        """
        start_time = time.time()
        error = None
        
        try:
            result = ragmh_run_rag_pipeline(
                query=question,
                source_filter=self.source_filter,
                save_log=False,
                verbose=False,
                llm=self.llm
            )
            
            if result is None:
                error = "No response from RAG pipeline"
                return "[ERROR: No response]", [], time.time() - start_time, error
            
            answer = result.get("response", "[ERROR: No response field]")
            contexts = result.get("contexts", [])
            
            return answer, contexts, time.time() - start_time, None
            
        except Exception as e:
            error = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Error processing question: {error}")
            logger.debug(traceback.format_exc())
            return f"[ERROR: {error}]", [], time.time() - start_time, error
    
    def process_single_entry(self, entry: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Process a single Q&A entry."""
        question = entry.get("question", "")
        reference = entry.get("answer", "")
        
        if not question or not reference:
            logger.warning(f"Skipping entry {index}: missing question or answer")
            return None
        
        # Run RAG pipeline
        answer, contexts, duration, error = self.run_rag_pipeline(question)
        
        # Update stats
        self.stats['total_time'] += duration
        if error:
            self.stats['failed'] += 1
        else:
            self.stats['success'] += 1
        
        return {
            "question": question,
            "answer": answer,
            "reference": reference,
            "contexts": contexts,
            "duration_seconds": round(duration, 3),
            "timestamp": datetime.now().isoformat(),
            "llm": self.llm,
            "error": error
        }
    
    def generate_evaluation_data(
        self,
        input_path: Path,
        output_path: Path,
        sample_size: Optional[int] = None,
        shuffle: bool = False,
        resume_from: Optional[Path] = None,
        parallel: bool = False,
        max_workers: int = 4
    ) -> None:
        """Generate evaluation data from test questions."""
        
        # Load input data
        logger.info(f"Loading data from {input_path}")
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Validate data
        if not isinstance(data, list):
            raise ValueError("Input data must be a list of Q&A entries")
        
        logger.info(f"Loaded {len(data)} entries")
        
        # Resume from previous run if specified
        existing_questions = set()
        results = []
        if resume_from and resume_from.exists():
            logger.info(f"Resuming from {resume_from}")
            with open(resume_from, "r", encoding="utf-8") as f:
                results = json.load(f)
                existing_questions = {r["question"] for r in results}
                logger.info(f"Found {len(results)} existing results")
        
        # Filter out already processed questions
        data = [d for d in data if d.get("question") not in existing_questions]
        
        # Sample and shuffle if requested
        if shuffle:
            random.shuffle(data)
        
        if sample_size and sample_size < len(data):
            logger.info(f"Sampling {sample_size} entries")
            data = data[:sample_size]
        
        self.stats['total'] = len(data)
        
        # Process entries
        logger.info(f"Processing {len(data)} entries with LLM: {self.llm}")
        
        if parallel and len(data) > 1:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_entry = {
                    executor.submit(self.process_single_entry, entry, i): (entry, i)
                    for i, entry in enumerate(data)
                }
                
                # Process results as they complete
                with tqdm(total=len(data), desc="Processing") as pbar:
                    for future in as_completed(future_to_entry):
                        try:
                            result = future.result()
                            if result:
                                results.append(result)
                        except Exception as e:
                            logger.error(f"Failed to process entry: {e}")
                            self.stats['failed'] += 1
                        pbar.update(1)
                        
                        # Save intermediate results every 10 entries
                        if len(results) % 10 == 0:
                            self._save_results(output_path, results, intermediate=True)
        else:
            # Sequential processing
            for i, entry in enumerate(tqdm(data, desc="Processing")):
                result = self.process_single_entry(entry, i)
                if result:
                    results.append(result)
                
                # Save intermediate results every 10 entries
                if (i + 1) % 10 == 0:
                    self._save_results(output_path, results, intermediate=True)
        
        # Save final results
        self._save_results(output_path, results)
        
        # Print summary statistics
        self._print_summary(results)
    
    def _save_results(self, output_path: Path, results: List[Dict], intermediate: bool = False):
        """Save results to file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use temporary file for intermediate saves
        save_path = output_path.with_suffix('.tmp.json') if intermediate else output_path
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        if not intermediate:
            logger.info(f"Saved {len(results)} evaluation entries to {output_path}")
            # Remove temporary file if exists
            tmp_path = output_path.with_suffix('.tmp.json')
            if tmp_path.exists():
                tmp_path.unlink()
    
    def _print_summary(self, results: List[Dict]):
        """Print summary statistics."""
        successful_results = [r for r in results if not r.get("error")]
        
        if successful_results:
            avg_duration = sum(r["duration_seconds"] for r in successful_results) / len(successful_results)
            avg_contexts = sum(len(r["contexts"]) for r in successful_results) / len(successful_results)
            avg_answer_length = sum(len(r["answer"].split()) for r in successful_results) / len(successful_results)
        else:
            avg_duration = avg_contexts = avg_answer_length = 0
        
        print("\n" + "="*50)
        print("EVALUATION DATA GENERATION SUMMARY")
        print("="*50)
        print(f"Total entries processed: {len(results)}")
        print(f"Successful: {self.stats['success']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Average response time: {avg_duration:.2f} seconds")
        print(f"Average contexts retrieved: {avg_contexts:.1f}")
        print(f"Average answer length: {avg_answer_length:.1f} words")
        print(f"Total time: {self.stats['total_time']:.2f} seconds")
        
        if self.stats['failed'] > 0:
            print(f"\n⚠️  {self.stats['failed']} entries failed. Check logs for details.")

def filter_by_keywords(data: List[Dict], keywords: List[str]) -> List[Dict]:
    """Filter questions containing specific keywords."""
    filtered = []
    for entry in data:
        question = entry.get("question", "").lower()
        if any(keyword.lower() in question for keyword in keywords):
            filtered.append(entry)
    return filtered

def main():
    parser = argparse.ArgumentParser(
        description="Generate evaluation data from CounselChat using your RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate evaluation data for all questions
  python generate_eval_data.py
  
  # Use a specific LLM and sample 50 questions
  python generate_eval_data.py --llm gpt-3.5-turbo-0125 --sample 50
  
  # Resume from previous run
  python generate_eval_data.py --resume logs/eval_counselchat.tmp.json
  
  # Filter by keywords
  python generate_eval_data.py --keywords anxiety depression --sample 20
  
  # Run in parallel with 8 workers
  python generate_eval_data.py --parallel --workers 8
        """
    )
    
    parser.add_argument("--llm", type=str, default="ollama", 
                       help="LLM to use (ollama, gpt-3.5-turbo-0125, gemini-1.5-flash-8b, etc.)")
    parser.add_argument("--input", type=str, default="data/counselchat/counselchat_qa.json",
                       help="Input JSON file with Q&A data")
    parser.add_argument("--output", type=str, default="logs/eval_counselchat.json",
                       help="Output JSON file for evaluation data")
    parser.add_argument("--sample", type=int, default=None,
                       help="Number of questions to sample (default: all)")
    parser.add_argument("--shuffle", action="store_true",
                       help="Shuffle questions before sampling")
    parser.add_argument("--source-filter", type=str, default=None,
                       help="Filter retrieved contexts by source (e.g., 'counselchat')")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from previous partial run")
    parser.add_argument("--keywords", nargs="+", default=None,
                       help="Filter questions containing these keywords")
    parser.add_argument("--parallel", action="store_true",
                       help="Process questions in parallel")
    parser.add_argument("--workers", type=int, default=4,
                       help="Number of parallel workers (default: 4)")
    
    args = parser.parse_args()
    
    # Convert paths
    input_path = Path(args.input)
    output_path = Path(args.output)
    resume_path = Path(args.resume) if args.resume else None
    
    # Check input file exists
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Load and optionally filter data
    if args.keywords:
        logger.info(f"Filtering by keywords: {args.keywords}")
        with open(input_path, "r", encoding="utf-8") as f:
            all_data = json.load(f)
        filtered_data = filter_by_keywords(all_data, args.keywords)
        logger.info(f"Found {len(filtered_data)} questions matching keywords")
        
        # Save filtered data to temporary file
        temp_path = Path("logs/filtered_temp.json")
        temp_path.parent.mkdir(exist_ok=True)
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f)
        input_path = temp_path
    
    # Create generator and run
    generator = RAGEvaluationGenerator(llm=args.llm, source_filter=args.source_filter)
    
    try:
        generator.generate_evaluation_data(
            input_path=input_path,
            output_path=output_path,
            sample_size=args.sample,
            shuffle=args.shuffle,
            resume_from=resume_path,
            parallel=args.parallel,
            max_workers=args.workers
        )
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user. Partial results saved.")
    except Exception as e:
        logger.error(f"Failed to generate evaluation data: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
    finally:
        # Clean up temporary file if created
        if args.keywords and Path("logs/filtered_temp.json").exists():
            Path("logs/filtered_temp.json").unlink()

if __name__ == "__main__":
    main()