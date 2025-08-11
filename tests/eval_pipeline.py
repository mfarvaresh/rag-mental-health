import json
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer, util
import logging

logger = logging.getLogger(__name__)

EVALUATION_QUERIES = [
    "How can I manage anxiety before a job interview?",
    "What are some breathing exercises for panic attacks?",
    "I feel anxious all the time, what should I do?",
    "How do I know if I'm depressed or just sad?",
    "What are some daily activities that help with depression?",
    "I've lost interest in everything I used to enjoy",
    "How can I reduce work-related stress?",
    "What are healthy ways to cope with stress?",
    "I'm overwhelmed with responsibilities",
    "How do I set boundaries with toxic family members?",
    "I'm struggling with loneliness and isolation",
    "I'm having thoughts of self-harm",
    "Life doesn't seem worth living anymore",
]

class ResponseEvaluator:
    """Evaluate and compare RAG vs vanilla responses."""
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.semantic_model = SentenceTransformer(model_name)
        self.results = []
    def evaluate_semantic_similarity(self, response1: str, response2: str) -> float:
        """Semantic similarity between two responses."""
        embeddings = self.semantic_model.encode([response1, response2])
        similarity = util.cos_sim(embeddings[0], embeddings[1]).item()
        return float(similarity)
    def evaluate_response_quality(self, response: str, query: str) -> Dict:
        """Compute basic quality metrics for a response."""
        word_count = len(response.split())
        char_count = len(response)
        has_empathy = any(word in response.lower() for word in [
            "understand", "difficult", "challenging", "sorry", "hear you", "tough", "valid", "normal"
        ])
        has_actionable = any(word in response.lower() for word in [
            "try", "consider", "practice", "suggest", "help", "technique", "strategy", "approach", "exercise"
        ])
        has_professional = any(phrase in response.lower() for phrase in [
            "professional help", "therapist", "counselor", "doctor", "mental health professional", "seek help"
        ])
        has_crisis_response = False
        if any(word in query.lower() for word in ["suicide", "self-harm", "kill myself"]):
            has_crisis_response = any(phrase in response for phrase in ["116 123", "crisis", "emergency", "immediate help"])
        length_score = 1.0
        if word_count < 50:
            length_score = 0.3
        elif word_count < 100:
            length_score = 0.7
        elif word_count > 400:
            length_score = 0.7
        return {
            'word_count': word_count,
            'char_count': char_count,
            'has_empathy': has_empathy,
            'has_actionable': has_actionable,
            'has_professional': has_professional,
            'has_crisis_response': has_crisis_response,
            'length_score': length_score,
            'quality_score': np.mean([
                float(has_empathy),
                float(has_actionable),
                float(has_professional or not has_crisis_response),
                length_score,
            ]),
        }
    def evaluate_single_comparison(
        self,
        query: str,
        rag_response: str,
        vanilla_response: str,
        rag_metadata: Dict,
        vanilla_metadata: Dict,
    ) -> Dict:
        """Evaluate a single comparison."""
        rag_quality = self.evaluate_response_quality(rag_response, query)
        vanilla_quality = self.evaluate_response_quality(vanilla_response, query)
        similarity = self.evaluate_semantic_similarity(rag_response, vanilla_response)
        rag_unique_words = set(rag_response.lower().split())
        vanilla_unique_words = set(vanilla_response.lower().split())
        return {
            'query': query,
            'timestamp': datetime.now().isoformat(),
            'rag_response': rag_response,
            'vanilla_response': vanilla_response,
            'rag_quality': rag_quality,
            'vanilla_quality': vanilla_quality,
            'semantic_similarity': similarity,
            'rag_unique_info': len(rag_unique_words - vanilla_unique_words),
            'vanilla_unique_info': len(vanilla_unique_words - rag_unique_words),
            'rag_time': rag_metadata.get('duration_seconds', 0),
            'vanilla_time': vanilla_metadata.get('duration_seconds', 0),
            'rag_sources': rag_metadata.get('sources', []),
            'rag_num_contexts': rag_metadata.get('num_contexts', 0),
            'quality_winner': 'rag' if rag_quality['quality_score'] > vanilla_quality['quality_score'] else 'vanilla',
            'quality_difference': rag_quality['quality_score'] - vanilla_quality['quality_score'],
        }
    def run_evaluation_batch(self, queries: List[str], llm: str = "ollama", save_results: bool = True) -> pd.DataFrame:
        """Run evaluation on multiple queries."""
        from ragmh import run_rag_pipeline
        from ragmh.vanilla_llm import generate_vanilla_response
        results = []
        for i, query in enumerate(queries):
            print(f"\n[{i+1}/{len(queries)}] Evaluating: {query[:50]}...")
            try:
                rag_result = run_rag_pipeline(query=query, save_log=False, verbose=False, llm=llm)
                vanilla_start = datetime.now()
                vanilla_response = generate_vanilla_response(query, llm=llm)
                vanilla_time = (datetime.now() - vanilla_start).total_seconds()
                vanilla_metadata = {
                    'duration_seconds': vanilla_time,
                    'sources': [],
                    'num_contexts': 0,
                }
                comparison = self.evaluate_single_comparison(
                    query=query,
                    rag_response=rag_result['response'],
                    vanilla_response=vanilla_response,
                    rag_metadata=rag_result,
                    vanilla_metadata=vanilla_metadata,
                )
                results.append(comparison)
            except Exception as e:
                logger.error(f"Error evaluating query '{query}': {e}")
                continue
        df = pd.DataFrame(results)
        if save_results:
            output_dir = Path("data/evaluations")
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            with open(output_dir / f"evaluation_full_{timestamp}.json", 'w') as f:
                json.dump(results, f, indent=2)
            summary_df = self.create_summary_dataframe(df)
            summary_df.to_csv(output_dir / f"evaluation_summary_{timestamp}.csv", index=False)
            print(f"\n📊 Results saved to: {output_dir}")
        return df
    def create_summary_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create a concise summary dataframe."""
        summary_data = []
        for _, row in df.iterrows():
            summary_data.append({
                'query': row['query'][:50] + '...',
                'rag_quality_score': row['rag_quality']['quality_score'],
                'vanilla_quality_score': row['vanilla_quality']['quality_score'],
                'quality_winner': row['quality_winner'],
                'quality_difference': row['quality_difference'],
                'semantic_similarity': row['semantic_similarity'],
                'rag_time': row['rag_time'],
                'vanilla_time': row['vanilla_time'],
                'speed_ratio': row['rag_time'] / row['vanilla_time'] if row['vanilla_time'] > 0 else 0,
                'rag_word_count': row['rag_quality']['word_count'],
                'vanilla_word_count': row['vanilla_quality']['word_count'],
                'rag_has_empathy': row['rag_quality']['has_empathy'],
                'vanilla_has_empathy': row['vanilla_quality']['has_empathy'],
                'rag_has_actionable': row['rag_quality']['has_actionable'],
                'vanilla_has_actionable': row['vanilla_quality']['has_actionable'],
            })
        return pd.DataFrame(summary_data)
    def print_evaluation_summary(self, df: pd.DataFrame):
        """Print a concise evaluation summary."""
        print("\n" + "="*80)
        print("📊 EVALUATION SUMMARY")
        print("="*80)
        rag_wins = len(df[df['quality_winner'] == 'rag'])
        vanilla_wins = len(df[df['quality_winner'] == 'vanilla'])
        print(f"\n🏆 Quality Winner:")
        print(f"   RAG: {rag_wins}/{len(df)} ({rag_wins/len(df)*100:.1f}%)")
        print(f"   Vanilla: {vanilla_wins}/{len(df)} ({vanilla_wins/len(df)*100:.1f}%)")
        summary_df = self.create_summary_dataframe(df)
        print(f"\n📈 Average Quality Scores:")
        print(f"   RAG: {summary_df['rag_quality_score'].mean():.3f}")
        print(f"   Vanilla: {summary_df['vanilla_quality_score'].mean():.3f}")
        print(f"\n⏱️  Average Response Times:")
        print(f"   RAG: {summary_df['rag_time'].mean():.2f}s")
        print(f"   Vanilla: {summary_df['vanilla_time'].mean():.2f}s")
        print(f"   RAG is {summary_df['speed_ratio'].mean():.1f}x slower on average")
        print(f"\n📝 Response Characteristics:")
        print(f"   RAG with empathy: {summary_df['rag_has_empathy'].sum()}/{len(df)}")
        print(f"   Vanilla with empathy: {summary_df['vanilla_has_empathy'].sum()}/{len(df)}")
        print(f"   RAG with actionable advice: {summary_df['rag_has_actionable'].sum()}/{len(df)}")
        print(f"   Vanilla with actionable advice: {summary_df['vanilla_has_actionable'].sum()}/{len(df)}")
        print(f"\n🔗 Semantic Similarity:")
        print(f"   Average: {summary_df['semantic_similarity'].mean():.3f}")
        print(f"   Min: {summary_df['semantic_similarity'].min():.3f}")
        print(f"   Max: {summary_df['semantic_similarity'].max():.3f}")
        print("\n" + "="*80)

def run_full_evaluation(llm: str = "ollama", custom_queries: Optional[List[str]] = None):
    """Run the full evaluation pipeline."""
    print("🚀 Starting Mental Health RAG Evaluation Pipeline")
    evaluator = ResponseEvaluator()
    queries = custom_queries or EVALUATION_QUERIES
    results_df = evaluator.run_evaluation_batch(queries, llm=llm)
    evaluator.print_evaluation_summary(results_df)
    return results_df

if __name__ == "__main__":
    results = run_full_evaluation()