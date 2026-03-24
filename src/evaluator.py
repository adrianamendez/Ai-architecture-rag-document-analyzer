"""
RAGAS Evaluator for RAG system.
Implements faithfulness, answer relevancy, context precision, and context recall metrics.
"""

import json
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

from config import EVAL_DIR, RAGAS_METRICS
from rag_engine import RAGEngine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGEvaluator:
    """Evaluate RAG system using RAGAS metrics."""

    def __init__(self, rag_engine: RAGEngine):
        """
        Initialize evaluator.

        Args:
            rag_engine: RAG engine instance to evaluate
        """
        self.rag_engine = rag_engine
        self.metrics = {
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision,
            'context_recall': context_recall,
        }

    def evaluate_single_query(
        self,
        question: str,
        ground_truth: str,
        ground_truth_contexts: List[str],
    ) -> Dict[str, Any]:
        """
        Evaluate a single query.

        Args:
            question: The question
            ground_truth: The expected answer
            ground_truth_contexts: The relevant contexts

        Returns:
            Dictionary with answer and contexts
        """
        # Query the RAG system
        result = self.rag_engine.query(question)

        # Extract generated answer and contexts
        answer = result['answer']
        contexts = [doc['content'] for doc in result['context_documents']]

        return {
            'question': question,
            'answer': answer,
            'contexts': contexts,
            'ground_truth': ground_truth,
            'ground_truth_contexts': ground_truth_contexts,
        }

    def evaluate_dataset(
        self,
        eval_dataset: List[Dict[str, Any]],
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate RAG system on a dataset.

        Args:
            eval_dataset: List of Q&A pairs with ground truth
            limit: Maximum number of samples to evaluate (for testing)

        Returns:
            Evaluation results with RAGAS metrics
        """
        if limit:
            eval_dataset = eval_dataset[:limit]

        logger.info(f"Evaluating {len(eval_dataset)} samples...")

        # Collect results
        results = []
        for i, sample in enumerate(eval_dataset, 1):
            if i % 5 == 0:
                logger.info(f"Processing sample {i}/{len(eval_dataset)}...")

            result = self.evaluate_single_query(
                question=sample['question'],
                ground_truth=sample.get('answer', sample.get('ground_truth', '')),
                ground_truth_contexts=sample.get('contexts', []),
            )
            results.append(result)

        # Prepare data for RAGAS
        ragas_dataset = {
            'question': [r['question'] for r in results],
            'answer': [r['answer'] for r in results],
            'contexts': [r['contexts'] for r in results],
            'ground_truth': [r['ground_truth'] for r in results],
        }

        # Convert to HuggingFace Dataset
        dataset = Dataset.from_dict(ragas_dataset)

        logger.info("Running RAGAS evaluation...")

        # Select metrics to evaluate
        metrics_to_use = [self.metrics[m] for m in RAGAS_METRICS if m in self.metrics]

        # Run RAGAS evaluation
        try:
            ragas_results = evaluate(dataset, metrics=metrics_to_use)

            # Extract scores
            scores = {
                'faithfulness': ragas_results.get('faithfulness', 0.0),
                'answer_relevancy': ragas_results.get('answer_relevancy', 0.0),
                'context_precision': ragas_results.get('context_precision', 0.0),
                'context_recall': ragas_results.get('context_recall', 0.0),
            }

            # Calculate average score
            scores['average'] = np.mean(list(scores.values()))

            logger.info("✓ RAGAS evaluation complete")
            logger.info(f"  Faithfulness: {scores['faithfulness']:.3f}")
            logger.info(f"  Answer Relevancy: {scores['answer_relevancy']:.3f}")
            logger.info(f"  Context Precision: {scores['context_precision']:.3f}")
            logger.info(f"  Context Recall: {scores['context_recall']:.3f}")
            logger.info(f"  Average: {scores['average']:.3f}")

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            scores = {
                'faithfulness': 0.0,
                'answer_relevancy': 0.0,
                'context_precision': 0.0,
                'context_recall': 0.0,
                'average': 0.0,
                'error': str(e),
            }

        return {
            'scores': scores,
            'samples_evaluated': len(results),
            'results': results[:5],  # Store first 5 for inspection
            'model': self.rag_engine.model_config['name'],
            'chunking_strategy': self.rag_engine.chunking_strategy,
            'reranking_used': self.rag_engine.use_reranking,
        }

    def compare_strategies(
        self,
        eval_dataset: List[Dict[str, Any]],
        strategies: List[Dict[str, Any]],
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Compare different RAG strategies on the same dataset.

        Args:
            eval_dataset: Evaluation dataset
            strategies: List of strategy configurations
                       [{'model_type': 'text_only', 'chunking': 'combined', 'rerank': True}, ...]
            limit: Max samples per strategy

        Returns:
            Comparison results
        """
        comparison_results = {}

        for strategy in strategies:
            strategy_name = f"{strategy['model_type']}_{strategy['chunking']}_rerank{strategy['rerank']}"

            logger.info(f"\nEvaluating strategy: {strategy_name}")
            logger.info("-" * 60)

            # Create new RAG engine with this strategy
            engine = RAGEngine(
                model_type=strategy['model_type'],
                use_reranking=strategy['rerank'],
                chunking_strategy=strategy['chunking'],
            )

            # Make sure documents are ingested
            if engine.collection.count() == 0:
                engine.ingest_documents()

            # Create evaluator for this engine
            evaluator = RAGEvaluator(engine)

            # Evaluate
            results = evaluator.evaluate_dataset(eval_dataset, limit=limit)

            comparison_results[strategy_name] = results

        # Calculate best strategy
        best_strategy = max(
            comparison_results.items(),
            key=lambda x: x[1]['scores']['average']
        )

        return {
            'comparison': comparison_results,
            'best_strategy': best_strategy[0],
            'best_score': best_strategy[1]['scores']['average'],
        }

    def save_results(self, results: Dict[str, Any], filename: str = "evaluation_results.json"):
        """Save evaluation results to file."""
        filepath = EVAL_DIR / filename

        # Make results JSON serializable
        serializable_results = json.loads(json.dumps(results, default=str))

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"✓ Saved evaluation results to {filepath}")
        return filepath


def main():
    """Test evaluator."""
    print("=" * 60)
    print("RAGAS Evaluator Test")
    print("=" * 60)

    # Check if evaluation dataset exists
    eval_file = EVAL_DIR / "eval_dataset.json"
    if not eval_file.exists():
        print("\n⚠ Evaluation dataset not found!")
        print("Run: python src/eval_dataset_generator.py first")
        return

    # Load dataset
    print("\nLoading evaluation dataset...")
    with open(eval_file, 'r') as f:
        eval_dataset = json.load(f)
    print(f"✓ Loaded {len(eval_dataset)} Q&A pairs")

    # Create RAG engine
    print("\nInitializing RAG engine...")
    rag_engine = RAGEngine(
        model_type="text_only",
        use_reranking=True,
        chunking_strategy="combined",
    )

    # Ingest documents if needed
    if rag_engine.collection.count() == 0:
        print("Ingesting documents...")
        rag_engine.ingest_documents()

    # Create evaluator
    evaluator = RAGEvaluator(rag_engine)

    # Evaluate on small subset
    print("\nEvaluating on 5 samples (this may take a few minutes)...")
    print("-" * 60)

    results = evaluator.evaluate_dataset(eval_dataset, limit=5)

    # Display results
    print("\nEvaluation Results:")
    print("-" * 60)
    print(f"Model: {results['model']}")
    print(f"Chunking Strategy: {results['chunking_strategy']}")
    print(f"Reranking: {results['reranking_used']}")
    print(f"\nScores:")
    for metric, score in results['scores'].items():
        if metric != 'error':
            print(f"  {metric}: {score:.3f}")

    # Save results
    print("\nSaving results...")
    evaluator.save_results(results, filename="test_evaluation_results.json")

    print("\n" + "=" * 60)
    print("✓ Evaluator test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
