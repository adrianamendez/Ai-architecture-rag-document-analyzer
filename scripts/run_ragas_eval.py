#!/usr/bin/env python3
"""
Run RAGAS evaluation in a standalone Python process.
This avoids Jupyter event-loop/contextvars issues.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from eval_dataset_generator import EvalDatasetGenerator
from rag_engine import RAGEngine
from evaluator import RAGEvaluator


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=10, help="Generated eval dataset size")
    parser.add_argument("--limit", type=int, default=5, help="Samples used per strategy")
    parser.add_argument(
        "--output",
        type=str,
        default="data/eval_dataset/ragas_runtime_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    # Conservative defaults for local Ollama stability.
    os.environ.setdefault("RAGAS_OLLAMA_LLM", "llama3:latest")
    os.environ.setdefault("RAGAS_OLLAMA_EMBEDDINGS", "nomic-embed-text:latest")
    os.environ.setdefault("RAGAS_MAX_WORKERS", "1")
    os.environ.setdefault("RAGAS_TIMEOUT_SEC", "300")
    os.environ.setdefault("RAGAS_MAX_RETRIES", "2")
    # Standalone process does not need nest_asyncio patching.
    os.environ.setdefault("RAGAS_ALLOW_NEST_ASYNCIO", "false")

    generator = EvalDatasetGenerator()
    dataset = generator.generate_dataset(num_samples=args.samples)

    strategies = [
        {"model_type": "text_only", "chunking": "combined", "rerank": False},
        {"model_type": "text_only", "chunking": "combined", "rerank": True},
        {"model_type": "multimodal", "chunking": "combined", "rerank": True},
    ]

    rows = []
    for s in strategies:
        name = f"{s['model_type']}_{s['chunking']}_rerank{s['rerank']}"
        print(f"Evaluating: {name}")
        engine = RAGEngine(
            model_type=s["model_type"],
            use_reranking=s["rerank"],
            chunking_strategy=s["chunking"],
        )
        if engine.collection.count() == 0:
            engine.ingest_documents()
        evaluator = RAGEvaluator(engine)
        result = evaluator.evaluate_dataset(dataset, limit=args.limit)
        scores = result.get("scores", {})
        rows.append(
            {
                "Strategy": name,
                "Faithfulness": float(scores.get("faithfulness", 0.0)),
                "Answer Relevancy": float(scores.get("answer_relevancy", 0.0)),
                "Context Precision": float(scores.get("context_precision", 0.0)),
                "Context Recall": float(scores.get("context_recall", 0.0)),
                "Average": float(scores.get("average", 0.0)),
                "Error": scores.get("error", ""),
            }
        )

    rows.sort(key=lambda x: x["Average"], reverse=True)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved results to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
