"""Evaluate sentence-transformer retrieval and semantic/BM25 hybrid weights.

This deliberately writes a separate artifact so historical lexical results remain
unchanged.  Model weights are downloaded by sentence-transformers when absent.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import load_evaluation_inputs, metrics
from src.retrieval import HybridRetriever, SemanticRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--artifact", default="evaluation/semantic_embeddings.json")
    parser.add_argument("--output", default="evaluation/semantic_retrieval_results.json")
    args = parser.parse_args()
    chunks, dataset, judgments = load_evaluation_inputs()
    semantic = SemanticRetriever(chunks, args.model, args.batch_size, artifact_path=args.artifact)
    result = {"model": args.model, "semantic": metrics(semantic, dataset, judgments)}
    for weight in (.25, .5, .75):
        result[f"semantic_{weight:.2f}_bm25_{1 - weight:.2f}"] = metrics(
            HybridRetriever(chunks, primary=semantic, semantic_weight=weight), dataset, judgments
        )
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
