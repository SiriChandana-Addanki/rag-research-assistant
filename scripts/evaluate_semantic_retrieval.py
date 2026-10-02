"""Evaluate sentence-transformer retrieval and semantic/BM25 hybrid weights.

This deliberately writes a separate artifact so historical lexical results remain
unchanged.  Model weights are downloaded by sentence-transformers when absent.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import load_evaluation_inputs, metrics, retrieval_trace
from src.retrieval import HybridRetriever, SemanticRetriever


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--artifact", default="evaluation/semantic_embeddings.json")
    parser.add_argument("--output", default="evaluation/semantic_retrieval_results.json")
    parser.add_argument("--trace-output", default="evaluation/semantic_retrieval_traces.json")
    args = parser.parse_args()
    chunks, dataset, judgments = load_evaluation_inputs()
    semantic = SemanticRetriever(chunks, args.model, args.batch_size, artifact_path=args.artifact)
    retrievers={"semantic":semantic}
    for weight in (.25, .5, .75):
        retrievers[f"semantic_{weight:.2f}_bm25_{1 - weight:.2f}"]=HybridRetriever(chunks, primary=semantic, semantic_weight=weight)
    result = {"model": args.model, **{name:metrics(retriever,dataset,judgments) for name,retriever in retrievers.items()}}
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    trace_output = Path(args.trace_output)
    traces={"model":args.model,"max_rank":10,"configurations":{name:retrieval_trace(retriever,dataset,judgments) for name,retriever in retrievers.items()}}
    trace_output.write_text(json.dumps(traces, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
