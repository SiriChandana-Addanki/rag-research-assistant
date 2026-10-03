"""Run the predeclared, single-query q009 terminology-alignment ablation.

Only the query text varies.  Corpus, model, embedding artifact, BM25, score
normalisation, hybrid weights, relevance labels, and ranking tie policy are the
same as the selected retrieval baseline.  This diagnostic is intentionally not
imported by the production retrieval path.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import load_evaluation_inputs
from src.retrieval import HybridRetriever, SemanticRetriever, normalize, rank

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_WEIGHT = 0.75
QUESTION_ID = "q009"
EXPECTED = tuple(f"paper1_c{number:04d}" for number in range(27, 31))
ALIGNED_QUERY = (
    "How can SELF-RAG use a Retrieve=Yes threshold, Critique/reflection-token "
    "weighted decoding including ISSUP support, and hard Critique-token filtering "
    "to control retrieval, evidence grounding, and generation quality at inference "
    "time with no additional training?"
)


def _ranking(retriever, query, relevant):
    semantic = retriever.primary.scores(query)
    bm25 = retriever.bm25.scores(query)
    hybrid = [SEMANTIC_WEIGHT * dense + (1 - SEMANTIC_WEIGHT) * lexical
              for dense, lexical in zip(normalize(semantic), normalize(bm25))]
    chunk_index = {chunk["chunk_id"]: index for index, chunk in enumerate(retriever.chunks)}
    semantic_ranks = {item.chunk["chunk_id"]: position for position, item in enumerate(rank(retriever.chunks, semantic, len(retriever.chunks)), 1)}
    bm25_ranks = {item.chunk["chunk_id"]: position for position, item in enumerate(rank(retriever.chunks, bm25, len(retriever.chunks)), 1)}
    ordered = rank(retriever.chunks, hybrid, len(retriever.chunks))
    rows = []
    for position, item in enumerate(ordered, 1):
        index = chunk_index[item.chunk["chunk_id"]]
        rows.append({"rank": position, "chunk_id": item.chunk["chunk_id"],
                     "semantic_score": semantic[index], "bm25_score": bm25[index],
                     "hybrid_score": hybrid[index], "is_relevant": item.chunk["chunk_id"] in relevant})
    by_id = {row["chunk_id"]: row for row in rows}
    expected = []
    for chunk_id in EXPECTED:
        row = by_id[chunk_id]
        expected.append({**row, "semantic_rank": semantic_ranks[chunk_id], "bm25_rank": bm25_ranks[chunk_id],
                         "hybrid_rank": row["rank"], "entered_top_10": row["rank"] <= 10,
                         "entered_top_30": row["rank"] <= 30})
    def recall_at(k): return bool(set(row["chunk_id"] for row in rows[:k]) & relevant)
    def coverage_at(k): return len(set(row["chunk_id"] for row in rows[:k]) & relevant) / len(relevant)
    first = next((row["rank"] for row in rows if row["is_relevant"]), None)
    return {"ranked_chunks": rows, "expected_chunk_measurements": expected,
            "metrics": {"recall@10": recall_at(10), "recall@30": recall_at(30),
                        "evidence_coverage@10": coverage_at(10), "evidence_coverage@30": coverage_at(30),
                        "mrr": 1 / first if first else 0.0}}


def run():
    chunks, dataset, judgments = load_evaluation_inputs()
    if len(chunks) != 127:
        raise ValueError(f"frozen corpus must have 127 chunks, found {len(chunks)}")
    question = next(item for item in dataset if item["id"] == QUESTION_ID)
    relevant = set(next(item["relevant_chunk_ids"] for item in judgments if item["question_id"] == QUESTION_ID))
    if tuple(sorted(relevant)) != EXPECTED:
        raise ValueError("q009 relevance labels drifted")
    retriever = HybridRetriever(chunks, primary=SemanticRetriever(chunks, MODEL, artifact_path="evaluation/semantic_embeddings.json"), semantic_weight=SEMANTIC_WEIGHT)
    control = _ranking(retriever, question["question"], relevant)
    aligned = _ranking(retriever, ALIGNED_QUERY, relevant)
    control_by_id = {row["chunk_id"]: row for row in control["expected_chunk_measurements"]}
    for row in aligned["expected_chunk_measurements"]:
        row["hybrid_rank_change_vs_control"] = control_by_id[row["chunk_id"]]["hybrid_rank"] - row["hybrid_rank"]
    return {"experiment_type": "controlled_q009_terminology_aligned_query_ablation", "status": "completed",
            "frozen_conditions": {"chunk_count": len(chunks), "model": MODEL, "semantic_weight": SEMANTIC_WEIGHT,
                                  "bm25_weight": 1 - SEMANTIC_WEIGHT, "reranking": False, "multi_query": False,
                                  "fusion": False, "final_k": 10, "ranking_tie_break": "descending score, ascending chunk_id"},
            "question_id": QUESTION_ID, "relevant_chunk_ids": list(EXPECTED),
            "variants": [{"id": "control", "query": question["question"], **control},
                         {"id": "terminology_aligned", "query": ALIGNED_QUERY, **aligned}]}


def main():
    output = Path("evaluation/q009_terminology_aligned_ablation.json")
    try:
        payload = run()
    except RuntimeError as error:
        payload = {"experiment_type": "controlled_q009_terminology_aligned_query_ablation", "status": "not_run",
                   "reason": str(error), "required_local_command": "python scripts/evaluate_q009_terminology_ablation.py",
                   "environment": {"python": platform.python_version()}, "predeclared_aligned_query": ALIGNED_QUERY}
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
