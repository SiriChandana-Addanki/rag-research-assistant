"""Measure full, label-blind 127-candidate rankings for the frozen q009 setup."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import load_evaluation_inputs
from src.retrieval import HybridRetriever, SemanticRetriever, normalize, rank

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_WEIGHT = 0.75
BM25_WEIGHT = 0.25
QUESTION_ID = "q009"
EXPECTED_CHUNK_IDS = tuple(f"paper1_c{number:04d}" for number in range(27, 31))
EXPECTED_VARIANT_IDS = (
    "original", "inference_control", "grounding_quality_tradeoff", "no_retraining_control",
)


def classify(rank_value: int | None) -> str:
    if rank_value is None:
        return "D"
    if rank_value <= 10:
        return "A"
    if rank_value <= 30:
        return "B"
    return "C"


def validate_variant_ids(variants: list[dict]) -> None:
    ids = tuple(item["id"] for item in variants)
    if ids != EXPECTED_VARIANT_IDS:
        raise ValueError(f"q009 variants must be {EXPECTED_VARIANT_IDS}, got {ids}")


def validate_ranking(records: list[dict], chunks: list[dict]) -> None:
    expected_count = len(chunks)
    ids = [record["chunk_id"] for record in records]
    ranks = [record["rank"] for record in records]
    if expected_count != 127:
        raise ValueError(f"frozen q009 corpus must contain 127 chunks, got {expected_count}")
    if len(records) != expected_count:
        raise ValueError(f"ranking must contain {expected_count} chunks, got {len(records)}")
    if len(set(ids)) != expected_count:
        raise ValueError("ranking contains duplicate chunk IDs")
    if set(ids) != {chunk["chunk_id"] for chunk in chunks}:
        raise ValueError("ranking chunk IDs do not match indexed corpus")
    if ranks != list(range(1, expected_count + 1)):
        raise ValueError("ranking ranks must be contiguous and ordered from 1")


def full_ranking(retriever, query: str, relevant_ids: set[str]) -> tuple[list[dict], list[dict], list[dict]]:
    """Use the frozen retriever score sources and production ``rank`` tie-breaker."""
    semantic = retriever.primary.scores(query)
    bm25 = retriever.bm25.scores(query)
    combined = [SEMANTIC_WEIGHT * value + BM25_WEIGHT * lexical
                for value, lexical in zip(normalize(semantic), normalize(bm25))]
    ordered = rank(retriever.chunks, combined, len(retriever.chunks))
    index_by_id = {chunk["chunk_id"]: index for index, chunk in enumerate(retriever.chunks)}
    records = []
    for position, result in enumerate(ordered, 1):
        chunk = result.chunk
        index = index_by_id[chunk["chunk_id"]]
        records.append({
            "rank": position, "chunk_id": chunk["chunk_id"],
            "semantic_score": semantic[index], "bm25_score": bm25[index],
            "combined_score": combined[index], "page_start": chunk["page_start"],
            "page_end": chunk["page_end"], "section": chunk["section"],
            "is_relevant": chunk["chunk_id"] in relevant_ids,
        })
    semantic_ranking = rank(retriever.chunks, semantic, len(retriever.chunks))
    bm25_ranking = rank(retriever.chunks, bm25, len(retriever.chunks))
    return records, [item.chunk["chunk_id"] for item in semantic_ranking], [item.chunk["chunk_id"] for item in bm25_ranking]


def extract_expected(rankings: list[dict]) -> list[dict]:
    summaries = []
    for chunk_id in EXPECTED_CHUNK_IDS:
        per_variant = []
        for ranking in rankings:
            entry = next((item for item in ranking["ranked_chunks"] if item["chunk_id"] == chunk_id), None)
            if entry is None:
                raise ValueError(f"expected chunk {chunk_id} missing from {ranking['id']} ranking")
            per_variant.append({**entry, "query_variant": ranking["id"], "top10": entry["rank"] <= 10,
                                "top20": entry["rank"] <= 20, "top30": entry["rank"] <= 30,
                                "classification": classify(entry["rank"]),
                                "semantic_rank": ranking["semantic_rank_by_chunk"][chunk_id],
                                "bm25_rank": ranking["bm25_rank_by_chunk"][chunk_id]})
        best = min(per_variant, key=lambda item: (item["rank"], item["query_variant"]))
        summaries.append({"chunk_id": chunk_id, "per_variant": per_variant,
                          "best_rank_across_variants": best["rank"], "best_variant": best["query_variant"],
                          "best_semantic_score": best["semantic_score"], "best_bm25_score": best["bm25_score"],
                          "best_combined_score": best["combined_score"],
                          "best_semantic_rank": min(item["semantic_rank"] for item in per_variant),
                          "best_bm25_rank": min(item["bm25_rank"] for item in per_variant),
                          "best_hybrid_rank": best["rank"]})
    return summaries


def fusion_comparison(expected_summary: list[dict], trace_path: Path) -> list[dict]:
    experiments = json.loads(trace_path.read_text(encoding="utf-8"))["experiments"]
    names = ("A_control_max_depth_10", "B_max_depth_20", "B_max_depth_30", "C_rrf_k60_depth_10", "D_round_robin_union_depth_10")
    output = []
    for summary in expected_summary:
        ranks = {}
        for name in names:
            row = next(row for row in experiments[name]["traces"] if row["question_id"] == QUESTION_ID)
            ranks[name] = next((item["rank"] for item in row["ranked_chunks"] if item["chunk_id"] == summary["chunk_id"]), None)
        output.append({"chunk_id": summary["chunk_id"], "best_individual_variant": summary["best_variant"],
                       "best_individual_rank": summary["best_rank_across_variants"], "final_ranks": ranks})
    return output


def root_cause_interpretation(payload: dict) -> str:
    summaries = payload["expected_chunk_summary"]
    competitive = [item["chunk_id"] for item in summaries if item["best_rank_across_variants"] <= 30]
    fusion_losses = [item["chunk_id"] for item in payload["fusion_comparison"]
                     if item["best_individual_rank"] <= 10 and all(rank is None for rank in item["final_ranks"].values())]
    if len(competitive) == 4:
        return "All four chunks are individually top-30 reachable; any final absence is downstream of individual retrieval."
    if competitive == ["paper1_c0027"]:
        return "Case 5: c0027 is the only individually competitive chunk. Treat its fusion loss as chunk-specific and do not generalize it to c0028–c0030."
    if not competitive:
        return "Case 3: all four chunks are individually weak, indicating retrieval/query-representation difficulty rather than a fusion-only explanation."
    return f"Case 2: {', '.join(competitive)} are individually competitive while the remaining evidence is weak; the q009 failure is mixed."


def render_report(payload: dict) -> str:
    lines = ["# q009 Full Candidate-Ranking Diagnostic", "", "## Objective", "",
             "Measure all 127 candidates for each frozen q009 query variant without changing retrieval, judgments, or fusion.",
             "", "## Frozen Experimental Conditions", "",
             f"* Model: `{payload['model']}`; hybrid weights: semantic {payload['semantic_weight']:.2f}, BM25 {payload['bm25_weight']:.2f}.",
             "* Ranking uses the existing `src.retrieval.rank` descending-score/ascending-chunk-ID tie-breaker.",
             "", "## Query Variants", ""]
    for variant in payload["query_variants"]:
        lines.append(f"* `{variant['id']}`: {variant['query']}")
    lines += ["", "## Candidate-Space Validation", "", "All four rankings contain exactly 127 unique indexed chunk IDs with contiguous ranks 1–127.",
              "", "## Full Ranking Results", "", "Complete 127-row rankings for all four variants (508 rows) are persisted in the JSON artifact.",
              "", "## Expected Evidence Reachability", "", "| Chunk | Variant | Rank | Semantic | BM25 | Hybrid | Class |", "| --- | --- | ---: | ---: | ---: | ---: | --- |"]
    for summary in payload["expected_chunk_summary"]:
        for item in summary["per_variant"]:
            lines.append(f"| `{summary['chunk_id']}` | `{item['query_variant']}` | {item['rank']} | {item['semantic_score']:.6f} | {item['bm25_score']:.6f} | {item['combined_score']:.6f} | {item['classification']} |")
    lines += ["", "## Individual Retrieval vs Fusion Comparison", "", "| Chunk | Best variant | Best rank | Max 10 | Max 20 | Max 30 | RRF 10 | Round-robin 10 |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for item in payload["fusion_comparison"]:
        ranks = item["final_ranks"]
        value = lambda name: ranks[name] if ranks[name] is not None else "N/A"
        lines.append(f"| `{item['chunk_id']}` | `{item['best_individual_variant']}` | {item['best_individual_rank']} | {value('A_control_max_depth_10')} | {value('B_max_depth_20')} | {value('B_max_depth_30')} | {value('C_rrf_k60_depth_10')} | {value('D_round_robin_union_depth_10')} |")
    lines += ["", "## Evidence Content Assessment", "",
              "* **c0027:** contrasts more frequent retrieval for factual accuracy with less retrieval for open-ended creativity/utility; it supports the high-level trade-off and uses related but not identical wording.",
              "* **c0028:** specifies the `Retrieve=Yes` threshold and critic-weighted decoding; it supports the inference-control mechanism and overlaps lexically with retrieval/control terminology.",
              "* **c0029:** makes critique weights inference-time hyperparameters and gives higher `ISSUP` weight as the evidence-grounding control; its reflection-token terminology differs materially from q009.",
              "* **c0030:** gives hard critique-token filtering and says SELF-RAG tailors behavior with no additional training; it directly supports the no-retraining qualifier but uses decoding/critique terminology.",
              "", "The measured semantic/BM25 ranks below, rather than wording alone, identify whether each chunk is helped by semantic similarity, lexical overlap, or both.",
              "", "## Retrieval Characteristics", "", "| Chunk | Best semantic rank | Best BM25 rank | Best hybrid rank | Best hybrid variant |", "| --- | ---: | ---: | ---: | --- |"]
    for summary in payload["expected_chunk_summary"]:
        lines.append(f"| `{summary['chunk_id']}` | {summary['best_semantic_rank']} | {summary['best_bm25_rank']} | {summary['best_hybrid_rank']} | `{summary['best_variant']}` |")
    lines += ["", "## Root-Cause Interpretation", "", root_cause_interpretation(payload),
              "", "## Limitations", "", "This is a fixed-corpus, fixed-model measurement, not a production retrieval change. The conclusion is limited to these four frozen formulations and 127 chunks.",
              "", "## Next Experiment Recommendation", "", "**One experiment — terminology-aligned q009 formulation ablation.** **Hypothesis:** the expected chunks that remain weak are missed because q009 uses different surface terminology from `Retrieve`, `ISSUP`, and `Critique`. **Independent variable:** one predeclared terminology-aligned query formulation versus the frozen original wording. **Frozen variables:** corpus, chunks, model, BM25, 0.75/0.25 weighting, judgments, and final-k. **Metrics:** per-chunk semantic/BM25/hybrid rank and recall@10/@30 for c0027–c0030. **Failure signal:** no meaningful rank improvement for the weak chunks. **Interpretation:** improvement supports query-representation difficulty; no improvement redirects investigation away from wording toward representation/chunk retrieval.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default="evaluation/semantic_embeddings.json")
    parser.add_argument("--queries", default="evaluation/multi_query_queries.json")
    parser.add_argument("--fusion-traces", default="evaluation/fusion_experiment_traces.json")
    parser.add_argument("--output", default="evaluation/q009_full_candidate_rankings.json")
    parser.add_argument("--analysis", default="evaluation/q009_full_candidate_rankings_analysis.md")
    args = parser.parse_args()
    chunks, dataset, judgments = load_evaluation_inputs()
    question = next(row for row in dataset if row["id"] == QUESTION_ID)
    relevant = next(set(row["relevant_chunk_ids"]) for row in judgments if row["question_id"] == QUESTION_ID)
    variants = json.loads(Path(args.queries).read_text(encoding="utf-8"))[QUESTION_ID]
    formulations = [{"id": "original", "query": question["question"]}, *variants]
    validate_variant_ids(formulations)
    if MODEL_NAME != "sentence-transformers/all-MiniLM-L6-v2" or (SEMANTIC_WEIGHT, BM25_WEIGHT) != (.75, .25):
        raise ValueError("frozen model or hybrid weights changed")
    if not set(EXPECTED_CHUNK_IDS).issubset({chunk["chunk_id"] for chunk in chunks}):
        raise ValueError("an expected q009 chunk is missing from the indexed corpus")
    retriever = HybridRetriever(chunks, primary=SemanticRetriever(chunks, MODEL_NAME, artifact_path=args.artifact), semantic_weight=SEMANTIC_WEIGHT)
    rankings = []
    for formulation in formulations:
        records, semantic_ids, bm25_ids = full_ranking(retriever, formulation["query"], relevant)
        validate_ranking(records, chunks)
        rankings.append({**formulation, "ranked_chunks": records,
                         "semantic_rank_by_chunk": {chunk_id: index + 1 for index, chunk_id in enumerate(semantic_ids)},
                         "bm25_rank_by_chunk": {chunk_id: index + 1 for index, chunk_id in enumerate(bm25_ids)}})
    expected_summary = extract_expected(rankings)
    payload = {"experiment_type": "frozen_q009_full_candidate_ranking", "model": MODEL_NAME,
               "semantic_weight": SEMANTIC_WEIGHT, "bm25_weight": BM25_WEIGHT, "question_id": QUESTION_ID,
               "question": question["question"], "query_variants": formulations, "expected_chunk_ids": list(EXPECTED_CHUNK_IDS),
               "full_rankings": rankings, "expected_chunk_summary": expected_summary,
               "fusion_comparison": fusion_comparison(expected_summary, Path(args.fusion_traces)),
               "evidence_chunks": {chunk_id: next(chunk for chunk in chunks if chunk["chunk_id"] == chunk_id) for chunk_id in EXPECTED_CHUNK_IDS}}
    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    Path(args.analysis).write_text(render_report(payload), encoding="utf-8")
    for item in expected_summary:
        print(f"{item['chunk_id']}: {item['best_variant']} rank {item['best_rank_across_variants']} ({classify(item['best_rank_across_variants'])})")


if __name__ == "__main__":
    main()
