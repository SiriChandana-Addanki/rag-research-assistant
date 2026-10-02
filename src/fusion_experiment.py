"""Isolated, label-independent fusion experiments for deterministic queries.

This module intentionally does not participate in ``src.retrieval``.  It only
uses retrieval scores to build candidate pools; judgments are accepted solely
by the trace/evaluation layer after rankings have been produced.
"""
from __future__ import annotations

from collections import defaultdict

from src.evaluation import KS, metrics_from_trace
from src.multi_query_evaluation import evidence_coverage
from src.retrieval import normalize


RRF_CONSTANT = 60


def ranked_indices(chunks, scores, depth):
    """Rank by descending score then ascending chunk ID, as production does."""
    return sorted(range(len(chunks)), key=lambda i: (-scores[i], chunks[i]["chunk_id"]))[:depth]


def max_score_fusion(candidate_lists):
    """Fuse retained candidates by their largest per-query hybrid score."""
    fused = {}
    for candidates in candidate_lists:
        for candidate in candidates:
            chunk_id = candidate["chunk_id"]
            fused[chunk_id] = max(fused.get(chunk_id, float("-inf")), candidate["hybrid_score"])
    return fused


def reciprocal_rank_fusion(candidate_lists, constant=RRF_CONSTANT):
    """Sum ``1 / (constant + rank)`` for every retained candidate occurrence."""
    if constant < 0:
        raise ValueError("RRF constant must be non-negative")
    fused = defaultdict(float)
    for candidates in candidate_lists:
        for candidate in candidates:
            fused[candidate["chunk_id"]] += 1 / (constant + candidate["source_rank"])
    return dict(fused)


def round_robin_union(candidate_lists, capacity):
    """Allocate a fixed union pool cyclically across sources, skipping duplicates.

    The allocation uses only source ordering and chunk IDs.  It deliberately
    knows nothing about questions' relevance judgments.
    """
    if capacity < 1:
        raise ValueError("candidate allocation capacity must be positive")
    selected, offsets = [], [0] * len(candidate_lists)
    while len(selected) < capacity:
        progressed = False
        existing = {item["chunk_id"] for item in selected}
        for position, candidates in enumerate(candidate_lists):
            while offsets[position] < len(candidates) and candidates[offsets[position]]["chunk_id"] in existing:
                offsets[position] += 1
            if offsets[position] < len(candidates) and len(selected) < capacity:
                selected.append(candidates[offsets[position]])
                existing.add(candidates[offsets[position]]["chunk_id"])
                offsets[position] += 1
                progressed = True
        if not progressed:
            break
    return selected


def _variant_scores(retriever, query):
    semantic = retriever.primary.scores(query)
    bm25 = retriever.bm25.scores(query)
    hybrid = [retriever.semantic_weight * a + (1 - retriever.semantic_weight) * b
              for a, b in zip(normalize(semantic), normalize(bm25))]
    return semantic, bm25, hybrid


def _candidates(chunks, formulation, scores, depth):
    semantic, bm25, hybrid = scores
    return [{"chunk_id": chunks[index]["chunk_id"], "source_variant_id": formulation["id"],
             "source_query": formulation["query"], "source_rank": rank,
             "semantic_score": semantic[index], "bm25_score": bm25[index],
             "hybrid_score": hybrid[index], "index": index}
            for rank, index in enumerate(ranked_indices(chunks, hybrid, depth), 1)]


def _trace_entry(chunk, rank, sources, fusion_score, relevant):
    # The highest hybrid-scoring source supplies the directly displayed scores;
    # source order is deterministic for ties.
    winner = sorted(sources, key=lambda item: (-item["hybrid_score"], item["source_variant_id"]))[0]
    return {"rank": rank, "chunk_id": chunk["chunk_id"], "semantic_score": winner["semantic_score"],
            "bm25_score": winner["bm25_score"], "hybrid_score": winner["hybrid_score"],
            "combined_score": winner["hybrid_score"], "fusion_score": fusion_score,
            "candidate_sources": [{key: value for key, value in source.items() if key != "index"} for source in sources],
            "page_start": chunk["page_start"], "page_end": chunk["page_end"], "section": chunk["section"],
            "is_relevant": chunk["chunk_id"] in relevant}


def build_experiment_trace(retriever, dataset, judgments, variants, *, name, fusion_strategy,
                           candidate_depth, rrf_constant=RRF_CONSTANT, allocation_capacity=None,
                           final_k=max(KS)):
    """Build one experiment trace.  ``judgments`` is not used until after fusion."""
    if fusion_strategy not in {"max_score", "rrf", "round_robin_union_max_score"}:
        raise ValueError("unknown fusion strategy")
    retrieval_rows = []
    for question in dataset:
        formulations = [{"id": "original", "query": question["question"]}, *variants.get(question["id"], [])]
        candidate_lists = [_candidates(retriever.chunks, formulation, _variant_scores(retriever, formulation["query"]), candidate_depth)
                           for formulation in formulations]
        if fusion_strategy == "round_robin_union_max_score":
            retained = round_robin_union(candidate_lists, allocation_capacity or candidate_depth * len(candidate_lists))
            allowed = {candidate["chunk_id"] for candidate in retained}
            candidate_lists = [[candidate for candidate in candidates if candidate["chunk_id"] in allowed] for candidates in candidate_lists]
        fusion_scores = reciprocal_rank_fusion(candidate_lists, rrf_constant) if fusion_strategy == "rrf" else max_score_fusion(candidate_lists)
        source_map = defaultdict(list)
        for candidates in candidate_lists:
            for candidate in candidates:
                source_map[candidate["chunk_id"]].append(candidate)
        ranked = sorted(fusion_scores, key=lambda chunk_id: (-fusion_scores[chunk_id], chunk_id))[:final_k]
        # Relevance is deliberately attached only after ranking/candidate allocation.
        relevant = next(set(row["relevant_chunk_ids"]) for row in judgments if row["question_id"] == question["id"])
        chunk_by_id = {chunk["chunk_id"]: chunk for chunk in retriever.chunks}
        entries = [_trace_entry(chunk_by_id[chunk_id], rank, source_map[chunk_id], fusion_scores[chunk_id], relevant)
                   for rank, chunk_id in enumerate(ranked, 1)]
        retrieval_rows.append({"question_id": question["id"], "question": question["question"], "experiment_name": name,
                               "fusion_strategy": fusion_strategy, "candidate_depth": candidate_depth,
                               "rrf_constant": rrf_constant if fusion_strategy == "rrf" else None,
                               "allocation_capacity": allocation_capacity if fusion_strategy == "round_robin_union_max_score" else None,
                               "query_variant_ids": [item["id"] for item in formulations], "query_variants": formulations,
                               "expected_relevant_chunk_ids": sorted(relevant), "ranked_chunks": entries})
    # Evaluation-only enrichment follows retrieval so labels cannot affect it.
    for row in retrieval_rows:
        row["evidence_coverage"] = evidence_coverage(row["ranked_chunks"], row["expected_relevant_chunk_ids"])
        row["evidence_coverage"]["coverage_ratio_at"] = {
            cutoff: row["evidence_coverage"]["retrieved_at"][cutoff] / row["evidence_coverage"]["expected_chunk_count"]
            for cutoff in map(str, KS)}
    return retrieval_rows


def summarize_experiment(traces, dataset, judgments):
    metrics = metrics_from_trace(traces, dataset, judgments)
    coverage = {str(k): sum(row["evidence_coverage"]["coverage_ratio_at"][str(k)] for row in traces) / len(traces) for k in KS}
    complete = sum(row["evidence_coverage"]["all_expected_recovered_at_10"] for row in traces)
    return {"metrics": metrics, "average_evidence_coverage": coverage,
            "complete_evidence_recovery_at_10": {"count": complete, "ratio": complete / len(traces)}}
