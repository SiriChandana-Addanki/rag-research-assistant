"""Deterministic trace construction for the explicit multi-query experiment."""
from src.evaluation import KS, metrics_from_trace
from src.retrieval import normalize


def _rank(chunks, scores, k):
    return sorted(range(len(chunks)), key=lambda index: (-scores[index], chunks[index]["chunk_id"]))[:k]


def _entry(chunk, rank, semantic, bm25, combined, relevant, fused=None, variants=None):
    return {
        "rank": rank, "chunk_id": chunk["chunk_id"], "semantic_score": semantic,
        "bm25_score": bm25, "combined_score": combined, "fused_score": fused,
        "winning_query_variant_ids": variants or [], "page_start": chunk["page_start"],
        "page_end": chunk["page_end"], "section": chunk["section"],
        "is_relevant": chunk["chunk_id"] in relevant,
    }


def _scores(retriever, query):
    semantic = retriever.primary.scores(query)
    bm25 = retriever.bm25.scores(query)
    combined = [retriever.semantic_weight * left + (1 - retriever.semantic_weight) * right
                for left, right in zip(normalize(semantic), normalize(bm25))]
    return semantic, bm25, combined


def evidence_coverage(ranked_chunks, expected_chunk_ids):
    """Report coverage without assuming a list or set representation of gold IDs."""
    expected = sorted(set(expected_chunk_ids))
    ranks = {chunk_id: next((entry["rank"] for entry in ranked_chunks if entry["chunk_id"] == chunk_id), None)
             for chunk_id in expected}
    return {"expected_chunk_count": len(expected), "ranks": ranks,
            "first_relevant_rank": min((rank for rank in ranks.values() if rank is not None), default=None),
            "retrieved_at": {str(k): sum(rank is not None and rank <= k for rank in ranks.values()) for k in KS},
            "all_expected_recovered_at_10": all(rank is not None and rank <= 10 for rank in ranks.values())}


def experiment_trace(retriever, dataset, judgments, variants, final_k=max(KS)):
    """Return baseline/multi traces and q009-style coverage without changing retrieval."""
    gold = {item["question_id"]: set(item["relevant_chunk_ids"]) for item in judgments}
    baseline, multi, coverage = [], [], {}
    for question in dataset:
        question_id, original = question["id"], question["question"]
        configured = variants.get(question_id, [])
        formulations = [{"id": "original", "query": original}, *configured]
        relevant = gold[question_id]
        variant_scores = []
        variant_traces = []
        for formulation in formulations:
            semantic, bm25, combined = _scores(retriever, formulation["query"])
            variant_scores.append((semantic, bm25, combined))
            indices = _rank(retriever.chunks, combined, final_k)
            variant_traces.append({"id": formulation["id"], "query": formulation["query"],
                "ranked_chunks": [_entry(retriever.chunks[index], rank, semantic[index], bm25[index], combined[index], relevant)
                                  for rank, index in enumerate(indices, 1)]})
        semantic, bm25, combined = variant_scores[0]
        baseline_indices = _rank(retriever.chunks, combined, final_k)
        base = {"question_id": question_id, "question": original, "expected_relevant_chunk_ids": sorted(relevant),
                "candidate_count": len(retriever.chunks), "query_count": 1, "final_top_k": final_k,
                "fusion_method": "single_query", "ranked_chunks": [_entry(retriever.chunks[index], rank, semantic[index], bm25[index], combined[index], relevant)
                for rank, index in enumerate(baseline_indices, 1)]}
        fused = [max(scores[2][index] for scores in variant_scores) for index in range(len(retriever.chunks))]
        fused_indices = _rank(retriever.chunks, fused, final_k)
        fused_entries = []
        for rank, index in enumerate(fused_indices, 1):
            winners = [formulations[position]["id"] for position, scores in enumerate(variant_scores) if scores[2][index] == fused[index]]
            winner = next(position for position, scores in enumerate(variant_scores) if scores[2][index] == fused[index])
            sem, lexical, combined_score = variant_scores[winner][0][index], variant_scores[winner][1][index], variant_scores[winner][2][index]
            fused_entries.append(_entry(retriever.chunks[index], rank, sem, lexical, combined_score, relevant, fused[index], winners))
        fused_trace = {**base, "query_count": len(formulations), "fusion_method": "max_combined_score", "query_variants": variant_traces, "ranked_chunks": fused_entries}
        baseline.append(base); multi.append(fused_trace)
        if question_id == "q009":
            coverage[question_id] = evidence_coverage(fused_entries, relevant)
    return {"baseline": baseline, "multi_query": multi, "q009_evidence_coverage": coverage,
            "metrics": {"baseline": metrics_from_trace(baseline, dataset, judgments), "multi_query": metrics_from_trace(multi, dataset, judgments)}}
