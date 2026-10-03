import importlib.util
from pathlib import Path

import pytest

from src.retrieval import HybridRetriever

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_q009_full_candidate_rankings.py"
spec = importlib.util.spec_from_file_location("q009_rankings", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def chunks(count=127):
    return [{"chunk_id": f"chunk-{index:03d}", "chunk_text": f"term {index}", "page_start": 1,
             "page_end": 1, "section": "S"} for index in range(count)]


def records(rows):
    return [{"rank": index + 1, "chunk_id": row["chunk_id"]} for index, row in enumerate(rows)]


def test_validate_ranking_requires_exactly_127_unique_contiguous_chunks():
    rows = chunks()
    module.validate_ranking(records(rows), rows)
    with pytest.raises(ValueError, match="127"):
        module.validate_ranking(records(rows[:-1]), rows[:-1])
    duplicate = records(rows)
    duplicate[-1]["chunk_id"] = duplicate[0]["chunk_id"]
    with pytest.raises(ValueError, match="duplicate"):
        module.validate_ranking(duplicate, rows)
    discontinuous = records(rows)
    discontinuous[-1]["rank"] = 129
    with pytest.raises(ValueError, match="contiguous"):
        module.validate_ranking(discontinuous, rows)


def test_classification_boundaries_are_deterministic():
    assert [module.classify(value) for value in (1, 10, 11, 30, 31, 127, None)] == ["A", "A", "B", "B", "C", "C", "D"]


def test_extract_expected_chunks_and_summary_match_persisted_ranking():
    expected = list(module.EXPECTED_CHUNK_IDS)
    ranking = []
    for variant_index, variant in enumerate(module.EXPECTED_VARIANT_IDS):
        ranked = []
        for position, chunk_id in enumerate(expected, 1):
            ranked.append({"rank": position + variant_index, "chunk_id": chunk_id, "semantic_score": .1 * position,
                           "bm25_score": float(position), "combined_score": .2 * position})
        ranking.append({"id": variant, "ranked_chunks": ranked,
                        "semantic_rank_by_chunk": {chunk_id: position + 10 for position, chunk_id in enumerate(expected, 1)},
                        "bm25_rank_by_chunk": {chunk_id: position + 20 for position, chunk_id in enumerate(expected, 1)}})
    summary = module.extract_expected(ranking)
    assert len(summary) == 4
    assert summary[0]["chunk_id"] == expected[0]
    assert summary[0]["best_variant"] == "original"
    assert summary[0]["best_rank_across_variants"] == 1
    assert summary[0]["per_variant"][0]["classification"] == "A"
    assert summary[0]["best_hybrid_rank"] == 1


def test_missing_expected_chunk_is_explicit_error():
    ranking = [{"id": variant, "ranked_chunks": [], "semantic_rank_by_chunk": {}, "bm25_rank_by_chunk": {}}
               for variant in module.EXPECTED_VARIANT_IDS]
    with pytest.raises(ValueError, match="expected chunk"):
        module.extract_expected(ranking)


def test_full_ranking_uses_all_chunks_once_and_existing_hybrid_scoring():
    rows = chunks()
    class Primary:
        def __init__(self, rows): self.chunks = rows
        def scores(self, query): return [float(index) for index in range(len(self.chunks))]
    retriever = HybridRetriever(rows, primary=Primary(rows), semantic_weight=.75)
    ranked, semantic_ids, bm25_ids = module.full_ranking(retriever, "term", set())
    module.validate_ranking(ranked, rows)
    assert len(ranked) == len(semantic_ids) == len(bm25_ids) == 127
    assert len({record["chunk_id"] for record in ranked}) == 127
