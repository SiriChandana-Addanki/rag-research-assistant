import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_q009_terminology_ablation.py"


def load_module():
    spec = importlib.util.spec_from_file_location("q009_ablation", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_q009_ablation_freezes_conditions_and_reports_expected_chunks(monkeypatch):
    module = load_module()
    expected = [f"paper1_c{number:04d}" for number in range(27, 31)]
    chunks = [
        {"document_id": "paper1", "chunk_id": chunk_id, "chunk_index": index,
         "page_start": 1, "page_end": 1, "section": "inference", "chunk_text": chunk_id}
        for index, chunk_id in enumerate(expected)
    ]
    chunks.extend(
        {"document_id": "paper2", "chunk_id": f"other_{index:03d}", "chunk_index": index + 4,
         "page_start": 1, "page_end": 1, "section": "other", "chunk_text": "other"}
        for index in range(123)
    )
    dataset = [{"id": "q009", "question": "control"}]
    judgments = [{"question_id": "q009", "relevant_chunk_ids": expected}]

    class Semantic:
        def __init__(self, chunks, *args, **kwargs): self.chunks = chunks
        def scores(self, query):
            bonus = 1 if query == module.ALIGNED_QUERY else 0
            return [float(index + bonus) for index, _ in enumerate(self.chunks)]

    monkeypatch.setattr(module, "load_evaluation_inputs", lambda: (chunks, dataset, judgments))
    monkeypatch.setattr(module, "SemanticRetriever", Semantic)
    payload = module.run()

    assert payload["status"] == "completed"
    assert payload["frozen_conditions"]["chunk_count"] == 127
    assert payload["frozen_conditions"]["semantic_weight"] == .75
    assert [item["id"] for item in payload["variants"]] == ["control", "terminology_aligned"]
    for variant in payload["variants"]:
        records = variant["expected_chunk_measurements"]
        assert [record["chunk_id"] for record in records] == expected
        assert all({"semantic_rank", "bm25_rank", "hybrid_rank", "semantic_score", "bm25_score", "hybrid_score", "entered_top_10", "entered_top_30"} <= set(record) for record in records)
    assert "hybrid_rank_change_vs_control" in payload["variants"][1]["expected_chunk_measurements"][0]
