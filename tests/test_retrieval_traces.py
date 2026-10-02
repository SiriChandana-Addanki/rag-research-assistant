import importlib.util
import json
import sys
from pathlib import Path

from src.evaluation import metrics, metrics_from_trace
from src.retrieval import HybridRetriever


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_semantic_retrieval.py"


class FakeSemanticRetriever:
    def __init__(self, chunks, *args, **kwargs):
        self.chunks = chunks

    def scores(self, query):
        return [float(chunk["semantic_scores"][query]) for chunk in self.chunks]

    def search(self, query, k=5):
        from src.retrieval import rank

        return rank(self.chunks, self.scores(query), k)


def load_script_module():
    spec = importlib.util.spec_from_file_location("evaluate_semantic_retrieval", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_semantic_evaluator_persists_ranked_retrieval_traces(tmp_path, monkeypatch):
    chunks = [
        {"document_id": "doc", "chunk_id": "chunk-a", "chunk_index": 0, "page_start": 1, "page_end": 1, "section": "Alpha", "chunk_text": "alpha", "semantic_scores": {"alpha question": .9, "beta question": .2}},
        {"document_id": "doc", "chunk_id": "chunk-b", "chunk_index": 1, "page_start": 2, "page_end": 3, "section": "Beta", "chunk_text": "beta", "semantic_scores": {"alpha question": .4, "beta question": .8}},
    ]
    dataset = [{"id": "q1", "question": "alpha question"}, {"id": "q2", "question": "beta question"}]
    judgments = [{"question_id": "q1", "relevant_chunk_ids": ["chunk-a"]}, {"question_id": "q2", "relevant_chunk_ids": ["chunk-b"]}]
    module = load_script_module()
    monkeypatch.setattr(module, "SemanticRetriever", FakeSemanticRetriever)
    monkeypatch.setattr(module, "load_evaluation_inputs", lambda: (chunks, dataset, judgments))
    output = tmp_path / "results.json"
    trace_output = tmp_path / "evaluation" / "semantic_retrieval_traces.json"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT_PATH), "--output", str(output)])

    module.main()

    result = json.loads(output.read_text(encoding="utf-8"))
    trace = json.loads(trace_output.read_text(encoding="utf-8"))
    assert result["semantic"] == metrics(FakeSemanticRetriever(chunks), dataset, judgments)
    for weight in (.25, .5, .75):
        name = f"semantic_{weight:.2f}_bm25_{1 - weight:.2f}"
        assert result[name] == metrics(
            HybridRetriever(chunks, primary=FakeSemanticRetriever(chunks), semantic_weight=weight),
            dataset,
            judgments,
        )
    assert set(trace["configurations"]) == {"semantic", "semantic_0.25_bm25_0.75", "semantic_0.50_bm25_0.50", "semantic_0.75_bm25_0.25"}
    for name, questions in trace["configurations"].items():
        assert [item["question_id"] for item in questions] == ["q1", "q2"]
        assert questions[0]["question"] == "alpha question"
        assert questions[0]["expected_relevant_chunk_ids"] == ["chunk-a"]
        ranked = questions[0]["ranked_chunks"]
        assert ranked[0]["rank"] == 1
        assert ranked[0]["chunk_id"] == "chunk-a"
        assert ranked[0]["page_start"] == 1 and ranked[0]["page_end"] == 1
        assert ranked[0]["section"] == "Alpha" and ranked[0]["is_relevant"] is True
        assert metrics_from_trace(questions, dataset, judgments) == result[name]
    semantic_chunk = trace["configurations"]["semantic"][0]["ranked_chunks"][0]
    assert semantic_chunk["semantic_score"] == .9
    assert semantic_chunk["bm25_score"] is None and semantic_chunk["combined_score"] is None
    hybrid_chunk = trace["configurations"]["semantic_0.50_bm25_0.50"][0]["ranked_chunks"][0]
    assert hybrid_chunk["semantic_score"] == .9
    assert hybrid_chunk["bm25_score"] > 0
    assert hybrid_chunk["combined_score"] == 1


def test_metrics_from_trace_rejects_relevance_drift():
    dataset = [{"id": "q1", "question": "question"}]
    judgments = [{"question_id": "q1", "relevant_chunk_ids": ["chunk-a"]}]
    trace = [{
        "question_id": "q1", "question": "question",
        "expected_relevant_chunk_ids": ["chunk-a"],
        "ranked_chunks": [{"rank": 1, "chunk_id": "chunk-a", "is_relevant": False}],
    }]

    import pytest

    with pytest.raises(ValueError, match="relevance flags"):
        metrics_from_trace(trace, dataset, judgments)


def test_multi_query_retriever_keeps_the_original_query_and_fuses_deterministically():
    from src.retrieval import MultiQueryRetriever

    chunks = [
        {"chunk_id": "chunk-a", "chunk_text": "a"},
        {"chunk_id": "chunk-b", "chunk_text": "b"},
    ]
    primary = FakeSemanticRetriever(chunks)
    chunks[0]["semantic_scores"] = {"original": .2, "aspect": .9}
    chunks[1]["semantic_scores"] = {"original": .8, "aspect": .1}
    retriever = MultiQueryRetriever(primary, lambda query: ["aspect", "aspect", ""])

    assert retriever.queries("original") == ["original", "aspect"]
    assert [result.chunk["chunk_id"] for result in retriever.search("original", 2)] == [
        "chunk-a", "chunk-b"
    ]
