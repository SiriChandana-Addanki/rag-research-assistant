from src.evaluation import retrieval_trace
from src.retrieval import HybridRetriever

CHUNKS=[
 {"document_id":"paper","chunk_id":"c2","chunk_index":1,"page_start":2,"page_end":2,"section":"METHOD","chunk_text":"beta"},
 {"document_id":"paper","chunk_id":"c1","chunk_index":0,"page_start":1,"page_end":1,"section":"INTRODUCTION","chunk_text":"alpha"},
]
DATASET=[{"id":"q1","question":"alpha","relevant_sections":[],"relevant_pages":[]}]
JUDGMENTS=[{"question_id":"q1","relevant_chunk_ids":["c1"],"judgment_basis":"fixture"}]

class Semantic:
 def __init__(self,chunks): self.chunks=chunks
 def scores(self,query): return [.25,.75]

def test_retrieval_trace_preserves_ranked_component_scores_and_metadata():
 trace=retrieval_trace(HybridRetriever(CHUNKS,primary=Semantic(CHUNKS),semantic_weight=.5),DATASET,JUDGMENTS)
 assert trace[0]["expected_relevant_chunk_ids"]==["c1"]
 first=trace[0]["ranked_chunks"][0]
 assert first=={"rank":1,"chunk_id":"c1","semantic_score":.75,"bm25_score":.6931471805599453,"combined_score":1.0,"page_start":1,"page_end":1,"section":"INTRODUCTION","is_relevant":True}
 assert trace[0]["ranked_chunks"][1]["combined_score"]==0.0

def test_semantic_trace_omits_inapplicable_component_scores_with_nulls():
 trace=retrieval_trace(Semantic(CHUNKS),DATASET,JUDGMENTS)
 assert trace[0]["ranked_chunks"][0]["semantic_score"]==.75
 assert trace[0]["ranked_chunks"][0]["bm25_score"] is None
 assert trace[0]["ranked_chunks"][0]["combined_score"] is None
