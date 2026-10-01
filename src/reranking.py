"""Optional reranking interfaces; external cross-encoders are loaded lazily."""
from typing import Protocol
from src.retrieval import Result
class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[Result], k: int) -> list[Result]: ...
class CrossEncoderReranker:
    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2", model=None):
        if model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as error: raise RuntimeError("sentence-transformers is required for CrossEncoderReranker") from error
            model=CrossEncoder(model_name)
        self.model=model
    def rerank(self, query, candidates, k):
        scores=self.model.predict([(query, item.chunk["chunk_text"]) for item in candidates])
        return [item for _,item in sorted(zip(scores,candidates), key=lambda pair: -pair[0])[:k]]
