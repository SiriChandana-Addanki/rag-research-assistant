"""Local lexical retrieval and optional sentence-transformer semantic retrieval."""
from __future__ import annotations
import hashlib, json, math, re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

TOKEN = re.compile(r"[a-z0-9]+")
def terms(text: str) -> list[str]: return TOKEN.findall(text.lower())
@dataclass(frozen=True)
class Result: chunk: dict; score: float

def rank(chunks, scores, k):
    return [Result(chunks[i], scores[i]) for i in sorted(range(len(chunks)), key=lambda i: (-scores[i], chunks[i]["chunk_id"]))[:k]]
def normalize(scores):
    low, high = min(scores), max(scores)
    return [(x-low)/(high-low) if high != low else 0.0 for x in scores]
def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b)); norm = math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
    return dot/norm if norm else 0.0
class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[Result]: ...
class TfidfRetriever:
    """Sparse lexical TF-IDF/cosine baseline, not a neural dense retriever."""
    def __init__(self, chunks):
        self.chunks, self.docs = chunks, [Counter(terms(c["chunk_text"])) for c in chunks]
        self.n = len(chunks); vocabulary = set().union(*self.docs)
        self.df = {term: sum(term in doc for doc in self.docs) for term in vocabulary}
    def vector(self, text):
        counts = Counter(terms(text))
        return {term: (1 + math.log(count))*math.log((self.n + 1)/(self.df.get(term, 0) + 1)) for term, count in counts.items()}
    def scores(self, query):
        if not query.strip(): raise ValueError("query must not be empty")
        query_vector = self.vector(query)
        return [self._score(query_vector, self.vector_from_counts(doc)) for doc in self.docs]
    def vector_from_counts(self, counts):
        return {t: (1+math.log(n))*math.log((self.n+1)/(self.df[t]+1)) for t,n in counts.items()}
    def _score(self, left, right):
        denominator = math.sqrt(sum(x*x for x in left.values())*sum(x*x for x in right.values()))
        return sum(value*right.get(term, 0) for term, value in left.items())/denominator if denominator else 0.0
    def search(self, query, k=5): return rank(self.chunks, self.scores(query), k)
class BM25Retriever:
    def __init__(self, chunks, k1=1.5, b=.75):
        self.chunks, self.k1, self.b = chunks, k1, b; self.docs=[Counter(terms(c["chunk_text"])) for c in chunks]
        self.lengths=[sum(d.values()) for d in self.docs]; self.average_length=sum(self.lengths)/len(chunks); vocab=set().union(*self.docs); self.df={t:sum(t in d for d in self.docs) for t in vocab}
    def scores(self, query):
        if not query.strip(): raise ValueError("query must not be empty")
        scores=[]
        for document, length in zip(self.docs, self.lengths):
            score=0.0
            for term in set(terms(query)):
                frequency=document[term]
                if frequency:
                    idf=math.log(1+(len(self.docs)-self.df.get(term,0)+.5)/(self.df.get(term,0)+.5))
                    score += idf*frequency*(self.k1+1)/(frequency+self.k1*(1-self.b+self.b*length/self.average_length))
            scores.append(score)
        return scores
    def search(self, query, k=5): return rank(self.chunks, self.scores(query), k)
class SemanticRetriever:
    """True neural semantic retrieval, loaded lazily to keep lexical use dependency-free."""
    def __init__(self, chunks, model_name="sentence-transformers/all-MiniLM-L6-v2", batch_size=32, encoder=None, artifact_path=None):
        self.chunks, self.model_name, self.batch_size = chunks, model_name, batch_size
        if batch_size <= 0: raise ValueError("batch_size must be positive")
        if encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError("sentence-transformers is required for SemanticRetriever") from error
            encoder = SentenceTransformer(model_name)
        self.encoder=encoder
        self.artifact_path=Path(artifact_path) if artifact_path else None
        self.embeddings=self._load_or_encode([c["chunk_text"] for c in chunks])
    def _fingerprint(self, texts):
        payload=json.dumps({"model_name":self.model_name,"texts":texts}, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
    def _normalize_vector(self, vector):
        values=[float(value) for value in vector]; magnitude=math.sqrt(sum(value*value for value in values))
        return [value/magnitude for value in values] if magnitude else values
    def _encode(self, texts):
        vectors=self.encoder.encode(texts, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
        return [self._normalize_vector(vector) for vector in vectors]
    def _load_or_encode(self, texts):
        fingerprint=self._fingerprint(texts)
        if self.artifact_path and self.artifact_path.is_file():
            cached=json.loads(self.artifact_path.read_text(encoding="utf-8"))
            if cached.get("fingerprint") == fingerprint:
                return cached["embeddings"]
        embeddings=self._encode(texts)
        if self.artifact_path:
            self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
            self.artifact_path.write_text(json.dumps({"fingerprint":fingerprint,"embeddings":embeddings})+"\n", encoding="utf-8")
        return embeddings
    def scores(self, query):
        if not query.strip(): raise ValueError("query must not be empty")
        query_vector=self._encode([query])[0]; return [cosine(query_vector, vector) for vector in self.embeddings]
    def search(self, query, k=5): return rank(self.chunks, self.scores(query), k)
class HybridRetriever:
    """Combines a supplied semantic/sparse retriever and BM25 after min-max scaling."""
    def __init__(self, chunks, primary=None, semantic_weight=.5):
        self.chunks=chunks; self.primary=primary or TfidfRetriever(chunks); self.bm25=BM25Retriever(chunks); self.semantic_weight=semantic_weight
    def search(self, query, k=5):
        first, second=normalize(self.primary.scores(query)), normalize(self.bm25.scores(query))
        return rank(self.chunks, [self.semantic_weight*a+(1-self.semantic_weight)*b for a,b in zip(first,second)], k)
def load_manifest(path="evaluation/chunk_manifest.json"):
    chunks=json.loads(Path(path).read_text(encoding="utf-8")); validate_manifest(chunks); return chunks
def validate_manifest(chunks):
    required={"document_id","chunk_id","chunk_index","page_start","page_end","section","chunk_text"}
    if not chunks or any(set(c)!=required or not c["chunk_text"].strip() or c["page_start"]>c["page_end"] for c in chunks): raise ValueError("malformed chunk metadata")
    if len({c["chunk_id"] for c in chunks}) != len(chunks): raise ValueError("duplicate chunk id")
