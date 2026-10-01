"""Dependency-light local retrieval baselines for the tracked chunk manifest."""
from __future__ import annotations
import hashlib, json, math, re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
TOKEN = re.compile(r"[a-z0-9]+")
def terms(text): return TOKEN.findall(text.lower())
def _cos(a,b):
 d=math.sqrt(sum(x*x for x in a.values())*sum(x*x for x in b.values()))
 return sum(v*b.get(k,0) for k,v in a.items())/d if d else 0.0
@dataclass(frozen=True)
class Result:
 chunk: dict; score: float
class DenseRetriever:
 """Stable hashed TF-IDF baseline; uses no network or model download."""
 def __init__(self,chunks):
  self.chunks=chunks; self.docs=[Counter(terms(c['chunk_text'])) for c in chunks]
  self.df=Counter({t:sum(t in d for d in self.docs) for t in set().union(*self.docs)})
  self.n=len(chunks)
 def vector(self,text):
  c=Counter(terms(text)); return {t:(1+math.log(v))*math.log((self.n+1)/(self.df.get(t,0)+1)) for t,v in c.items()}
 def scores(self,query):
  if not query.strip(): raise ValueError('query must not be empty')
  q=self.vector(query); return [_cos(q,{t:(1+math.log(v))*math.log((self.n+1)/(self.df[t]+1)) for t,v in d.items()}) for d in self.docs]
 def search(self,q,k=5): return rank(self.chunks,self.scores(q),k)
class BM25Retriever:
 def __init__(self,chunks,k1=1.5,b=0.75):
  self.chunks,self.k1,self.b=chunks,k1,b; self.docs=[Counter(terms(c['chunk_text'])) for c in chunks]; self.lengths=[sum(x.values()) for x in self.docs]; self.avg=sum(self.lengths)/len(chunks); self.df=Counter({t:sum(t in d for d in self.docs) for t in set().union(*self.docs)})
 def scores(self,q):
  if not q.strip(): raise ValueError('query must not be empty')
  out=[]
  for d,l in zip(self.docs,self.lengths):
   s=0
   for t in set(terms(q)):
    f=d[t]; idf=math.log(1+(len(self.docs)-self.df.get(t,0)+.5)/(self.df.get(t,0)+.5)); s+=idf*(f*(self.k1+1)/(f+self.k1*(1-self.b+self.b*l/self.avg))) if f else 0
   out.append(s)
  return out
 def search(self,q,k=5): return rank(self.chunks,self.scores(q),k)
class HybridRetriever:
 """Min-max normalize each score list then combine `dense_weight` and BM25."""
 def __init__(self,chunks,dense_weight=.5): self.dense=DenseRetriever(chunks); self.bm25=BM25Retriever(chunks); self.chunks=chunks; self.dense_weight=dense_weight
 def search(self,q,k=5):
  a,b=self.dense.scores(q),self.bm25.scores(q)
  norm=lambda x:[(v-min(x))/(max(x)-min(x)) if max(x)!=min(x) else 0 for v in x]
  a,b=norm(a),norm(b); return rank(self.chunks,[self.dense_weight*x+(1-self.dense_weight)*y for x,y in zip(a,b)],k)
def rank(chunks,scores,k):
 return [Result(chunks[i],scores[i]) for i in sorted(range(len(chunks)),key=lambda i:(-scores[i],chunks[i]['chunk_id']))[:k]]
def load_manifest(path='evaluation/chunk_manifest.json'):
 d=json.loads(Path(path).read_text()); validate_manifest(d); return d
def validate_manifest(chunks):
 required={'document_id','chunk_id','chunk_index','page_start','page_end','section','chunk_text'}
 if not chunks or any(set(c)!=required or not c['chunk_text'].strip() or c['page_start']>c['page_end'] for c in chunks): raise ValueError('malformed chunk metadata')
 if len({c['chunk_id'] for c in chunks})!=len(chunks): raise ValueError('duplicate chunk id')
