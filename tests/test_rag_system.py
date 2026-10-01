import json
from pathlib import Path
import pytest
from src.evaluation import evaluate_all,validate_judgments
from src.rag import MAX_CONTEXT,RAGPipeline,build_context,validate_citations,Citation
from src.retrieval import BM25Retriever,TfidfRetriever,HybridRetriever,Result,SemanticRetriever,load_manifest
ROOT=Path(__file__).parents[1]
@pytest.fixture
def chunks(): return load_manifest(ROOT/'evaluation/chunk_manifest.json')
def test_manifest_and_judgments_are_valid(chunks):
 d=json.loads((ROOT/'evaluation/retrieval_dataset.json').read_text()); j=json.loads((ROOT/'evaluation/relevance_judgments.json').read_text()); validate_judgments(j,d,chunks)
def test_retrievers_rank_and_reject_empty_query(chunks):
 for cls in (TfidfRetriever,BM25Retriever,HybridRetriever):
  r=cls(chunks); assert len(r.search('reflection tokens',3))==3
  with pytest.raises(ValueError): r.search('')
def test_retrieval_metrics_are_bounded():
 for values in evaluate_all().values(): assert all(0<=x<=1 for x in values.values())
def test_grounded_pipeline_uses_retrieved_citations(chunks):
 class Provider:
  def generate(self, prompt, timeout_seconds): return 'Grounded fixture.'
 p=RAGPipeline(chunks,provider=Provider())
 out=p.answer('What are reflection tokens?',2); assert out['answer']=='Grounded fixture.' and len(out['citations'])==2
 with pytest.raises(ValueError): validate_citations([Citation('paper1',1,'missing')],[])
def test_pipeline_input_limits(chunks):
 with pytest.raises(ValueError): RAGPipeline(chunks).answer(' ')
def test_pipeline_handles_empty_retrieval_and_oversized_context(chunks):
 class EmptyRetriever:
  def search(self, query, k): return []
 assert RAGPipeline(chunks,retriever=EmptyRetriever()).answer('question')['citations']==[]
 huge={**chunks[0],"chunk_text":"x"*(MAX_CONTEXT+1)}
 assert build_context([Result(huge,1.0)])==''
def test_pipeline_retries_timeouts_and_rejects_malformed_provider_response(chunks):
 class RetryingProvider:
  calls=0
  def generate(self, prompt, timeout_seconds):
   self.calls+=1
   if self.calls==1: raise TimeoutError()
   return 'Recovered.'
 provider=RetryingProvider()
 assert RAGPipeline(chunks,provider=provider,retries=1).answer('reflection tokens')['answer']=='Recovered.'
 class MalformedProvider:
  def generate(self, prompt, timeout_seconds): return ' '
 with pytest.raises(ValueError,match='malformed model output'):
  RAGPipeline(chunks,provider=MalformedProvider()).answer('reflection tokens')
def test_semantic_retriever_normalizes_and_reuses_an_embedding_artifact(chunks,tmp_path):
 class Encoder:
  calls=0
  def encode(self,texts,**kwargs):
   self.calls+=1; return [[3,4] for _ in texts]
 encoder=Encoder(); artifact=tmp_path/'embeddings.json'
 first=SemanticRetriever(chunks[:2],encoder=encoder,artifact_path=artifact)
 second=SemanticRetriever(chunks[:2],encoder=encoder,artifact_path=artifact)
 assert encoder.calls==1 and first.embeddings==second.embeddings==[[.6,.8],[.6,.8]]
 assert len(first.search('reflection',1))==1
