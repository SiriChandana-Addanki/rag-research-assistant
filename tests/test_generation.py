import json,logging
import pytest
from src.answer_evaluation import deterministic_evaluation
from src.config import RAGConfig
from src.providers import GeminiProvider,ProviderResponse,RateLimitError,estimate_cost
from src.rag import RAGPipeline
from src.retrieval import Result

CHUNK={"document_id":"paper1","chunk_id":"c1","chunk_index":0,"page_start":2,"page_end":2,"section":"TEST","chunk_text":"Evidence."}
class Retriever:
 def search(self,q,k): return [Result(CHUNK,1.)]
def payload(citations=None): return json.dumps({"answer":"Evidence-based answer.","citations":citations if citations is not None else [{"document_id":"paper1","chunk_id":"c1","page":2}]})
def test_gemini_requires_key(monkeypatch):
 monkeypatch.delenv("GEMINI_API_KEY",raising=False)
 with pytest.raises(ValueError,match="GEMINI_API_KEY"): GeminiProvider()
def test_gemini_normalizes_mocked_sdk_response():
 class Models:
  def generate_content(self,**kwargs):
   assert kwargs["http_options"]["timeout"]==2500
   return type("R",(),{"text":payload(),"usage_metadata":type("U",(),{"prompt_token_count":3,"candidates_token_count":4,"total_token_count":7})()})()
 provider=GeminiProvider(api_key="not-a-secret",client=type("C",(),{"models":Models()})())
 response=provider.generate("prompt",2.5)
 assert response.token_usage=={"prompt_tokens":3,"completion_tokens":4,"total_tokens":7}
def test_pipeline_rejects_invalid_model_citation_and_secret_safe_log(caplog):
 class Provider:
  def generate(self,*_): return ProviderResponse(payload([{"document_id":"paper1","chunk_id":"bad","page":2}]))
 caplog.set_level(logging.INFO)
 with pytest.raises(ValueError,match="invalid citation"): RAGPipeline([],provider=Provider(),retriever=Retriever()).answer("q")
 assert "not-a-secret" not in caplog.text
def test_rate_limit_is_bounded_not_silently_retried_forever():
 class Provider:
  calls=0
  def generate(self,*_): self.calls+=1; raise RateLimitError("limited")
 provider=Provider()
 with pytest.raises(RateLimitError): RAGPipeline([],provider=provider,retriever=Retriever(),retries=1).answer("q")
 assert provider.calls==2
def test_deterministic_answer_evaluation():
 result={"answer":"yes","citations":[{"document_id":"paper1","chunk_id":"c1","page":2}],"context":"x"}
 assert deterministic_evaluation(result,[Result(CHUNK,1)],10)=={"answer_present":True,"context_non_empty":True,"context_within_limit":True,"citations_valid":True,"evaluation_status":"deterministic"}
def test_environment_config(monkeypatch):
 monkeypatch.setenv("RAG_TOP_K","2"); monkeypatch.setenv("RAG_SEMANTIC_WEIGHT","0.75")
 assert RAGConfig.from_env().top_k==2
def test_cost_requires_configured_prices(monkeypatch):
 monkeypatch.setenv("RAG_INPUT_COST_PER_MILLION","1"); monkeypatch.setenv("RAG_OUTPUT_COST_PER_MILLION","2")
 assert estimate_cost({"prompt_tokens":3,"completion_tokens":4})==11/1_000_000
