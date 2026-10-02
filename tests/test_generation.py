import json,logging
import pytest
from src.answer_evaluation import deterministic_evaluation,run_answer_evaluation
from src.config import RAGConfig
from src.providers import GeminiProvider,ProviderError,ProviderResponse,RateLimitError,estimate_cost
from src.rag import RAGPipeline
from src.retrieval import Result

CHUNK={"document_id":"paper1","chunk_id":"c1","chunk_index":0,"page_start":2,"page_end":2,"section":"TEST","chunk_text":"Evidence."}
class Retriever:
 def search(self,q,k): return [Result(CHUNK,1.)]
def payload(citations=None): return json.dumps({"answer":"Evidence-based answer.","citations":citations if citations is not None else [{"document_id":"paper1","chunk_id":"c1","page":2}]})
def test_gemini_requires_key(monkeypatch):
 monkeypatch.delenv("GEMINI_API_KEY",raising=False)
 with pytest.raises(ValueError,match="GEMINI_API_KEY"): GeminiProvider()
def test_gemini_configures_timeout_on_client_not_generate_content():
 class Models:
  def generate_content(self,model,contents,config):
   assert model=="gemini-2.5-flash" and contents=="prompt"
   assert config=={"response_mime_type":"application/json"}
   return type("R",(),{"text":payload(),"usage_metadata":type("U",(),{"prompt_token_count":3,"candidates_token_count":4,"total_token_count":7})()})()
 client=type("C",(),{"models":Models()})()
 configured=[]
 provider=GeminiProvider(api_key="not-a-secret",client_factory=lambda timeout: configured.append(timeout) or client)
 response=provider.generate("prompt",2.5)
 assert configured==[2.5]
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
def test_batch_evaluation_preserves_successes_after_a_transient_failure():
 class Pipeline:
  retries=0; calls=[]
  def answer(self,question,*_):
   self.calls.append(question)
   if question=="first":
    error=RateLimitError("temporary"); error.status_code=503; error.request_id="request-1"; error.latency_ms=12.5; raise error
   return {"answer":"safe answer","citations":[],"citation_validation":True,"retrieved_chunk_ids":["c1"],"token_usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5},"cost":"unavailable","latency":{"total_ms":20.0},"evaluation_status":"generated"}
 pipeline=Pipeline(); results,summary=run_answer_evaluation(pipeline,[{"id":"q1","question":"first"},{"id":"q2","question":"second"}],1,1,"gemini-test")
 assert pipeline.calls==["first","second"]
 assert results[0]["evaluation_status"]=="transient_failure" and results[0]["failure"]["http_status"]==503
 assert results[0]["request_id"]=="request-1" and "temporary" not in json.dumps(results[0])
 assert results[1]["answer"]=="safe answer"
 assert summary["successful_generations"]==1 and summary["transient_failures"]==1 and summary["token_usage"]=={"prompt_tokens":2,"completion_tokens":3,"total_tokens":5}
def test_gemini_429_is_conservative_but_preserves_safe_sdk_diagnostics():
 class Response: headers={"retry-after":"30","authorization":"secret"}
 class ClientError(Exception):
  code=429; response=Response(); body={"error":{"status":"RESOURCE_EXHAUSTED","message":"secret detail","details":[{"@type":"type.googleapis.com/google.rpc.QuotaFailure"}]}}
 class Models:
  def generate_content(self,**_): raise ClientError()
 provider=GeminiProvider(api_key="not-a-secret",client=type("C",(),{"models":Models()})())
 with pytest.raises(ProviderError) as raised: provider.generate("prompt",1)
 assert not isinstance(raised.value,RateLimitError)
 assert raised.value.provider_diagnostics=={"http_status":429,"provider_status":"RESOURCE_EXHAUSTED","detail_types":["type.googleapis.com/google.rpc.QuotaFailure"],"retry_after":"30"}
 assert "secret" not in json.dumps(raised.value.provider_diagnostics)
