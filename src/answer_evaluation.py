"""Offline deterministic answer checks and resilient answer-run evaluation."""
import statistics
from typing import Protocol
from src.providers import ProviderError, TransientProviderError
from src.rag import Citation, validate_citations

class AnswerJudge(Protocol):
 def evaluate(self, question, answer, context): ...

def deterministic_evaluation(result, retrieved, context_limit):
 citations=[Citation(**citation) for citation in result.get("citations",[])]
 valid=False
 try:
  validate_citations(citations,retrieved); valid=True
 except ValueError: pass
 context=result.get("context","")
 return {"answer_present":bool(result.get("answer"," ").strip()),"context_non_empty":bool(context),"context_within_limit":len(context)<=context_limit,"citations_valid":valid,"evaluation_status":"deterministic"}

def failure_metadata(error, provider_model, retry_count):
 status_code=getattr(error,"status_code",getattr(error,"code",None))
 status_code=status_code if isinstance(status_code,int) else None
 category="transient_failure" if isinstance(error,TransientProviderError) else "permanent_failure" if isinstance(error,ProviderError) else "evaluation_error"
 return {"evaluation_status":category,"failure":{"error_category":category,"error_type":type(error).__name__,"http_status":status_code,"provider_diagnostics":getattr(error,"provider_diagnostics",None),"provider_model":provider_model,"application_retry_count":retry_count,"latency_ms":getattr(error,"latency_ms",None)},"request_id":getattr(error,"request_id",None)}

def summarize_answer_evaluation(results):
 counts={status:sum(result["evaluation_status"]==status for result in results) for status in ("generated","transient_failure","permanent_failure","evaluation_error")}
 successful=[result["latency"]["total_ms"] for result in results if result["evaluation_status"]=="generated" and "total_ms" in result.get("latency",{})]
 usage=[result["token_usage"] for result in results if isinstance(result.get("token_usage"),dict)]
 tokens={key:sum(item.get(key,0) for item in usage) for key in ("prompt_tokens","completion_tokens","total_tokens")} if usage else "unavailable"
 latency={"mean_ms":statistics.mean(successful)} if successful else "unavailable"
 if len(successful)>=2:
  ordered=sorted(successful); latency.update({"p50_ms":statistics.median(ordered),"p95_ms":ordered[round((len(ordered)-1)*.95)]})
 return {"total_questions":len(results),"successful_generations":counts["generated"],"transient_failures":counts["transient_failure"],"permanent_failures":counts["permanent_failure"],"evaluation_failures":counts["evaluation_error"],"success_rate":counts["generated"]/len(results) if results else 0,"successful_generation_latency":latency,"token_usage":tokens}

def run_answer_evaluation(pipeline, dataset, top_k, candidate_k, provider_model=None):
 """Continue after one provider failure and preserve every question's result."""
 results=[]
 for item in dataset:
  try:
   result=pipeline.answer(item["question"],top_k,candidate_k)
   status=result.get("evaluation_status")
   if status!="generated": result["evaluation_status"]="evaluation_error"
  except Exception as error:
   result=failure_metadata(error,provider_model,getattr(pipeline,"retries",0))
   result.update({"answer":None,"citations":[],"citation_validation":False,"retrieved_chunk_ids":[],"token_usage":"unavailable","cost":"unavailable","latency":{"total_ms":result["failure"]["latency_ms"]}})
  results.append({"question_id":item["id"],"question":item["question"],**result})
 return results,summarize_answer_evaluation(results)
