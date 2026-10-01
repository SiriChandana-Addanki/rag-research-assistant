"""Offline deterministic answer checks and a pluggable model-judge interface."""
from typing import Protocol
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
