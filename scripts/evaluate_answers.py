"""Run end-to-end answer generation over the 15-question retrieval dataset."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import RAGConfig
from src.providers import GeminiProvider
from src.rag import RAGPipeline
from src.retrieval import HybridRetriever,SemanticRetriever,load_manifest

def main():
 config=RAGConfig.from_env(); chunks=load_manifest(); semantic=SemanticRetriever(chunks,artifact_path="evaluation/semantic_embeddings.json")
 pipeline=RAGPipeline(chunks,provider=GeminiProvider(config.gemini_api_key,config.gemini_model),retriever=HybridRetriever(chunks,semantic,config.semantic_weight),timeout_seconds=config.timeout_seconds,retries=config.max_retries,context_limit=config.context_limit)
 dataset=json.loads(Path("evaluation/retrieval_dataset.json").read_text())
 results=[]
 for item in dataset:
  result=pipeline.answer(item["question"],config.top_k,config.candidate_k)
  results.append({"question_id":item["id"],"question":item["question"],**result})
 Path("evaluation/answer_evaluation_results.json").write_text(json.dumps(results,indent=2)+"\n")
 print(json.dumps({"questions":len(results),"output":"evaluation/answer_evaluation_results.json"}))
if __name__=="__main__": main()
