"""Ask the local paper corpus, with optional grounded Gemini generation."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.config import RAGConfig
from src.providers import GeminiProvider
from src.rag import RAGPipeline
from src.retrieval import HybridRetriever,SemanticRetriever,load_manifest

def main():
 parser=argparse.ArgumentParser(); parser.add_argument("question"); parser.add_argument("--no-generation",action="store_true",help="return traceable retrieved evidence without calling an LLM"); args=parser.parse_args(); config=RAGConfig.from_env(); chunks=load_manifest()
 semantic=SemanticRetriever(chunks,batch_size=32,artifact_path="evaluation/semantic_embeddings.json")
 retriever=HybridRetriever(chunks,primary=semantic,semantic_weight=config.semantic_weight)
 provider=None if args.no_generation else GeminiProvider(config.gemini_api_key,config.gemini_model)
 output=RAGPipeline(chunks,provider=provider,retriever=retriever,timeout_seconds=config.timeout_seconds,retries=config.max_retries,context_limit=config.context_limit).answer(args.question,config.top_k,config.candidate_k)
 print(json.dumps(output,indent=2))
if __name__=="__main__": main()
