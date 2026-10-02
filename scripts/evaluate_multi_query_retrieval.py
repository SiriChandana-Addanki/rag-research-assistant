"""Run the controlled explicit-query max-fusion experiment."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import load_evaluation_inputs
from src.multi_query_evaluation import experiment_trace
from src.retrieval import HybridRetriever, SemanticRetriever

def main():
 parser=argparse.ArgumentParser(); parser.add_argument('--model',default='sentence-transformers/all-MiniLM-L6-v2'); parser.add_argument('--batch-size',type=int,default=32); parser.add_argument('--artifact',default='evaluation/semantic_embeddings.json'); parser.add_argument('--queries',default='evaluation/multi_query_queries.json'); parser.add_argument('--results',default='evaluation/multi_query_retrieval_results.json'); parser.add_argument('--traces',default='evaluation/multi_query_retrieval_traces.json'); args=parser.parse_args()
 chunks,dataset,judgments=load_evaluation_inputs(); variants=json.loads(Path(args.queries).read_text())
 semantic=SemanticRetriever(chunks,args.model,args.batch_size,artifact_path=args.artifact)
 payload=experiment_trace(HybridRetriever(chunks,primary=semantic,semantic_weight=.75),dataset,judgments,variants)
 metadata={"model":args.model,"semantic_weight":.75,"bm25_weight":.25,"candidate_count":len(chunks),"final_top_k":10,"query_source":args.queries,"fusion_method":"max_combined_score"}
 Path(args.traces).write_text(json.dumps({**metadata,**payload},indent=2)+'\n')
 Path(args.results).write_text(json.dumps({**metadata,"metrics":payload["metrics"],"q009_evidence_coverage":payload["q009_evidence_coverage"]},indent=2)+'\n')
 print(json.dumps(payload["metrics"],indent=2))
if __name__=='__main__': main()
