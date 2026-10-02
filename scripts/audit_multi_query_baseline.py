"""Validate the persisted 75/25 baseline used by the multi-query experiment."""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluation import load_evaluation_inputs, metrics_from_trace
from src.multi_query_evaluation import evidence_coverage

def main():
 chunks, dataset, judgments = load_evaluation_inputs()
 trace = json.loads(Path('evaluation/semantic_retrieval_traces.json').read_text())['configurations']['semantic_0.75_bm25_0.25']
 metrics = metrics_from_trace(trace, dataset, judgments)
 q009 = next(item for item in trace if item['question_id'] == 'q009')
 coverage = evidence_coverage(q009['ranked_chunks'], q009['expected_relevant_chunk_ids'])
 payload = {'baseline_metrics': metrics, 'q009_evidence_coverage': coverage}
 print(json.dumps(payload, indent=2))
if __name__ == '__main__': main()
