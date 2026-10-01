import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import json,statistics,time
from src.retrieval import HybridRetriever,load_manifest
if __name__ == '__main__':
 chunks=load_manifest(); queries=[x['question'] for x in json.load(open('evaluation/retrieval_dataset.json'))]; cold=time.perf_counter(); r=HybridRetriever(chunks); cold=(time.perf_counter()-cold)*1000
 samples=[]
 for _ in range(10):
  for q in queries:
   t=time.perf_counter(); r.search(q); samples.append((time.perf_counter()-t)*1000)
 pct=lambda p: sorted(samples)[min(len(samples)-1,round((len(samples)-1)*p))]
 data={'cold_index_build_ms':cold,'warm_retrieval_ms':{'samples':len(samples),'p50':pct(.5),'p95':pct(.95),'p99':pct(.99)}}
 open('evaluation/benchmark_results.json','w').write(json.dumps(data,indent=2)+'\n'); print(json.dumps(data,indent=2))
