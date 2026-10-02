from src.multi_query_evaluation import experiment_trace
from src.retrieval import HybridRetriever

class FakePrimary:
    def __init__(self, chunks): self.chunks=chunks
    def scores(self, query): return [chunk['semantic_scores'][query] for chunk in self.chunks]

def test_multi_query_trace_preserves_scores_association_ties_and_coverage():
    chunks=[
      {'chunk_id':'a','chunk_text':'alpha','page_start':1,'page_end':1,'section':'A','semantic_scores':{'question':.2,'aspect':.9}},
      {'chunk_id':'b','chunk_text':'beta','page_start':2,'page_end':2,'section':'B','semantic_scores':{'question':.8,'aspect':.9}},
      {'chunk_id':'c','chunk_text':'gamma','page_start':3,'page_end':3,'section':'C','semantic_scores':{'question':.1,'aspect':.1}},
    ]
    dataset=[{'id':'q009','question':'question'}]
    judgments=[{'question_id':'q009','relevant_chunk_ids':['a','b']}]
    # Empty lexical terms ensure the test controls only deterministic semantic fusion.
    payload=experiment_trace(HybridRetriever(chunks,primary=FakePrimary(chunks),semantic_weight=.75),dataset,judgments,{'q009':[{'id':'aspect','query':'aspect'}]})
    multi=payload['multi_query'][0]
    assert multi['query_count']==2 and multi['fusion_method']=='max_combined_score'
    assert [item['chunk_id'] for item in multi['ranked_chunks'][:2]]==['a','b']
    assert multi['ranked_chunks'][0]['winning_query_variant_ids']==['aspect']
    assert multi['query_variants'][1]['ranked_chunks'][0]['semantic_score']==.9
    assert payload['q009_evidence_coverage']['q009']['retrieved_at']['1']==1
    assert payload['q009_evidence_coverage']['q009']['retrieved_at']['3']==2
    assert payload['metrics']['multi_query']['recall@1']==1

def test_experiment_runner_persists_trace_and_results(tmp_path, monkeypatch):
    import importlib.util, json, sys
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[1] / 'scripts' / 'evaluate_multi_query_retrieval.py'
    spec = importlib.util.spec_from_file_location('evaluate_multi_query_retrieval', script_path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    chunks=[{'chunk_id':'a','chunk_text':'alpha','page_start':1,'page_end':1,'section':'A','semantic_scores':{'question':.2,'aspect':.9}}, {'chunk_id':'b','chunk_text':'beta','page_start':2,'page_end':2,'section':'B','semantic_scores':{'question':.8,'aspect':.1}}]
    class FakeSemantic:
        def __init__(self, chunks, *args, **kwargs): self.chunks=chunks
        def scores(self, query): return [chunk['semantic_scores'][query] for chunk in self.chunks]
    monkeypatch.setattr(module, 'SemanticRetriever', FakeSemantic)
    monkeypatch.setattr(module, 'load_evaluation_inputs', lambda: (chunks,[{'id':'q009','question':'question'}],[{'question_id':'q009','relevant_chunk_ids':['a']}]))
    queries=tmp_path/'queries.json'; queries.write_text(json.dumps({'q009':[{'id':'aspect','query':'aspect'}]}))
    results, traces=tmp_path/'results.json', tmp_path/'traces.json'
    monkeypatch.setattr(sys, 'argv', [str(script_path),'--queries',str(queries),'--results',str(results),'--traces',str(traces)])
    module.main()
    assert json.loads(results.read_text())['metrics']['multi_query']['recall@1'] == 1
    trace=json.loads(traces.read_text())
    assert trace['multi_query'][0]['query_variants'][1]['id'] == 'aspect'
    assert trace['multi_query'][0]['ranked_chunks'][0]['fused_score'] is not None
