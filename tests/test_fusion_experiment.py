from src.fusion_experiment import max_score_fusion, reciprocal_rank_fusion, round_robin_union


def candidate(chunk_id, score, rank, source='a'):
    return {'chunk_id': chunk_id, 'hybrid_score': score, 'source_rank': rank, 'source_variant_id': source}


def test_fusion_formulas_and_ties_are_deterministic():
    lists = [[candidate('b', .5, 1), candidate('a', .4, 2)], [candidate('a', .8, 1, 'b')]]
    assert max_score_fusion(lists) == {'b': .5, 'a': .8}
    assert reciprocal_rank_fusion(lists, 60)['a'] == 1 / 62 + 1 / 61
    assert sorted(max_score_fusion([[candidate('b', .5, 1)], [candidate('a', .5, 1)]]), key=lambda x: (-.5, x)) == ['a', 'b']


def test_round_robin_union_preserves_sources_without_labels():
    lists = [[candidate('a', .9, 1), candidate('b', .8, 2)], [candidate('a', .7, 1, 'b'), candidate('c', .6, 2, 'b')]]
    assert [item['chunk_id'] for item in round_robin_union(lists, 3)] == ['a', 'c', 'b']


def test_trace_metrics_and_coverage_are_consistent_and_labels_do_not_change_retrieval():
    from src.fusion_experiment import build_experiment_trace, summarize_experiment
    from src.retrieval import HybridRetriever

    chunks = [
        {'chunk_id': 'a', 'chunk_text': 'a', 'page_start': 1, 'page_end': 1, 'section': 'A', 'scores': {'q': .9, 'v': .1}},
        {'chunk_id': 'b', 'chunk_text': 'b', 'page_start': 2, 'page_end': 2, 'section': 'B', 'scores': {'q': .2, 'v': .9}},
    ]
    class Primary:
        def __init__(self, rows): self.chunks = rows
        def scores(self, query): return [row['scores'][query] for row in self.chunks]
    retriever = HybridRetriever(chunks, primary=Primary(chunks), semantic_weight=.75)
    dataset = [{'id': 'q1', 'question': 'q'}]
    variants = {'q1': [{'id': 'variant', 'query': 'v'}]}
    first = build_experiment_trace(retriever, dataset, [{'question_id': 'q1', 'relevant_chunk_ids': ['a']}], variants,
                                   name='test', fusion_strategy='max_score', candidate_depth=2)
    second = build_experiment_trace(retriever, dataset, [{'question_id': 'q1', 'relevant_chunk_ids': ['b']}], variants,
                                    name='test', fusion_strategy='max_score', candidate_depth=2)
    assert [item['chunk_id'] for item in first[0]['ranked_chunks']] == [item['chunk_id'] for item in second[0]['ranked_chunks']]
    assert first[0]['evidence_coverage']['retrieved_at']['1'] == 1
    assert summarize_experiment(first, dataset, [{'question_id': 'q1', 'relevant_chunk_ids': ['a']}])['metrics']['recall@1'] == 1
