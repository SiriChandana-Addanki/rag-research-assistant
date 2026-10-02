"""Run isolated candidate-allocation and fusion experiments; never alters retrieval."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation import load_evaluation_inputs
from src.fusion_experiment import RRF_CONSTANT, build_experiment_trace, summarize_experiment
from src.retrieval import HybridRetriever, SemanticRetriever


EXPERIMENTS = (
    ("A_control_max_depth_10", "max_score", 10, None),
    ("B_max_depth_20", "max_score", 20, None),
    ("B_max_depth_30", "max_score", 30, None),
    ("C_rrf_k60_depth_10", "rrf", 10, None),
    ("D_round_robin_union_depth_10", "round_robin_union_max_score", 10, 10),
)


def analysis(payload):
    control = payload["experiments"]["A_control_max_depth_10"]["summary"]
    lines = ["# MultiQuery Fusion & Candidate Allocation Experiment", "", "## 1. Hypothesis", "",
             "Different deterministic query variants may discover useful candidates that max-score fusion or a shallow candidate pool discards; controlled allocation or fusion may improve complete judged-evidence coverage without unacceptable ranking regressions.", "",
             "## 2. Fixed variables", "", "Corpus, chunks/metadata, embedding model and parameters, semantic and BM25 implementations, 0.75/0.25 hybrid weights, questions, judgments, deterministic variants, and the audited baseline are fixed. Retrieval logic in `src/retrieval.py` is unchanged.", "",
             "## 3. Experimental variables", "", "A uses the existing max hybrid-score fusion at depth 10. B changes only per-query retained depth to 20 and 30. C changes only fusion to RRF (constant 60, predeclared) at depth 10. D uses a label-independent round-robin union capacity of 10 before max-score ranking.", "",
             "## 4. Aggregate results", "", "| Strategy | R@1 | R@3 | R@5 | R@10 | MRR |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, experiment in payload["experiments"].items():
        metric = experiment["summary"]["metrics"]
        lines.append(f"| {name} | {metric['recall@1']:.4f} ({metric['recall@1']-control['metrics']['recall@1']:+.4f}) | {metric['recall@3']:.4f} ({metric['recall@3']-control['metrics']['recall@3']:+.4f}) | {metric['recall@5']:.4f} ({metric['recall@5']-control['metrics']['recall@5']:+.4f}) | {metric['recall@10']:.4f} ({metric['recall@10']-control['metrics']['recall@10']:+.4f}) | {metric['mrr']:.4f} ({metric['mrr']-control['metrics']['mrr']:+.4f}) |")
    lines += ["", "## 5. Evidence coverage", "", "| Strategy | Avg @1 | Avg @3 | Avg @5 | Avg @10 | Complete @10 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, experiment in payload["experiments"].items():
        summary = experiment["summary"]
        coverage = summary["average_evidence_coverage"]
        complete = summary["complete_evidence_recovery_at_10"]
        lines.append(f"| {name} | {coverage['1']:.4f} | {coverage['3']:.4f} | {coverage['5']:.4f} | {coverage['10']:.4f} | {complete['count']}/15 ({complete['ratio']:.4f}) |")
    lines += ["", "## 6. Per-question analysis", "", "The trace artifact records every q001–q015 rank, contributing source, scores, and coverage. Meaningful changes are summarized below from those persisted rankings.", ""]
    control_rows = {row['question_id']: row for row in payload['experiments']['A_control_max_depth_10']['traces']}
    for qid in [f"q{i:03d}" for i in range(1, 16)]:
        pieces = []
        for name, experiment in payload['experiments'].items():
            row = next(row for row in experiment['traces'] if row['question_id'] == qid)
            if [x['chunk_id'] for x in row['ranked_chunks']] != [x['chunk_id'] for x in control_rows[qid]['ranked_chunks']]:
                pieces.append(f"{name}: first relevant {row['evidence_coverage']['first_relevant_rank']}, @10 {row['evidence_coverage']['retrieved_at']['10']}/{row['evidence_coverage']['expected_chunk_count']}")
        lines.append(f"* **{qid}:** " + ("; ".join(pieces) if pieces else "unchanged from control."))
    lines += ["", "## 7. q009 deep dive", ""]
    wanted = ["paper1_c0027", "paper1_c0028", "paper1_c0029", "paper1_c0030"]
    for name, experiment in payload['experiments'].items():
        row = next(row for row in experiment['traces'] if row['question_id'] == 'q009')
        found = {entry['chunk_id']: entry for entry in row['ranked_chunks']}
        detail = "; ".join(f"{chunk}: rank {found[chunk]['rank']}, sources {','.join(source['source_variant_id'] for source in found[chunk]['candidate_sources'])}" if chunk in found else f"{chunk}: not top-10" for chunk in wanted)
        cov = row['evidence_coverage']
        lines.append(f"* **{name}:** {detail}. Coverage @1/@3/@5/@10 = {cov['retrieved_at']['1']}/{cov['expected_chunk_count']}, {cov['retrieved_at']['3']}/{cov['expected_chunk_count']}, {cov['retrieved_at']['5']}/{cov['expected_chunk_count']}, {cov['retrieved_at']['10']}/{cov['expected_chunk_count']}; complete @10: {cov['all_expected_recovered_at_10']}.")
    lines += ["", "## 8. q014 regression analysis", ""]
    for name, experiment in payload['experiments'].items():
        row = next(row for row in experiment['traces'] if row['question_id'] == 'q014'); ranks = {x['chunk_id']: x['rank'] for x in row['ranked_chunks']}
        lines.append(f"* **{name}:** c0043 rank {ranks.get('paper1_c0043', 'not top-10')}; c0046 rank {ranks.get('paper1_c0046', 'not top-10')}.")
    lines += ["", "## 9. q010 analysis", "", "The q010 per-strategy ranks and coverage are stored in the trace; compare its c0024 rank to the control to determine whether the prior @5 boundary gain survives.", "", "## 10. Fusion behavior", "", "Max fusion retains the best hybrid score of a candidate. Larger-depth variants test upstream truncation. RRF rewards repeated high ranks rather than raw scores. The round-robin policy makes source contribution cyclic before max-score ranking; it uses no labels.", "", "## 11. Regressions", "", "Any metric or per-question movement below the control is a regression; the per-question section and traces preserve the exact evidence needed to inspect it.", "", "## 12. Limitations", "", "This is a 15-question set with deterministic human-authored variants, fixed embedding model and hybrid weights, and current relevance judgments. It is a candidate/fusion experiment only; it uses no LLM-generated expansion. Evidence coverage depends on the completeness of current gold judgments.", "", "## 13. Engineering conclusion", "", "Do not integrate any strategy. This small controlled result should keep MultiQuery/fusion experimental unless a strategy improves aggregate ranking and complete-evidence coverage without material per-question regressions. A follow-up should be justified only by the persisted measurements, not q009 alone.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='sentence-transformers/all-MiniLM-L6-v2'); parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--artifact', default='evaluation/semantic_embeddings.json'); parser.add_argument('--queries', default='evaluation/multi_query_queries.json')
    parser.add_argument('--results', default='evaluation/fusion_experiment_results.json'); parser.add_argument('--traces', default='evaluation/fusion_experiment_traces.json'); parser.add_argument('--analysis', default='evaluation/fusion_experiment_analysis.md')
    args = parser.parse_args(); chunks, dataset, judgments = load_evaluation_inputs(); variants = json.loads(Path(args.queries).read_text())
    retriever = HybridRetriever(chunks, primary=SemanticRetriever(chunks, args.model, args.batch_size, artifact_path=args.artifact), semantic_weight=.75)
    experiments = {}
    for name, strategy, depth, capacity in EXPERIMENTS:
        traces = build_experiment_trace(retriever, dataset, judgments, variants, name=name, fusion_strategy=strategy, candidate_depth=depth, allocation_capacity=capacity)
        experiments[name] = {'configuration': {'fusion_strategy': strategy, 'candidate_depth': depth, 'rrf_constant': RRF_CONSTANT if strategy == 'rrf' else None, 'allocation_capacity': capacity}, 'summary': summarize_experiment(traces, dataset, judgments), 'traces': traces}
    payload = {'experiment_type': 'isolated_multi_query_candidate_allocation_and_fusion', 'model': args.model, 'semantic_weight': .75, 'bm25_weight': .25, 'final_top_k': 10, 'query_source': args.queries, 'experiments': experiments}
    Path(args.traces).write_text(json.dumps(payload, indent=2) + '\n'); Path(args.results).write_text(json.dumps({**{key: value for key, value in payload.items() if key != 'experiments'}, 'experiments': {name: {'configuration': item['configuration'], 'summary': item['summary']} for name, item in experiments.items()}}, indent=2) + '\n'); Path(args.analysis).write_text(analysis(payload), encoding='utf-8')
    print(json.dumps({name: item['summary'] for name, item in experiments.items()}, indent=2))


if __name__ == '__main__': main()
