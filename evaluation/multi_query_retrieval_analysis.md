# Controlled multi-query retrieval experiment (not yet executed)

## Hypothesis and scope

This experiment tests one variable only: explicit, deterministic query diversity
followed by max fusion. It keeps the audited single-query semantic .75 / BM25
.25 baseline, corpus, labels, evaluator, and top-10 cutoff unchanged. q009 is
the motivating all-configuration miss; q001, q010, and q014 are additional
multi-aspect questions with legitimate reformulations. The method runs over all
15 questions: the four targeted questions receive configured variants and all
other questions retain only their original query, making their fused ranking
identical to their baseline ranking.

## Reproducible query formulations and leakage controls

`multi_query_queries.json` is a hand-authored, versioned configuration. Its
variants come only from each question's stated information need, not from chunk
IDs, relevance judgments, answer text, or target chunk sentences. No LLM is
used. q009 separates inference control, grounding-versus-quality tradeoffs, and
control without retraining; this probes the question's distinct facets rather
than wording a known answer. q001 separates on-demand retrieval from fixed
retrieval limits; q010 separates critic supervision from generator data; q014
separates retrieval relevance from support assessment.

## Scoring, fusion, and traces

For each original/variant query, the unchanged semantic retriever and BM25
retriever score all 127 chunks. Each score vector is independently min-max
normalized exactly as `HybridRetriever` does, then combined as
`.75 * semantic + .25 * BM25`. Each per-query candidate list sorts by descending
combined score then ascending `chunk_id`. For every chunk, max fusion takes the
maximum combined score over the original query and all variants. A duplicate
chunk is represented once in the fused list; all tied winning formulation IDs
are retained. The fused list uses the same descending-score/ascending-ID tie
break, returns top 10, and retains score, page, section, relevance, and
query-to-result metadata. Per-query lists and fused lists are persisted by the
experiment runner.

The script emits baseline and multi-query traces, query count, candidate count,
final top-k, query texts/IDs, per-query scores, winning variant IDs, and fused
scores. It also recomputes standard R@1/R@3/R@5/R@10/MRR from each trace. For
q009 it reports expected-chunk ranks, coverage at every cutoff, first relevant
rank, and whether all expected chunks appear in the top 10.

## Execution status, result, and decision

**Exact inconsistency found: stale experiment trace data.** The prior
`multi_query_retrieval_traces.json` was only a `not_run` placeholder: it did
not contain either a baseline trace or a multi-query trace. It was not evidence
of a relevance lookup, question-ID, expected-ID representation, or coverage
calculation mismatch. The corrected artifacts now embed the audited persisted
75/25 baseline trace and its q009 coverage: R@1/R@3/R@5/R@10/MRR are
.5333/.7333/.8667/.9333/.6633, and q009 retrieves 0 of 4 expected chunks at
all evaluated cutoffs. No retrieval behavior or relevance judgment changed.

The multi-query run is still **not executed in this environment**:
`sentence_transformers` is unavailable and package installation was denied by
the configured package mirror. Query embeddings cannot be inferred from the
persisted document embedding artifact, so multi-query rankings, metrics,
coverage, latency, false positives, or a recommendation would be fabricated.

Therefore the answer to whether multi-query recovered q009, whether coverage
improved, what happened to aggregate metrics, whether noise increased, and
whether integration is justified is **unknown**. Multi-query is **promising but
insufficiently validated**, not a pipeline change. The next action is to run
`scripts/evaluate_multi_query_retrieval.py` in an environment with the exact
baseline model available, inspect its persisted trace, and only then decide
whether a broader query-diversity experiment is warranted.
