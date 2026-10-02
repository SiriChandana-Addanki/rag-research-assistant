# Retrieval evaluation baseline audit

## Scope and reproducibility

This audit uses the tracked 15-question dataset, chunk manifest, corrected
judgments, and the persisted top-10 rankings in
`semantic_retrieval_traces.json`. The trace was produced by
`evaluate_semantic_retrieval.py` using `sentence-transformers/all-MiniLM-L6-v2`
and its corpus-fingerprinted embedding artifact. In this environment the
`sentence_transformers` dependency is unavailable, so the neural evaluator
cannot be re-executed here. The rankings and numeric scores are therefore
preserved from the real prior run; this audit regenerated the trace's
judgment-derived fields (`expected_relevant_chunk_ids` and `is_relevant`) after
the q011 correction and recomputed every metric from those rankings. It does
**not** claim that retrieval changed.

`metrics_from_trace` now rejects a trace whose question order, expected IDs,
ranks, or relevance flags drift from the dataset/judgments. The evaluator
asserts that its live metrics equal metrics recomputed from its emitted trace.
Recall is question-level hit recall: a question counts when **any** judged chunk
appears within k. MRR is the reciprocal rank of the first such chunk. Thus these
metrics evaluate retrieval of at least one answer-bearing chunk; they do not
prove that all evidence necessary for a multi-part answer was retrieved.

## Judgment audit

All 15 existing judged chunk sets were checked against their manifest text. The
judged chunks provide direct evidence for their respective questions. The one
correction is q011: `paper1_c0012` states that SELF-RAG trains on critic-produced
reflection-token examples offline at lower cost than RLHF and contrasts
inference control with RLHF's preference-alignment goal. It directly answers
the question and is now relevant. `paper1_c0011` introduces PPO/RLHF but does
not itself state the offline-training contrast, so it remains unlabelled.

q009 correctly retains four separate relevant chunks (`c0027`--`c0030`): they
cover the factuality/utility tradeoff, threshold, weighted critique scores, and
no-retraining control. The current any-hit metric supports multiple relevant
chunks, but it cannot measure complete multi-chunk coverage. The chunks were
not collapsed.

## Before and after the q011 correction

| Configuration | Before R@1 / R@3 / R@5 / R@10 / MRR | After R@1 / R@3 / R@5 / R@10 / MRR |
| --- | --- | --- |
| semantic | .2667 / .6667 / .6667 / .9333 / .4817 | .3333 / .6667 / .6667 / .9333 / .5151 |
| semantic .25 / BM25 .75 | .4000 / .8000 / .8667 / .9333 / .6040 | .4000 / .8000 / .8667 / .9333 / .6040 |
| semantic .50 / BM25 .50 | .4667 / .8667 / .8667 / .9333 / .6417 | .4667 / .8667 / .8667 / .9333 / .6528 |
| semantic .75 / BM25 .25 | .5333 / .7333 / .8667 / .9333 / .6522 | .5333 / .7333 / .8667 / .9333 / .6633 |

The changed values are an evaluation correction only. No model, embeddings,
chunk text, ranking algorithm, or hybrid weight changed. The selected **current
measured baseline for this evaluation set** is semantic .75 / BM25 .25 because
it has the highest observed R@1 and MRR after the correction; 15 questions is
far too small to claim general superiority.

## Per-question behavior for the selected measured baseline

| question_id | question_type | first relevant rank | R@1 | R@3 | R@5 | R@10 | failure category | evidence | confidence |
| --- | --- | ---: | --- | --- | --- | --- | --- | --- | --- |
| q001 | conventional-RAG limitation | 5 | no | no | yes | yes | semantic mismatch | Direct fixed-retrieval evidence `c0005` is rank 5. | high |
| q002 | causal limitation | 1 | yes | yes | yes | yes | none observed | `c0000` is top-1. | high |
| q003 | strategy comparison | 1 | yes | yes | yes | yes | none observed | `c0004` is top-1. | high |
| q004 | mechanism | 1 | yes | yes | yes | yes | none observed | `c0004` is top-1. | high |
| q005 | definition/list | 1 | yes | yes | yes | yes | none observed | `c0015` is top-1. | high |
| q006 | inference signals | 4 | no | no | yes | yes | hybrid ranking | `c0017`/`c0016` arrive at ranks 4/5; lower semantic weighting retrieves one at rank 1. | high |
| q007 | procedural sequence | 1 | yes | yes | yes | yes | none observed | `c0017` is top-1. | high |
| q008 | passage assessment | 2 | no | yes | yes | yes | hybrid ranking | `c0017` is rank 2; .25 semantic makes it rank 1. | medium |
| q009 | inference-time tradeoff/control | — | no | no | no | no | multiple-chunk evidence; ranking miss | None of `c0027`--`c0030` occurs in the top 10 in any configuration. | high |
| q010 | critic/generator training pipeline | 6 | no | no | no | yes | ranking mismatch | First direct judged training chunk is `c0024` at rank 6. | high |
| q011 | PPO/RLHF comparison | 2 | no | yes | yes | yes | corrected incomplete ground truth | Newly judged `c0012` is rank 2 and directly gives the requested comparison. | high |
| q012 | datasets/metrics | 1 | yes | yes | yes | yes | none observed | `c0031` is top-1. | high |
| q013 | table lookup | 1 | yes | yes | yes | yes | none observed | `c0041` is top-1. | high |
| q014 | ablation interpretation | 3 | no | yes | yes | yes | hybrid ranking | Direct ablation evidence `c0043` is rank 3; .25 weighting reaches rank 2. | high |
| q015 | training-method comparison | 1 | yes | yes | yes | yes | none observed | `c0026` is top-1. | high |

## Strategy tradeoffs

* More lexical weight improves q001 (rank 5 to 3), q005 (1 versus 8 for
  semantic), q008 (1 versus 3), q012 (1 versus 7), and q014 (2 versus 3).
* More semantic weight improves q004 (1 versus 4 for .25), q007 (1 versus 2),
  q010 (6 versus 7--8), and q013 (1 versus 2).
* The .50 configuration has the strongest observed R@3 (.8667), while .75 has
  the strongest R@1 (.5333) and MRR (.6633). q006 illustrates the tradeoff:
  .50 ranks direct evidence first, while .75 delays it to rank 4.
* q009 is a common miss, so changing the existing blend alone did not address
  the most complete failure. Differences correspond to one or a few questions
  and should be treated as diagnostic, not statistically general.

## Section and chunk audit

Section assignment is metadata-only: every manifest chunk is indexed and
neither retriever reads `section`. `REFERENCES` begins at the literal heading;
the parser recognizes literal headings and numeric all-caps headings, but not
appendix headings such as `A SELF-RAG DETAILS`, so later appendix chunks inherit
that label. This is a stale metadata label, not a bad boundary: chunks remain
sentence-aware and do not cross the section boundary the parser recognized.

The trace contains no verified bibliography chunks `c0060`--`c0072` in any
top-10 ranking. It does contain 20--29 appendix chunks per configuration that
are labelled `REFERENCES`, including substantive material. Consequently there
is no evidence of bibliography contamination and a blanket `REFERENCES` filter
would be harmful. The smallest justified follow-up correction is appendix-heading
recognition in chunking metadata, with an expected retrieval-score effect of
zero unless a future section-aware retriever deliberately uses the metadata.
It is intentionally not implemented in this baseline phase because the current
retrievers do not consume the label and the available runner cannot regenerate
the source-PDF manifest.

## Cross-encoder experiment audit

The reported cross-encoder result (R@1/R@3/R@5/R@10/MRR .4000/.7333/.8000/.9333/.5822)
is a useful negative observation relative to the old .75/.25 hybrid
(.5333/.7333/.8667/.9333/.6522), but it is **not reproducible enough to be a
controlled conclusion**. No tracked command or trace records candidate count,
candidate IDs/pool, final-k handling, relevance-label version, reranker scores,
or deterministic tie policy. `CrossEncoderReranker` sorts only by score, and
although `RAGPipeline` can invoke an injected reranker, production defaults to
no reranker. It was not integrated into the live default path. Do not treat the
result as evidence that cross-encoders are generally harmful or integrate it.

## Next experiment: controlled multi-query retrieval for multi-aspect questions

* **Hypothesis:** combining deterministic subqueries for a multi-aspect question
  will retrieve at least one of q009's distributed inference-control chunks
  where a single query misses all four.
* **Targeted failure:** q009 is the only question missed through rank 10 by all
  four existing strategies, and its answer spans threshold, weighting, and
  quality-tradeoff evidence.
* **Expected benefit:** improve q009 R@10 and inspect whether all four evidence
  aspects gain coverage without perturbing the baseline's single-query ranks.
* **Downside:** generated/decomposed queries can dilute intent, add latency, and
  create an overfit rule on a 15-question set.
* **Success metrics:** unchanged baseline trace plus a separately named
  multi-query trace; q009 R@10 becomes a hit and aggregate R@10/MRR do not
  regress materially. Record per-subquery candidate IDs and deterministic fusion
  tie breaks.
* **Failure condition:** q009 remains a miss, or gains are offset by regressions
  on other questions/only occur with hand-authored q009-specific wording.

The experimental `MultiQueryRetriever` is now implemented as an explicit-query,
deterministic max-fusion wrapper, while the live baseline remains unchanged. It
requires a versioned, query-independent expansion source and a separate emitted
trace before any metric claim; no hand-authored q009 expansion or result is
reported here.
