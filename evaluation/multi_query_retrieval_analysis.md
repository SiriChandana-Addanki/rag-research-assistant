# MultiQuery Retrieval Experiment Analysis

## 1. Experimental setup

This report analyzes the completed local run represented by
`multi_query_retrieval_results.json` and `multi_query_retrieval_traces.json`.
It does not modify the production retrieval path, chunking, labels, questions,
model, or weights.

| Item | Configuration |
| --- | --- |
| Baseline | One original evaluation question, scored against all 127 chunks with the existing hybrid retriever: 0.75 semantic / 0.25 BM25. |
| Isolated variable | For q001, q009, q010, and q014 only, add the deterministic human-authored formulations in `multi_query_queries.json`; the remaining 11 questions use the original formulation only. |
| MultiQuery method | Score every original/variant independently using the unchanged per-query min-max normalization and hybrid score; fuse each chunk by the maximum combined score across formulations. |
| Deterministic fusion | Sort the fused scores descending, then `chunk_id` ascending on ties; retain the top 10 unique chunks. The trace records every per-query top-10 ranking and the winning formulation ID(s) for each fused entry. |
| Evaluation set | 15 questions (`q001`–`q015`) and the current, unchanged relevance judgments. Recall is a question-level hit metric: a question counts when at least one judged-relevant chunk is in the cutoff. MRR uses the first judged-relevant rank. |

### Artifact-integrity verification

The artifacts are populated run outputs, not the prior `not_run` placeholder:
the trace contains 15 ordered baseline traces, 15 ordered multi-query traces,
per-variant rankings, 127 candidates, and score/fusion metadata; the results
file has the same model, weights, top-10 cutoff, query source, fusion method,
metrics, and q009 coverage. `metrics_from_trace` independently accepts both
trace branches against the current dataset and judgments, and exactly
recomputes the metrics persisted in the results JSON. The baseline branch also
has the same question IDs, rankings, expected IDs, and relevance flags as the
persisted semantic-0.75/BM25-0.25 baseline; numeric score fields agree within
`1e-12` (the largest serialized floating-point difference is approximately
`3.6e-15`). Its additional fusion-schema keys and that insignificant
floating-point serialization difference mean the raw JSON objects are not
byte-for-byte identical. These checks
establish that this is a completed, internally consistent run using the stated
configuration; no inconsistency was found.

## 2. Aggregate results

Absolute change is MultiQuery minus baseline (percentage points for recall,
raw reciprocal-rank units for MRR).

| Metric | Baseline | MultiQuery | Absolute change |
| --- | ---: | ---: | ---: |
| Recall@1 | 0.5333333333 | 0.5333333333 | +0.0000000000 |
| Recall@3 | 0.7333333333 | 0.7333333333 | +0.0000000000 |
| Recall@5 | 0.8666666667 | 0.9333333333 | +0.0666666666 |
| Recall@10 | 0.9333333333 | 0.9333333333 | +0.0000000000 |
| MRR | 0.6633333333 | 0.6766666667 | +0.0133333334 |

## 3. Per-question impact

“Relevant-at-5” lists the judged-relevant chunk IDs present in the top five,
not merely the hit flag. “Unchanged” means the complete fused top-10 sequence
equals baseline; “outcome unchanged” means its relevant evidence did not
change even though non-relevant ordering did.

| Question | Baseline first relevant rank | MultiQuery first relevant rank | Baseline relevant-at-5 | MultiQuery relevant-at-5 | Change/category |
| --- | ---: | ---: | --- | --- | --- |
| q001 | 5 | 2 | c0005 | c0000, c0005 | Earlier rank; newly retrieves c0000 into @5. |
| q002 | 1 | 1 | c0000, c0002 | c0000, c0002 | Unchanged. |
| q003 | 1 | 1 | c0004, c0017 | c0004, c0017 | Unchanged. |
| q004 | 1 | 1 | c0004, c0017 | c0004, c0017 | Unchanged. |
| q005 | 1 | 1 | c0015 | c0015 | Unchanged. |
| q006 | 4 | 4 | c0017, c0016 | c0017, c0016 | Unchanged. |
| q007 | 1 | 1 | c0017 | c0017 | Unchanged. |
| q008 | 2 | 2 | c0017 | c0017 | Unchanged. |
| q009 | — | — | none | none | Outcome unchanged: no judged evidence at @10. |
| q010 | 6 | 5 | none | c0024 | Earlier rank; newly retrieves c0024 into @5. |
| q011 | 2 | 2 | c0012, c0025, c0026 | c0012, c0025, c0026 | Unchanged. |
| q012 | 1 | 1 | c0031, c0033 | c0031, c0033 | Unchanged. |
| q013 | 1 | 1 | c0041 | c0041 | Unchanged. |
| q014 | 3 | 5 | c0043, c0046 | c0043 | Regression: c0043 moves later; c0046 falls out of @5 (but remains rank 6). |
| q015 | 1 | 1 | c0026, c0025 | c0026, c0025 | Unchanged. |

There are three improvements in individual relevant ranks/chunk coverage:
q001 c0000 rank 7 -> 2, and q010 c0024 rank 6 -> 5. There is one worsening:
q014 c0043 rank 3 -> 5 and c0046 rank 4 -> 6. No relevant chunk disappears
from a top-10 result. q009’s fused top-10 changes, but it contains no judged
relevant chunk in either condition.

## 4. q009 Deep Dive

q009 asks how factual grounding and generation quality can be balanced at
inference without retraining. The four expected chunks are
`paper1_c0027`, `paper1_c0028`, `paper1_c0029`, and `paper1_c0030`.

| Expected chunk | What the judgment covers | Baseline fused rank | MultiQuery fused rank | Per-variant retrieval observation |
| --- | --- | ---: | ---: | --- |
| c0027 | Inference-time controllability and the retrieval threshold. | — | — | `inference_control` retrieves it at rank 5 in its own top-10, but max fusion scores it below the final fused cutoff. |
| c0028 | Threshold mechanics and tree decoding. | — | — | No q009 formulation retrieves it in its per-query top-10. |
| c0029 | Critique-score weights are adjustable at inference. | — | — | No q009 formulation retrieves it in its per-query top-10. |
| c0030 | Higher ISSUP weight/hard constraints, without additional training. | — | — | No q009 formulation retrieves it in its per-query top-10. |

| Cutoff | Baseline expected-evidence coverage | MultiQuery fused expected-evidence coverage |
| --- | ---: | ---: |
| @1 | 0/4 | 0/4 |
| @3 | 0/4 | 0/4 |
| @5 | 0/4 | 0/4 |
| @10 | 0/4 | 0/4 |

**Result:** MultiQuery did **not** address the q009 distributed-evidence
failure in the delivered fused ranking. It recovers c0027 only in the
unfused `inference_control` diagnostic list; it recovers none of the four in
the final top-10. Thus fused evidence coverage is unchanged at every reported
cutoff, complete recovery is false, and the final result is neither partial nor
complete recovery. The diagnostic per-variant hit is useful evidence that one
formulation can surface one facet, but it is not an evaluation success under
the configured max-score fusion and top-10 policy.

## 5. Why the metric changed

* **Recall@5 (+1/15 = +0.0666666666): q010 only.** The `critic_supervision`
  formulation makes judged-relevant c0024 rank 2 in its own list; max fusion
  puts c0024 at fused rank 5, compared with baseline rank 6. q010 therefore
  becomes an @5 hit. q001 was already an @5 hit, q014 remains an @5 hit, and
  q009 remains a miss, so none changes Recall@5.
* **MRR (+0.0133333334): q001, q010, and q014.** q001 contributes
  `(1/2 - 1/5) / 15 = +0.0200000000`; q010 contributes
  `(1/5 - 1/6) / 15 = +0.0022222222`; and q014 offsets them by
  `(1/5 - 1/3) / 15 = -0.0088888889`. All other first relevant ranks are
  unchanged. q009 contributes zero because it has no relevant top-10 entry in
  either condition.

The measured benefit is therefore a mixture of a new question-level @5 hit
(q010) and better first-relevant ranking (especially q001), not improved q009
distributed-evidence coverage.

## 6. Regressions

* **q014:** The baseline’s judged ablation evidence c0043/c0046 at ranks 3/4
  becomes ranks 5/6 after fusion. c0046 is consequently lost from @5, and the
  first relevant rank worsens from 3 to 5. Both still appear at @10, so the
  question-level recall metrics do not expose this evidence-coverage loss.
* **q009:** This is not a baseline-to-MultiQuery regression in relevance
  coverage (both are 0/4), but it is an important failed target: the
  `inference_control` list’s c0027 rank-5 diagnostic hit is displaced by
  higher max scores from non-relevant chunks in the final fused list.
* No other question changes its complete top-10 sequence, and no relevant
  chunk disappears from top 10.

## 7. Query-variant analysis

Only q001, q009, q010, and q014 have added formulations. The other eleven
questions retain the original query alone, so their deterministic fusion is
identical to baseline.

| Question and formulation | Retrieval intent | Concrete retrieved relationship to judgment and fused rank |
| --- | --- | --- |
| q001 original | Broad conventional-RAG problem. | Retrieves judged c0005 at 5 and c0000 at 7; its top score retains non-relevant c0004 at fused 1. |
| q001 `on_demand_retrieval` | Isolate adaptive retrieval necessity. | Retrieves only judged c0005 at variant rank 9; it does not create the improvement. |
| q001 `fixed_retrieval_limit` | Isolate the harm of always retrieving a fixed passage count. | Retrieves judged c0000 at 1 and c0005 at 4. Its c0000 score wins fusion and moves c0000 to fused 2 (c0000 is a judged conventional-RAG limitation), yielding q001’s MRR gain. |
| q009 original | Entire multi-part balance question. | Retrieves none of c0027–c0030; none is fused. |
| q009 `inference_control` | Target inference-time behavioral control. | Retrieves judged c0027 at 5, consistent with c0027’s controllability/threshold content, but c0027 is absent from fused @10. |
| q009 `grounding_quality_tradeoff` | Target factual-grounding versus quality/utility preference. | Retrieves no judged c0027–c0030; its winning fused c0002 at rank 1 is not q009-relevant. |
| q009 `no_retraining_control` | Target adjustment without model retraining. | Retrieves no judged c0027–c0030; its winning fused c0003 at rank 4 and c0040 at rank 10 are not q009-relevant. |
| q010 original | Full critic-token production and generator-data process. | Retrieves c0024 at 6, just outside @5. |
| q010 `critic_supervision` | Focus on creating reflection-signal supervision before generator training. | Retrieves judged c0024 at 2. Its score wins c0024’s fused entry, moving it to 5; c0024 describes critic/retrieval augmentation for generator data, matching the judgment. |
| q010 `generator_data` | Focus on augmentation of generator training data. | Also retrieves judged c0024 at 2, supporting the same relevant training-data facet, but the fused c0024 winner is `critic_supervision` (the higher score). |
| q014 original | Combined retrieval-relevance and ISSUP ablation result. | Retrieves judged c0043 at 3 and c0046 at 4. These original-query scores still win their fused entries, but intervening variant-promoted non-relevant chunks lower them to 5 and 6. |
| q014 `retrieval_relevance_ablation` | Isolate the relevant-passage/top-passage ablation. | Retrieves c0043 at 4, but not c0046; its high-scoring non-relevant c0004 and c0089 occupy fused ranks 3 and 4. |
| q014 `support_assessment_ablation` | Isolate ISSUP support-assessment effect. | Retrieves c0043 at 3, but not c0046; its non-relevant c0050 enters fused rank 7. It does not preserve the support-effect chunk c0046 at @5. |

This establishes concrete causes rather than a generic “diversity” claim:
the fixed-retrieval wording aligns with q001 c0000, and the training-process
wordings align with q010 c0024; q009’s control wording aligns only with c0027
before fusion; q014’s narrower wordings add non-relevant competitors and lower
two already retrieved ablation chunks.

## 8. Limitations

* This is only 15 questions. One q010 @5 boundary crossing changes Recall@5
  by 1/15, and a single q014 degradation materially offsets MRR. It is too
  small to generalize a performance claim.
* Variants are deterministic, hand-authored, and supplied for only four
  selected questions. They test these formulations, not LLM-generated query
  expansion, automatic decomposition, or robustness to generated-query noise.
* The test holds the `all-MiniLM-L6-v2` embedding model and the 0.75/0.25
  semantic/BM25 weighting fixed. It cannot attribute outcomes to MultiQuery
  independently of this scorer/fusion interaction, especially c0027’s
  per-variant-but-not-fused outcome.
* Results depend on the present relevance judgments. The evaluation correctly
  uses those labels, but it does not establish that every unjudged chunk is
  irrelevant or every judged chunk is equally necessary for an answer.
* Recall and MRR are first-hit metrics. They do not measure complete evidence
  coverage for a multi-part question. q009 proves the limitation directly:
  a diagnostic variant can retrieve one of four necessary chunks while the
  fused evaluation has 0/4; q014 loses one of two @5 evidence chunks with no
  change in question-level Recall@5. The explicit q009 coverage table is
  therefore descriptive evidence for this case, not a replacement metric.

## 9. Engineering conclusion

Do not integrate MultiQuery into the live retrieval pipeline. This controlled
run does **not** justify rejecting all multi-query approaches—q001 and q010
show formulation-specific gains—but it also does **not** solve the motivating
q009 failure and has a measurable q014 evidence regression. Keep it as an
**experimental candidate** and run another controlled experiment focused on
the fusion/candidate-selection failure: preserve the same corpus, labels,
questions, embedding model, and hybrid weights; compare max fusion with a
deterministic evidence-preserving fusion/candidate-allocation policy; and
evaluate per-question complete judged-evidence coverage alongside the existing
Recall/MRR. Do not infer universal superiority from this 15-question run.
