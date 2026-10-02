# MultiQuery Fusion & Candidate Allocation Experiment

## 1. Hypothesis

Different deterministic query variants may discover useful candidates that max-score fusion or a shallow candidate pool discards; controlled allocation or fusion may improve complete judged-evidence coverage without unacceptable ranking regressions.

## 2. Fixed variables

Corpus, chunks/metadata, embedding model and parameters, semantic and BM25 implementations, 0.75/0.25 hybrid weights, questions, judgments, deterministic variants, and the audited baseline are fixed. Retrieval logic in `src/retrieval.py` is unchanged.

## 3. Experimental variables

A uses the existing max hybrid-score fusion at depth 10. B changes only per-query retained depth to 20 and 30. C changes only fusion to RRF (constant 60, predeclared) at depth 10. D uses a label-independent round-robin union capacity of 10 before max-score ranking.

## 4. Aggregate results

| Strategy | R@1 | R@3 | R@5 | R@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_control_max_depth_10 | 0.5333 (+0.0000) | 0.7333 (+0.0000) | 0.9333 (+0.0000) | 0.9333 (+0.0000) | 0.6767 (+0.0000) |
| B_max_depth_20 | 0.5333 (+0.0000) | 0.7333 (+0.0000) | 0.9333 (+0.0000) | 0.9333 (+0.0000) | 0.6767 (+0.0000) |
| B_max_depth_30 | 0.5333 (+0.0000) | 0.7333 (+0.0000) | 0.9333 (+0.0000) | 0.9333 (+0.0000) | 0.6767 (+0.0000) |
| C_rrf_k60_depth_10 | 0.6000 (+0.0667) | 0.8667 (+0.1333) | 0.9333 (+0.0000) | 0.9333 (+0.0000) | 0.7500 (+0.0733) |
| D_round_robin_union_depth_10 | 0.5333 (+0.0000) | 0.7333 (+0.0000) | 0.9333 (+0.0000) | 1.0000 (+0.0667) | 0.6833 (+0.0067) |

## 5. Evidence coverage

| Strategy | Avg @1 | Avg @3 | Avg @5 | Avg @10 | Complete @10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| A_control_max_depth_10 | 0.1667 | 0.2989 | 0.4533 | 0.5978 | 2/15 (0.1333) |
| B_max_depth_20 | 0.1667 | 0.2989 | 0.4533 | 0.5978 | 2/15 (0.1333) |
| B_max_depth_30 | 0.1667 | 0.2989 | 0.4533 | 0.5978 | 2/15 (0.1333) |
| C_rrf_k60_depth_10 | 0.1800 | 0.3344 | 0.4311 | 0.5978 | 2/15 (0.1333) |
| D_round_robin_union_depth_10 | 0.1667 | 0.2989 | 0.4533 | 0.6144 | 2/15 (0.1333) |

## 6. Per-question analysis

The trace artifact records every q001–q015 rank, contributing source, scores, and coverage. Meaningful changes are summarized below from those persisted rankings.

* **q001:** C_rrf_k60_depth_10: first relevant 2, @10 2/3; D_round_robin_union_depth_10: first relevant 2, @10 2/3
* **q002:** unchanged from control.
* **q003:** unchanged from control.
* **q004:** unchanged from control.
* **q005:** unchanged from control.
* **q006:** unchanged from control.
* **q007:** unchanged from control.
* **q008:** unchanged from control.
* **q009:** C_rrf_k60_depth_10: first relevant None, @10 0/4; D_round_robin_union_depth_10: first relevant 10, @10 1/4
* **q010:** C_rrf_k60_depth_10: first relevant 1, @10 1/5; D_round_robin_union_depth_10: first relevant 5, @10 1/5
* **q011:** unchanged from control.
* **q012:** unchanged from control.
* **q013:** unchanged from control.
* **q014:** C_rrf_k60_depth_10: first relevant 2, @10 2/3; D_round_robin_union_depth_10: first relevant 5, @10 2/3
* **q015:** unchanged from control.

## 7. q009 deep dive

* **A_control_max_depth_10:** paper1_c0027: not top-10; paper1_c0028: not top-10; paper1_c0029: not top-10; paper1_c0030: not top-10. Coverage @1/@3/@5/@10 = 0/4, 0/4, 0/4, 0/4; complete @10: False.
* **B_max_depth_20:** paper1_c0027: not top-10; paper1_c0028: not top-10; paper1_c0029: not top-10; paper1_c0030: not top-10. Coverage @1/@3/@5/@10 = 0/4, 0/4, 0/4, 0/4; complete @10: False.
* **B_max_depth_30:** paper1_c0027: not top-10; paper1_c0028: not top-10; paper1_c0029: not top-10; paper1_c0030: not top-10. Coverage @1/@3/@5/@10 = 0/4, 0/4, 0/4, 0/4; complete @10: False.
* **C_rrf_k60_depth_10:** paper1_c0027: not top-10; paper1_c0028: not top-10; paper1_c0029: not top-10; paper1_c0030: not top-10. Coverage @1/@3/@5/@10 = 0/4, 0/4, 0/4, 0/4; complete @10: False.
* **D_round_robin_union_depth_10:** paper1_c0027: rank 10, sources inference_control; paper1_c0028: not top-10; paper1_c0029: not top-10; paper1_c0030: not top-10. Coverage @1/@3/@5/@10 = 0/4, 0/4, 0/4, 1/4; complete @10: False.

## 8. q014 regression analysis

* **A_control_max_depth_10:** c0043 rank 5; c0046 rank 6.
* **B_max_depth_20:** c0043 rank 5; c0046 rank 6.
* **B_max_depth_30:** c0043 rank 5; c0046 rank 6.
* **C_rrf_k60_depth_10:** c0043 rank 2; c0046 rank 9.
* **D_round_robin_union_depth_10:** c0043 rank 5; c0046 rank 6.

## 9. q010 analysis

The q010 per-strategy ranks and coverage are stored in the trace; compare its c0024 rank to the control to determine whether the prior @5 boundary gain survives.

## 10. Fusion behavior

Max fusion retains the best hybrid score of a candidate. Larger-depth variants test upstream truncation. RRF rewards repeated high ranks rather than raw scores. The round-robin policy makes source contribution cyclic before max-score ranking; it uses no labels.

## 11. Regressions

Any metric or per-question movement below the control is a regression; the per-question section and traces preserve the exact evidence needed to inspect it.

## 12. Limitations

This is a 15-question set with deterministic human-authored variants, fixed embedding model and hybrid weights, and current relevance judgments. It is a candidate/fusion experiment only; it uses no LLM-generated expansion. Evidence coverage depends on the completeness of current gold judgments.

## 13. Engineering conclusion

Do not integrate any strategy. This small controlled result should keep MultiQuery/fusion experimental unless a strategy improves aggregate ranking and complete-evidence coverage without material per-question regressions. A follow-up should be justified only by the persisted measurements, not q009 alone.
