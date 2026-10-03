# q009 Full Candidate-Ranking Diagnostic

## Objective

Measure all 127 candidates for each frozen q009 query variant without changing retrieval, judgments, or fusion.

## Frozen Experimental Conditions

* Model: `sentence-transformers/all-MiniLM-L6-v2`; hybrid weights: semantic 0.75, BM25 0.25.
* Ranking uses the existing `src.retrieval.rank` descending-score/ascending-chunk-ID tie-breaker.

## Query Variants

* `original`: How can SELF-RAG change the balance between factual grounding and generation quality at inference time without retraining the model?
* `inference_control`: How does SELF-RAG control its behavior at inference time?
* `grounding_quality_tradeoff`: How can a system trade off factual grounding and generation quality during inference?
* `no_retraining_control`: How can retrieval and generation behavior be adjusted without retraining?

## Candidate-Space Validation

All four rankings contain exactly 127 unique indexed chunk IDs with contiguous ranks 1–127.

## Full Ranking Results

Complete 127-row rankings for all four variants (508 rows) are persisted in the JSON artifact.

## Expected Evidence Reachability

| Chunk | Variant | Rank | Semantic | BM25 | Hybrid | Class |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `paper1_c0027` | `original` | 26 | 0.425826 | 9.064440 | 0.666136 | B |
| `paper1_c0027` | `inference_control` | 5 | 0.424012 | 15.284454 | 0.763294 | A |
| `paper1_c0027` | `grounding_quality_tradeoff` | 17 | 0.372250 | 8.448029 | 0.611846 | B |
| `paper1_c0027` | `no_retraining_control` | 18 | 0.415597 | 5.995860 | 0.696083 | B |
| `paper1_c0028` | `original` | 91 | 0.209535 | 3.944381 | 0.337918 | C |
| `paper1_c0028` | `inference_control` | 83 | 0.153139 | 2.537965 | 0.241465 | C |
| `paper1_c0028` | `grounding_quality_tradeoff` | 76 | 0.285036 | 1.759275 | 0.382222 | C |
| `paper1_c0028` | `no_retraining_control` | 44 | 0.341672 | 3.676185 | 0.539377 | C |
| `paper1_c0029` | `original` | 44 | 0.324386 | 9.576857 | 0.554164 | C |
| `paper1_c0029` | `inference_control` | 35 | 0.277464 | 8.135212 | 0.476833 | C |
| `paper1_c0029` | `grounding_quality_tradeoff` | 43 | 0.338231 | 4.952120 | 0.506323 | C |
| `paper1_c0029` | `no_retraining_control` | 47 | 0.257478 | 8.057162 | 0.532083 | C |
| `paper1_c0030` | `original` | 15 | 0.481218 | 12.416234 | 0.779579 | B |
| `paper1_c0030` | `inference_control` | 17 | 0.471040 | 2.238528 | 0.604310 | B |
| `paper1_c0030` | `grounding_quality_tradeoff` | 14 | 0.357116 | 11.600913 | 0.649603 | B |
| `paper1_c0030` | `no_retraining_control` | 48 | 0.370183 | 1.699484 | 0.529827 | C |

## Individual Retrieval vs Fusion Comparison

| Chunk | Best variant | Best rank | Max 10 | Max 20 | Max 30 | RRF 10 | Round-robin 10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `paper1_c0027` | `inference_control` | 5 | N/A | N/A | N/A | N/A | 10 |
| `paper1_c0028` | `no_retraining_control` | 44 | N/A | N/A | N/A | N/A | N/A |
| `paper1_c0029` | `inference_control` | 35 | N/A | N/A | N/A | N/A | N/A |
| `paper1_c0030` | `grounding_quality_tradeoff` | 14 | N/A | N/A | N/A | N/A | N/A |

## Evidence Content Assessment

* **c0027:** contrasts more frequent retrieval for factual accuracy with less retrieval for open-ended creativity/utility; it supports the high-level trade-off and uses related but not identical wording.
* **c0028:** specifies the `Retrieve=Yes` threshold and critic-weighted decoding; it supports the inference-control mechanism and overlaps lexically with retrieval/control terminology.
* **c0029:** makes critique weights inference-time hyperparameters and gives higher `ISSUP` weight as the evidence-grounding control; its reflection-token terminology differs materially from q009.
* **c0030:** gives hard critique-token filtering and says SELF-RAG tailors behavior with no additional training; it directly supports the no-retraining qualifier but uses decoding/critique terminology.

The measured semantic/BM25 ranks below, rather than wording alone, identify whether each chunk is helped by semantic similarity, lexical overlap, or both.

## Retrieval Characteristics

| Chunk | Best semantic rank | Best BM25 rank | Best hybrid rank | Best hybrid variant |
| --- | ---: | ---: | ---: | --- |
| `paper1_c0027` | 15 | 1 | 5 | `inference_control` |
| `paper1_c0028` | 53 | 22 | 44 | `no_retraining_control` |
| `paper1_c0029` | 49 | 2 | 35 | `inference_control` |
| `paper1_c0030` | 8 | 4 | 14 | `grounding_quality_tradeoff` |

## Root-Cause Interpretation

Case 2: paper1_c0027, paper1_c0030 are individually competitive while the remaining evidence is weak; the q009 failure is mixed.

## Limitations

This is a fixed-corpus, fixed-model measurement, not a production retrieval change. The conclusion is limited to these four frozen formulations and 127 chunks.

## Next Experiment Recommendation

**One experiment — terminology-aligned q009 formulation ablation.** **Hypothesis:** the expected chunks that remain weak are missed because q009 uses different surface terminology from `Retrieve`, `ISSUP`, and `Critique`. **Independent variable:** one predeclared terminology-aligned query formulation versus the frozen original wording. **Frozen variables:** corpus, chunks, model, BM25, 0.75/0.25 weighting, judgments, and final-k. **Metrics:** per-chunk semantic/BM25/hybrid rank and recall@10/@30 for c0027–c0030. **Failure signal:** no meaningful rank improvement for the weak chunks. **Interpretation:** improvement supports query-representation difficulty; no improvement redirects investigation away from wording toward representation/chunk retrieval.
