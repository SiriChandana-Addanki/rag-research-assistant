# q009 Candidate Reachability Analysis

## 1. Question and expected evidence

> How can SELF-RAG change the balance between factual grounding and generation quality at inference time without retraining the model?

The current judgment expects `paper1_c0027` through `paper1_c0030` (four consecutive chunks in §3.3, page 6). This diagnostic does **not** change that judgment or the retrieval implementation. The persisted per-variant trace has only ten rows per query, although the corpus has 127 chunks. Consequently, it can establish exact reachability for c0027, but it cannot honestly provide exact 11–127 ranks or semantic/hybrid scores for c0028–c0030 without recreating the unavailable query embeddings.

## 2. Query variants

| ID | Query |
| --- | --- |
| `original` | How can SELF-RAG change the balance between factual grounding and generation quality at inference time without retraining the model? |
| `inference_control` | How does SELF-RAG control its behavior at inference time? |
| `grounding_quality_tradeoff` | How can a system trade off factual grounding and generation quality during inference? |
| `no_retraining_control` | How can retrieval and generation behavior be adjusted without retraining? |

## 3. Full candidate reachability

**Important artifact limitation.** The requested full 127-candidate lists were not persisted in any supplied trace: `multi_query_retrieval_traces.json` records the final ten candidates for each variant, and `fusion_experiment_traces.json` records the final ten fused candidates. The document-embedding cache has no query vectors. The table therefore reports exact values where they are persisted and `not persisted` rather than inventing values. BM25 was recomputed from the fixed 127 indexed chunks; semantic and hybrid values require the missing query encoding.

| Chunk | Variant | Rank | Hybrid score | Top10 | Top20 | Top30 | Classification |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| `paper1_c0027` | `original` | not persisted (not top-10) | not persisted | no | unknown | unknown | A |
| `paper1_c0028` | `original` | not persisted (not top-10) | not persisted | no | unknown | unknown | indeterminate (B/C/D cannot be assigned from supplied top-10-only traces) |
| `paper1_c0029` | `original` | not persisted (not top-10) | not persisted | no | unknown | unknown | indeterminate (B/C/D cannot be assigned from supplied top-10-only traces) |
| `paper1_c0030` | `original` | not persisted (not top-10) | not persisted | no | unknown | unknown | indeterminate (B/C/D cannot be assigned from supplied top-10-only traces) |
| `paper1_c0027` | `inference_control` | 5 | 0.763294 | yes | yes | yes | A |
| `paper1_c0028` | `inference_control` | not persisted (not top-10) | not persisted | no | unknown | unknown | indeterminate (B/C/D cannot be assigned from supplied top-10-only traces) |
| `paper1_c0029` | `inference_control` | not persisted (not top-10) | not persisted | no | unknown | unknown | indeterminate (B/C/D cannot be assigned from supplied top-10-only traces) |
| `paper1_c0030` | `inference_control` | not persisted (not top-10) | not persisted | no | unknown | unknown | indeterminate (B/C/D cannot be assigned from supplied top-10-only traces) |
| `paper1_c0027` | `grounding_quality_tradeoff` | not persisted (not top-10) | not persisted | no | unknown | unknown | A |
| `paper1_c0028` | `grounding_quality_tradeoff` | not persisted (not top-10) | not persisted | no | unknown | unknown | indeterminate (B/C/D cannot be assigned from supplied top-10-only traces) |
| `paper1_c0029` | `grounding_quality_tradeoff` | not persisted (not top-10) | not persisted | no | unknown | unknown | indeterminate (B/C/D cannot be assigned from supplied top-10-only traces) |
| `paper1_c0030` | `grounding_quality_tradeoff` | not persisted (not top-10) | not persisted | no | unknown | unknown | indeterminate (B/C/D cannot be assigned from supplied top-10-only traces) |
| `paper1_c0027` | `no_retraining_control` | not persisted (not top-10) | not persisted | no | unknown | unknown | A |
| `paper1_c0028` | `no_retraining_control` | not persisted (not top-10) | not persisted | no | unknown | unknown | indeterminate (B/C/D cannot be assigned from supplied top-10-only traces) |
| `paper1_c0029` | `no_retraining_control` | not persisted (not top-10) | not persisted | no | unknown | unknown | indeterminate (B/C/D cannot be assigned from supplied top-10-only traces) |
| `paper1_c0030` | `no_retraining_control` | not persisted (not top-10) | not persisted | no | unknown | unknown | indeterminate (B/C/D cannot be assigned from supplied top-10-only traces) |

Exact persisted semantic/BM25/hybrid values for the one observed expected hit are: c0027 under `inference_control` = 0.424012 / 15.284454 / 0.763294. The JSON includes the recomputed BM25 value for every row, plus explicit `null` values for unrecoverable semantic/hybrid/rank fields.

## 4. Best query variant per evidence chunk

* **c0027 — `inference_control`, rank 5, class A.** This is directly observed in the individual-query trace.
* **c0028, c0029, c0030 — indeterminate from the supplied artifacts.** Each is absent from each persisted variant top-10, but that fact alone cannot distinguish rank 11–20 (B), 21–30 (B), 31–127 (C), or a noncompetitive zero/near-zero result (D). No best variant can be assigned without full candidate traces.

## 5. Where evidence is lost

### Demonstrated fusion/final-top-k loss: c0027

`inference_control` retrieves c0027 at individual rank 5. It does not occur in the final top-10 for max-score at depths 10, 20, or 30, or for RRF at depth 10. It does occur at final rank 10 for round-robin. Thus the evidence conclusively rules out an underlying single-variant retrieval failure for c0027. Its disappearance in the max-score and RRF outputs is a fusion/final-top-k allocation outcome; round-robin’s inclusion demonstrates that a source-balanced candidate allocation can preserve it.

### Not established: c0028–c0030

The artifacts show only that these chunks do not reach any *final* fused top-10. They do not show whether individual variants ranked them 11–30, 31–127, or noncompetitively. It would be incorrect to call this a retrieval failure, candidate-depth failure, fusion failure, or final-top-k failure before measuring their full per-variant ranks.

| Fused output | c0027 | c0028 | c0029 | c0030 |
| --- | --- | --- | --- | --- |
| max-score depth 10 | not final top-10 | not final top-10 | not final top-10 | not final top-10 |
| max-score depth 20 | not final top-10 | not final top-10 | not final top-10 | not final top-10 |
| max-score depth 30 | not final top-10 | not final top-10 | not final top-10 | not final top-10 |
| RRF depth 10 | not final top-10 | not final top-10 | not final top-10 | not final top-10 |
| round-robin depth 10 | rank 10 | not final top-10 | not final top-10 | not final top-10 |

## 6. q009 evidence-content analysis

* **c0027 — retrieval-frequency control for factuality versus openness.** It says SELF-RAG can retrieve more frequently for factual-accuracy tasks to align output with evidence, or retrieve less for open-ended tasks to emphasize creativity/utility; it introduces a configurable retrieval threshold.
* **c0028 — threshold mechanics and decoding setup.** It specifies the normalized `Retrieve=Yes` probability threshold that triggers retrieval, then describes retrieved-passage continuations and a critic-weighted segment score.
* **c0029 — inference-time critique weighting.** It states that critique-token weights are inference-time hyperparameters and gives the concrete grounding mechanism: increase `ISSUP` weight to favor evidence-supported output over other aspects.
* **c0030 — hard constraints and no-additional-training claim.** It gives the alternative of filtering undesirable critique tokens during decoding and contrasts SELF-RAG’s no-additional-training tailoring with training-based preference balancing.

The four chunks are **complementary**, not merely duplicates: c0027 supplies the high-level retrieval-frequency trade-off; c0028 supplies the threshold operation; c0029 supplies soft critique-weight control; and c0030 supplies hard decoding control and the answer to “without retraining.” The question can receive a short, partially supported answer from c0027 alone, but a complete answer to its mechanisms and no-retraining qualifier genuinely benefits from all four. The current relevance judgment is therefore defensible on the indexed text, with the qualification that c0028 is more implementation detail and c0030 carries the clearest no-retraining statement.

## 7. Comparison with fusion experiment

* **Identical max-score depth 10/20/30 final results** mean that no additional retained candidate displaced the same ten high max-hybrid-score chunks. This does **not** prove c0028–c0030 were absent from depths 20/30, because the experiment persists only the final top-10 rather than the retained pool.
* **RRF did not recover q009** because c0027 appears in just one observed source list while several non-expected chunks are repeated across variants; rank-frequency aggregation can favor those repeated candidates. This statement is limited to c0027, the only expected chunk with observed individual reachability.
* **Round-robin recovered c0027** because its source cycling retains the `inference_control` rank-5 candidate before max-score ranking; it landed at final rank 10. This is direct evidence that allocation/final-top-k, rather than query formulation alone, loses c0027.

## 8. Root-cause hypothesis

The strongest evidence-supported conclusion is **mixed and incomplete**: for c0027, the failure is downstream of individual retrieval—max-score/RRF final ranking or candidate allocation excludes a known rank-5 variant hit, whereas round-robin preserves it. For c0028–c0030, the cause cannot be assigned from the supplied top-10-only artifacts. The data do not support claiming that fusion, query generation, or the relevance judgment is the single root cause of the entire four-chunk failure. The judgment itself is textually defensible.

## 9. Next experiment recommendation

Run **one full-depth, label-blind diagnostic trace** with the same frozen retriever/model and four existing variants, persisting all 127 candidates per variant (rank, raw semantic score, BM25 score, and hybrid score) and the complete retained pool for each already-run fusion strategy. Evaluate against labels only after serialization. This single measurement resolves c0028–c0030’s B/C/D classification and separates candidate-depth, fusion, and final-top-k loss without changing production retrieval, relevance judgments, query variants, or fusion strategy.
