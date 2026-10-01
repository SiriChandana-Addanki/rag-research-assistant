# Architecture decisions

## ADR-001 — Structure-aware chunking
Chunks preserve page and section metadata so retrieval evidence can be cited and audited.

## ADR-002 — Dense baseline
A local TF-IDF cosine baseline provides deterministic semantic-like term weighting without requiring network model downloads. A sentence-transformers encoder is a future configurable upgrade.

## ADR-003 — BM25
BM25 adds a transparent lexical baseline that is strong for exact technical terminology.

## ADR-004 — Hybrid retrieval
Min-max normalized dense and BM25 scores are combined with a configurable 0.5 dense weight; evaluation, not assumption, determines whether it is useful.

## ADR-005 — Reranking after retrieval
Reranking is intentionally deferred: no lightweight reranker/model is installed. A reranker belongs after candidate retrieval to bound cost.

## ADR-006 — Recall@K and MRR
These metrics expose evidence coverage and rank of first useful evidence against chunk-level judgments.

## ADR-007 — Retrieved-chunk citations
Citations are validated against the actual retrieved chunk IDs, preventing invented references.

## ADR-008 — Observable failures
Input, metadata, retrieval, citation, and provider configuration failures fail explicitly and logs contain request metadata—not secrets.
