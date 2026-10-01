# RAG Research Assistant

A small, inspectable RAG system over the SELF-RAG paper. It preserves the existing PDF ingestion and structure-aware chunking pipeline, then evaluates local dense-style, BM25, and hybrid retrieval against manually reviewed chunk-level evidence labels.

## Architecture
`PDF → ingestion → chunks + page/section metadata → local retrieval → optional reranking (not configured) → bounded context → configurable generation → retrieved-chunk citations`.

Retrieved text is untrusted evidence, never executable instructions. Inputs and contexts are bounded; citations are validated against retrieved chunk IDs; structured logs intentionally omit secrets.

## Components
- `src/ingestion.py`, `src/chunking.py`: existing page-aware extraction and structure-aware chunks.
- `src/retrieval.py`: deterministic TF-IDF/cosine "dense" baseline, BM25, and normalized hybrid retrieval (default dense weight 0.5).
- `src/rag.py`: grounded orchestration, limits, citation validation, request IDs, and stage-safe errors. LLMs are injected/configured externally—no key is hard-coded.
- `src/evaluation.py`: judgment validation and Recall@1/3/5/10 plus MRR.

The dense baseline is intentionally dependency-free because sentence-transformers is not installed; it is not a claim that TF-IDF is a neural embedding model. No reranker or real LLM evaluation was executed in this environment.

## Reproducible evaluation
```bash
python3 scripts/evaluate_retrieval.py
python3 scripts/benchmark_retrieval.py
pytest -q
```
Machine-readable outputs are `evaluation/retrieval_results.json` and `evaluation/benchmark_results.json`. Scores and latency files are generated from the local 15-question dataset and 127 tracked chunks; re-run commands after retrieval changes.

## Measured results
See the JSON artifacts above for actual numbers. The benchmark separates cold index construction from 150 warm retrieval searches and reports p50/p95/p99. It does not measure end-to-end LLM latency because no provider was configured.

## Failure handling and limitations
Empty/oversized queries, malformed manifests, invalid relevance labels, invalid citations, and empty context fail clearly. LLM timeouts/API failures and malformed provider output require a concrete provider adapter and are not silently masked. The system has no authentication, deployment service, neural embedding model, reranker, or production load test.

## Decision record
The concise rationale for chunking, baselines, hybrid scoring, deferred reranking, metrics, citations, and observability is in [docs/adr/ADR-001-through-008.md](docs/adr/ADR-001-through-008.md).
