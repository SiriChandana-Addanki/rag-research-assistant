# RAG Research Assistant

A small, inspectable RAG system over the SELF-RAG paper. It preserves the existing PDF ingestion and structure-aware chunking pipeline, then evaluates local TF-IDF/cosine, BM25, and hybrid retrieval against manually reviewed chunk-level evidence labels.

## Architecture
`PDF → ingestion → chunks + page/section metadata → BM25 + semantic embeddings → weighted hybrid retrieval → optional cross-encoder reranking → bounded structured context → Gemini (or another Provider) → grounded JSON answer → validated citations → evaluation and observability`.

Retrieved text is untrusted evidence, never executable instructions. Inputs and contexts are bounded; citations are validated against retrieved chunk IDs; structured logs intentionally omit secrets.

## Components
- `src/ingestion.py`, `src/chunking.py`: existing page-aware extraction and structure-aware chunks.
- `src/retrieval.py`: deterministic TF-IDF/cosine lexical baseline, BM25, normalized hybrid retrieval, and an optional sentence-transformer semantic retriever. Semantic embeddings are normalized and may be cached in a model-and-corpus-fingerprinted local artifact.
- `src/rag.py`: grounded orchestration, limits, citation validation, request IDs, and stage-safe errors. LLMs are injected/configured externally—no key is hard-coded.
- `src/evaluation.py`: judgment validation and Recall@1/3/5/10 plus MRR.

The TF-IDF baseline is intentionally dependency-free; it is not a claim that TF-IDF is a neural embedding model.

## Reproducible evaluation
```bash
python3 scripts/evaluate_retrieval.py
python3 scripts/benchmark_retrieval.py
python3 scripts/evaluate_semantic_retrieval.py
pytest -q
```
`evaluate_semantic_retrieval.py` writes `evaluation/semantic_retrieval_results.json` and, by default, reuses `evaluation/semantic_embeddings.json` when its model and chunk corpus fingerprint match. It keeps the historical lexical artifact untouched. Scores and latency files are generated from the local 15-question dataset and 127 tracked chunks; re-run commands after retrieval changes.

## Verification status
### VERIFIED
The lexical evaluation was run against the tracked 15-question dataset in the available runner:

| Retriever | Recall@1 | Recall@3 | Recall@5 | Recall@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| TF-IDF/cosine | 0.3333 | 0.7333 | 0.9333 | 1.0000 | 0.5883 |
| BM25 | 0.4000 | 0.8000 | 0.9333 | 0.9333 | 0.6156 |
| TF-IDF + BM25 hybrid (0.50 / 0.50) | 0.3333 | 0.8000 | 0.8000 | 1.0000 | 0.5956 |
| Semantic (all-MiniLM-L6-v2) | 0.2667 | 0.6667 | 0.6667 | 0.9333 | 0.4817 |
| Semantic 75% + BM25 25% | 0.5333 | 0.7333 | 0.8667 | 0.9333 | 0.6522 |

The temporary reranking experiment measured R@1/R@3/R@5/R@10/MRR of 0.4000/0.7333/0.8000/0.9333/0.5822. It is labelled experimental because it is not yet emitted by a reproducible tracked evaluation command. The evaluation set has only 15 questions, so these results guide development rather than establish universal superiority.

The same run measured lexical hybrid index construction at 115.786 ms. Across 150 warm searches, p50/p95/p99 were 8.205/8.943/9.704 ms. These are local, single-run measurements, not a production latency claim. Deterministic RAG tests cover retrieved citations, invalid citation rejection, empty retrieval, oversized context, malformed provider output, and timeout retry behavior.

### IMPLEMENTED BUT NOT VERIFIED
`SemanticRetriever` accepts a configurable sentence-transformers model name and batch size, normalizes embeddings before cosine scoring, preserves the original chunk dictionaries, and can reuse a local embedding artifact. `evaluate_semantic_retrieval.py` evaluates semantic retrieval and semantic/BM25 weights of 0.25/0.75, 0.50/0.50, and 0.75/0.25. `CrossEncoderReranker` remains an optional interface for hybrid top-N candidates followed by final-K reranking.

The generation layer uses the official `google-genai` SDK behind a provider protocol. `scripts/ask.py` runs one question and `scripts/evaluate_answers.py` writes per-question answer records without reference answers or fabricated quality scores. Provider output must be JSON with an answer and citations, and citations are rejected unless they identify a retrieved chunk and a matching document/page. Token usage is recorded when Gemini supplies it; cost is reported only when both local `RAG_INPUT_COST_PER_MILLION` and `RAG_OUTPUT_COST_PER_MILLION` are configured.

Copy `.env.example` to a local ignored `.env` only through your own environment loader, or set `GEMINI_API_KEY`, `GEMINI_MODEL`, `RAG_TOP_K`, `RAG_CANDIDATE_K`, `RAG_SEMANTIC_WEIGHT`, `RAG_CONTEXT_LIMIT`, `RAG_TIMEOUT`, and `RAG_MAX_RETRIES` in the shell. No key is read from source files or logged.

## Engineering trade-offs
BM25 remains useful for exact terminology while embeddings capture related phrasing; weighted hybrid scoring deliberately combines those signals and must be measured, not assumed. Reranking can improve ordering but adds latency. The provider protocol avoids vendor lock-in and mocks keep tests deterministic. Local retrieval bounds API use and cost; context is length-limited and treated as untrusted data, while validated citations make the evidence trail inspectable.

### BLOCKED BY ENVIRONMENT
The available command runner is Python 3.14.4 and cannot execute the repository's Windows `.res/Scripts/python.exe` (`Permission denied`); its configured Python is 3.10.0. In the available runner, `sentence_transformers`, `torch`, `rank_bm25`, `numpy`, and `sklearn` were absent. Semantic evaluation therefore stopped with `RuntimeError: sentence-transformers is required for SemanticRetriever`; no semantic, semantic-hybrid, or cross-encoder metrics were written or claimed. No package was installed because the accessible interpreter is outside the project Windows virtual environment. No real provider credential was configured or required.

## Failure handling and limitations
Empty/oversized queries, malformed manifests, invalid relevance labels, invalid citations, and empty context fail clearly. LLM timeouts/API failures and malformed provider output require a concrete provider adapter and are not silently masked. The system has no authentication, deployment service, neural embedding model, reranker, or production load test.

## Decision record
The concise rationale for chunking, baselines, hybrid scoring, deferred reranking, metrics, citations, and observability is in [docs/adr/ADR-001-through-008.md](docs/adr/ADR-001-through-008.md).

## Status
**Implemented and verified:** lexical TF-IDF, BM25, historical lexical hybrid evaluation, citation checks, bounded context, deterministic provider behavior, and retrieval benchmark.

**Implemented but not executed:** live `SemanticRetriever` and `CrossEncoderReranker` evaluations. They require `sentence-transformers` and model downloads, unavailable in this runtime.

**Future work:** configure a real LLM provider via environment-managed credentials, then measure generation latency, tokens, cost, and model-based groundedness.
