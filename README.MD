# RAG Research Assistant

An incremental research-paper assistant intended to produce grounded answers
with citations. The project is being built and evaluated in small, testable
milestones rather than as a single end-to-end implementation.

## Problem

Language models can produce plausible answers that are unsupported by source
documents. This project will evaluate whether retrieval, reranking, citation
construction, and explicit evaluation improve grounding for questions over
research papers.

## Planned architecture

```text
PDF documents
  -> layout-aware text extraction
  -> sentence-aware, section-bounded chunks
  -> embeddings and vector retrieval
  -> BM25 and hybrid retrieval
  -> reranking
  -> context construction
  -> grounded generation with citations
  -> evaluation and tracing
```

## Current implementation

Implemented:

- PDF ingestion extracts page-level text and source/page metadata.
- Chunking keeps sentences within their detected section and preserves source,
  page range, section, document ID, and chunk ID metadata.
- `evaluation/retrieval_dataset.json` contains manually authored retrieval
  questions with expected section and page labels for one local paper.

Not implemented yet: embeddings, vector retrieval, BM25, hybrid retrieval,
reranking, generation, citations, and runtime observability.

## Evaluation foundation

### Decision

Keep a version-controlled, question-level retrieval dataset and validate its
schema in the test suite.

### Reason

The retrieval stages have not been implemented. Preserving the questions and
their expected source regions now gives later retrieval experiments a stable
evaluation input.

### Alternatives considered

- Evaluate retrieval only with ad-hoc manual questions.
- Label exact chunk IDs before the final chunking configuration is selected.

### Evidence

The dataset currently provides expected sections and pages, while chunk IDs are
derived from chunking configuration. No retrieval-quality, latency, or cost
measurements exist yet.

### Tradeoffs

The current labels can support source-region checks but cannot yet calculate
exact chunk-level Recall@K or MRR. Exact relevance judgments should be added
after the chunking configuration is finalized and before retrieval strategies
are compared.

## Local development

Create an environment and install dependencies:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run the tests:

```bash
python -m pytest -q
```

The ingestion test creates a temporary PDF at test time. Local PDFs under
`data/raw/` are intentionally ignored by Git and are not required to run the
test suite.

## Known limitations

- The evaluation dataset targets one paper and has question-level source-region
  labels only.
- Extraction and chunking have unit coverage, but the local corpus is not
  version-controlled, so corpus-level ingestion has not been reproduced here.
- No retrieval or generation quality, latency, token, or cost metrics have
  been measured.
