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
- `scripts/generate_chunk_manifest.py` projects the existing ingestion and
  fixed-default chunking output into a deterministic chunk manifest.

Not implemented yet: embeddings, vector retrieval, BM25, hybrid retrieval,
reranking, generation, citations, and runtime observability.

## Evaluation foundation

### Decision

Keep a version-controlled, question-level retrieval dataset and validate its
schema in the test suite. Generate a tracked chunk manifest from the ignored
local PDF before adding exact chunk-level relevance judgments.

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

### Chunk-manifest workflow

Chunk-level judgments need stable chunk IDs and the exact extracted text they
refer to; section/page labels alone cannot verify that a chunk contains enough
evidence for an answer. The manifest generator deliberately reuses
`extract_text_from_pdf` and `chunk_documents` with their current defaults, so
it does not define a second chunking configuration.

With the local, ignored corpus available at `data/raw/paper1.pdf`, generate
the tracked artifact with:

```bash
python scripts/generate_chunk_manifest.py
```

This writes `evaluation/chunk_manifest.json`, containing each chunk's document
ID, chunk ID/index, page range, section, and exact chunk text. JSON is emitted
in the chunking pipeline's order with stable indentation, making the artifact
suitable for review and later relevance judgments. The manifest-schema test
uses a temporary fixture PDF; it does **not** validate the absent real corpus.

Do not add relevance judgments until the real-PDF manifest has been generated
and each proposed mapping has been checked against its `chunk_text`.

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
  labels only until the real local corpus is used to generate and review
  `evaluation/chunk_manifest.json`.
- This checkout does not contain `data/raw/paper1.pdf`; no real-corpus manifest
  or chunk-level judgments have been generated here.
- Extraction and chunking have unit coverage, but the local corpus is not
  version-controlled, so corpus-level ingestion has not been reproduced here.
- No retrieval or generation quality, latency, token, or cost metrics have
  been measured.
