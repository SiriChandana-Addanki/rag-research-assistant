# RAG Research Assistant

## Purpose and engineering impact

A CLI research assistant that answers questions over a small corpus of research PDFs with retrieved evidence and validated page/chunk citations. The project demonstrates document-aware ingestion, deterministic hybrid retrieval, grounded generation, failure handling, and evaluation artifacts that make trade-offs inspectable.

## Stack and flow

- **Tech stack:** Python 3.10, PyMuPDF, sentence-transformers, BM25, pytest, Docker.
- **LLM:** Google Gemini through the official `google-genai` SDK. Generation is optional.
- **Input:** Research PDFs in `data/raw/` when rebuilding the corpus; a question string when using the assistant.
- **Output:** JSON answer, citations, retrieved evidence, retrieval trace, and execution status on stdout.
- **Architecture:** PDFs → page-aware ingestion and structure-aware chunks → checked-in chunk manifest → `all-MiniLM-L6-v2` semantic scores plus BM25 → normalized-score hybrid retrieval → bounded evidence context → optional Gemini JSON response → citation validation. Production remains at semantic/BM25 weights **0.75/0.25**, `RAG_TOP_K=5`, and `RAG_CANDIDATE_K=10`; experimental reranking, MultiQuery, and fusion are not in the production path.

## Measured results

On the frozen 15-question dataset and relevance judgments, the recorded selected hybrid baseline is Recall@1 **0.5333**, Recall@3 **0.7333**, Recall@5 **0.8667**, Recall@10 **0.9333**, and MRR **0.6633**. These are repository-specific measurements, not general performance claims. See `evaluation/semantic_retrieval_results.json` and `evaluation/retrieval_baseline_analysis.md`. The q009 aligned ablation remains `not_run`; no result is inferred for it.

## Run locally

Install dependencies with `python -m pip install -r requirements.txt`. Then run:

```powershell
python scripts/ask.py --no-generation "What are reflection tokens?"
```

Run with live generation by supplying the key through the environment. Do not put it in source or the image:

```powershell
$env:GEMINI_API_KEY = "your-key"
python scripts/ask.py "What are reflection tokens?"
```

`GEMINI_MODEL` optionally selects the model; default is `gemini-3.8-flash`. Retrieval settings remain configurable through existing `RAG_*` environment variables, whose defaults preserve the selected production configuration.

## Docker

Build from the repository root:

```powershell
docker build -t rag-research-assistant:local .
```

Run the CLI without generation (no API key needed):

```powershell
docker run --rm rag-research-assistant:local scripts/ask.py --no-generation "What are reflection tokens?"
```

For live generation, pass credentials only at runtime:

```powershell
docker run --rm -e GEMINI_API_KEY="your-key" rag-research-assistant:local scripts/ask.py "What are reflection tokens?"
```

Run tests in the image:

```powershell
docker run --rm rag-research-assistant:local -m pytest -q
```

Run the lexical retrieval evaluation without secrets:

```powershell
docker run --rm rag-research-assistant:local scripts/evaluate_retrieval.py
```

The image contains source, scripts, the reviewed chunk manifest, and evaluation inputs. It excludes `.env`, local Python environments, model/download caches, and persisted result/trace artifacts. The sentence-transformer model is downloaded on first semantic use and cached in the container's temporary cache. To retain that cache or generated outputs, mount an appropriate writable volume.

## Limitations

The evaluation set is small and domain-specific. PDF extraction can lose information in scans, tables, equations, and unusual layouts. Model downloads and Gemini generation need network access; generation needs a user-supplied API key. Retrieval or citation validation reduces unsupported output but cannot guarantee factual correctness. The container is a CLI deployment, not an HTTP service, and local inference resources may constrain performance.
