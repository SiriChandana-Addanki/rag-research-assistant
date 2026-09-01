# PRODUCTION RAG RESEARCH ASSISTANT

## Research Assistant with Grounded Answers + Citations + Evaluation 

An AI-powered research assistant that answers questions over a collection of research papers and technical documents using retrieval-augmented generation (RAG).

USECASE  :  

Basically like after uploading bunch of research papers , i could ask compare the approcahes they have taken to reduce hallucinations of RAG.

### Problem

Large Language Models can generate plausible but unsupported answers.(kIND OF HALLUCINATIONS)

How I Want actually to Solve this :

This project explores how retrieval, reranking, citations, and evaluation can improve the factual grounding and reliability of answers.

### Planned Architecture

Documents
→ Chunking
→ Embeddings
→ Retrieval
→ Reranking
→ Context Construction
→ LLM
→ Grounded Answer + Citations
→ Evaluation + Tracing

### Experiments

The project will compare:

#### Baseline

Dense / vector retrieval
→ LLM

#### Improved Pipeline

Hybrid retrieval
→ Reranking
→ LLM

The systems will be evaluated using a manually created evaluation dataset.

### Evaluation Metrics

- Retrieval Recall
- Context Precision
- Faithfulness
- Answer Relevance
- Latency
- Token Usage
- Cost

### Tech Stack



## Project Status

