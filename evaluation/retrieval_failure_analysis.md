# Retrieval failure analysis (evidence only)

## Scope and evidence limitation

This review used the tracked retrieval dataset, relevance judgments, chunk manifest,
source/chunk-generation code, and the rank facts in the request. The claimed
`evaluation/semantic_retrieval_traces.json` was **not present in this working
tree** when inspected. Its complete top-10 lists and scores therefore could not
be independently reproduced: the runtime lacks `sentence_transformers` and the
model package could not be downloaded. This report does not infer missing rank
IDs or claim text for unprovided competitors; it maps every explicitly named ID
to the manifest evidence.

## Why `REFERENCES` chunks are retrievable

The index is the manifest generated from *all* chunks returned by
`chunk_documents`; `build_manifest` projects every such chunk and applies no
section exclusion. `split_into_sections` sets the section when it sees the
literal `REFERENCES` heading and continues assigning that section until it
recognizes another heading. The appendix headings in this PDF, such as
`A SELF-RAG DETAILS`, do not match the supported literal headings or the
numeric-heading pattern. Thus chunks `c0073` onward retain a stale
`REFERENCES` label even though many are appendix/body content.

| chunk ID | manifest metadata | actual chunk-text evidence |
|---|---|---|
| `paper1_c0073` | `REFERENCES`, pp. 16–17 | Table-of-contents material followed by “A SELF-RAG DETAILS” and reflection-token definitions. |
| `paper1_c0074` | `REFERENCES`, p. 17 | Defines retrieval-on-demand and `ISREL`; substantive appendix content. |
| `paper1_c0081` | `REFERENCES`, p. 19 | Describes critic predictions and “Algorithm 3 Mgen Data creation.” |
| `paper1_c0082` | `REFERENCES`, p. 19 | Gives `ISREL`, `ISSUP`, and `ISUSE` score calculations. |
| `paper1_c0089` | `REFERENCES`, pp. 20–21 | Reports evidence-grounding results and begins human-evaluation examples. |

By contrast, `c0060`–`c0072` contain conventional author/title/venue
bibliographic entries. A blanket `REFERENCES` filter would also discard
substantive appendix evidence; this section label is not a reliable bibliography
indicator for this document.

## Question-level assessment

| question_id | failure_pattern | relevant_chunks | competing_chunks | ground_truth_assessment | recommended_next_experiment |
|---|---|---|---|---|---|
| q001 | Direct conventional-RAG evidence is only at ranks 5 (`c0005`) and 7 (`c0000`) in the supplied summary. | `c0000,c0002,c0005` | Full top-10 IDs unavailable; supplied relevant ranks: `c0005`=5, `c0000`=7. | clearly correct ground truth | Inspect the complete persisted trace, then compare lexical/semantic terms for unseen ranks 1–4. |
| q006 | `c0089` (irrelevant) is rank 2; direct inference-overview chunks `c0017`/`c0016` are ranks 4/5. | `c0016,c0017,c0018` | `c0089`=2; other top-10 IDs unavailable. | clearly correct ground truth | Separate verified bibliography from appendix content before testing any section filter. |
| q009 | No judged chunk is in the supplied top 10; introduction chunks dominate and one `REFERENCES`-labelled chunk is rank 6. | `c0027,c0028,c0029,c0030` | Exact introduction/reference IDs unavailable. | clearly correct ground truth | Recover query-to-chunk scores; use a decomposed query only as a diagnostic. |
| q010 | Direct generator-data evidence `c0024` is rank 6; introduction chunks rank above it and a `REFERENCES`-labelled chunk is rank 7. | `c0021,c0022,c0023,c0024,c0025` | `c0024`=6; remaining IDs unavailable. | clearly correct ground truth | Compare score contribution by verified content class; do not equate `REFERENCES` with bibliography. |
| q011 | `c0011`/`c0012` rank 1–2 but are unlabelled; judged `c0025`/`c0026` are ranks 3/5. | `c0021,c0023,c0025,c0026` | `c0011`=1, `c0012`=2, `c0025`=3, `c0026`=5. | likely incomplete ground truth | Human-review `c0011`/`c0012`; both explicitly contrast offline critic training with PPO/RLHF. |
| q014 | Direct ablation chunks `c0043`/`c0046` are ranks 3/4; `REFERENCES` chunks occupy ranks 5/6. | `c0043,c0046,c0047` | `c0043`=3, `c0046`=4; reference IDs unavailable. | clearly correct ground truth | Recover rank-5/6 IDs and distinguish appendix content from bibliography before filtering. |

## Chunk evidence map

The full chunk text is retained verbatim in `evaluation/chunk_manifest.json`.
This table maps every judged and explicitly named competing chunk to its actual
text subject and metadata.

| chunk ID | section; pages | actual chunk-text evidence |
|---|---|---|
| `c0000` | ABSTRACT; 1–1 | Fixed-number retrieval regardless of necessity/relevance can reduce versatility or cause unhelpful responses. |
| `c0002` | INTRODUCTION; 1–1 | Indiscriminate retrieval can introduce unnecessary/off-topic passages and outputs need not follow evidence. |
| `c0005` | INTRODUCTION; 1–2 | Conventional RAG retrieves a fixed number regardless of necessity and does not revisit generation quality. |
| `c0011` | INTRODUCTION; 3–3 | Introduces PPO/RLHF and contrasts SELF-RAG’s fine-grained reflection/customizable inference. |
| `c0012` | INTRODUCTION; 3–3 | Explicitly says offline critic-token augmentation has lower cost than RLHF and differs in inference control. |
| `c0016` | 3.1 PROBLEM FORMALIZATION AND OVERVIEW; 4–4 | Algorithm 1: Retrieve, ISREL, ISSUP, ISUSE, and ranking. |
| `c0017` | 3.1 PROBLEM FORMALIZATION AND OVERVIEW; 4–4 | Explains retrieval decision, relevance/support/utility tokens, and soft/hard control. |
| `c0018` | 3.1 PROBLEM FORMALIZATION AND OVERVIEW; 4–4 | Explains parallel passages, ISREL/ISSUP selection, and critic-produced training tokens. |
| `c0021` | 3.2.1 TRAINING THE CRITIC MODEL; 4–5 | GPT-4 produces reflection-token supervision distilled into critic `C`. |
| `c0022` | 3.2.1 TRAINING THE CRITIC MODEL; 5–5 | Prompts, collected critic data, and critic-learning objective. |
| `c0023` | 3.2.1 TRAINING THE CRITIC MODEL; 5–5 | Critic training and agreement with GPT-4 predictions. |
| `c0024` | 3.2.2 TRAINING THE GENERATOR MODEL; 5–5 | Generator-data augmentation, retrieval, `ISREL`, `ISSUP`, `ISUSE`, and `Dgen`. |
| `c0025` | 3.2.2 TRAINING THE GENERATOR MODEL; 5–6 | Generator training on `Dgen` and offline critique versus PPO. |
| `c0026` | 3.2.2 TRAINING THE GENERATOR MODEL; 6–6 | Special tokens and inference re-ranking/hard constraints. |
| `c0027` | 3.3 SELF-RAG INFERENCE; 6–6 | More retrieval for factual accuracy versus less retrieval/utility for open-ended tasks. |
| `c0028` | 3.3 SELF-RAG INFERENCE; 6–6 | Adaptive retrieval threshold and critique-token tree decoding. |
| `c0029` | 3.3 SELF-RAG INFERENCE; 6–6 | Weighted `ISREL`/`ISSUP`/`ISUSE` score and adjustable inference weights. |
| `c0030` | 3.3 SELF-RAG INFERENCE; 6–6 | Hard constraints and weights tailor behavior without additional training. |
| `c0043` | 5.2 ANALYSIS; 9–9 | No Retriever and No Critic/top-one-passage ablations. |
| `c0046` | 5.2 ANALYSIS; 9–9 | Deterioration from top-one irrelevant retrieval and removing ISSUP. |
| `c0047` | 5.2 ANALYSIS; 9–10 | ISSUP-weight effects and test-time customization without retraining. |
| `c0073,c0074,c0081,c0082,c0089` | REFERENCES; pp. 16–21 | Appendix/results material, not solely bibliography; see the preceding table. |

## q009 answerability

q009 is answerable from the indexed document, but the evidence is **distributed
across multiple chunks**, not contained in one chunk. `c0027` supplies the
factual-grounding versus open-ended-quality trade-off; `c0028` supplies the
retrieval threshold; `c0029` supplies adjustable critique weights; and
`c0030` states the no-additional-training result. This matches the four
existing judgments and does not support a conclusion that the evidence is absent.

## Conclusion

A blanket `REFERENCES` filter is **not justified**: it would remove substantive
appendix text in `c0073`–`c0089`. A controlled diagnostic that excludes only
verified bibliography chunks after a section-label audit is justified. q011 is
the only case that clearly needs human label review: its top-ranked `c0011` and
`c0012` directly discuss the asked PPO/RLHF comparison but are not judged
relevant.
