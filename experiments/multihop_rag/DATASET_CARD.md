# MultiHop-RAG dataset card for the TreeRAG study

## Source and license

- Dataset: MultiHop-RAG.
- Authors: Yixuan Tang and Yi Yang.
- Venue: COLM 2024.
- Official paper: https://arxiv.org/abs/2401.15391
- Official repository: https://github.com/yixuantt/MultiHop-RAG
- Declared dataset license: Open Data Commons Attribution License (ODC-BY).

The official repository's license statement governs the released database.
Because the knowledge base contains news articles, redistributors must still
review whether database licensing covers every underlying text and intended
form of redistribution. The TreeRAG artifact should preferentially provide
download/materialization code, qids, checksums, and derived aggregate results
rather than republishing upstream article text.

The final artifact must record the exact upstream commit or immutable archive
digest used. A moving repository URL alone is insufficient provenance.

## Original dataset

The official paper and repository report:

- 609 English news articles from six categories;
- 2,556 multi-hop queries;
- evidence spanning two to four documents for answerable queries;
- 816 inference queries (31.92 percent);
- 856 comparison queries (33.49 percent);
- 583 temporal queries (22.81 percent);
- 301 null queries (11.78 percent);
- 1,078 questions requiring two evidence pieces;
- 779 requiring three;
- 398 requiring four;
- 301 null questions with no supporting evidence.

The articles were collected from September 26 through December 26, 2023.
Queries and answers were generated with GPT-4 from extracted claims and bridge
entities/topics, followed by automated checks and manual review of a subset.

## Frozen TreeRAG sample

TreeRAG uses a deterministic 200-question sample with seed 20260806:

| Query type | Sample count | Sample share |
|---|---:|---:|
| Comparison | 50 | 25 percent |
| Inference | 50 | 25 percent |
| Temporal | 50 | 25 percent |
| Null | 50 | 25 percent |
| Total | 200 | 100 percent |

This is a balanced stratified sample, not a simple random sample from the
original query distribution. The aggregate headline score therefore estimates
equal-weight performance across the four query types. It must not be described
as an estimate of accuracy under the dataset's natural type prevalence.

Question-type results are exploratory because each stratum has only 50
questions. The frozen qid manifest and construction script are release
artifacts; qids cannot be changed after observing system outcomes.

## Public hierarchy transformation

The original dataset is a news-article knowledge base, not the private
deployment's governed filesystem. The TreeRAG public pipeline deterministically
materializes a metadata-derived hierarchy:

    category -> source -> document -> section -> chunk

This provides a reproducible corpus-scale structural test, but it is not
evidence that the public corpus came with an operational governance hierarchy.
The manuscript must distinguish:

- ecological evidence from the existing private governed hierarchy;
- reproducible public evidence from a metadata-derived hierarchy.

No claim should imply that the public benchmark reproduces every property of
the regulated deployment.

## Outcomes and comparability

The original paper reports answer accuracy for short answers and also evaluates
retrieval against supporting evidence. The TreeRAG study uses blinded joint
scores in {0, 0.5, 1} for free-form system answers. These outcome definitions
are not numerically interchangeable. TreeRAG results cannot be compared
directly with the original paper's accuracy table as though they used the same
judge, prompts, answer format, retrieval context, or model.

Primary TreeRAG inference uses gpt-oss:120b. The same model is used for the
main model judge, with balanced answer ordering and a separate order-sensitivity
audit. Human/LLM agreement is reported only after the study team supplies
approved aggregate validation statistics.

## Known limitations

### Generated questions and answer style

The benchmark was partly generated with GPT-4 and emphasizes short answers such
as entities, yes/no comparisons, and temporal relations. Performance may not
transfer to long-form synthesis, procedural completeness, or heterogeneous
enterprise questions.

### Evidence depth

Answerable questions use at most four evidence pieces. The benchmark does not
directly test retrieval requiring dozens of dispersed records or exhaustive
coverage of a large folder.

### Temporal contamination

The 2023 article window reduced contamination for models available when the
dataset was created. A model released or trained later may have encountered the
articles, paper, repository, or generated questions. The closed-book control
quantifies answerability from model priors on the frozen sample but cannot prove
that no training contamination occurred.

### Null oversampling

Null questions are 25 percent of the TreeRAG sample but 11.78 percent of the
full dataset. This deliberately gives insufficient-evidence behavior equal
weight; it also changes the aggregate relative to natural prevalence.

### News-to-governance transfer

News metadata is cleaner and less domain-specific than many organizational
file systems. The public hierarchy does not reproduce access controls, version
histories, duplicated policies, scanned documents, or institution-specific
naming conventions.

### LLM judging

The answer model and primary judge share a model family and endpoint. Blinding,
balanced order, order reversal, and human validation reduce but do not eliminate
shared-model bias.

## Required release provenance

The completed public package must include:

- upstream repository commit or archive checksum;
- download date and license snapshot;
- hashes of raw upstream files without republishing restricted content;
- sample seed, script, and frozen qid manifest;
- counts by query type;
- hierarchy materialization code and build metadata;
- system configurations and model digests;
- immutable answer and judge artifact hashes;
- aggregate analysis and random seeds;
- a statement that the balanced score is macro-averaged across query types.
