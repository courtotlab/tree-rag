# Reproducibility contract

TreeQuest keeps implementation versions and result artifacts separate. A reported
number is valid only for the named implementation, data manifest, model, and evaluator.

## Version boundary

- **evaluated-v0** is the immutable implementation that generated the 294-question
  restricted deployment study, 63-question ablations, and frozen 200-question public
  run. Exact source snapshots are under `reference/evaluated_v0/`.
- **modular-v1** is the documented package under `src/treequest/`. It preserves the
  controller design while adding bounded retries, queue hygiene, unique-ID validation,
  exact evidence/call caps, and lexical-only wide-node ordering by default.
- No evaluated-v0 result may be presented as modular-v1 evidence. A new run receives a
  new artifact path and implementation label.

## Public experiment provenance

- Dataset: MultiHop-RAG, ODC-BY.
- Corpus: 609 public articles, materialized to DOCX without content changes.
- Frozen sample: 200 qids, seed 20260806, 50 per question type.
- Tree: 20,495 nodes, 19,212 chunks, maximum depth 5 edges, maximum fan-out 236.
- Build: 49,401 seconds and 19,976 LLM calls.
- TreeQuest/answer model: `gpt-oss:120b`.
- Official evaluator: upstream commit
  `cde8e844af14b3012f20158abc2854fe8458212a`.
- Immutable aggregate: `experiments/multihop_rag/results/treequest_official_multihop_eval_v2_20260813.json`.

The balanced sample is not the benchmark's natural-prevalence test distribution.
Published full-dataset results are contextual, not directly comparable leaderboard
entries. Official retrieval excludes null questions. The matched LLM judge is a
separate within-study semantic measure and is never called an official metric.

## Result artifact requirements

Every new run records:

- immutable code revision and implementation version;
- input dataset revision and qid manifest hash;
- full nonsecret configuration;
- model names, digests, quantization, context limit, and server version;
- hardware, start/end timestamps, wall time, successful calls, failures, and tokens;
- immutable answer checkpoint and evaluator output;
- statistical seed, paired unit, uncertainty method, and deviations.

Outputs are append-only or versioned. Never overwrite a report, answer checkpoint,
tree, index, judge artifact, or provenance record.

## Restricted deployment evidence

No private corpus, hierarchy, question, answer, evidence passage, trace, document
identifier, path, or per-question statistic is released. Restricted claims use only
approved aggregates. The public demo tree and public result files must never share a
cache directory with private inference.

## Build-cost record

Record cold and warm builds separately with document/chunk/node counts, model digest,
hardware, software versions, wall time, peak memory (or `unavailable`), output bytes,
cache state, UTC timestamp, and SHA-256. Missing values remain unavailable rather than
being inferred.

## Failure policy

All frozen questions stay in the denominator. Endpoint failures, malformed model output,
timeouts, or absent answers are recorded rather than silently dropped. Retrying follows
the frozen bounded policy. Any protocol change creates a dated deviation record and a
new output path.
