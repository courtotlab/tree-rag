# TreeQuest canonical-v1

This document defines the publication-default implementation used for new experiments.
It is part of the reproducibility contract, not a claim that earlier deployment results
were produced by the corrected code.

## Version boundary

- `evaluated-v0`: the preserved monolithic implementation that produced the original
  regulated-deployment benchmark and initial ablations.
- `canonical-v1`: the corrected modular release and the implementation used for the frozen
  public evaluation and new versioned private regression.

Never relabel a v0 report as v1. Every new report must record `canonical-v1` explicitly.

## Canonical query-time configuration

| Setting | Value | Meaning |
|---|---:|---|
| Model | `gpt-oss:120b` | Open-weight routing and answering model |
| `max_embed_per_decision` | `0` | No query-time vector similarity in tree routing |
| `max_rank_candidates` | `60` | Maximum children scored at one node |
| `score_batch` | `5` | Candidates per routing call |
| `noise_floor` | `0.20` | Active/reserve boundary |
| `max_steps` | `40` | Traversal-iteration ceiling |
| `max_files` | `8` | Distinct retained-source ceiling under valid metadata |
| `max_evidence` | `50` | Internal and returned evidence-piece ceiling |
| `max_llm_calls` | `400` | Exact successful-call ceiling, including answer generation |

At fan-out above 60, lexical overlap and stable source order choose the scored shortlist.
Overflow remains reachable in reserve with an explicit unscored marker.

## Queue invariants

- A node ID occurs at most once across active and reserve after pruning.
- Dispatched or collected nodes are stale and cannot be selected again.
- Forced contrast targets are removed from either active or reserve before dispatch.
- Default selection takes the best valid active entry, then the best valid reserve entry.
- Contrast, breadth, and residual teleports are explicit bounded policy overrides.

The loader rejects empty or duplicate node IDs. Internal collection refuses additions at
the evidence cap.

## Resource semantics

The successful context-level LLM-call cap is exact. Navigation reserves answer capacity
inside that cap. Failed HTTP attempts internal to the bounded client retry policy are not
counted as successful calls.

Wall-clock limits are cooperative checks between requests. They do not interrupt an
in-flight model request and must not be described as hard real-time deadlines.

## Non-guarantees

Canonical-v1 does not guarantee retrieval completeness, semantic correctness, score
calibration, global best-first order, optimality, logarithmic query time, or a preemptive
wall-clock deadline. Lexical seeding may scan a corpus-level lexical index.

## Reproducibility mapping

- Modular implementation: `src/treequest/`
- Frozen public runner: `experiments/multihop_rag/benchmark_treequest_public_canonical_v1.py`
- Formal specification and proofs: supplied with the paper appendix
- Public reports: new versioned artifacts only
- Private reports and corpus artifacts: excluded from this repository

The public runner and the modular package share vector-free routing, fan-out control,
frontier hygiene, forced-target handling, and evidence limits. Experimental reports retain
observed calls, latency, trace events, and implementation version so deviations are
auditable.

