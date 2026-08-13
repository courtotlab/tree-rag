# TreeQuest

**Scaling interactive reading from long documents to governed collections.**

TreeQuest lets an open-weight language model navigate a folder -> document ->
section -> passage hierarchy, collect evidence, recover from an early wrong turn,
and answer with source references. The query-time controller uses LLM relevance
judgments, a corpus-global teleport frontier, same-document sweeps, an evidence
sufficiency gate, and bounded search budgets.

![TreeQuest architecture](assets/treequest-system.svg)

## What this release contains

- `src/treequest/`: a documented modular API and CLI for interactive use.
- `reference/evaluated_v0/`: immutable source snapshots behind the original
  deployment and public experiments.
- `experiments/multihop_rag/`: the frozen public sample, exact benchmark runner,
  official evaluator adapter, aggregate results, and pinned upstream evaluator.
- `data/multihop_rag_demo/corpus_tree.json`: the committed public demonstration
  tree, containing 20,495 nodes over 609 public articles.

The reported experiments use **evaluated-v0**. The modular package is a hardened
release port with bounded retries, queue hygiene, tree validation, and a
lexical-only default for over-wide candidate presentation. Results are never
retrospectively relabeled across those versions.

## Results at a glance

### Public MultiHop-RAG study

The frozen sample has 200 questions, balanced across comparison, inference,
temporal, and null types. Retrieval metrics use the official MultiHop-RAG code
at pinned commit `cde8e844af14b3012f20158abc2854fe8458212a`.

| System | Official QA accuracy | Hits@4 | Hits@10 | MAP@10 | MRR@10 |
|---|---:|---:|---:|---:|---:|
| TreeQuest evaluated-v0 | **0.495** | **0.667** | **0.693** | **0.226** | **0.461** |
| Flat hybrid | 0.415 | unavailable | unavailable | unavailable | unavailable |
| Collapsed-tree control | 0.325 | 0.353 | 0.480 | 0.106 | 0.254 |
| Oracle gold-context diagnostic | 0.435 | 1.000 | 1.000 | 0.670 | 1.000 |

Official retrieval excludes the 50 null questions. Flat-hybrid retrieval metrics
cannot be reconstructed from its stored title-level artifact and are therefore
reported as unavailable, never as zero. Under the separate matched semantic judge,
TreeQuest scored 0.639 versus 0.598 for flat hybrid (paired difference 0.041,
95% CI [-0.029, 0.112], `p=0.262`). See [RESULTS.md](RESULTS.md) for the complete
interpretation and limitations.

### Restricted deployment study

On 294 held-out operational questions, evaluated-v0 scored 0.563 versus 0.469
for the deployed hybrid retriever (paired difference 0.094, 95% CI
[0.063, 0.127]) at 6.95x mean wall-clock latency. Only approved aggregate
statistics are released; no private question, answer, path, trace, or corpus
artifact is present here.

## Install

```bash
git clone https://github.com/asharma391/treerag.git
cd treerag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[build,test]'
```

TreeQuest expects an Ollama-compatible endpoint. For confidential material, run
the code and Ollama inside the same approved compute boundary; do not send corpus
text to a third-party endpoint.

## Demo 1: query the committed public tree

Run this command **on the compute host that has Ollama**, not on a laptop that
must remain awake. `127.0.0.1:11434` then refers to Ollama on that compute host.

```bash
ollama pull gpt-oss:120b
export TREEQUEST_OLLAMA_URL=http://127.0.0.1:11434
export TREEQUEST_MODEL=gpt-oss:120b
./scripts/query_demo.sh "Which developments are compared across multiple reports?"
```

Equivalent direct CLI invocation:

```bash
treequest \
  --tree data/multihop_rag_demo/corpus_tree.json \
  --ollama-url http://127.0.0.1:11434 \
  --model gpt-oss:120b \
  --mode thorough \
  "Your question"
```

The JSON response contains the answer, source identifiers, elapsed time, model
calls, and mode. Thorough mode reproduces the research operating point more
closely; quick mode is cheaper but was materially less accurate in the ablation.

## Demo 2: rebuild the MultiHop-RAG tree

The committed tree is never overwritten by this workflow. A fresh build is written
under `experiments/multihop_rag/tree_cache/`.

```bash
export TREEQUEST_OLLAMA_URL=http://127.0.0.1:11434
export TREEQUEST_MODEL=gpt-oss:120b
./scripts/build_multihop_demo.sh

treequest \
  --tree experiments/multihop_rag/tree_cache/corpus_tree.json \
  --mode thorough \
  "Your question"
```

The reference build used 19,976 model calls and 13 h 42 min for 609 articles.
The builder is resumable through parse and node caches. Use `tmux` or a scheduler
on remote compute so a laptop sleep or disconnect cannot interrupt it; see
[docs/REMOTE_COMPUTE.md](docs/REMOTE_COMPUTE.md).

## Reproduce the public metrics

```bash
cd experiments/multihop_rag
python official_multihop_eval.py --help
```

The exact executed runner is preserved at
`reference/evaluated_v0/benchmark_treequest_public_frozen_v0.py`. The runnable
copy in `experiments/multihop_rag/` differs only in runtime path/model/endpoint
configuration. Read the experiment README before launching a full 200-question
run; it is expensive and all outputs must be written to new versioned paths.

## Query-time vector-signal boundary

TreeQuest does not use a dense index to retrieve evidence: the LLM scores tree
branches and all evidence is reached through navigation. For scientific precision,
the evaluated-v0 source does use local embeddings to order names shown inside an
over-wide `contains:` preview. The modular release default sets
`max_embed_per_decision=0` and uses lexical ordering instead. Accordingly, the
measured method is described as **LLM-routed tree retrieval without dense-vector
evidence retrieval**, not as having no vector computation anywhere.

## Repository layout

```text
src/treequest/                 modular controller and CLI
scripts/                       public demo/build launchers
data/multihop_rag_demo/        committed public demonstration tree
experiments/multihop_rag/      frozen sample, runners, evaluator, public results
reference/evaluated_v0/        immutable evaluated source snapshots
docs/                          architecture and remote-compute guidance
tests/                         synthetic unit and controller tests
```

## Data, privacy, and licensing

MultiHop-RAG is licensed ODC-BY. TreeQuest source is GPL-3.0-or-later. See
[DATA_POLICY.md](DATA_POLICY.md), [SECURITY.md](SECURITY.md), and
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). Do not commit private corpora, trees,
questions, answers, traces, credentials, endpoints, or result files.

## Next steps

Next steps can consider running the remaining public controls and evaluating other
systems like Psi-RAG as a separate external baseline. They would broaden
public-benchmark coverage and help position TreeQuest against a recent
vector-assisted retrieval system.

## Citation

Use [CITATION.cff](CITATION.cff). The manuscript itself is intentionally not
stored in either GitHub repository before publication.
