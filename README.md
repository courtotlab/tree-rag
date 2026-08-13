# TreeQuest: Agentic Hierarchical Retrieval for Governed Document Collections

<div align="center">

[![CI](https://github.com/courtotlab/tree-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/courtotlab/tree-rag/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-GPL--3.0--or--later-2F6B3B)](LICENSE)

</div>

**Recursive summarization and agentic tree traversal over large, structured corpora.**

TreeQuest lets an open-weight language model navigate a corpus-level hierarchy,
retain relevant evidence, recover from an early wrong turn, and answer with source
references. It is designed for document collections where structure, privacy,
and auditability matter.

![TreeQuest architecture](assets/treequest-system.svg)

- [Overview](#overview)
- [Getting started](#getting-started)
- [How TreeQuest retrieves evidence](#how-treequest-retrieves-evidence)
- [Reproducing the public evaluation](#reproducing-the-public-evaluation)
- [Advanced settings](#advanced-settings)
- [Evaluated implementation and vector boundary](#evaluated-implementation-and-vector-boundary)
- [Repository layout](#repository-layout)
- [Data, privacy, and licensing](#data-privacy-and-licensing)
- [Next steps](#next-steps)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)

---

## Overview

Clinical laboratories keep many controlled documents in nested folder
hierarchies, and staff need effective methods to find specific information
inside them. More generally, governed organizations face the same challenge
across large collections of policies, procedures, reports, and records.

Single-shot ranking does not explicitly consider document structure, so queries
whose answers are found in tables, span sections, or require multiple documents
can perform poorly. TreeQuest recursively summarizes the corpus into a
hierarchical tree. From there, an open-weight LLM agent traverses the tree to
find and retain the most relevant passages for a question.

### Key features

- **Corpus-level summarization tree:** preserves folder, document, section, and
  passage structure instead of flattening the collection into independent chunks.
- **Agentic tree traversal:** recursively scores visible branches and follows the
  strongest route toward relevant evidence.
- **Cross-document retention:** accumulates passages across traversal steps and
  documents before composing an answer.
- **Corpus-global recovery:** a teleport frontier resumes from the strongest
  unexplored node at any depth after an unproductive branch.
- **Evidence-aware reading:** scope selection, same-document sweeps, and an
  evidence-sufficiency gate determine whether to answer or continue searching.
- **Bounded and auditable execution:** explicit budgets limit model calls and
  visited nodes, while traces and source identifiers support inspection.
- **Open-weight deployment:** TreeQuest uses an Ollama-compatible endpoint and
  can run inside the same approved compute boundary as confidential data.
- **No dense-vector evidence retrieval:** evidence is reached through tree
  navigation rather than a dense retrieval index.

### What this release contains

- `src/treequest/`: the documented modular controller and command-line interface.
- `reference/evaluated_v0/`: immutable source snapshots for the evaluated method.
- `experiments/multihop_rag/`: the frozen public sample, runner, official
  evaluator adapter, aggregate outputs, and pinned upstream evaluator.
- `data/multihop_rag_demo/corpus_tree.json`: a committed public tree with 20,495
  nodes over 609 MultiHop-RAG articles.
- `tests/`: synthetic unit and controller tests that do not require corpus data.

## Getting started

### Package requirements

- Python 3.11 or 3.12
- OICR VPN and SSH access to the approved Ollama host
- Local port `11528` available for the SSH tunnel
- Approximately 30 MB of free space for the committed public tree

### Install

```bash
git clone git@github.com:courtotlab/tree-rag.git
cd tree-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[build,test]'
pytest -q
```

### Open the OICR Ollama tunnel

TreeQuest runs on the workstation. Model requests reach the approved OICR Ollama
service through an SSH local forward; no TreeQuest process or corpus is placed on
the server.

Keep this command running in a dedicated terminal while using TreeQuest:

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3 \
  -o IdentitiesOnly=yes \
  -i "$HOME/.ssh/id_ed25519_oicr" \
  -L 127.0.0.1:11528:127.0.0.1:11434 \
  asharma@10.30.134.39
```

The command is silent after connecting. In a second terminal, configure and verify
the tunneled endpoint:

```bash
export TREEQUEST_OLLAMA_URL=http://127.0.0.1:11528
export TREEQUEST_MODEL=gpt-oss:120b
curl -fsS "$TREEQUEST_OLLAMA_URL/api/version"
```

Do not run Ollama locally or bind the forward to a non-loopback address. See the
[private OICR tunnel runbook](docs/OICR_CLUSTER.md) for both demos.

### Query the existing MultiHop-RAG tree

Querying the committed tree exercises retrieval without rebuilding the index:

```bash
./scripts/query_demo.sh "Which developments are compared across multiple reports?"
```

Equivalent direct invocation:

```bash
treequest \
  --tree data/multihop_rag_demo/corpus_tree.json \
  --ollama-url "$TREEQUEST_OLLAMA_URL" \
  --model "$TREEQUEST_MODEL" \
  --mode thorough \
  "Your question"
```

The JSON response contains the answer, source identifiers, elapsed time, model
calls, and operating mode.

### Build a new MultiHop-RAG tree

The build workflow writes to a fresh cache path and never overwrites the committed
tree:

```bash
./scripts/build_multihop_demo.sh
```

Query the new tree:

```bash
treequest \
  --tree experiments/multihop_rag/tree_cache/corpus_tree.json \
  --mode thorough \
  "Your question"
```

Tree construction is resumable through parse and node caches. Use `tmux` or a
cluster scheduler so laptop sleep or a network disconnect cannot interrupt a
remote run. See [Remote compute](docs/REMOTE_COMPUTE.md).

## How TreeQuest retrieves evidence

1. **Build the summarization tree.** Each node describes its children while
   preserving the path from corpus to folder, document, section, and passage.
2. **Traverse and retain.** The model scores visible branches, descends into the
   strongest candidate, and retains relevant passages.
3. **Decide scope.** When evidence is reached, the controller can retain the
   passage, its section, or its document according to the question.
4. **Sweep locally.** Other high-scoring sections from the same document can be
   read while unrelated sections are skipped.
5. **Check sufficiency.** A gate asks whether the retained evidence answers the
   question completely.
6. **Answer or revisit the frontier.** If evidence is incomplete, TreeQuest jumps
   to the best unexplored node at any depth and continues within budget.

## Reproducing the public evaluation

The artifact includes the frozen question sample, exact benchmark runner,
official MultiHop-RAG evaluator adapter, aggregate outputs, and pinned upstream
evaluator:

```bash
cd experiments/multihop_rag
python official_multihop_eval.py --help
```

The exact executed runner is preserved at
`reference/evaluated_v0/benchmark_treequest_public_frozen_v0.py`. The runnable
experiment copy changes only runtime path, model, and endpoint configuration.
Read [the experiment guide](experiments/multihop_rag/README.md) before launching
a full run. It is expensive, and each output must use a new versioned path.

See [RESULTS.md](RESULTS.md) for metrics, uncertainty, controls, and limitations.
See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the artifact checklist.

## Advanced settings

### Search modes

- `thorough`: the primary research operating point, with a larger traversal
  budget and full evidence checks.
- `quick`: a lower-cost sensitivity setting with a smaller search budget.

### Runtime configuration

```bash
export TREEQUEST_OLLAMA_URL=http://127.0.0.1:11434
export TREEQUEST_MODEL=gpt-oss:120b
treequest --help
```

Do not edit a frozen result file in place. Use a new output path for every run so
earlier experiments remain recoverable.

## Evaluated implementation and vector boundary

Reported experiments use **evaluated-v0**, preserved as an immutable source
snapshot. The modular package is a hardened release port with bounded retries,
queue hygiene, tree validation, and a lexical-only default for over-wide
candidate presentation. Results are never retrospectively relabeled across
these versions.

TreeQuest does not use a dense index to retrieve evidence: the LLM scores tree
branches, and evidence is reached through navigation. For scientific precision,
evaluated-v0 used local embeddings only to order names displayed inside an
over-wide `contains:` preview. The modular release defaults
`max_embed_per_decision=0` and uses lexical ordering. The evaluated method is
therefore described as **LLM-routed tree retrieval without dense-vector evidence
retrieval**, not as having no vector computation anywhere.

## Repository layout

```text
src/treequest/                 modular controller and CLI
scripts/                       query and tree-build launchers
data/multihop_rag_demo/        committed public demonstration tree
experiments/multihop_rag/      frozen sample, runners, evaluator, public outputs
reference/evaluated_v0/        immutable evaluated source snapshots
docs/                          architecture and remote-compute guidance
tests/                         synthetic unit and controller tests
```

## Data, privacy, and licensing

MultiHop-RAG is licensed under ODC-BY. TreeQuest source is GPL-3.0-or-later. See
[DATA_POLICY.md](DATA_POLICY.md) and [SECURITY.md](SECURITY.md).

Never commit private corpora, trees, questions, answers, traces, credentials,
endpoints, or result files. For confidential collections, run both TreeQuest and
the open-weight model inside the approved organizational compute boundary.

## Next steps

Next steps can consider running the remaining public controls and evaluating
other systems like Psi-RAG as a separate external baseline. They would broaden
public-benchmark coverage and help position TreeQuest against a recent
vector-assisted retrieval system.

## Acknowledgements

TreeQuest grew from the TreeRAG project in Genome Informatics at the Ontario
Institute for Cancer Research, with work by Arjun Sharma, Jochen Weile, Kayla
Marsh, and Melanie Courtot. The project was supported by the University of
Toronto Data Sciences Institute and the Government of Ontario.

## Citation

Citation metadata is provided in [CITATION.cff](CITATION.cff). The manuscript is
intentionally excluded from this repository before publication.
