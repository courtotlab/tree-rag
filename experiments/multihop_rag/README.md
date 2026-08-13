# MultiHop-RAG public experiment

This directory reproduces the completed evaluated-v0 public study. It contains only
public ODC-BY MultiHop-RAG inputs and public model outputs.

## Completed artifacts

- Frozen sample: `input/sample_200.json` (50 questions per type, seed 20260806).
- Exact executed TreeQuest source:
  `../../reference/evaluated_v0/benchmark_treequest_public_frozen_v0.py`.
- Immutable full TreeQuest report:
  `results/treequest_public_frozen_v0_200_20260811.json`.
- Official aggregate:
  `results/treequest_official_multihop_eval_v2_20260813.json`.
- Pinned upstream evaluator: `vendor/multihop_rag/`.
- Build record: `results/build_metadata_20260810.txt`.

Both completed controls have 200/200 answers: collapsed-tree retrieval and an oracle
gold-context diagnostic. The flat-hybrid system also has 200/200 answers. Exact ranked
flat passages cannot be recovered from the title-level stored artifact, so official
flat retrieval metrics are unavailable; official flat QA remains available.

## Rebuild the public hierarchy

Run on a compute host with Ollama, preferably inside `tmux`:

```bash
python download_data.py
export TREEQUEST_OLLAMA_URL=http://127.0.0.1:11434
export TREEQUEST_MODEL=gpt-oss:120b
python build_tree_public.py
```

This writes to this directory's `tree_cache/`; it never replaces the committed demo
tree at the repository root. Parse and node caches make the build resumable.

## Run the evaluated-v0 benchmark

The runnable benchmark uses environment-configured paths and endpoint:

```bash
export TREEQUEST_OLLAMA_URL=http://127.0.0.1:11434
export TREEQUEST_MODEL=gpt-oss:120b
export IMPROVE_REPORT_PATH=results/treequest_public_rerun_$(date -u +%Y%m%dT%H%M%SZ).json
python benchmark_treequest_public.py
```

Do not run the immutable file in `reference/` directly; it preserves the original
machine-specific constants solely for provenance. Never overwrite the released report.

## Official evaluation

The adapter pins and hashes the official `retrieval_evaluate.py` and `qa_evaluate.py`.
It maps TreeQuest's first-visit `read_file` events to ordered public passages, excludes
navigation-only nodes, preserves first-visit order, and reports mapping coverage.

```bash
python official_multihop_eval.py --help
```

Report official retrieval, official QA, and the matched semantic judge separately.
Do not call the semantic judge retrieval accuracy, and do not compare this balanced
sample numerically to a full natural-prevalence leaderboard without qualification.

## Signal boundary

The evaluated-v0 runner makes branch choices with LLM scores and does not retrieve
evidence from a dense index. It does use embeddings to order names displayed in an
over-wide candidate preview. The modular release disables that auxiliary ordering by
default. This distinction must remain explicit in papers and derivative results.
