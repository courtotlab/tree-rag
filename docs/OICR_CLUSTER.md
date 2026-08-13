# OICR Ollama SSH-tunnel setup (private operations)

This document belongs only in the private Courtot Lab repository. TreeRAG runs on the
workstation; the OICR server supplies only the approved Ollama endpoint. The corpus,
repository, caches, and TreeRAG process remain local.

## Prerequisites

- Connect the workstation to the OICR VPN.
- Keep the private key at `~/.ssh/id_ed25519` with mode `600`.
- Install this repository and Python 3.11 or 3.12 on the workstation.
- Do not copy credentials, private corpora, or private trees to the server.

## Terminal 1: open local port 11528

Keep this command running for the duration of either demo:

```bash
ssh -NT \
  -o ExitOnForwardFailure=yes \
  -o ServerAliveInterval=60 \
  -o ServerAliveCountMax=3 \
  -o IdentitiesOnly=yes \
  -i "$HOME/.ssh/id_ed25519" \
  -L 127.0.0.1:11528:172.17.0.1:11434 \
  asharma@ollama.res.oicr.on.ca
```

A successful tunnel prints nothing and keeps the terminal occupied. Stop it with
`Ctrl-C` only after the local TreeRAG command has finished. The loopback binding
prevents other machines from connecting to the forwarded port.

## Terminal 2: install and verify locally

Run these commands on the workstation, not inside an OICR SSH shell:

```bash
cd /path/to/tree-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[build,test]'

export TREERAG_OLLAMA_URL=http://127.0.0.1:11528
export TREERAG_MODEL=gpt-oss:120b
curl -fsS "$TREERAG_OLLAMA_URL/api/version"
curl -fsS "$TREERAG_OLLAMA_URL/api/tags"
```

Do not run `ollama pull` on the workstation. Models are managed on the approved OICR
Ollama host.

## Demo 1: start the MultiHop-RAG tree builder, observe progress, then stop

This is a capability smoke test, not a complete rebuild. The frozen public builder uses
`gpt-oss:120b` for text summaries and bottom-up combines. MultiHop-RAG is text-only, so
its existing `DESCRIBE_IMAGES=False` setting correctly makes no `gemma3:27b` calls.
Do not change that setting.

```bash
cd /path/to/tree-rag
source .venv/bin/activate

export TREERAG_OLLAMA_URL=http://127.0.0.1:11528
export TREERAG_MODEL=gpt-oss:120b
export TREERAG_BUILD_WORKERS=4
RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$HOME/treerag-runs/build-smoke-$RUN_ID"
mkdir -p "$RUN_ROOT"
export TREERAG_CACHE_DIR="$RUN_ROOT/tree_cache"

./scripts/build_multihop_demo.sh
```

The script first downloads and parses the public articles. Wait until its live `tqdm`
progress bar reports completed calls, rate, and ETA. After several model calls complete,
press `Ctrl-C` once. The versioned partial cache remains available and neither the
committed demonstration tree nor any previous run is overwritten.

## Demo 2: query the already-built public tree

Leave Terminal 1's tunnel running. In Terminal 2:

```bash
cd /path/to/tree-rag
source .venv/bin/activate

export TREERAG_OLLAMA_URL=http://127.0.0.1:11528
export TREERAG_MODEL=gpt-oss:120b
export TREERAG_MODE=thorough
export TREERAG_TREE_PATH="$PWD/data/multihop_rag_demo/corpus_tree.json"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$HOME/treerag-runs/query-$RUN_ID"
mkdir -p "$RUN_ROOT"

./scripts/query_demo.sh \
  "Which developments are compared across multiple reports?" \
  2>&1 | tee "$RUN_ROOT/query.log"

printf 'Query output: %s\n' "$RUN_ROOT/query.log"
```

The local agent traverses the committed public tree, retains evidence, tests
sufficiency, performs bounded recovery when needed, and sends model calls through local
port `11528` to OICR `gpt-oss:120b` for traversal and final answer synthesis.

## Troubleshooting

If port `11528` is already occupied:

```bash
lsof -nP -iTCP:11528 -sTCP:LISTEN
```

If the tunnel exits immediately, confirm the VPN is connected and test SSH directly:

```bash
ssh -o IdentitiesOnly=yes -i "$HOME/.ssh/id_ed25519" asharma@ollama.res.oicr.on.ca
```

If the tunnel is running but the API check fails, verify that the remote Ollama service
is reachable from the SSH host at `172.17.0.1:11434`; do not expose that port directly.
