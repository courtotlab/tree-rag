#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUESTION="${*:-Which developments are compared across multiple reports?}"
TREE="${TREERAG_TREE_PATH:-$ROOT/data/multihop_rag_demo/corpus_tree.json}"
export TREERAG_OLLAMA_URL="${TREERAG_OLLAMA_URL:-http://127.0.0.1:11528}"
export TREERAG_MODEL="${TREERAG_MODEL:-gpt-oss:120b}"

if [[ ! -s "$TREE" ]]; then
  printf 'Public demo tree not found: %s\n' "$TREE" >&2
  exit 66
fi

exec python "$ROOT/scripts/query_with_progress.py" \
  --tree "$TREE" --ollama-url "$TREERAG_OLLAMA_URL" \
  --model "$TREERAG_MODEL" --mode "${TREERAG_MODE:-thorough}" "$QUESTION"
