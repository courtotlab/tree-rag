#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
QUESTION="${*:-Which developments are compared across multiple reports?}"
TREE="${TREEQUEST_TREE_PATH:-$ROOT/data/multihop_rag_demo/corpus_tree.json}"
export TREEQUEST_OLLAMA_URL="${TREEQUEST_OLLAMA_URL:-http://127.0.0.1:11528}"
export TREEQUEST_MODEL="${TREEQUEST_MODEL:-gpt-oss:120b}"

if [[ ! -s "$TREE" ]]; then
  printf 'Public demo tree not found: %s\n' "$TREE" >&2
  exit 66
fi

exec treequest --tree "$TREE" --ollama-url "$TREEQUEST_OLLAMA_URL" \
  --model "$TREEQUEST_MODEL" --mode "${TREEQUEST_MODE:-thorough}" "$QUESTION"
