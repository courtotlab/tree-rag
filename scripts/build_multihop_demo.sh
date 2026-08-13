#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="$ROOT/experiments/multihop_rag"
export TREERAG_OLLAMA_URL="${TREERAG_OLLAMA_URL:-http://127.0.0.1:11528}"
export TREERAG_MODEL="${TREERAG_MODEL:-gpt-oss:120b}"
export TREERAG_DOCS_ROOT="${TREERAG_DOCS_ROOT:-folders}"
export TREERAG_CACHE_DIR="${TREERAG_CACHE_DIR:-tree_cache}"

cd "$EXP"
python download_data.py
python build_tree_public.py
