#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="$ROOT/experiments/multihop_rag"
export TREEQUEST_OLLAMA_URL="${TREEQUEST_OLLAMA_URL:-http://127.0.0.1:11528}"
export TREEQUEST_MODEL="${TREEQUEST_MODEL:-gpt-oss:120b}"
export TREEQUEST_DOCS_ROOT="${TREEQUEST_DOCS_ROOT:-folders}"
export TREEQUEST_CACHE_DIR="${TREEQUEST_CACHE_DIR:-tree_cache}"

cd "$EXP"
python download_data.py
python build_tree_public.py
