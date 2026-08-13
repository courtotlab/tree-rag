#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="$ROOT/experiments/multihop_rag"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

export TREEQUEST_OLLAMA_URL="${TREEQUEST_OLLAMA_URL:-http://127.0.0.1:11528}"
export TREEQUEST_MODEL="${TREEQUEST_MODEL:-gpt-oss:120b}"
export TREEQUEST_CACHE_DIR="${TREEQUEST_CACHE_DIR:-$ROOT/data/multihop_rag_demo}"
export IMPROVE_REPORT_PATH="${TREEQUEST_BENCHMARK_REPORT:-$EXP/results/treequest_public_rerun_${STAMP}.json}"
export TREEQUEST_DOSSIER_PATH="${TREEQUEST_BENCHMARK_DOSSIER:-$EXP/results/treequest_failure_dossiers_${STAMP}.json}"

mkdir -p "$(dirname "$IMPROVE_REPORT_PATH")" "$(dirname "$TREEQUEST_DOSSIER_PATH")"

printf 'Tree: %s/corpus_tree.json\n' "$TREEQUEST_CACHE_DIR"
printf 'Benchmark responses: %s\n' "$IMPROVE_REPORT_PATH"
printf 'Failure dossiers: %s\n' "$TREEQUEST_DOSSIER_PATH"

cd "$EXP"
exec python benchmark_treequest_public.py
