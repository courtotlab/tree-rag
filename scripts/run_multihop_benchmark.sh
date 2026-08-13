#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXP="$ROOT/experiments/multihop_rag"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

export TREERAG_OLLAMA_URL="${TREERAG_OLLAMA_URL:-http://127.0.0.1:11528}"
export TREERAG_MODEL="${TREERAG_MODEL:-gpt-oss:120b}"
export TREERAG_CACHE_DIR="${TREERAG_CACHE_DIR:-$ROOT/data/multihop_rag_demo}"
export IMPROVE_REPORT_PATH="${TREERAG_BENCHMARK_REPORT:-$EXP/results/treerag_public_rerun_${STAMP}.json}"
export TREERAG_DOSSIER_PATH="${TREERAG_BENCHMARK_DOSSIER:-$EXP/results/treerag_failure_dossiers_${STAMP}.json}"

mkdir -p "$(dirname "$IMPROVE_REPORT_PATH")" "$(dirname "$TREERAG_DOSSIER_PATH")"

printf 'Tree: %s/corpus_tree.json\n' "$TREERAG_CACHE_DIR"
printf 'Benchmark responses: %s\n' "$IMPROVE_REPORT_PATH"
printf 'Failure dossiers: %s\n' "$TREERAG_DOSSIER_PATH"

cd "$EXP"
exec python benchmark_treerag_public.py
