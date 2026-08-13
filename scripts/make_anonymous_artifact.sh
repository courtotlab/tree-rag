#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  printf 'usage: %s /absolute/new/destination\n' "$0" >&2
  exit 64
fi

source_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="$1"

if [[ "$destination" != /* ]]; then
  printf 'destination must be absolute\n' >&2
  exit 64
fi
if [[ -e "$destination" ]]; then
  printf 'refusing existing destination: %s\n' "$destination" >&2
  exit 73
fi

mkdir -p "$destination"
rsync -a --safe-links \
  --exclude='.git/' \
  --exclude='.env' \
  --exclude='.venv/' \
  --exclude='.DS_Store' \
  --exclude='__pycache__/' \
  --exclude='*.py[cod]' \
  --exclude='.pytest_cache/' \
  --exclude='.ruff_cache/' \
  --exclude='tree_cache/' \
  --exclude='folders/' \
  --exclude='documents/' \
  --exclude='input/' \
  --exclude='results/' \
  --exclude='logs/' \
  --exclude='failure_logs/' \
  --exclude='*.jsonl' \
  --exclude='corpus_tree*.json' \
  --exclude='questions*.json' \
  --exclude='qms_answers*.json' \
  --exclude='benchmark_report*.json' \
  --exclude='*.pre_*' \
  "$source_root/" "$destination/"

cp "$source_root/packaging/pyproject.anonymous.toml" "$destination/pyproject.toml"
cp "$source_root/packaging/CITATION.anonymous.cff" "$destination/CITATION.cff"

cat > "$destination/ARTIFACT_BUILD.txt" <<META
artifact: TreeQuest anonymous review artifact
built_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)
source: sanitized working-tree snapshot
identity_metadata: anonymous
generated_or_private_data: excluded
META

for forbidden in 'asharma' 'courtotlab' 'Project Algorithm' '/Users/' '@oicr.on.ca'; do
  if rg -l -i --hidden \
      --glob '!.git/**' \
      --glob '!MANIFEST.sha256' \
      --fixed-strings "$forbidden" "$destination" >/dev/null; then
    printf 'anonymization check failed for forbidden marker: %s\n' "$forbidden" >&2
    exit 65
  fi
done

(
  cd "$destination"
  find . -type f ! -name MANIFEST.sha256 -print0 \
    | LC_ALL=C sort -z \
    | xargs -0 shasum -a 256 > MANIFEST.sha256
)

printf 'anonymous artifact created at %s\n' "$destination"
