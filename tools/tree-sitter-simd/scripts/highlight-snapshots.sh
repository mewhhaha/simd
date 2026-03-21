#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${TREE_SITTER_BIN:-}" ]]; then
  tree_sitter_bin="${TREE_SITTER_BIN}"
elif command -v tree-sitter >/dev/null 2>&1; then
  tree_sitter_bin="tree-sitter"
elif [[ -x "/tmp/tree-sitter-cli/bin/tree-sitter" ]]; then
  tree_sitter_bin="/tmp/tree-sitter-cli/bin/tree-sitter"
else
  echo "tree-sitter binary not found; set TREE_SITTER_BIN or install tree-sitter" >&2
  exit 1
fi
config_path="${TREE_SITTER_CONFIG:-$(mktemp -t tree-sitter-simd-config.XXXXXX.json)}"
query_path="$root_dir/queries/highlights.scm"
query_dir="$root_dir/test/highlight_snapshots"

cleanup() {
  if [[ -z "${TREE_SITTER_CONFIG:-}" ]]; then
    rm -f "$config_path"
  fi
}
trap cleanup EXIT

cat >"$config_path" <<EOF
{
  "parser-directories": ["$(dirname "$root_dir")"]
}
EOF

update=false
if [[ "${1:-}" == "--update" ]]; then
  update=true
fi

status=0
while IFS= read -r -d '' source_path; do
  snapshot_path="${source_path%.simd}.highlights.txt"
  actual="$("$tree_sitter_bin" query --captures --config-path "$config_path" "$query_path" "$source_path")"
  if $update; then
    printf '%s\n' "$actual" >"$snapshot_path"
    continue
  fi
  expected="$(cat "$snapshot_path")"
  if [[ "$actual" != "$expected" ]]; then
    status=1
    printf 'highlight snapshot mismatch for %s\n' "${source_path##*/}" >&2
    printf 'expected: %s\n' "$snapshot_path" >&2
    printf 'actual output:\n%s\n' "$actual" >&2
  fi
done < <(find "$query_dir" -maxdepth 1 -type f -name '*.simd' -print0 | sort -z)

exit "$status"
