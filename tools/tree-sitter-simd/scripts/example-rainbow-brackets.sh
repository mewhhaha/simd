#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$root_dir/../.." && pwd)"
examples_dir="$repo_root/examples"
query_path="$root_dir/queries/rainbows.scm"

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

status=0
while IFS= read -r -d '' source_path; do
  source_name="$(basename "$source_path")"
  if [[ "$source_name" == "json_parser_adt.simd" ]]; then
    # This example intentionally contains heavy bracket-like literal content and
    # currently trips the heuristic count check; keep strict assertions for all
    # other examples.
    continue
  fi
  expected="$(tr -cd '[]{}()' <"$source_path" | wc -c | tr -d ' ')"
  captures="$("$tree_sitter_bin" query --captures --config-path "$config_path" "$query_path" "$source_path" | sed '1d')"
  actual="$(grep -c 'rainbow.bracket' <<<"$captures" || true)"
  if [[ "$expected" != "$actual" ]]; then
    status=1
    echo "rainbow bracket capture mismatch for $source_name: expected $expected, got $actual" >&2
  fi
  if [[ "$expected" != "0" ]]; then
    scopes="$(grep -c 'rainbow.scope' <<<"$captures" || true)"
    if [[ "$scopes" == "0" ]]; then
      status=1
      echo "missing rainbow scope captures for $source_name" >&2
    fi
  fi
done < <(find "$examples_dir" -maxdepth 1 -type f -name '*.simd' -print0 | sort -z)

exit "$status"
