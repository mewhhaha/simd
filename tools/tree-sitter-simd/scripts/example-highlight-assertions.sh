#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo_root="$(cd "$root_dir/../.." && pwd)"
examples_dir="$repo_root/examples"
query_path="$root_dir/queries/highlights.scm"

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
  captures="$("$tree_sitter_bin" query --captures --config-path "$config_path" "$query_path" "$source_path" | sed '1d')"
  base_name="$(basename "$source_path")"

  signature_line="$(awk '/^main[[:space:]]*:/ { print NR; exit }' "$source_path")"
  if [[ -n "$signature_line" ]]; then
    signature_line=$((signature_line - 1))
    if ! grep -Eq "capture: .* - function, start: \($signature_line, 0\),.*text: \`main\`" <<<"$captures"; then
      status=1
      echo "missing @function capture for signature 'main' in $base_name" >&2
    fi
  fi

  clause_line="$(awk '/^main[[:space:]].*=/{ print NR; exit }' "$source_path")"
  if [[ -n "$clause_line" ]]; then
    clause_line=$((clause_line - 1))
    if ! grep -Eq "capture: .* - function, start: \($clause_line, 0\),.*text: \`main\`" <<<"$captures"; then
      status=1
      echo "missing @function capture for clause 'main' in $base_name" >&2
    fi
  fi

  if grep -q '{' "$source_path"; then
    if ! grep -q ' - property,' <<<"$captures"; then
      status=1
      echo "missing @property captures for record-heavy example $base_name" >&2
    fi
  fi

  if grep -Eq '(\+|\*|/|%|<|>|==|<=|>=)' "$source_path"; then
    if ! grep -q ' - operator,' <<<"$captures"; then
      status=1
      echo "missing @operator capture in $base_name" >&2
    fi
  fi
done < <(find "$examples_dir" -maxdepth 1 -type f -name '*.simd' -print0 | sort -z)

exit "$status"
