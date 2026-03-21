#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
grammar_dir="$repo_root/tools/tree-sitter-simd"

resolve_tree_sitter_bin() {
  if [[ -n "${TREE_SITTER_BIN:-}" ]]; then
    printf '%s\n' "$TREE_SITTER_BIN"
    return 0
  fi
  if command -v tree-sitter >/dev/null 2>&1; then
    command -v tree-sitter
    return 0
  fi
  local vendored="$repo_root/.tools/tree-sitter-cli/bin/tree-sitter"
  if [[ ! -x "$vendored" ]]; then
    cargo install tree-sitter-cli --root "$repo_root/.tools/tree-sitter-cli"
  fi
  printf '%s\n' "$vendored"
}

tree_sitter_bin="$(resolve_tree_sitter_bin)"
if [[ ! -x "$tree_sitter_bin" ]]; then
  echo "tree-sitter binary not found at $tree_sitter_bin" >&2
  exit 1
fi

cd "$grammar_dir"
"$tree_sitter_bin" generate
"$tree_sitter_bin" test
TREE_SITTER_BIN="$tree_sitter_bin" ./scripts/query-snapshots.sh
TREE_SITTER_BIN="$tree_sitter_bin" ./scripts/highlight-snapshots.sh
TREE_SITTER_BIN="$tree_sitter_bin" ./scripts/rainbow-snapshots.sh
TREE_SITTER_BIN="$tree_sitter_bin" ./scripts/example-highlight-snapshots.sh
TREE_SITTER_BIN="$tree_sitter_bin" ./scripts/example-rainbow-brackets.sh
TREE_SITTER_BIN="$tree_sitter_bin" ./scripts/example-highlight-assertions.sh
