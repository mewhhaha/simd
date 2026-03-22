#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
helix_config_dir="${HELIX_CONFIG:-$HOME/.config/helix}"
helix_queries_dir="$helix_config_dir/runtime/queries/simd"
helix_grammars_dir="$helix_config_dir/runtime/grammars"
helix_languages_toml="$helix_config_dir/languages.toml"
grammar_dir="$repo_root/tools/tree-sitter-simd"
parser_c="$grammar_dir/src/parser.c"

if [[ ! -f "$parser_c" ]]; then
  echo "missing parser source at $parser_c (run tree-sitter generate first)" >&2
  exit 1
fi

if ! command -v cc >/dev/null 2>&1; then
  echo "missing 'cc' compiler needed to build Helix tree-sitter grammar" >&2
  exit 1
fi

case "$(uname -s)" in
  Darwin)
    grammar_ext="dylib"
    compile_cmd=(cc -O3 -fPIC -I"$grammar_dir/src" -dynamiclib "$parser_c")
    ;;
  *)
    grammar_ext="so"
    compile_cmd=(cc -O3 -fPIC -I"$grammar_dir/src" -shared "$parser_c")
    ;;
esac

mkdir -p "$helix_queries_dir"
cp "$repo_root/tools/tree-sitter-simd/queries/highlights.scm" \
  "$helix_queries_dir/highlights.scm"
cp "$repo_root/tools/tree-sitter-simd/queries/rainbows.scm" \
  "$helix_queries_dir/rainbows.scm"

mkdir -p "$helix_grammars_dir"
"${compile_cmd[@]}" -o "$helix_grammars_dir/simd.$grammar_ext"

mkdir -p "$(dirname "$helix_languages_toml")"
if [[ ! -f "$helix_languages_toml" ]] || ! grep -q 'name = "simd"' "$helix_languages_toml"; then
  {
    if [[ -f "$helix_languages_toml" ]]; then
      printf '\n'
    fi
    cat "$repo_root/editors/helix/languages.toml"
  } >>"$helix_languages_toml"
fi

echo "installed simd language config, query, and grammar runtime into $helix_config_dir"
