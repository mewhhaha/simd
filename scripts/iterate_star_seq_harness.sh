#!/usr/bin/env sh
set -eu

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

mode="${1:-quick}"
if [ "$mode" != "quick" ] && [ "$mode" != "full" ]; then
  echo "usage: $0 [quick|full]" >&2
  exit 2
fi

artifact_dir=".tmp/harness"
mkdir -p "$artifact_dir"
run_id="$(date -u +"%Y%m%dT%H%M%SZ")"
log_file="$artifact_dir/star_seq_harness_$run_id.ndjson"
summary_file="$artifact_dir/star_seq_harness_$run_id.summary.txt"
last_log="$artifact_dir/star_seq_last.ndjson"
last_summary="$artifact_dir/star_seq_last.summary.txt"

now_ms() {
  if ts="$(date +%s%3N 2>/dev/null)"; then
    printf '%s\n' "$ts"
    return
  fi
  s="$(date +%s)"
  printf '%s\n' "$((s * 1000))"
}

log_step() {
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf '{"ts":"%s","event":"step","name":"%s","status":"%s","duration_ms":%s}\n' "$ts" "$1" "$2" "$3" >> "$log_file"
}

run_step() {
  name="$1"
  shift
  start="$(now_ms)"
  if "$@"; then
    status="ok"
    code=0
  else
    status="fail"
    code=$?
  fi
  end="$(now_ms)"
  duration_ms="$((end - start))"
  log_step "$name" "$status" "$duration_ms"
  if [ "$code" -ne 0 ]; then
    cp "$log_file" "$last_log"
    cp "$summary_file" "$last_summary" 2>/dev/null || true
    exit "$code"
  fi
}

run_capture_expect_error() {
  name="$1"
  out_file="$2"
  shift 2
  start="$(now_ms)"
  if "$@" >"$out_file" 2>&1; then
    status="unexpected-success"
    code=1
  else
    status="ok"
    code=0
  fi
  end="$(now_ms)"
  duration_ms="$((end - start))"
  log_step "$name" "$status" "$duration_ms"
  if [ "$code" -ne 0 ]; then
    cp "$log_file" "$last_log"
    cp "$summary_file" "$last_summary" 2>/dev/null || true
    exit "$code"
  fi
}

printf '[star-seq-harness] run_id=%s mode=%s\n' "$run_id" "$mode"
printf 'run_id=%s\nmode=%s\n' "$run_id" "$mode" > "$summary_file"

echo "[star-seq-harness] language and evaluator checks"
run_step "test.star_seq.parser_accepts_star_seq_types_and_literals" cargo test -q parser_accepts_star_seq_types_and_literals
run_step "test.star_seq.parser_rejects_reserved_borrow_seq_syntax" cargo test -q parser_rejects_reserved_borrow_seq_syntax
run_step "test.star_seq.star_seq_example_runs" cargo test -q star_seq_example_runs
run_step "test.star_seq.star_seq_recursive_enum_example_runs" cargo test -q star_seq_recursive_enum_example_runs
run_step "test.star_seq.star_seq_runtime_boundary_uses_json_arrays" cargo test -q star_seq_runtime_boundary_uses_json_arrays
run_step "test.star_seq.star_seq_wasm_scalar_consumer_runs" cargo test -q star_seq_wasm_scalar_consumer_runs
run_step "test.star_seq.star_seq_wasm_scalar_result_runs" cargo test -q star_seq_wasm_scalar_result_runs
run_step "test.star_seq.star_seq_wasm_scalar_enum_field_runs" cargo test -q star_seq_wasm_scalar_enum_field_runs
run_step "test.star_seq.star_seq_wasm_prepared_scalar_result_runs" cargo test -q star_seq_wasm_prepared_scalar_result_runs
run_step "test.star_seq.star_seq_recursive_enum_wasm_runs" cargo test -q star_seq_recursive_enum_wasm_runs
run_step "test.star_seq.star_seq_recursive_enum_wasm_prepared_runs" cargo test -q star_seq_recursive_enum_wasm_prepared_runs

echo "[star-seq-harness] cli smoke checks"
run_step "cli.star_seq_basics.run" cargo run -q -- run examples/star_seq_basics.simd --main main --args '[[4,5,6]]'
run_step "cli.star_seq_tree.run" cargo run -q -- run examples/star_seq_tree.simd --main main --args '[]'
run_step "cli.star_seq_basics.main.run_wasm" cargo run -q -- run-wasm examples/star_seq_basics.simd --main main --args '[[4,5,6]]'
run_step "cli.star_seq_basics.main_boxed.run_wasm" cargo run -q -- run-wasm examples/star_seq_basics.simd --main main_boxed --args '[[4,5,6]]'
run_step "cli.star_seq_basics.sum_seq.run_wasm" cargo run -q -- run-wasm examples/star_seq_basics.simd --main sum_seq --args '[[4,5,6]]'
run_step "cli.star_seq_basics.main.run_wasm_prepared" cargo run -q -- run-wasm-prepared examples/star_seq_basics.simd --main main --args '[[4,5,6]]' --iters 2
run_step "cli.star_seq_basics.main_boxed.run_wasm_prepared" cargo run -q -- run-wasm-prepared examples/star_seq_basics.simd --main main_boxed --args '[[4,5,6]]' --iters 2
run_step "cli.star_seq_basics.sum_seq.run_wasm_prepared" cargo run -q -- run-wasm-prepared examples/star_seq_basics.simd --main sum_seq --args '[[4,5,6]]' --iters 2
run_step "cli.star_seq_tree.main.run_wasm" cargo run -q -- run-wasm examples/star_seq_tree.simd --main main --args '[]'
run_step "cli.star_seq_tree.main_tree.run_wasm" cargo run -q -- run-wasm examples/star_seq_tree.simd --main main_tree --args '[]'
run_step "cli.star_seq_tree.main.run_wasm_prepared" cargo run -q -- run-wasm-prepared examples/star_seq_tree.simd --main main --args '[]' --iters 2
run_step "cli.star_seq_tree.main_tree.run_wasm_prepared" cargo run -q -- run-wasm-prepared examples/star_seq_tree.simd --main main_tree --args '[]' --iters 2

if [ "$mode" = "full" ]; then
  echo "[star-seq-harness] full-mode extras"
  run_step "inspect_html" cargo run -q -- inspect-html --out docs/inspector.html
fi

cp "$log_file" "$last_log"
cp "$summary_file" "$last_summary"
printf '[star-seq-harness] ok summary=%s\n' "$summary_file"
