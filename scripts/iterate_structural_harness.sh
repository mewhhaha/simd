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
log_file="$artifact_dir/structural_harness_$run_id.ndjson"
last_log="$artifact_dir/last.ndjson"
summary_file="$artifact_dir/structural_harness_$run_id.summary.txt"
last_summary="$artifact_dir/last.summary.txt"

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

log_metric() {
  ts="$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
  printf '{"ts":"%s","event":"metric","name":"%s","value":%s}\n' "$ts" "$1" "$2" >> "$log_file"
}

read_kv() {
  file="$1"
  key="$2"
  awk -F= -v k="$key" '$1 == k { print substr($0, length(k) + 2); exit }' "$file"
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

run_step_quiet() {
  name="$1"
  shift
  start="$(now_ms)"
  if "$@" > /dev/null 2>&1; then
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

run_step_capture() {
  name="$1"
  out_file="$2"
  shift 2
  start="$(now_ms)"
  if "$@" > "$out_file" 2>&1; then
    status="ok"
    code=0
  else
    status="fail"
    code=$?
  fi
  end="$(now_ms)"
  duration_ms="$((end - start))"
  cat "$out_file"
  log_step "$name" "$status" "$duration_ms"
  if [ "$code" -ne 0 ]; then
    cp "$log_file" "$last_log"
    cp "$summary_file" "$last_summary" 2>/dev/null || true
    exit "$code"
  fi
}

measure_repeat_ms() {
  iters="$1"
  shift
  start="$(now_ms)"
  i=0
  while [ "$i" -lt "$iters" ]; do
    "$@" > /dev/null
    i=$((i + 1))
  done
  end="$(now_ms)"
  total_ms="$((end - start))"
  if [ "$iters" -gt 0 ]; then
    avg_ms="$((total_ms / iters))"
  else
    avg_ms=0
  fi
  printf '%s %s\n' "$total_ms" "$avg_ms"
}

printf '[harness] run_id=%s mode=%s\n' "$run_id" "$mode"
printf 'run_id=%s\nmode=%s\n' "$run_id" "$mode" > "$summary_file"

echo "[harness] structural lowering checks"
run_step "test.structural.recursive_enum_function_lowers_to_structural" cargo test -q recursive_enum_function_lowers_to_structural
run_step "test.structural.slice_pattern_function_lowers_to_structural" cargo test -q slice_pattern_function_lowers_to_structural
run_step "test.structural.lowered_and_checked_execution_agree" cargo test -q lowered_and_checked_execution_agree
run_step "test.structural.json_parser_helpers_do_not_fallback_and_form_structural_cluster" cargo test -q json_parser_helpers_do_not_fallback_and_form_structural_cluster
run_step "test.structural.structural_optimizer_reports_use_structural_loop_exec" cargo test -q structural_optimizer_reports_use_structural_loop_exec

echo "[harness] wasm recursion + parser checks"
run_step "test.wasm.wasm_matches_recursive_enum_list_length" cargo test -q wasm_matches_recursive_enum_list_length
run_step "test.wasm.wasm_matches_recursive_enum_tree_sum" cargo test -q wasm_matches_recursive_enum_tree_sum
run_step "test.wasm.wasm_json_parser_adt_example_runs" cargo test -q wasm_json_parser_adt_example_runs
run_step "test.wasm.wasm_json_parser_adt_string_leaf_chars_harness_runs" cargo test -q wasm_json_parser_adt_string_leaf_chars_harness_runs

echo "[harness] json parser perf (cli loop)"
json_iters=2
if [ "$mode" = "full" ]; then
  json_iters=5
fi
json_input='["[1,2,3,4,5,6,7,8,9,10]"]'
simd_bin="./target/debug/simd"
cargo build -q --bin simd > /dev/null
eval_cmd_main() {
  "$simd_bin" run examples/json_parser_adt.simd --main main --args "$json_input"
}
wasm_cmd_main() {
  "$simd_bin" run-wasm examples/json_parser_adt.simd --main main --args "$json_input"
}
prepared_iters=10
if [ "$mode" = "full" ]; then
  prepared_iters=50
fi
wasm_prepared_cmd_main() {
  "$simd_bin" run-wasm-prepared examples/json_parser_adt.simd --main main --args "$json_input" --iters "$prepared_iters"
}
run_step_quiet "json_perf.warmup.eval" eval_cmd_main
run_step_quiet "json_perf.warmup.wasm" wasm_cmd_main
run_step_quiet "json_perf.warmup.wasm_prepared" wasm_prepared_cmd_main
set -- $(measure_repeat_ms "$json_iters" eval_cmd_main)
eval_total_ms="$1"
eval_avg_ms="$2"
set -- $(measure_repeat_ms "$json_iters" wasm_cmd_main)
wasm_total_ms="$1"
wasm_avg_ms="$2"
set -- $(measure_repeat_ms "$json_iters" wasm_prepared_cmd_main)
wasm_prepared_total_ms="$1"
if [ "$json_iters" -gt 0 ] && [ "$prepared_iters" -gt 0 ]; then
  wasm_prepared_avg_ms="$((wasm_prepared_total_ms / json_iters / prepared_iters))"
else
  wasm_prepared_avg_ms=0
fi
if [ "$eval_avg_ms" -gt 0 ]; then
  wasm_vs_eval_milli="$((wasm_avg_ms * 1000 / eval_avg_ms))"
  wasm_prepared_vs_eval_milli="$((wasm_prepared_avg_ms * 1000 / eval_avg_ms))"
else
  wasm_vs_eval_milli=0
  wasm_prepared_vs_eval_milli=0
fi
printf '[harness] json parser perf: iters=%s eval_avg_ms=%s wasm_avg_ms=%s wasm_vs_eval_x1000=%s wasm_prepared_iters=%s wasm_prepared_avg_ms=%s wasm_prepared_vs_eval_x1000=%s\n' \
  "$json_iters" "$eval_avg_ms" "$wasm_avg_ms" "$wasm_vs_eval_milli" "$prepared_iters" "$wasm_prepared_avg_ms" "$wasm_prepared_vs_eval_milli"
printf 'json_iters=%s\njson_eval_total_ms=%s\njson_eval_avg_ms=%s\njson_wasm_total_ms=%s\njson_wasm_avg_ms=%s\njson_wasm_vs_eval_x1000=%s\njson_wasm_prepared_iters=%s\njson_wasm_prepared_total_ms=%s\njson_wasm_prepared_avg_ms=%s\njson_wasm_prepared_vs_eval_x1000=%s\n' \
  "$json_iters" "$eval_total_ms" "$eval_avg_ms" "$wasm_total_ms" "$wasm_avg_ms" "$wasm_vs_eval_milli" "$prepared_iters" "$wasm_prepared_total_ms" "$wasm_prepared_avg_ms" "$wasm_prepared_vs_eval_milli" >> "$summary_file"
log_metric "json_iters" "$json_iters"
log_metric "json_eval_total_ms" "$eval_total_ms"
log_metric "json_eval_avg_ms" "$eval_avg_ms"
log_metric "json_wasm_total_ms" "$wasm_total_ms"
log_metric "json_wasm_avg_ms" "$wasm_avg_ms"
log_metric "json_wasm_vs_eval_x1000" "$wasm_vs_eval_milli"
log_metric "json_wasm_prepared_iters" "$prepared_iters"
log_metric "json_wasm_prepared_total_ms" "$wasm_prepared_total_ms"
log_metric "json_wasm_prepared_avg_ms" "$wasm_prepared_avg_ms"
log_metric "json_wasm_prepared_vs_eval_x1000" "$wasm_prepared_vs_eval_milli"

echo "[harness] tracked nested JSON benchmark"
json_bench_iters=25
if [ "$mode" = "full" ]; then
  json_bench_iters=100
fi
json_bench_out="$artifact_dir/structural_harness_$run_id.json_bench.txt"
run_step_capture "json_perf.bench.nested" "$json_bench_out" \
  "$simd_bin" bench json-parser-adt-nested --size 4096 --iters "$json_bench_iters" --no-contract
json_bench_rust_line="$(grep -m1 '^  rust:' "$json_bench_out" || true)"
json_bench_wasm_hot_line="$(grep -m1 '^  wasm-hot:' "$json_bench_out" || true)"
json_bench_wasm_prepared_line="$(grep -m1 '^  wasm-prepared-hot:' "$json_bench_out" || true)"
printf '[harness] nested json bench: %s | %s | %s\n' \
  "$json_bench_rust_line" "$json_bench_wasm_hot_line" "$json_bench_wasm_prepared_line"
printf 'json_bench_iters=%s\njson_bench_rust_line=%s\njson_bench_wasm_hot_line=%s\njson_bench_wasm_prepared_line=%s\n' \
  "$json_bench_iters" "$json_bench_rust_line" "$json_bench_wasm_hot_line" "$json_bench_wasm_prepared_line" >> "$summary_file"

echo "[harness] evaluator stage profile"
eval_profile_out="$artifact_dir/structural_harness_$run_id.eval_profile.txt"
run_step_capture "json_perf.profile.eval" "$eval_profile_out" \
  "$simd_bin" run-profile examples/json_parser_adt.simd --main main --args "$json_input"
eval_profile_frontend_ms="$(read_kv "$eval_profile_out" frontend_ms)"
eval_profile_frontend_us="$(read_kv "$eval_profile_out" frontend_us)"
eval_profile_parse_args_ms="$(read_kv "$eval_profile_out" parse_args_ms)"
eval_profile_parse_args_us="$(read_kv "$eval_profile_out" parse_args_us)"
eval_profile_execute_ms="$(read_kv "$eval_profile_out" execute_ms)"
eval_profile_execute_us="$(read_kv "$eval_profile_out" execute_us)"
eval_profile_total_ms="$(read_kv "$eval_profile_out" total_ms)"
eval_profile_total_us="$(read_kv "$eval_profile_out" total_us)"
printf '[harness] evaluator profile: frontend_ms=%s frontend_us=%s parse_args_ms=%s parse_args_us=%s execute_ms=%s execute_us=%s total_ms=%s total_us=%s\n' \
  "$eval_profile_frontend_ms" "$eval_profile_frontend_us" "$eval_profile_parse_args_ms" "$eval_profile_parse_args_us" "$eval_profile_execute_ms" "$eval_profile_execute_us" "$eval_profile_total_ms" "$eval_profile_total_us"
printf 'json_eval_profile_frontend_ms=%s\njson_eval_profile_frontend_us=%s\njson_eval_profile_parse_args_ms=%s\njson_eval_profile_parse_args_us=%s\njson_eval_profile_execute_ms=%s\njson_eval_profile_execute_us=%s\njson_eval_profile_total_ms=%s\njson_eval_profile_total_us=%s\n' \
  "$eval_profile_frontend_ms" "$eval_profile_frontend_us" "$eval_profile_parse_args_ms" "$eval_profile_parse_args_us" "$eval_profile_execute_ms" "$eval_profile_execute_us" "$eval_profile_total_ms" "$eval_profile_total_us" >> "$summary_file"
log_metric "json_eval_profile_frontend_ms" "$eval_profile_frontend_ms"
log_metric "json_eval_profile_frontend_us" "$eval_profile_frontend_us"
log_metric "json_eval_profile_parse_args_ms" "$eval_profile_parse_args_ms"
log_metric "json_eval_profile_parse_args_us" "$eval_profile_parse_args_us"
log_metric "json_eval_profile_execute_ms" "$eval_profile_execute_ms"
log_metric "json_eval_profile_execute_us" "$eval_profile_execute_us"
log_metric "json_eval_profile_total_ms" "$eval_profile_total_ms"
log_metric "json_eval_profile_total_us" "$eval_profile_total_us"

echo "[harness] cold wasm stage profile"
profile_out="$artifact_dir/structural_harness_$run_id.profile.txt"
profile_cache_root="$repo_root/$artifact_dir/wasmtime-cache-$run_id"
run_step_capture "json_perf.profile.wasm_cold" "$profile_out" \
  env SIMD_WASMTIME_CACHE_ROOT="$profile_cache_root" \
  "$simd_bin" run-wasm-profile examples/json_parser_adt.simd --main main --args "$json_input"
profile_artifact_bytes="$(read_kv "$profile_out" artifact_bytes)"
profile_frontend_ms="$(read_kv "$profile_out" frontend_ms)"
profile_specialize_ms="$(read_kv "$profile_out" specialize_ms)"
profile_lambda_lower_ms="$(read_kv "$profile_out" lambda_lower_ms)"
profile_canonicalize_ms="$(read_kv "$profile_out" canonicalize_ms)"
profile_parse_args_ms="$(read_kv "$profile_out" parse_args_ms)"
profile_compile_artifact_ms="$(read_kv "$profile_out" compile_artifact_ms)"
profile_compile_artifact_us="$(read_kv "$profile_out" compile_artifact_us)"
profile_compile_enum_layouts_ms="$(read_kv "$profile_out" compile_enum_layouts_ms)"
profile_compile_enum_layouts_us="$(read_kv "$profile_out" compile_enum_layouts_us)"
profile_compile_plan_ms="$(read_kv "$profile_out" compile_plan_ms)"
profile_compile_plan_us="$(read_kv "$profile_out" compile_plan_us)"
profile_compile_normalize_ms="$(read_kv "$profile_out" compile_normalize_ms)"
profile_compile_normalize_us="$(read_kv "$profile_out" compile_normalize_us)"
profile_compile_lower_ms="$(read_kv "$profile_out" compile_lower_ms)"
profile_compile_lower_us="$(read_kv "$profile_out" compile_lower_us)"
profile_compile_group_ms="$(read_kv "$profile_out" compile_group_ms)"
profile_compile_group_us="$(read_kv "$profile_out" compile_group_us)"
profile_compile_intents_ms="$(read_kv "$profile_out" compile_intents_ms)"
profile_compile_intents_us="$(read_kv "$profile_out" compile_intents_us)"
profile_engine_ms="$(read_kv "$profile_out" engine_ms)"
profile_module_compile_ms="$(read_kv "$profile_out" module_compile_ms)"
profile_runtime_build_ms="$(read_kv "$profile_out" runtime_build_ms)"
profile_execute_ms="$(read_kv "$profile_out" execute_ms)"
profile_total_ms="$(read_kv "$profile_out" total_ms)"
profile_total_us="$(read_kv "$profile_out" total_us)"
printf '[harness] cold wasm profile: artifact_bytes=%s frontend_ms=%s specialize_ms=%s lambda_lower_ms=%s canonicalize_ms=%s parse_args_ms=%s compile_artifact_ms=%s compile_artifact_us=%s compile_enum_layouts_ms=%s compile_enum_layouts_us=%s compile_plan_ms=%s compile_plan_us=%s compile_normalize_ms=%s compile_normalize_us=%s compile_lower_ms=%s compile_lower_us=%s compile_group_ms=%s compile_group_us=%s compile_intents_ms=%s compile_intents_us=%s engine_ms=%s module_compile_ms=%s runtime_build_ms=%s execute_ms=%s total_ms=%s total_us=%s\n' \
  "$profile_artifact_bytes" "$profile_frontend_ms" "$profile_specialize_ms" "$profile_lambda_lower_ms" "$profile_canonicalize_ms" "$profile_parse_args_ms" "$profile_compile_artifact_ms" "$profile_compile_artifact_us" "$profile_compile_enum_layouts_ms" "$profile_compile_enum_layouts_us" "$profile_compile_plan_ms" "$profile_compile_plan_us" "$profile_compile_normalize_ms" "$profile_compile_normalize_us" "$profile_compile_lower_ms" "$profile_compile_lower_us" "$profile_compile_group_ms" "$profile_compile_group_us" "$profile_compile_intents_ms" "$profile_compile_intents_us" "$profile_engine_ms" "$profile_module_compile_ms" "$profile_runtime_build_ms" "$profile_execute_ms" "$profile_total_ms" "$profile_total_us"
profile_cached_out="$artifact_dir/structural_harness_$run_id.profile_cached.txt"
run_step_capture "json_perf.profile.wasm_cached" "$profile_cached_out" \
  env SIMD_WASMTIME_CACHE_ROOT="$profile_cache_root" \
  "$simd_bin" run-wasm-profile examples/json_parser_adt.simd --main main --args "$json_input"
profile_cached_frontend_ms="$(read_kv "$profile_cached_out" frontend_ms)"
profile_cached_specialize_ms="$(read_kv "$profile_cached_out" specialize_ms)"
profile_cached_lambda_lower_ms="$(read_kv "$profile_cached_out" lambda_lower_ms)"
profile_cached_canonicalize_ms="$(read_kv "$profile_cached_out" canonicalize_ms)"
profile_cached_parse_args_ms="$(read_kv "$profile_cached_out" parse_args_ms)"
profile_cached_compile_artifact_ms="$(read_kv "$profile_cached_out" compile_artifact_ms)"
profile_cached_compile_artifact_us="$(read_kv "$profile_cached_out" compile_artifact_us)"
profile_cached_compile_enum_layouts_ms="$(read_kv "$profile_cached_out" compile_enum_layouts_ms)"
profile_cached_compile_enum_layouts_us="$(read_kv "$profile_cached_out" compile_enum_layouts_us)"
profile_cached_compile_plan_ms="$(read_kv "$profile_cached_out" compile_plan_ms)"
profile_cached_compile_plan_us="$(read_kv "$profile_cached_out" compile_plan_us)"
profile_cached_compile_normalize_ms="$(read_kv "$profile_cached_out" compile_normalize_ms)"
profile_cached_compile_normalize_us="$(read_kv "$profile_cached_out" compile_normalize_us)"
profile_cached_compile_lower_ms="$(read_kv "$profile_cached_out" compile_lower_ms)"
profile_cached_compile_lower_us="$(read_kv "$profile_cached_out" compile_lower_us)"
profile_cached_compile_group_ms="$(read_kv "$profile_cached_out" compile_group_ms)"
profile_cached_compile_group_us="$(read_kv "$profile_cached_out" compile_group_us)"
profile_cached_compile_intents_ms="$(read_kv "$profile_cached_out" compile_intents_ms)"
profile_cached_compile_intents_us="$(read_kv "$profile_cached_out" compile_intents_us)"
profile_cached_engine_ms="$(read_kv "$profile_cached_out" engine_ms)"
profile_cached_module_compile_ms="$(read_kv "$profile_cached_out" module_compile_ms)"
profile_cached_runtime_build_ms="$(read_kv "$profile_cached_out" runtime_build_ms)"
profile_cached_execute_ms="$(read_kv "$profile_cached_out" execute_ms)"
profile_cached_structural_functions="$(read_kv "$profile_cached_out" structural_functions)"
profile_cached_fallback_functions="$(read_kv "$profile_cached_out" fallback_functions)"
profile_cached_parser_fallback_functions="$(read_kv "$profile_cached_out" parser_fallback_functions)"
profile_cached_structural_sccs="$(read_kv "$profile_cached_out" structural_sccs)"
profile_cached_structural_state_count="$(read_kv "$profile_cached_out" structural_state_count)"
profile_cached_structural_transition_count="$(read_kv "$profile_cached_out" structural_transition_count)"
profile_cached_structural_span_ops="$(read_kv "$profile_cached_out" structural_span_ops)"
profile_cached_structural_enum_ops="$(read_kv "$profile_cached_out" structural_enum_ops)"
profile_cached_total_ms="$(read_kv "$profile_cached_out" total_ms)"
profile_cached_total_us="$(read_kv "$profile_cached_out" total_us)"
printf '[harness] cached wasm profile: frontend_ms=%s specialize_ms=%s lambda_lower_ms=%s canonicalize_ms=%s parse_args_ms=%s compile_artifact_ms=%s compile_artifact_us=%s compile_enum_layouts_ms=%s compile_enum_layouts_us=%s compile_plan_ms=%s compile_plan_us=%s compile_normalize_ms=%s compile_normalize_us=%s compile_lower_ms=%s compile_lower_us=%s compile_group_ms=%s compile_group_us=%s compile_intents_ms=%s compile_intents_us=%s engine_ms=%s module_compile_ms=%s runtime_build_ms=%s execute_ms=%s structural_functions=%s fallback_functions=%s parser_fallback_functions=%s structural_sccs=%s structural_state_count=%s structural_transition_count=%s structural_span_ops=%s structural_enum_ops=%s total_ms=%s total_us=%s\n' \
  "$profile_cached_frontend_ms" "$profile_cached_specialize_ms" "$profile_cached_lambda_lower_ms" "$profile_cached_canonicalize_ms" "$profile_cached_parse_args_ms" "$profile_cached_compile_artifact_ms" "$profile_cached_compile_artifact_us" "$profile_cached_compile_enum_layouts_ms" "$profile_cached_compile_enum_layouts_us" "$profile_cached_compile_plan_ms" "$profile_cached_compile_plan_us" "$profile_cached_compile_normalize_ms" "$profile_cached_compile_normalize_us" "$profile_cached_compile_lower_ms" "$profile_cached_compile_lower_us" "$profile_cached_compile_group_ms" "$profile_cached_compile_group_us" "$profile_cached_compile_intents_ms" "$profile_cached_compile_intents_us" "$profile_cached_engine_ms" "$profile_cached_module_compile_ms" "$profile_cached_runtime_build_ms" "$profile_cached_execute_ms" "$profile_cached_structural_functions" "$profile_cached_fallback_functions" "$profile_cached_parser_fallback_functions" "$profile_cached_structural_sccs" "$profile_cached_structural_state_count" "$profile_cached_structural_transition_count" "$profile_cached_structural_span_ops" "$profile_cached_structural_enum_ops" "$profile_cached_total_ms" "$profile_cached_total_us"
printf 'json_profile_artifact_bytes=%s\njson_profile_frontend_ms=%s\njson_profile_specialize_ms=%s\njson_profile_lambda_lower_ms=%s\njson_profile_canonicalize_ms=%s\njson_profile_parse_args_ms=%s\njson_profile_compile_artifact_ms=%s\njson_profile_compile_artifact_us=%s\njson_profile_compile_enum_layouts_ms=%s\njson_profile_compile_enum_layouts_us=%s\njson_profile_compile_plan_ms=%s\njson_profile_compile_plan_us=%s\njson_profile_compile_normalize_ms=%s\njson_profile_compile_normalize_us=%s\njson_profile_compile_lower_ms=%s\njson_profile_compile_lower_us=%s\njson_profile_compile_group_ms=%s\njson_profile_compile_group_us=%s\njson_profile_compile_intents_ms=%s\njson_profile_compile_intents_us=%s\njson_profile_engine_ms=%s\njson_profile_module_compile_ms=%s\njson_profile_runtime_build_ms=%s\njson_profile_execute_ms=%s\njson_profile_total_ms=%s\njson_profile_total_us=%s\n' \
  "$profile_artifact_bytes" "$profile_frontend_ms" "$profile_specialize_ms" "$profile_lambda_lower_ms" "$profile_canonicalize_ms" "$profile_parse_args_ms" "$profile_compile_artifact_ms" "$profile_compile_artifact_us" "$profile_compile_enum_layouts_ms" "$profile_compile_enum_layouts_us" "$profile_compile_plan_ms" "$profile_compile_plan_us" "$profile_compile_normalize_ms" "$profile_compile_normalize_us" "$profile_compile_lower_ms" "$profile_compile_lower_us" "$profile_compile_group_ms" "$profile_compile_group_us" "$profile_compile_intents_ms" "$profile_compile_intents_us" "$profile_engine_ms" "$profile_module_compile_ms" "$profile_runtime_build_ms" "$profile_execute_ms" "$profile_total_ms" "$profile_total_us" >> "$summary_file"
printf 'json_profile_cached_frontend_ms=%s\njson_profile_cached_specialize_ms=%s\njson_profile_cached_lambda_lower_ms=%s\njson_profile_cached_canonicalize_ms=%s\njson_profile_cached_parse_args_ms=%s\njson_profile_cached_compile_artifact_ms=%s\njson_profile_cached_compile_artifact_us=%s\njson_profile_cached_compile_enum_layouts_ms=%s\njson_profile_cached_compile_enum_layouts_us=%s\njson_profile_cached_compile_plan_ms=%s\njson_profile_cached_compile_plan_us=%s\njson_profile_cached_compile_normalize_ms=%s\njson_profile_cached_compile_normalize_us=%s\njson_profile_cached_compile_lower_ms=%s\njson_profile_cached_compile_lower_us=%s\njson_profile_cached_compile_group_ms=%s\njson_profile_cached_compile_group_us=%s\njson_profile_cached_compile_intents_ms=%s\njson_profile_cached_compile_intents_us=%s\njson_profile_cached_engine_ms=%s\njson_profile_cached_module_compile_ms=%s\njson_profile_cached_runtime_build_ms=%s\njson_profile_cached_execute_ms=%s\njson_profile_cached_total_ms=%s\njson_profile_cached_total_us=%s\n' \
  "$profile_cached_frontend_ms" "$profile_cached_specialize_ms" "$profile_cached_lambda_lower_ms" "$profile_cached_canonicalize_ms" "$profile_cached_parse_args_ms" "$profile_cached_compile_artifact_ms" "$profile_cached_compile_artifact_us" "$profile_cached_compile_enum_layouts_ms" "$profile_cached_compile_enum_layouts_us" "$profile_cached_compile_plan_ms" "$profile_cached_compile_plan_us" "$profile_cached_compile_normalize_ms" "$profile_cached_compile_normalize_us" "$profile_cached_compile_lower_ms" "$profile_cached_compile_lower_us" "$profile_cached_compile_group_ms" "$profile_cached_compile_group_us" "$profile_cached_compile_intents_ms" "$profile_cached_compile_intents_us" "$profile_cached_engine_ms" "$profile_cached_module_compile_ms" "$profile_cached_runtime_build_ms" "$profile_cached_execute_ms" "$profile_cached_total_ms" "$profile_cached_total_us" >> "$summary_file"
printf 'json_profile_cached_structural_functions=%s\njson_profile_cached_fallback_functions=%s\njson_profile_cached_parser_fallback_functions=%s\njson_profile_cached_structural_sccs=%s\njson_profile_cached_structural_state_count=%s\njson_profile_cached_structural_transition_count=%s\njson_profile_cached_structural_span_ops=%s\njson_profile_cached_structural_enum_ops=%s\n' \
  "$profile_cached_structural_functions" "$profile_cached_fallback_functions" "$profile_cached_parser_fallback_functions" "$profile_cached_structural_sccs" "$profile_cached_structural_state_count" "$profile_cached_structural_transition_count" "$profile_cached_structural_span_ops" "$profile_cached_structural_enum_ops" >> "$summary_file"
log_metric "json_profile_artifact_bytes" "$profile_artifact_bytes"
log_metric "json_profile_frontend_ms" "$profile_frontend_ms"
log_metric "json_profile_specialize_ms" "$profile_specialize_ms"
log_metric "json_profile_lambda_lower_ms" "$profile_lambda_lower_ms"
log_metric "json_profile_canonicalize_ms" "$profile_canonicalize_ms"
log_metric "json_profile_parse_args_ms" "$profile_parse_args_ms"
log_metric "json_profile_compile_artifact_ms" "$profile_compile_artifact_ms"
log_metric "json_profile_compile_artifact_us" "$profile_compile_artifact_us"
log_metric "json_profile_compile_enum_layouts_ms" "$profile_compile_enum_layouts_ms"
log_metric "json_profile_compile_enum_layouts_us" "$profile_compile_enum_layouts_us"
log_metric "json_profile_compile_plan_ms" "$profile_compile_plan_ms"
log_metric "json_profile_compile_plan_us" "$profile_compile_plan_us"
log_metric "json_profile_compile_normalize_ms" "$profile_compile_normalize_ms"
log_metric "json_profile_compile_normalize_us" "$profile_compile_normalize_us"
log_metric "json_profile_compile_lower_ms" "$profile_compile_lower_ms"
log_metric "json_profile_compile_lower_us" "$profile_compile_lower_us"
log_metric "json_profile_compile_group_ms" "$profile_compile_group_ms"
log_metric "json_profile_compile_group_us" "$profile_compile_group_us"
log_metric "json_profile_compile_intents_ms" "$profile_compile_intents_ms"
log_metric "json_profile_compile_intents_us" "$profile_compile_intents_us"
log_metric "json_profile_engine_ms" "$profile_engine_ms"
log_metric "json_profile_module_compile_ms" "$profile_module_compile_ms"
log_metric "json_profile_runtime_build_ms" "$profile_runtime_build_ms"
log_metric "json_profile_execute_ms" "$profile_execute_ms"
log_metric "json_profile_total_ms" "$profile_total_ms"
log_metric "json_profile_total_us" "$profile_total_us"
log_metric "json_profile_cached_frontend_ms" "$profile_cached_frontend_ms"
log_metric "json_profile_cached_specialize_ms" "$profile_cached_specialize_ms"
log_metric "json_profile_cached_lambda_lower_ms" "$profile_cached_lambda_lower_ms"
log_metric "json_profile_cached_canonicalize_ms" "$profile_cached_canonicalize_ms"
log_metric "json_profile_cached_parse_args_ms" "$profile_cached_parse_args_ms"
log_metric "json_profile_cached_compile_artifact_ms" "$profile_cached_compile_artifact_ms"
log_metric "json_profile_cached_compile_artifact_us" "$profile_cached_compile_artifact_us"
log_metric "json_profile_cached_compile_enum_layouts_ms" "$profile_cached_compile_enum_layouts_ms"
log_metric "json_profile_cached_compile_enum_layouts_us" "$profile_cached_compile_enum_layouts_us"
log_metric "json_profile_cached_compile_plan_ms" "$profile_cached_compile_plan_ms"
log_metric "json_profile_cached_compile_plan_us" "$profile_cached_compile_plan_us"
log_metric "json_profile_cached_compile_normalize_ms" "$profile_cached_compile_normalize_ms"
log_metric "json_profile_cached_compile_normalize_us" "$profile_cached_compile_normalize_us"
log_metric "json_profile_cached_compile_lower_ms" "$profile_cached_compile_lower_ms"
log_metric "json_profile_cached_compile_lower_us" "$profile_cached_compile_lower_us"
log_metric "json_profile_cached_compile_group_ms" "$profile_cached_compile_group_ms"
log_metric "json_profile_cached_compile_group_us" "$profile_cached_compile_group_us"
log_metric "json_profile_cached_compile_intents_ms" "$profile_cached_compile_intents_ms"
log_metric "json_profile_cached_compile_intents_us" "$profile_cached_compile_intents_us"
log_metric "json_profile_cached_engine_ms" "$profile_cached_engine_ms"
log_metric "json_profile_cached_module_compile_ms" "$profile_cached_module_compile_ms"
log_metric "json_profile_cached_runtime_build_ms" "$profile_cached_runtime_build_ms"
log_metric "json_profile_cached_execute_ms" "$profile_cached_execute_ms"
log_metric "json_profile_cached_structural_functions" "$profile_cached_structural_functions"
log_metric "json_profile_cached_fallback_functions" "$profile_cached_fallback_functions"
log_metric "json_profile_cached_parser_fallback_functions" "$profile_cached_parser_fallback_functions"
log_metric "json_profile_cached_structural_sccs" "$profile_cached_structural_sccs"
log_metric "json_profile_cached_structural_state_count" "$profile_cached_structural_state_count"
log_metric "json_profile_cached_structural_transition_count" "$profile_cached_structural_transition_count"
log_metric "json_profile_cached_structural_span_ops" "$profile_cached_structural_span_ops"
log_metric "json_profile_cached_structural_enum_ops" "$profile_cached_structural_enum_ops"
log_metric "json_profile_cached_total_ms" "$profile_cached_total_ms"
log_metric "json_profile_cached_total_us" "$profile_cached_total_us"

if [ "${profile_cached_parser_fallback_functions:-0}" -ne 0 ]; then
  echo "[harness] parser hot path still contains fallback functions" >&2
  cp "$log_file" "$last_log"
  cp "$summary_file" "$last_summary"
  exit 1
fi

if [ "$mode" = "full" ]; then
  prepared_slowdown="$(printf '%s\n' "$json_bench_wasm_prepared_line" | sed -n 's/.*slowdown=\([0-9.]*\)x.*/\1/p')"
  if [ -n "$prepared_slowdown" ] && awk "BEGIN { exit !($prepared_slowdown <= 1.25) }"; then
    :
  else
    echo "[harness] prepared wasm json benchmark exceeded 1.25x rust baseline" >&2
    cp "$log_file" "$last_log"
    cp "$summary_file" "$last_summary"
    exit 1
  fi
  echo "[harness] bench snapshot"
  bench_out="$artifact_dir/structural_harness_$run_id.bench.txt"
  run_step_capture "bench.core" "$bench_out" cargo run --quiet -- bench
  printf 'bench_output=%s\n' "$bench_out" >> "$summary_file"
fi

cp "$log_file" "$last_log"
cp "$summary_file" "$last_summary"
printf '[harness] artifacts: %s %s\n' "$log_file" "$summary_file"
echo "[harness] done ($mode)"
