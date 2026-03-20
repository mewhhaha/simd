use super::*;
use crate::wasm_backend::{WasmLeafExport, WasmLeafResultAbi, WasmOptimizationReport};
use std::hint::black_box;
use std::time::{Duration, Instant};
use wasm_encoder::{
    BlockType, CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction,
    MemArg, MemorySection, MemoryType, Module, TypeSection, ValType,
};

const DEFAULT_SIZE: usize = 262_144;
const MATRIX_SIZES: [(&str, usize); 3] = [("small", 4_096), ("medium", 65_536), ("large", 262_144)];

#[derive(Debug, Clone)]
pub struct BenchOptions {
    pub selection: String,
    pub size: usize,
    pub iterations: Option<usize>,
    pub report_contract: bool,
}

impl Default for BenchOptions {
    fn default() -> Self {
        Self {
            selection: "all".to_string(),
            size: DEFAULT_SIZE,
            iterations: None,
            report_contract: true,
        }
    }
}

impl BenchOptions {
    fn from_selection(selection: &str, size: usize, iterations: Option<usize>) -> Self {
        Self {
            selection: selection.to_string(),
            size: if size == 0 { DEFAULT_SIZE } else { size },
            iterations,
            report_contract: true,
        }
    }
}

struct ExampleSpec {
    name: &'static str,
    description: &'static str,
    file: &'static str,
    main: &'static str,
    source: &'static str,
    build_input: fn(usize) -> BenchInput,
    rust_impl: fn(&[Value]) -> Result<Value>,
}

struct MatrixSpec {
    name: String,
    description: String,
    file: String,
    main: &'static str,
    source: String,
    build_input: Box<dyn Fn(usize) -> BenchInput>,
    rust_impl: Box<dyn Fn(&[Value]) -> Result<Value>>,
    size_label: &'static str,
    size: usize,
    is_core_suite: bool,
}

struct BenchInput {
    args: Vec<Value>,
    logical_elements: usize,
    leaf_elements: usize,
}

#[derive(Clone, Copy)]
struct BenchStats {
    iterations: usize,
    total: Duration,
}

pub fn bench_command(selection: &str, size: usize, iterations: Option<usize>) -> Result<String> {
    bench_command_with_options(BenchOptions::from_selection(selection, size, iterations))
}

pub fn bench_command_with_options(options: BenchOptions) -> Result<String> {
    if options.selection == "all" {
        let specs = benchmark_specs();
        return run_example_specs(
            &specs,
            options.size,
            options
                .iterations
                .unwrap_or_else(|| default_iterations(options.size)),
            options.report_contract,
        );
    }

    if options.selection == "matrix" {
        let matrix_specs = matrix_specs();
        return if options.report_contract {
            run_matrix_specs(&matrix_specs, options.iterations)
        } else {
            run_matrix_specs_with_contract(&matrix_specs, options.iterations, false)
        };
    }

    let specs = benchmark_specs();
    let spec = specs
        .into_iter()
        .find(|spec| spec.name == options.selection)
        .ok_or_else(|| {
            SimdError::new(format!(
                "unknown benchmark '{}'; available cases: {}, matrix",
                options.selection,
                benchmark_specs()
                    .iter()
                    .map(|spec| spec.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;

    run_example_specs(
        &[spec],
        options.size,
        options
            .iterations
            .unwrap_or_else(|| default_iterations(options.size)),
        options.report_contract,
    )
}

fn run_example_specs(
    specs: &[ExampleSpec],
    size: usize,
    iterations: usize,
    report_contract: bool,
) -> Result<String> {
    let mut sections = Vec::new();
    sections.push(format!(
        "benchmarks size={} hot-iterations={} e2e-iterations={} cases={}",
        size,
        iterations,
        end_to_end_iterations(iterations),
        specs
            .iter()
            .map(|spec| spec.name)
            .collect::<Vec<_>>()
            .join(", ")
    ));
    for spec in specs {
        let section = run_example_spec(spec, size, iterations)?;
        sections.push(section);
    }
    if report_contract && specs.len() > 1 {
        sections.push(render_core_contract_summary(specs, size, iterations)?);
    }
    Ok(sections.join("\n\n"))
}

fn run_example_spec(spec: &ExampleSpec, size: usize, iterations: usize) -> Result<String> {
    let input = (spec.build_input)(size);

    let frontend_start = Instant::now();
    let compiled = compile_source(spec.source)?;
    let frontend_compile = frontend_start.elapsed();

    let artifact_start = Instant::now();
    let artifact = compile_wasm_main(spec.source, spec.main)?;
    let artifact_compile = artifact_start.elapsed();
    let wasm_bytes = artifact.bytes.len();
    let optimizer_summary = summarize_optimizer_reports(&artifact.optimizer_reports);

    let prepare_start = Instant::now();
    let executable = prepare_wasm_artifact(artifact)?;
    let module_prepare = prepare_start.elapsed();
    let prepared_start = Instant::now();
    let prepared = prepare_wasm_main(spec.source, spec.main)?;
    let prepared_prepare = prepared_start.elapsed();
    let layout = prepared.layout_from_args(&input.args)?;
    let mut prepared_bound = prepared.bind(layout)?;
    prepared_bound.load_args(&input.args)?;

    let expected = (spec.rust_impl)(&input.args)?;
    let eval = run_compiled_main(&compiled, spec.main, &input.args)?;
    if eval != expected {
        return Err(SimdError::new(format!(
            "evaluator parity failed for benchmark '{}'",
            spec.name
        )));
    }
    let wasm = executable.run_hot(&input.args)?;
    if wasm != expected {
        return Err(SimdError::new(format!(
            "Wasm parity failed for benchmark '{}'",
            spec.name
        )));
    }
    prepared_bound.run()?;
    let prepared_value = prepared_bound.read_result()?;
    if prepared_value != expected {
        return Err(SimdError::new(format!(
            "Wasm prepared parity failed for benchmark '{}'",
            spec.name
        )));
    }

    let e2e_iterations = end_to_end_iterations(iterations);
    let rust_stats = measure(iterations, || (spec.rust_impl)(&input.args))?;
    let wasm_e2e_stats = measure(e2e_iterations, || {
        let artifact = compile_wasm_main(spec.source, spec.main)?;
        let executable = prepare_wasm_artifact(artifact)?;
        executable.run_hot(&input.args)
    })?;
    let wasm_hot_stats = measure(iterations, || executable.run_hot(&input.args))?;
    let prepared_hot_stats = measure_void(iterations, || prepared_bound.run())?;
    let mut prepared_read_scratch = build_prepared_bulk_read_scratch(&prepared_bound);
    let prepared_hot_read_stats = measure_void(iterations, || {
        prepared_bound.run()?;
        prepared_typed_readback(&prepared_bound, &mut prepared_read_scratch)
    })?;
    let handwritten_hot_line = if let Some(handcrafted_artifact) =
        handcrafted_baseline_artifact(spec.name)?
    {
        let handwritten_executable = prepare_wasm_artifact(handcrafted_artifact)?;
        let handwritten_value = handwritten_executable.run_hot(&input.args)?;
        if handwritten_value != expected {
            return Err(SimdError::new(format!(
                "handwritten Wasm parity failed for benchmark '{}'",
                spec.name
            )));
        }
        let handwritten_hot_stats =
            measure(iterations, || handwritten_executable.run_hot(&input.args))?;
        let vs_generated = avg_duration(&handwritten_hot_stats).as_secs_f64()
            / avg_duration(&wasm_hot_stats).as_secs_f64();
        format!(
            "  wasm-handwritten-hot: iterations={} avg/run={} ns/logical-elem={:.2} mlogical/s={:.2} slowdown={:.2}x vs-generated={:.2}x\n",
            iterations,
            format_duration(avg_duration(&handwritten_hot_stats)),
            nanos_per_element(&handwritten_hot_stats, input.logical_elements),
            melems_per_second(&handwritten_hot_stats, input.logical_elements),
            slowdown(&rust_stats, &handwritten_hot_stats),
            vs_generated,
        )
    } else {
        String::new()
    };

    Ok(format!(
        concat!(
            "{}\n",
            "  description: {}\n",
            "  source: {}\n",
            "  setup: frontend={} compile-wasm={} prepare-module={} prepare-bound={} wasm-bytes={}\n",
            "  optimizer: {}\n",
            "  rust: avg/run={} ns/logical-elem={:.2} mlogical/s={:.2}\n",
            "  wasm-e2e: iterations={} avg/run={} ns/logical-elem={:.2} mlogical/s={:.2} slowdown={:.2}x\n",
            "  wasm-hot: iterations={} avg/run={} ns/logical-elem={:.2} mlogical/s={:.2} slowdown={:.2}x\n",
            "{}",
            "  wasm-prepared-hot: iterations={} avg/run={} ns/logical-elem={:.2} mlogical/s={:.2} slowdown={:.2}x\n",
            "  wasm-prepared-hot+read: iterations={} avg/run={} ns/logical-elem={:.2} mlogical/s={:.2} slowdown={:.2}x{}\n",
            "  parity: evaluator, wasm, and rust agree"
        ),
        spec.name,
        spec.description,
        spec.file,
        format_duration(frontend_compile),
        format_duration(artifact_compile),
        format_duration(module_prepare),
        format_duration(prepared_prepare),
        wasm_bytes,
        optimizer_summary,
        format_duration(avg_duration(&rust_stats)),
        nanos_per_element(&rust_stats, input.logical_elements),
        melems_per_second(&rust_stats, input.logical_elements),
        e2e_iterations,
        format_duration(avg_duration(&wasm_e2e_stats)),
        nanos_per_element(&wasm_e2e_stats, input.logical_elements),
        melems_per_second(&wasm_e2e_stats, input.logical_elements),
        slowdown(&rust_stats, &wasm_e2e_stats),
        iterations,
        format_duration(avg_duration(&wasm_hot_stats)),
        nanos_per_element(&wasm_hot_stats, input.logical_elements),
        melems_per_second(&wasm_hot_stats, input.logical_elements),
        slowdown(&rust_stats, &wasm_hot_stats),
        handwritten_hot_line,
        iterations,
        format_duration(avg_duration(&prepared_hot_stats)),
        nanos_per_element(&prepared_hot_stats, input.logical_elements),
        melems_per_second(&prepared_hot_stats, input.logical_elements),
        slowdown(&rust_stats, &prepared_hot_stats),
        iterations,
        format_duration(avg_duration(&prepared_hot_read_stats)),
        nanos_per_element(&prepared_hot_read_stats, input.logical_elements),
        melems_per_second(&prepared_hot_read_stats, input.logical_elements),
        slowdown(&rust_stats, &prepared_hot_read_stats),
        format_leaf_metrics(
            &rust_stats,
            &wasm_e2e_stats,
            &wasm_hot_stats,
            &prepared_hot_stats,
            &prepared_hot_read_stats,
            &input,
        ),
    ))
}

fn run_matrix_specs(specs: &[MatrixSpec], iterations: Option<usize>) -> Result<String> {
    run_matrix_specs_with_contract(specs, iterations, true)
}

fn run_matrix_specs_with_contract(
    specs: &[MatrixSpec],
    iterations: Option<usize>,
    report_contract: bool,
) -> Result<String> {
    let mut sections = Vec::new();
    sections.push(format!(
        "benchmarks matrix sizes={} hot-iterations={:?} cases={}",
        MATRIX_SIZES
            .iter()
            .map(|(label, size)| format!("{}={}", label, size))
            .collect::<Vec<_>>()
            .join(", "),
        iterations,
        specs
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<Vec<_>>()
            .join(", ")
    ));
    let mut summary = MatrixSummary::default();
    for spec in specs {
        let (section, result) = run_matrix_spec(spec, iterations)?;
        summary.observe(&result);
        sections.push(section);
    }
    if report_contract {
        sections.push(summary.render());
    }
    Ok(sections.join("\n\n"))
}

fn run_matrix_spec(spec: &MatrixSpec, iterations: Option<usize>) -> Result<(String, MatrixResult)> {
    let input = (spec.build_input)(spec.size);

    let frontend_start = Instant::now();
    let compiled = compile_source(&spec.source)?;
    let frontend_compile = frontend_start.elapsed();

    let artifact_start = Instant::now();
    let artifact = compile_wasm_main(&spec.source, spec.main)?;
    let artifact_compile = artifact_start.elapsed();
    let wasm_bytes = artifact.bytes.len();
    let optimizer_summary = summarize_optimizer_reports(&artifact.optimizer_reports);

    let prepare_start = Instant::now();
    let executable = prepare_wasm_artifact(artifact)?;
    let module_prepare = prepare_start.elapsed();
    let prepared_start = Instant::now();
    let prepared = prepare_wasm_main(&spec.source, spec.main)?;
    let prepared_prepare = prepared_start.elapsed();
    let layout = prepared.layout_from_args(&input.args)?;
    let mut prepared_bound = prepared.bind(layout)?;
    prepared_bound.load_args(&input.args)?;

    let expected = (spec.rust_impl)(&input.args)?;
    let eval = run_compiled_main(&compiled, spec.main, &input.args)?;
    if eval != expected {
        return Err(SimdError::new(format!(
            "evaluator parity failed for benchmark '{}'",
            spec.name
        )));
    }
    let wasm = executable.run_hot(&input.args)?;
    if wasm != expected {
        return Err(SimdError::new(format!(
            "Wasm parity failed for benchmark '{}'",
            spec.name
        )));
    }
    prepared_bound.run()?;
    let prepared_value = prepared_bound.read_result()?;
    if prepared_value != expected {
        return Err(SimdError::new(format!(
            "Wasm prepared parity failed for benchmark '{}'",
            spec.name
        )));
    }

    let iterations = iterations.unwrap_or_else(|| default_iterations(spec.size));
    let e2e_iterations = end_to_end_iterations(iterations);
    let rust_stats = measure(iterations, || (spec.rust_impl)(&input.args))?;
    let wasm_e2e_stats = measure(e2e_iterations, || {
        let artifact = compile_wasm_main(&spec.source, spec.main)?;
        let executable = prepare_wasm_artifact(artifact)?;
        executable.run_hot(&input.args)
    })?;
    let wasm_hot_stats = measure(iterations, || executable.run_hot(&input.args))?;
    let prepared_hot_stats = measure_void(iterations, || prepared_bound.run())?;
    let mut prepared_read_scratch = build_prepared_bulk_read_scratch(&prepared_bound);
    let prepared_hot_read_stats = measure_void(iterations, || {
        prepared_bound.run()?;
        prepared_typed_readback(&prepared_bound, &mut prepared_read_scratch)
    })?;
    let handwritten_hot_line = if let Some(handcrafted_artifact) =
        handcrafted_baseline_artifact(&spec.name)?
    {
        let handwritten_executable = prepare_wasm_artifact(handcrafted_artifact)?;
        let handwritten_value = handwritten_executable.run_hot(&input.args)?;
        if handwritten_value != expected {
            return Err(SimdError::new(format!(
                "handwritten Wasm parity failed for benchmark '{}'",
                spec.name
            )));
        }
        let handwritten_hot_stats =
            measure(iterations, || handwritten_executable.run_hot(&input.args))?;
        let vs_generated = avg_duration(&handwritten_hot_stats).as_secs_f64()
            / avg_duration(&wasm_hot_stats).as_secs_f64();
        format!(
            "  wasm-handwritten-hot: iterations={} avg/run={} ns/logical-elem={:.2} mlogical/s={:.2} slowdown={:.2}x vs-generated={:.2}x\n",
            iterations,
            format_duration(avg_duration(&handwritten_hot_stats)),
            nanos_per_element(&handwritten_hot_stats, input.logical_elements),
            melems_per_second(&handwritten_hot_stats, input.logical_elements),
            slowdown(&rust_stats, &handwritten_hot_stats),
            vs_generated,
        )
    } else {
        String::new()
    };

    let contract = matrix_case_contract(spec.size_label, &rust_stats, &prepared_hot_stats);
    let result = MatrixResult {
        name: spec.name.clone(),
        size_label: spec.size_label,
        is_core_suite: spec.is_core_suite,
        rust_stats,
        wasm_hot_stats,
        prepared_hot_stats,
        prepared_hot_read_stats,
        leaf_elements: input.leaf_elements,
    };

    Ok((
        format!(
            concat!(
                "{}\n",
                "  description: {}\n",
                "  source: {}\n",
                "  setup: frontend={} compile-wasm={} prepare-module={} prepare-bound={} wasm-bytes={}\n",
                "  optimizer: {}\n",
                "  rust: avg/run={} ns/logical-elem={:.2} mlogical/s={:.2}\n",
                "  wasm-e2e: iterations={} avg/run={} ns/logical-elem={:.2} mlogical/s={:.2} slowdown={:.2}x\n",
                "  wasm-hot: iterations={} avg/run={} ns/logical-elem={:.2} mlogical/s={:.2} slowdown={:.2}x\n",
                "{}",
                "  wasm-prepared-hot: iterations={} avg/run={} ns/logical-elem={:.2} mlogical/s={:.2} slowdown={:.2}x\n",
                "  wasm-prepared-hot+read: iterations={} avg/run={} ns/logical-elem={:.2} mlogical/s={:.2} slowdown={:.2}x{}\n",
                "  contract: {}\n",
                "  parity: evaluator, wasm, and rust agree"
            ),
            spec.name,
            spec.description,
            spec.file,
            format_duration(frontend_compile),
            format_duration(artifact_compile),
            format_duration(module_prepare),
            format_duration(prepared_prepare),
            wasm_bytes,
            optimizer_summary,
            format_duration(avg_duration(&result.rust_stats)),
            nanos_per_element(&result.rust_stats, input.logical_elements),
            melems_per_second(&result.rust_stats, input.logical_elements),
            e2e_iterations,
            format_duration(avg_duration(&wasm_e2e_stats)),
            nanos_per_element(&wasm_e2e_stats, input.logical_elements),
            melems_per_second(&wasm_e2e_stats, input.logical_elements),
            slowdown(&result.rust_stats, &wasm_e2e_stats),
            iterations,
            format_duration(avg_duration(&result.wasm_hot_stats)),
            nanos_per_element(&result.wasm_hot_stats, input.logical_elements),
            melems_per_second(&result.wasm_hot_stats, input.logical_elements),
            slowdown(&result.rust_stats, &result.wasm_hot_stats),
            handwritten_hot_line,
            iterations,
            format_duration(avg_duration(&result.prepared_hot_stats)),
            nanos_per_element(&result.prepared_hot_stats, input.logical_elements),
            melems_per_second(&result.prepared_hot_stats, input.logical_elements),
            slowdown(&result.rust_stats, &result.prepared_hot_stats),
            iterations,
            format_duration(avg_duration(&prepared_hot_read_stats)),
            nanos_per_element(&prepared_hot_read_stats, input.logical_elements),
            melems_per_second(&prepared_hot_read_stats, input.logical_elements),
            slowdown(&result.rust_stats, &prepared_hot_read_stats),
            format_leaf_metrics(
                &result.rust_stats,
                &wasm_e2e_stats,
                &result.wasm_hot_stats,
                &result.prepared_hot_stats,
                &prepared_hot_read_stats,
                &input
            ),
            contract,
        ),
        result,
    ))
}

fn summarize_optimizer_reports(reports: &[WasmOptimizationReport]) -> String {
    if reports.is_empty() {
        return "none".to_string();
    }
    let mut reports = reports.to_vec();
    reports.sort_by(|left, right| {
        left.function
            .cmp(&right.function)
            .then(intent_label(&left.intent).cmp(intent_label(&right.intent)))
            .then(left.vector_unroll.cmp(&right.vector_unroll))
            .then(left.vectorizable.cmp(&right.vectorizable))
            .then(left.fallback_reason.cmp(&right.fallback_reason))
    });
    reports
        .into_iter()
        .map(|report| {
            let plan = match report.vector_unroll {
                0 => "scalar",
                1 => "vec1",
                2 => "vec2",
                4 => "vec4",
                _ => "vecN",
            };
            match report.fallback_reason {
                Some(reason) => format!(
                    "{}:{} plan={} fallback={}",
                    report.function,
                    intent_label(&report.intent),
                    plan,
                    reason
                ),
                None => format!(
                    "{}:{} plan={}",
                    report.function,
                    intent_label(&report.intent),
                    plan
                ),
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

fn intent_label(intent: &IntentClass) -> &'static str {
    match intent {
        IntentClass::MapUnary => "MapUnary",
        IntentClass::MapBinaryBroadcast => "MapBinaryBroadcast",
        IntentClass::MapTernaryBroadcast => "MapTernaryBroadcast",
        IntentClass::GroupedMap => "GroupedMap",
        IntentClass::ScalarTailRec => "ScalarTailRec",
        IntentClass::Fallback => "Fallback",
    }
}

fn matrix_case_contract(size_label: &str, rust: &BenchStats, wasm_hot: &BenchStats) -> String {
    if size_label == "small" {
        return format!(
            "SKIP baseline-regression gate unavailable in this run (observed rust/wasm-hot {:.2}x)",
            slowdown(rust, wasm_hot)
        );
    }
    let threshold = match size_label {
        "medium" => 1.05,
        "large" => 1.0,
        _ => 1.0,
    };
    let slowdown = slowdown(rust, wasm_hot);
    let status = if slowdown <= threshold {
        "PASS"
    } else {
        "FAIL"
    };
    format!(
        "{} {} ({:.2}x <= {:.2}x)",
        status,
        threshold_label(size_label),
        slowdown,
        threshold
    )
}

#[derive(Default)]
struct MatrixSummary {
    small_total: usize,
    small_within_rust_ten_percent: usize,
    medium_total: usize,
    medium_pass: usize,
    large_total: usize,
    large_pass: usize,
    core_large_total: usize,
    core_large_fast: usize,
    axpy_large_leaf: Option<f64>,
    axpy2_large_leaf: Option<f64>,
}

struct MatrixResult {
    name: String,
    size_label: &'static str,
    is_core_suite: bool,
    rust_stats: BenchStats,
    wasm_hot_stats: BenchStats,
    prepared_hot_stats: BenchStats,
    prepared_hot_read_stats: BenchStats,
    leaf_elements: usize,
}

impl MatrixSummary {
    fn observe(&mut self, result: &MatrixResult) {
        let slowdown = slowdown(&result.rust_stats, &result.prepared_hot_stats);
        let pass = match result.size_label {
            "small" => slowdown <= 1.10,
            "medium" => slowdown <= 1.05,
            "large" => slowdown < 1.0,
            _ => false,
        };
        match result.size_label {
            "small" => {
                self.small_total += 1;
                if pass {
                    self.small_within_rust_ten_percent += 1;
                }
            }
            "medium" => {
                self.medium_total += 1;
                if pass {
                    self.medium_pass += 1;
                }
            }
            "large" => {
                self.large_total += 1;
                if pass {
                    self.large_pass += 1;
                }
            }
            _ => {}
        }
        if result.is_core_suite && result.size_label == "large" {
            self.core_large_total += 1;
            if slowdown < 1.0 {
                self.core_large_fast += 1;
            }
            let leaf_nanos =
                nanos_per_element(&result.prepared_hot_read_stats, result.leaf_elements);
            if result.name == "axpy-i64" {
                self.axpy_large_leaf = Some(leaf_nanos);
            } else if result.name == "axpy2-record-i64" {
                self.axpy2_large_leaf = Some(leaf_nanos);
            }
        }
    }

    fn render(&self) -> String {
        let large_ok = self.large_pass == self.large_total && self.large_total != 0;
        let medium_ok = self.medium_pass == self.medium_total && self.medium_total != 0;
        let core_ok = self.core_large_fast >= 4 && self.core_large_total == 5;
        let axpy_ok = self.axpy_per_leaf_within_ten_percent();
        let axpy_leaf_summary = match (self.axpy_large_leaf, self.axpy2_large_leaf) {
            (Some(axpy), Some(axpy2)) => format!(
                "{} axpy2 per-leaf {:.2} ns/leaf-elem vs axpy {:.2} ns/leaf-elem ratio={:.2}x",
                if axpy_ok { "PASS" } else { "FAIL" },
                axpy2,
                axpy,
                axpy2 / axpy
            ),
            _ => "FAIL missing axpy or axpy2 large case".to_string(),
        };
        format!(
            concat!(
                "contract summary:\n",
                "  large slowdown<1.0: {} ({}/{} pass)\n",
                "  medium slowdown<=1.05: {} ({}/{} pass)\n",
                "  small baseline-regression gate: SKIP (baseline unavailable in this run; reference-only rust/wasm<=1.10 observed for {}/{} cases)\n",
                "  core-suite large gate: {} ({}/{} faster)\n",
                "  axpy2 per-leaf within 10% of axpy per-leaf at large: {}\n",
                "  overall: {}"
            ),
            if large_ok { "PASS" } else { "FAIL" },
            self.large_pass,
            self.large_total,
            if medium_ok { "PASS" } else { "FAIL" },
            self.medium_pass,
            self.medium_total,
            self.small_within_rust_ten_percent,
            self.small_total,
            if core_ok { "PASS" } else { "FAIL" },
            self.core_large_fast,
            self.core_large_total,
            axpy_leaf_summary,
            if large_ok && medium_ok && core_ok && axpy_ok {
                "PASS"
            } else {
                "FAIL"
            }
        )
    }

    fn axpy_per_leaf_within_ten_percent(&self) -> bool {
        match (self.axpy_large_leaf, self.axpy2_large_leaf) {
            (Some(axpy), Some(axpy2)) => axpy2 <= axpy * 1.10,
            _ => false,
        }
    }
}

fn threshold_label(size_label: &str) -> &'static str {
    match size_label {
        "small" => "slowdown<=1.10",
        "medium" => "slowdown<=1.05",
        "large" => "slowdown<1.0",
        _ => "slowdown",
    }
}

fn render_core_contract_summary(
    specs: &[ExampleSpec],
    size: usize,
    iterations: usize,
) -> Result<String> {
    let mut core_large_total = 0usize;
    let mut core_large_fast = 0usize;
    let mut axpy_large_leaf = None::<f64>;
    let mut axpy2_large_leaf = None::<f64>;
    for spec in specs {
        let input = (spec.build_input)(size);
        let compiled = compile_source(spec.source)?;
        let artifact = compile_wasm_main(spec.source, spec.main)?;
        let executable = prepare_wasm_artifact(artifact)?;
        let prepared = prepare_wasm_main(spec.source, spec.main)?;
        let layout = prepared.layout_from_args(&input.args)?;
        let mut prepared_bound = prepared.bind(layout)?;
        prepared_bound.load_args(&input.args)?;
        let expected = (spec.rust_impl)(&input.args)?;
        let eval = run_compiled_main(&compiled, spec.main, &input.args)?;
        if eval != expected {
            return Err(SimdError::new(format!(
                "evaluator parity failed for benchmark '{}'",
                spec.name
            )));
        }
        let wasm = executable.run_hot(&input.args)?;
        if wasm != expected {
            return Err(SimdError::new(format!(
                "Wasm parity failed for benchmark '{}'",
                spec.name
            )));
        }
        prepared_bound.run()?;
        let prepared_value = prepared_bound.read_result()?;
        if prepared_value != expected {
            return Err(SimdError::new(format!(
                "Wasm prepared parity failed for benchmark '{}'",
                spec.name
            )));
        }
        let rust_stats = measure(iterations, || (spec.rust_impl)(&input.args))?;
        let prepared_hot_stats = measure_void(iterations, || prepared_bound.run())?;
        let mut prepared_read_scratch = build_prepared_bulk_read_scratch(&prepared_bound);
        let prepared_hot_read_stats = measure_void(iterations, || {
            prepared_bound.run()?;
            prepared_typed_readback(&prepared_bound, &mut prepared_read_scratch)
        })?;
        let slowdown = slowdown(&rust_stats, &prepared_hot_stats);
        core_large_total += 1;
        if slowdown < 1.0 {
            core_large_fast += 1;
        }
        let leaf_nanos = nanos_per_element(&prepared_hot_read_stats, input.leaf_elements);
        if spec.name == "axpy-i64" {
            axpy_large_leaf = Some(leaf_nanos);
        } else if spec.name == "axpy2-record-i64" {
            axpy2_large_leaf = Some(leaf_nanos);
        }
    }
    let axpy_ratio = match (axpy_large_leaf, axpy2_large_leaf) {
        (Some(axpy), Some(axpy2)) => format!(
            "{} axpy2 per-leaf {:.2} ns/leaf-elem vs axpy {:.2} ns/leaf-elem ratio={:.2}x",
            if axpy2 <= axpy * 1.10 { "PASS" } else { "FAIL" },
            axpy2,
            axpy,
            axpy2 / axpy
        ),
        _ => "SKIP axpy2-record per-leaf gate unavailable".to_string(),
    };
    Ok(format!(
        concat!(
            "contract summary:\n",
            "  core-large gate: {} ({}/{} faster)\n",
            "  axpy2 per-leaf within 10% of axpy per-leaf at large: {}\n",
            "  matrix-large gate: SKIP (core suite mode)\n",
            "  matrix-medium gate: SKIP (core suite mode)\n",
            "  matrix-small gate: SKIP (core suite mode)\n",
            "  overall: {}"
        ),
        if core_large_fast >= 4 && core_large_total == 5 {
            "PASS"
        } else {
            "FAIL"
        },
        core_large_fast,
        core_large_total,
        axpy_ratio,
        if core_large_fast >= 4 && core_large_total == 5 && axpy_ratio.starts_with("PASS") {
            "PASS"
        } else {
            "FAIL"
        }
    ))
}

fn matrix_specs() -> Vec<MatrixSpec> {
    let mut specs = Vec::new();
    for &(size_label, size) in &MATRIX_SIZES {
        if size_label == "large" {
            specs.extend(core_suite_matrix_specs(size_label, size));
        }
    }
    specs.extend(expanded_matrix_specs());
    specs
}

fn core_suite_matrix_specs(size_label: &'static str, size: usize) -> Vec<MatrixSpec> {
    let mut specs = Vec::new();
    for spec in benchmark_specs() {
        let ExampleSpec {
            name,
            description,
            file,
            main,
            source,
            build_input,
            rust_impl,
        } = spec;
        specs.push(MatrixSpec {
            name: name.to_string(),
            description: format!("{} [core suite {}]", description, size_label),
            file: file.to_string(),
            main,
            source: source.to_string(),
            build_input: Box::new(move |_| build_input(size)),
            rust_impl: Box::new(move |args| rust_impl(args)),
            size_label,
            size,
            is_core_suite: true,
        });
    }
    specs
}

fn expanded_matrix_specs() -> Vec<MatrixSpec> {
    let mut specs = Vec::new();
    for &(size_label, size) in &MATRIX_SIZES {
        for prim in [Prim::I32, Prim::I64, Prim::F32, Prim::F64] {
            specs.push(MatrixSpec {
                name: format!("matrix-unary-{}-{}", prim_label(prim), size_label),
                description: format!(
                    "Unary pointwise add-one over {} bulk values.",
                    prim_label(prim)
                ),
                file: format!("matrix/unary-{}-{}.simd", prim_label(prim), size_label),
                main: "main",
                source: build_matrix_unary_source(prim),
                build_input: Box::new(move |actual_size| {
                    build_matrix_unary_input(prim, actual_size)
                }),
                rust_impl: Box::new(move |args| rust_matrix_unary(args, prim)),
                size_label,
                size,
                is_core_suite: false,
            });
            specs.push(MatrixSpec {
                name: format!("matrix-axpy-{}-{}", prim_label(prim), size_label),
                description: format!(
                    "Scalar broadcast axpy over {} bulk values.",
                    prim_label(prim)
                ),
                file: format!("matrix/axpy-{}-{}.simd", prim_label(prim), size_label),
                main: "main",
                source: build_matrix_axpy_source(prim),
                build_input: Box::new(move |actual_size| {
                    build_matrix_axpy_input(prim, actual_size)
                }),
                rust_impl: Box::new(move |args| rust_matrix_axpy(args, prim)),
                size_label,
                size,
                is_core_suite: false,
            });
            specs.push(MatrixSpec {
                name: format!("matrix-axpy2-record-{}-{}", prim_label(prim), size_label),
                description: format!(
                    "Grouped record axpy2 over {} bulk fields.",
                    prim_label(prim)
                ),
                file: format!(
                    "matrix/axpy2-record-{}-{}.simd",
                    prim_label(prim),
                    size_label
                ),
                main: "main",
                source: build_matrix_axpy2_source(prim),
                build_input: Box::new(move |actual_size| {
                    build_matrix_axpy2_input(prim, actual_size)
                }),
                rust_impl: Box::new(move |args| rust_matrix_axpy2(args, prim)),
                size_label,
                size,
                is_core_suite: false,
            });
            specs.push(MatrixSpec {
                name: format!("matrix-square-rank2-{}-{}", prim_label(prim), size_label),
                description: format!(
                    "Rank-2 pointwise square over {} bulk values.",
                    prim_label(prim)
                ),
                file: format!(
                    "matrix/square-rank2-{}-{}.simd",
                    prim_label(prim),
                    size_label
                ),
                main: "main",
                source: build_matrix_rank2_square_source(prim),
                build_input: Box::new(move |actual_size| {
                    build_matrix_rank2_square_input(prim, actual_size)
                }),
                rust_impl: Box::new(move |args| rust_matrix_rank2_square(args, prim)),
                size_label,
                size,
                is_core_suite: false,
            });
            specs.push(MatrixSpec {
                name: format!("matrix-affine-rank2-{}-{}", prim_label(prim), size_label),
                description: format!(
                    "Rank-2 affine transform (scale+bias) over {} matrix values.",
                    prim_label(prim)
                ),
                file: format!(
                    "matrix/affine-rank2-{}-{}.simd",
                    prim_label(prim),
                    size_label
                ),
                main: "main",
                source: build_matrix_rank2_affine_source(prim),
                build_input: Box::new(move |actual_size| {
                    build_matrix_rank2_affine_input(prim, actual_size)
                }),
                rust_impl: Box::new(move |args| rust_matrix_rank2_affine(args, prim)),
                size_label,
                size,
                is_core_suite: false,
            });
        }
        specs.push(MatrixSpec {
            name: format!("matrix-image-tone-rgb-f32-{}", size_label),
            description: "Image-style RGB tone mapping over rank-2 pixels.".to_string(),
            file: format!("matrix/image-tone-rgb-f32-{}.simd", size_label),
            main: "main",
            source: build_matrix_image_tone_source(),
            build_input: Box::new(build_matrix_image_tone_input),
            rust_impl: Box::new(rust_matrix_image_tone),
            size_label,
            size,
            is_core_suite: false,
        });
        specs.push(MatrixSpec {
            name: format!("matrix-image-blend-rgb-f32-{}", size_label),
            description: "Image-style RGB alpha blend over two rank-2 images.".to_string(),
            file: format!("matrix/image-blend-rgb-f32-{}.simd", size_label),
            main: "main",
            source: build_matrix_image_blend_source(),
            build_input: Box::new(build_matrix_image_blend_input),
            rust_impl: Box::new(rust_matrix_image_blend),
            size_label,
            size,
            is_core_suite: false,
        });
    }
    specs
}

fn prim_label(prim: Prim) -> &'static str {
    match prim {
        Prim::I32 => "i32",
        Prim::I64 => "i64",
        Prim::F32 => "f32",
        Prim::F64 => "f64",
    }
}

fn build_matrix_unary_source(prim: Prim) -> String {
    let ty = prim_label(prim);
    let one = match prim {
        Prim::F32 | Prim::F64 => "1.0",
        _ => "1",
    };
    format!(
        "add1 : {ty} -> {ty}\nadd1 x = x + {one}\n\nmain : {ty}[n] -> {ty}[n]\nmain xs = add1 xs\n"
    )
}

fn build_matrix_axpy_source(prim: Prim) -> String {
    let ty = prim_label(prim);
    format!(
        "axpy : {ty} -> {ty} -> {ty} -> {ty}\naxpy a x y = a * x + y\n\nmain : {ty} -> {ty}[n] -> {ty}[n] -> {ty}[n]\nmain a xs ys = axpy a xs ys\n"
    )
}

fn build_matrix_axpy2_source(prim: Prim) -> String {
    let ty = prim_label(prim);
    format!(
        "axpy2 : {ty} -> {{x:{ty},y:{ty}}} -> {{x:{ty},y:{ty}}} -> {{x:{ty},y:{ty}}}\naxpy2 a u v = {{ x = a * u.x + v.x, y = a * u.y + v.y }}\n\nmain : {ty} -> {{x:{ty},y:{ty}}}[n] -> {{x:{ty},y:{ty}}}[n] -> {{x:{ty},y:{ty}}}[n]\nmain a us vs = axpy2 a us vs\n"
    )
}

fn build_matrix_rank2_square_source(prim: Prim) -> String {
    let ty = prim_label(prim);
    format!(
        "square : {ty} -> {ty}\nsquare x = x * x\n\nmain : {ty}[n,m] -> {ty}[n,m]\nmain xs = square xs\n"
    )
}

fn build_matrix_rank2_affine_source(prim: Prim) -> String {
    let ty = prim_label(prim);
    format!(
        "affine : {ty} -> {ty} -> {ty} -> {ty}\naffine scale bias x = scale * x + bias\n\nmain : {ty} -> {ty} -> {ty}[n,m] -> {ty}[n,m]\nmain scale bias xs = affine scale bias xs\n"
    )
}

fn build_matrix_image_tone_source() -> String {
    "tone : f32 -> {r:f32,g:f32,b:f32} -> {r:f32,g:f32,b:f32}\ntone exposure p = { r = exposure * p.r + 0.05, g = exposure * p.g + 0.03, b = exposure * p.b + 0.01 }\n\nmain : f32 -> {r:f32,g:f32,b:f32}[h,w] -> {r:f32,g:f32,b:f32}[h,w]\nmain exposure img = tone exposure img\n".to_string()
}

fn build_matrix_image_blend_source() -> String {
    "blend : f32 -> {r:f32,g:f32,b:f32} -> {r:f32,g:f32,b:f32} -> {r:f32,g:f32,b:f32}\nblend alpha left right = let inv = 1.0 - alpha in { r = alpha * left.r + inv * right.r, g = alpha * left.g + inv * right.g, b = alpha * left.b + inv * right.b }\n\nmain : f32 -> {r:f32,g:f32,b:f32}[h,w] -> {r:f32,g:f32,b:f32}[h,w] -> {r:f32,g:f32,b:f32}[h,w]\nmain alpha left right = blend alpha left right\n".to_string()
}

fn build_matrix_unary_input(prim: Prim, size: usize) -> BenchInput {
    let elements = match prim {
        Prim::I32 => (0..size)
            .map(|index| ScalarValue::I32((index % 1024) as i32))
            .collect::<Vec<_>>(),
        Prim::I64 => (0..size)
            .map(|index| ScalarValue::I64((index % 1024) as i64))
            .collect::<Vec<_>>(),
        Prim::F32 => (0..size)
            .map(|index| ScalarValue::F32(((index % 97) as u32 + 1) as f32))
            .collect::<Vec<_>>(),
        Prim::F64 => (0..size)
            .map(|index| ScalarValue::F64(((index % 97) as u32 + 1) as f64))
            .collect::<Vec<_>>(),
    };
    BenchInput {
        args: vec![Value::Bulk(BulkValue {
            prim,
            shape: vec![size],
            elements,
        })],
        logical_elements: size,
        leaf_elements: size,
    }
}

fn build_matrix_axpy_input(prim: Prim, size: usize) -> BenchInput {
    let xs = match prim {
        Prim::I32 => (0..size)
            .map(|index| ScalarValue::I32((index % 257) as i32))
            .collect::<Vec<_>>(),
        Prim::I64 => (0..size)
            .map(|index| ScalarValue::I64((index % 257) as i64))
            .collect::<Vec<_>>(),
        Prim::F32 => (0..size)
            .map(|index| ScalarValue::F32(((index % 257) as u32) as f32))
            .collect::<Vec<_>>(),
        Prim::F64 => (0..size)
            .map(|index| ScalarValue::F64(((index % 257) as u32) as f64))
            .collect::<Vec<_>>(),
    };
    let ys = match prim {
        Prim::I32 => (0..size)
            .map(|index| ScalarValue::I32(((index * 3) % 257) as i32))
            .collect::<Vec<_>>(),
        Prim::I64 => (0..size)
            .map(|index| ScalarValue::I64(((index * 3) % 257) as i64))
            .collect::<Vec<_>>(),
        Prim::F32 => (0..size)
            .map(|index| ScalarValue::F32(((index * 3) % 257) as f32))
            .collect::<Vec<_>>(),
        Prim::F64 => (0..size)
            .map(|index| ScalarValue::F64(((index * 3) % 257) as f64))
            .collect::<Vec<_>>(),
    };
    BenchInput {
        args: vec![
            match prim {
                Prim::F32 => Value::Scalar(ScalarValue::F32(7.0)),
                Prim::F64 => Value::Scalar(ScalarValue::F64(7.0)),
                Prim::I32 => Value::Scalar(ScalarValue::I32(7)),
                Prim::I64 => Value::Scalar(ScalarValue::I64(7)),
            },
            Value::Bulk(BulkValue {
                prim,
                shape: vec![size],
                elements: xs,
            }),
            Value::Bulk(BulkValue {
                prim,
                shape: vec![size],
                elements: ys,
            }),
        ],
        logical_elements: size,
        leaf_elements: size,
    }
}

fn build_matrix_axpy2_input(prim: Prim, size: usize) -> BenchInput {
    let ux = match prim {
        Prim::I32 => (0..size)
            .map(|index| ScalarValue::I32((index % 257) as i32))
            .collect::<Vec<_>>(),
        Prim::I64 => (0..size)
            .map(|index| ScalarValue::I64((index % 257) as i64))
            .collect::<Vec<_>>(),
        Prim::F32 => (0..size)
            .map(|index| ScalarValue::F32(((index % 257) as u32) as f32))
            .collect::<Vec<_>>(),
        Prim::F64 => (0..size)
            .map(|index| ScalarValue::F64(((index % 257) as u32) as f64))
            .collect::<Vec<_>>(),
    };
    let uy = match prim {
        Prim::I32 => (0..size)
            .map(|index| ScalarValue::I32(((index * 3) % 257) as i32))
            .collect::<Vec<_>>(),
        Prim::I64 => (0..size)
            .map(|index| ScalarValue::I64(((index * 3) % 257) as i64))
            .collect::<Vec<_>>(),
        Prim::F32 => (0..size)
            .map(|index| ScalarValue::F32(((index * 3) % 257) as f32))
            .collect::<Vec<_>>(),
        Prim::F64 => (0..size)
            .map(|index| ScalarValue::F64(((index * 3) % 257) as f64))
            .collect::<Vec<_>>(),
    };
    let vx = match prim {
        Prim::I32 => (0..size)
            .map(|index| ScalarValue::I32(((index * 5) % 257) as i32))
            .collect::<Vec<_>>(),
        Prim::I64 => (0..size)
            .map(|index| ScalarValue::I64(((index * 5) % 257) as i64))
            .collect::<Vec<_>>(),
        Prim::F32 => (0..size)
            .map(|index| ScalarValue::F32(((index * 5) % 257) as f32))
            .collect::<Vec<_>>(),
        Prim::F64 => (0..size)
            .map(|index| ScalarValue::F64(((index * 5) % 257) as f64))
            .collect::<Vec<_>>(),
    };
    let vy = match prim {
        Prim::I32 => (0..size)
            .map(|index| ScalarValue::I32(((index * 7) % 257) as i32))
            .collect::<Vec<_>>(),
        Prim::I64 => (0..size)
            .map(|index| ScalarValue::I64(((index * 7) % 257) as i64))
            .collect::<Vec<_>>(),
        Prim::F32 => (0..size)
            .map(|index| ScalarValue::F32(((index * 7) % 257) as f32))
            .collect::<Vec<_>>(),
        Prim::F64 => (0..size)
            .map(|index| ScalarValue::F64(((index * 7) % 257) as f64))
            .collect::<Vec<_>>(),
    };
    BenchInput {
        args: vec![
            match prim {
                Prim::F32 => Value::Scalar(ScalarValue::F32(7.0)),
                Prim::F64 => Value::Scalar(ScalarValue::F64(7.0)),
                Prim::I32 => Value::Scalar(ScalarValue::I32(7)),
                Prim::I64 => Value::Scalar(ScalarValue::I64(7)),
            },
            Value::Record(std::collections::BTreeMap::from([
                (
                    "x".to_string(),
                    Value::Bulk(BulkValue {
                        prim,
                        shape: vec![size],
                        elements: ux,
                    }),
                ),
                (
                    "y".to_string(),
                    Value::Bulk(BulkValue {
                        prim,
                        shape: vec![size],
                        elements: uy,
                    }),
                ),
            ])),
            Value::Record(std::collections::BTreeMap::from([
                (
                    "x".to_string(),
                    Value::Bulk(BulkValue {
                        prim,
                        shape: vec![size],
                        elements: vx,
                    }),
                ),
                (
                    "y".to_string(),
                    Value::Bulk(BulkValue {
                        prim,
                        shape: vec![size],
                        elements: vy,
                    }),
                ),
            ])),
        ],
        logical_elements: size,
        leaf_elements: size * 2,
    }
}

fn build_matrix_rank2_square_input(prim: Prim, size: usize) -> BenchInput {
    let side = (size as f64).sqrt() as usize;
    debug_assert_eq!(side * side, size);
    let elements = match prim {
        Prim::I32 => (0..size)
            .map(|index| ScalarValue::I32((index % 4096) as i32))
            .collect::<Vec<_>>(),
        Prim::I64 => (0..size)
            .map(|index| ScalarValue::I64((index % 4096) as i64))
            .collect::<Vec<_>>(),
        Prim::F32 => (0..size)
            .map(|index| ScalarValue::F32(((index % 97) as u32 + 1) as f32))
            .collect::<Vec<_>>(),
        Prim::F64 => (0..size)
            .map(|index| ScalarValue::F64(((index % 97) as u32 + 1) as f64))
            .collect::<Vec<_>>(),
    };
    BenchInput {
        args: vec![Value::Bulk(BulkValue {
            prim,
            shape: vec![side, side],
            elements,
        })],
        logical_elements: size,
        leaf_elements: size,
    }
}

fn build_matrix_rank2_affine_input(prim: Prim, size: usize) -> BenchInput {
    let side = (size as f64).sqrt() as usize;
    debug_assert_eq!(side * side, size);
    let elements = match prim {
        Prim::I32 => (0..size)
            .map(|index| ScalarValue::I32((index % 4096) as i32))
            .collect::<Vec<_>>(),
        Prim::I64 => (0..size)
            .map(|index| ScalarValue::I64((index % 4096) as i64))
            .collect::<Vec<_>>(),
        Prim::F32 => (0..size)
            .map(|index| ScalarValue::F32(((index % 511) as u32) as f32 / 17.0))
            .collect::<Vec<_>>(),
        Prim::F64 => (0..size)
            .map(|index| ScalarValue::F64(((index % 511) as u32) as f64 / 17.0))
            .collect::<Vec<_>>(),
    };
    let scale = match prim {
        Prim::I32 => Value::Scalar(ScalarValue::I32(3)),
        Prim::I64 => Value::Scalar(ScalarValue::I64(3)),
        Prim::F32 => Value::Scalar(ScalarValue::F32(1.75)),
        Prim::F64 => Value::Scalar(ScalarValue::F64(1.75)),
    };
    let bias = match prim {
        Prim::I32 => Value::Scalar(ScalarValue::I32(11)),
        Prim::I64 => Value::Scalar(ScalarValue::I64(11)),
        Prim::F32 => Value::Scalar(ScalarValue::F32(0.125)),
        Prim::F64 => Value::Scalar(ScalarValue::F64(0.125)),
    };
    BenchInput {
        args: vec![
            scale,
            bias,
            Value::Bulk(BulkValue {
                prim,
                shape: vec![side, side],
                elements,
            }),
        ],
        logical_elements: size,
        leaf_elements: size,
    }
}

fn build_matrix_image_tone_input(size: usize) -> BenchInput {
    let side = (size as f64).sqrt() as usize;
    debug_assert_eq!(side * side, size);
    let r = (0..size)
        .map(|index| ScalarValue::F32((index % 251) as f32 / 255.0))
        .collect::<Vec<_>>();
    let g = (0..size)
        .map(|index| ScalarValue::F32(((index * 3) % 251) as f32 / 255.0))
        .collect::<Vec<_>>();
    let b = (0..size)
        .map(|index| ScalarValue::F32(((index * 7) % 251) as f32 / 255.0))
        .collect::<Vec<_>>();
    BenchInput {
        args: vec![
            Value::Scalar(ScalarValue::F32(1.15)),
            Value::Record(std::collections::BTreeMap::from([
                (
                    "r".to_string(),
                    Value::Bulk(BulkValue {
                        prim: Prim::F32,
                        shape: vec![side, side],
                        elements: r,
                    }),
                ),
                (
                    "g".to_string(),
                    Value::Bulk(BulkValue {
                        prim: Prim::F32,
                        shape: vec![side, side],
                        elements: g,
                    }),
                ),
                (
                    "b".to_string(),
                    Value::Bulk(BulkValue {
                        prim: Prim::F32,
                        shape: vec![side, side],
                        elements: b,
                    }),
                ),
            ])),
        ],
        logical_elements: size,
        leaf_elements: size * 3,
    }
}

fn build_matrix_image_blend_input(size: usize) -> BenchInput {
    let side = (size as f64).sqrt() as usize;
    debug_assert_eq!(side * side, size);
    let left_r = (0..size)
        .map(|index| ScalarValue::F32((index % 251) as f32 / 255.0))
        .collect::<Vec<_>>();
    let left_g = (0..size)
        .map(|index| ScalarValue::F32(((index * 3) % 251) as f32 / 255.0))
        .collect::<Vec<_>>();
    let left_b = (0..size)
        .map(|index| ScalarValue::F32(((index * 7) % 251) as f32 / 255.0))
        .collect::<Vec<_>>();
    let right_r = (0..size)
        .map(|index| ScalarValue::F32(((index * 11) % 251) as f32 / 255.0))
        .collect::<Vec<_>>();
    let right_g = (0..size)
        .map(|index| ScalarValue::F32(((index * 13) % 251) as f32 / 255.0))
        .collect::<Vec<_>>();
    let right_b = (0..size)
        .map(|index| ScalarValue::F32(((index * 17) % 251) as f32 / 255.0))
        .collect::<Vec<_>>();
    let left = Value::Record(std::collections::BTreeMap::from([
        (
            "r".to_string(),
            Value::Bulk(BulkValue {
                prim: Prim::F32,
                shape: vec![side, side],
                elements: left_r,
            }),
        ),
        (
            "g".to_string(),
            Value::Bulk(BulkValue {
                prim: Prim::F32,
                shape: vec![side, side],
                elements: left_g,
            }),
        ),
        (
            "b".to_string(),
            Value::Bulk(BulkValue {
                prim: Prim::F32,
                shape: vec![side, side],
                elements: left_b,
            }),
        ),
    ]));
    let right = Value::Record(std::collections::BTreeMap::from([
        (
            "r".to_string(),
            Value::Bulk(BulkValue {
                prim: Prim::F32,
                shape: vec![side, side],
                elements: right_r,
            }),
        ),
        (
            "g".to_string(),
            Value::Bulk(BulkValue {
                prim: Prim::F32,
                shape: vec![side, side],
                elements: right_g,
            }),
        ),
        (
            "b".to_string(),
            Value::Bulk(BulkValue {
                prim: Prim::F32,
                shape: vec![side, side],
                elements: right_b,
            }),
        ),
    ]));
    BenchInput {
        args: vec![Value::Scalar(ScalarValue::F32(0.35)), left, right],
        logical_elements: size,
        leaf_elements: size * 3,
    }
}

fn rust_matrix_unary(args: &[Value], prim: Prim) -> Result<Value> {
    let bulk = expect_bulk_prim(args, 0, prim)?;
    let elements = bulk
        .elements
        .iter()
        .map(|value| match (prim, value) {
            (Prim::I32, ScalarValue::I32(value)) => Ok(ScalarValue::I32(value + 1)),
            (Prim::I64, ScalarValue::I64(value)) => Ok(ScalarValue::I64(value + 1)),
            (Prim::F32, ScalarValue::F32(value)) => Ok(ScalarValue::F32(value + 1.0)),
            (Prim::F64, ScalarValue::F64(value)) => Ok(ScalarValue::F64(value + 1.0)),
            _ => Err(SimdError::new(format!(
                "expected {} element in unary benchmark, found {:?}",
                prim_label(prim),
                value
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Value::Bulk(BulkValue {
        prim,
        shape: bulk.shape.clone(),
        elements,
    }))
}

fn rust_matrix_axpy(args: &[Value], prim: Prim) -> Result<Value> {
    let a = expect_scalar_prim(args, 0, prim)?;
    let xs = expect_bulk_prim(args, 1, prim)?;
    let ys = expect_bulk_prim(args, 2, prim)?;
    if xs.shape != ys.shape || xs.elements.len() != ys.elements.len() {
        return Err(SimdError::new(
            "matrix axpy benchmark expected matching bulk shapes",
        ));
    }
    let elements = xs
        .elements
        .iter()
        .zip(&ys.elements)
        .map(|(x, y)| match (prim, x, y) {
            (Prim::I32, ScalarValue::I32(x), ScalarValue::I32(y)) => {
                Ok(ScalarValue::I32((a as i32 * x) + y))
            }
            (Prim::I64, ScalarValue::I64(x), ScalarValue::I64(y)) => {
                Ok(ScalarValue::I64((a as i64 * x) + y))
            }
            (Prim::F32, ScalarValue::F32(x), ScalarValue::F32(y)) => {
                Ok(ScalarValue::F32((a as f32 * x) + y))
            }
            (Prim::F64, ScalarValue::F64(x), ScalarValue::F64(y)) => {
                Ok(ScalarValue::F64((a as f64 * x) + y))
            }
            _ => Err(SimdError::new(format!(
                "matrix axpy benchmark expected {} lanes",
                prim_label(prim)
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Value::Bulk(BulkValue {
        prim,
        shape: xs.shape.clone(),
        elements,
    }))
}

fn rust_matrix_axpy2(args: &[Value], prim: Prim) -> Result<Value> {
    let a = expect_scalar_prim(args, 0, prim)?;
    let u = expect_record(args, 1)?;
    let v = expect_record(args, 2)?;
    let ux = expect_record_bulk_prim_field(u, "x", prim)?;
    let uy = expect_record_bulk_prim_field(u, "y", prim)?;
    let vx = expect_record_bulk_prim_field(v, "x", prim)?;
    let vy = expect_record_bulk_prim_field(v, "y", prim)?;
    if ux.shape != uy.shape
        || vx.shape != vy.shape
        || ux.shape != vx.shape
        || ux.elements.len() != uy.elements.len()
        || vx.elements.len() != vy.elements.len()
        || ux.elements.len() != vx.elements.len()
    {
        return Err(SimdError::new(
            "matrix axpy2 benchmark expected matching bulk shapes",
        ));
    }
    let mut x_elements = Vec::with_capacity(ux.elements.len());
    let mut y_elements = Vec::with_capacity(uy.elements.len());
    for ((ux, uy), (vx, vy)) in ux
        .elements
        .iter()
        .zip(&uy.elements)
        .zip(vx.elements.iter().zip(&vy.elements))
    {
        match (prim, ux, uy, vx, vy) {
            (
                Prim::I32,
                ScalarValue::I32(ux),
                ScalarValue::I32(uy),
                ScalarValue::I32(vx),
                ScalarValue::I32(vy),
            ) => {
                x_elements.push(ScalarValue::I32((a as i32 * ux) + vx));
                y_elements.push(ScalarValue::I32((a as i32 * uy) + vy));
            }
            (
                Prim::I64,
                ScalarValue::I64(ux),
                ScalarValue::I64(uy),
                ScalarValue::I64(vx),
                ScalarValue::I64(vy),
            ) => {
                x_elements.push(ScalarValue::I64((a as i64 * ux) + vx));
                y_elements.push(ScalarValue::I64((a as i64 * uy) + vy));
            }
            (
                Prim::F32,
                ScalarValue::F32(ux),
                ScalarValue::F32(uy),
                ScalarValue::F32(vx),
                ScalarValue::F32(vy),
            ) => {
                x_elements.push(ScalarValue::F32((a as f32 * ux) + vx));
                y_elements.push(ScalarValue::F32((a as f32 * uy) + vy));
            }
            (
                Prim::F64,
                ScalarValue::F64(ux),
                ScalarValue::F64(uy),
                ScalarValue::F64(vx),
                ScalarValue::F64(vy),
            ) => {
                x_elements.push(ScalarValue::F64((a as f64 * ux) + vx));
                y_elements.push(ScalarValue::F64((a as f64 * uy) + vy));
            }
            _ => {
                return Err(SimdError::new(format!(
                    "matrix axpy2 benchmark expected {} lanes",
                    prim_label(prim)
                )));
            }
        }
    }
    Ok(Value::Record(std::collections::BTreeMap::from([
        (
            "x".to_string(),
            Value::Bulk(BulkValue {
                prim,
                shape: ux.shape.clone(),
                elements: x_elements,
            }),
        ),
        (
            "y".to_string(),
            Value::Bulk(BulkValue {
                prim,
                shape: uy.shape.clone(),
                elements: y_elements,
            }),
        ),
    ])))
}

fn rust_matrix_rank2_square(args: &[Value], prim: Prim) -> Result<Value> {
    let xs = expect_bulk_prim(args, 0, prim)?;
    let elements = xs
        .elements
        .iter()
        .map(|value| match (prim, value) {
            (Prim::I32, ScalarValue::I32(value)) => Ok(ScalarValue::I32(value * value)),
            (Prim::I64, ScalarValue::I64(value)) => Ok(ScalarValue::I64(value * value)),
            (Prim::F32, ScalarValue::F32(value)) => Ok(ScalarValue::F32(value * value)),
            (Prim::F64, ScalarValue::F64(value)) => Ok(ScalarValue::F64(value * value)),
            _ => Err(SimdError::new(format!(
                "matrix rank2 square benchmark expected {} element",
                prim_label(prim)
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Value::Bulk(BulkValue {
        prim,
        shape: xs.shape.clone(),
        elements,
    }))
}

fn rust_matrix_rank2_affine(args: &[Value], prim: Prim) -> Result<Value> {
    let scale = expect_scalar_prim(args, 0, prim)?;
    let bias = expect_scalar_prim(args, 1, prim)?;
    let xs = expect_bulk_prim(args, 2, prim)?;
    let elements = xs
        .elements
        .iter()
        .map(|value| match (prim, value) {
            (Prim::I32, ScalarValue::I32(value)) => {
                Ok(ScalarValue::I32((scale as i32 * value) + (bias as i32)))
            }
            (Prim::I64, ScalarValue::I64(value)) => {
                Ok(ScalarValue::I64((scale as i64 * value) + (bias as i64)))
            }
            (Prim::F32, ScalarValue::F32(value)) => {
                Ok(ScalarValue::F32((scale as f32 * value) + (bias as f32)))
            }
            (Prim::F64, ScalarValue::F64(value)) => {
                Ok(ScalarValue::F64((scale as f64 * value) + (bias as f64)))
            }
            _ => Err(SimdError::new(format!(
                "matrix rank2 affine benchmark expected {} element",
                prim_label(prim)
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Value::Bulk(BulkValue {
        prim,
        shape: xs.shape.clone(),
        elements,
    }))
}

fn rust_matrix_image_tone(args: &[Value]) -> Result<Value> {
    let exposure = expect_scalar_prim(args, 0, Prim::F32)? as f32;
    let image = expect_record(args, 1)?;
    let r = expect_record_bulk_prim_field(image, "r", Prim::F32)?;
    let g = expect_record_bulk_prim_field(image, "g", Prim::F32)?;
    let b = expect_record_bulk_prim_field(image, "b", Prim::F32)?;
    if r.shape != g.shape
        || r.shape != b.shape
        || r.elements.len() != g.elements.len()
        || r.elements.len() != b.elements.len()
    {
        return Err(SimdError::new(
            "matrix image tone benchmark expected matching RGB shapes",
        ));
    }
    let r_elements = r
        .elements
        .iter()
        .map(|value| match value {
            ScalarValue::F32(value) => Ok(ScalarValue::F32((exposure * value) + 0.05)),
            other => Err(SimdError::new(format!(
                "matrix image tone benchmark expected f32 r lane, found {:?}",
                other
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    let g_elements = g
        .elements
        .iter()
        .map(|value| match value {
            ScalarValue::F32(value) => Ok(ScalarValue::F32((exposure * value) + 0.03)),
            other => Err(SimdError::new(format!(
                "matrix image tone benchmark expected f32 g lane, found {:?}",
                other
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    let b_elements = b
        .elements
        .iter()
        .map(|value| match value {
            ScalarValue::F32(value) => Ok(ScalarValue::F32((exposure * value) + 0.01)),
            other => Err(SimdError::new(format!(
                "matrix image tone benchmark expected f32 b lane, found {:?}",
                other
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Value::Record(std::collections::BTreeMap::from([
        (
            "r".to_string(),
            Value::Bulk(BulkValue {
                prim: Prim::F32,
                shape: r.shape.clone(),
                elements: r_elements,
            }),
        ),
        (
            "g".to_string(),
            Value::Bulk(BulkValue {
                prim: Prim::F32,
                shape: g.shape.clone(),
                elements: g_elements,
            }),
        ),
        (
            "b".to_string(),
            Value::Bulk(BulkValue {
                prim: Prim::F32,
                shape: b.shape.clone(),
                elements: b_elements,
            }),
        ),
    ])))
}

fn rust_matrix_image_blend(args: &[Value]) -> Result<Value> {
    let alpha = expect_scalar_prim(args, 0, Prim::F32)? as f32;
    let inv_alpha = 1.0f32 - alpha;
    let left = expect_record(args, 1)?;
    let right = expect_record(args, 2)?;
    let left_r = expect_record_bulk_prim_field(left, "r", Prim::F32)?;
    let left_g = expect_record_bulk_prim_field(left, "g", Prim::F32)?;
    let left_b = expect_record_bulk_prim_field(left, "b", Prim::F32)?;
    let right_r = expect_record_bulk_prim_field(right, "r", Prim::F32)?;
    let right_g = expect_record_bulk_prim_field(right, "g", Prim::F32)?;
    let right_b = expect_record_bulk_prim_field(right, "b", Prim::F32)?;
    if left_r.shape != left_g.shape
        || left_r.shape != left_b.shape
        || right_r.shape != right_g.shape
        || right_r.shape != right_b.shape
        || left_r.shape != right_r.shape
        || left_r.elements.len() != left_g.elements.len()
        || left_r.elements.len() != left_b.elements.len()
        || right_r.elements.len() != right_g.elements.len()
        || right_r.elements.len() != right_b.elements.len()
        || left_r.elements.len() != right_r.elements.len()
    {
        return Err(SimdError::new(
            "matrix image blend benchmark expected matching RGB shapes",
        ));
    }
    let blend_channel =
        |left: &BulkValue, right: &BulkValue, channel: &str| -> Result<Vec<ScalarValue>> {
            left.elements
                .iter()
                .zip(&right.elements)
                .map(|(left, right)| match (left, right) {
                    (ScalarValue::F32(left), ScalarValue::F32(right)) => {
                        Ok(ScalarValue::F32((alpha * left) + (inv_alpha * right)))
                    }
                    _ => Err(SimdError::new(format!(
                        "matrix image blend benchmark expected f32 {} lanes",
                        channel
                    ))),
                })
                .collect::<Result<Vec<_>>>()
        };
    let r_elements = blend_channel(left_r, right_r, "r")?;
    let g_elements = blend_channel(left_g, right_g, "g")?;
    let b_elements = blend_channel(left_b, right_b, "b")?;
    Ok(Value::Record(std::collections::BTreeMap::from([
        (
            "r".to_string(),
            Value::Bulk(BulkValue {
                prim: Prim::F32,
                shape: left_r.shape.clone(),
                elements: r_elements,
            }),
        ),
        (
            "g".to_string(),
            Value::Bulk(BulkValue {
                prim: Prim::F32,
                shape: left_g.shape.clone(),
                elements: g_elements,
            }),
        ),
        (
            "b".to_string(),
            Value::Bulk(BulkValue {
                prim: Prim::F32,
                shape: left_b.shape.clone(),
                elements: b_elements,
            }),
        ),
    ])))
}

fn expect_scalar_prim(args: &[Value], index: usize, prim: Prim) -> Result<f64> {
    match args.get(index) {
        Some(Value::Scalar(ScalarValue::I32(value))) if prim == Prim::I32 => Ok(*value as f64),
        Some(Value::Scalar(ScalarValue::I64(value))) if prim == Prim::I64 => Ok(*value as f64),
        Some(Value::Scalar(ScalarValue::F32(value))) if prim == Prim::F32 => Ok(*value as f64),
        Some(Value::Scalar(ScalarValue::F64(value))) if prim == Prim::F64 => Ok(*value),
        Some(other) => Err(SimdError::new(format!(
            "expected {} scalar at argument {}, found {:?}",
            prim_label(prim),
            index,
            other
        ))),
        None => Err(SimdError::new(format!("missing argument {}", index))),
    }
}

fn expect_bulk_prim(args: &[Value], index: usize, prim: Prim) -> Result<&BulkValue> {
    match args.get(index) {
        Some(Value::Bulk(value)) if value.prim == prim => Ok(value),
        Some(other) => Err(SimdError::new(format!(
            "expected {} bulk at argument {}, found {:?}",
            prim_label(prim),
            index,
            other
        ))),
        None => Err(SimdError::new(format!("missing argument {}", index))),
    }
}

fn expect_record_bulk_prim_field<'a>(
    record: &'a std::collections::BTreeMap<String, Value>,
    field: &str,
    prim: Prim,
) -> Result<&'a BulkValue> {
    match record.get(field) {
        Some(Value::Bulk(value)) if value.prim == prim => Ok(value),
        Some(other) => Err(SimdError::new(format!(
            "expected {} bulk field '{}' in record, found {:?}",
            prim_label(prim),
            field,
            other
        ))),
        None => Err(SimdError::new(format!("missing record field '{}'", field))),
    }
}

fn benchmark_specs() -> Vec<ExampleSpec> {
    vec![
        ExampleSpec {
            name: "inc-i64",
            description: "Pointwise i64 increment over one bulk input.",
            file: "examples/inc_i64.simd",
            main: "main",
            source: include_str!("../examples/inc_i64.simd"),
            build_input: build_inc_i64_input,
            rust_impl: rust_inc_i64,
        },
        ExampleSpec {
            name: "square-f32",
            description: "Pointwise f32 square over one bulk input.",
            file: "examples/square_f32.simd",
            main: "main",
            source: include_str!("../examples/square_f32.simd"),
            build_input: build_square_f32_input,
            rust_impl: rust_square_f32,
        },
        ExampleSpec {
            name: "axpy-i64",
            description: "Scalar broadcast plus two bulk i64 inputs.",
            file: "examples/axpy_i64.simd",
            main: "main",
            source: include_str!("../examples/axpy_i64.simd"),
            build_input: build_axpy_i64_input,
            rust_impl: rust_axpy_i64,
        },
        ExampleSpec {
            name: "axpy2-record-i64",
            description: "Lifted record-valued i64 axpy over bulk fields.",
            file: "examples/axpy2_record_i64.simd",
            main: "main",
            source: include_str!("../examples/axpy2_record_i64.simd"),
            build_input: build_axpy2_record_i64_input,
            rust_impl: rust_axpy2_record_i64,
        },
        ExampleSpec {
            name: "pow2-i64",
            description: "Tail-recursive scalar helper lifted across an i64 bulk input.",
            file: "examples/pow2_i64.simd",
            main: "main",
            source: include_str!("../examples/pow2_i64.simd"),
            build_input: build_pow2_i64_input,
            rust_impl: rust_pow2_i64,
        },
    ]
}

fn handcrafted_baseline_artifact(name: &str) -> Result<Option<WasmArtifact>> {
    match name {
        "inc-i64" => Ok(Some(handcrafted_inc_i64_artifact()?)),
        "square-f32" => Ok(Some(handcrafted_square_f32_artifact()?)),
        "axpy-i64" => Ok(Some(handcrafted_axpy_i64_artifact()?)),
        "axpy2-record-i64" => Ok(Some(handcrafted_axpy2_record_i64_artifact()?)),
        "pow2-i64" => Ok(Some(handcrafted_pow2_i64_artifact()?)),
        _ => handcrafted_matrix_baseline_artifact(name),
    }
}

fn handcrafted_matrix_baseline_artifact(name: &str) -> Result<Option<WasmArtifact>> {
    let Some((base_name, _size_label)) = strip_matrix_size_suffix(name) else {
        return Ok(None);
    };
    if let Some(prim) = base_name
        .strip_prefix("matrix-unary-")
        .and_then(parse_prim_label)
    {
        return Ok(Some(handcrafted_matrix_unary_artifact(prim)?));
    }
    if let Some(prim) = base_name
        .strip_prefix("matrix-square-rank2-")
        .and_then(parse_prim_label)
    {
        return Ok(Some(handcrafted_matrix_square_rank2_artifact(prim)?));
    }
    if let Some(prim) = base_name
        .strip_prefix("matrix-axpy-")
        .and_then(parse_prim_label)
    {
        return Ok(Some(handcrafted_matrix_axpy_artifact(prim)?));
    }
    if let Some(prim) = base_name
        .strip_prefix("matrix-affine-rank2-")
        .and_then(parse_prim_label)
    {
        return Ok(Some(handcrafted_matrix_affine_rank2_artifact(prim)?));
    }
    if let Some(prim) = base_name
        .strip_prefix("matrix-axpy2-record-")
        .and_then(parse_prim_label)
    {
        return Ok(Some(handcrafted_matrix_axpy2_record_artifact(prim)?));
    }
    if base_name == "matrix-image-tone-rgb-f32" {
        return Ok(Some(handcrafted_matrix_image_tone_rgb_f32_artifact()?));
    }
    if base_name == "matrix-image-blend-rgb-f32" {
        return Ok(Some(handcrafted_matrix_image_blend_rgb_f32_artifact()?));
    }
    Ok(None)
}

fn strip_matrix_size_suffix(name: &str) -> Option<(&str, &'static str)> {
    if let Some(base) = name.strip_suffix("-small") {
        return Some((base, "small"));
    }
    if let Some(base) = name.strip_suffix("-medium") {
        return Some((base, "medium"));
    }
    if let Some(base) = name.strip_suffix("-large") {
        return Some((base, "large"));
    }
    None
}

fn parse_prim_label(label: &str) -> Option<Prim> {
    match label {
        "i32" => Some(Prim::I32),
        "i64" => Some(Prim::I64),
        "f32" => Some(Prim::F32),
        "f64" => Some(Prim::F64),
        _ => None,
    }
}

#[derive(Clone, Copy)]
enum HandcraftedUnaryOp {
    AddOne,
    Square,
}

#[derive(Clone, Copy)]
enum HandcraftedYMode {
    Unity,
    OneMinusScale,
}

fn handcrafted_matrix_unary_artifact(prim: Prim) -> Result<WasmArtifact> {
    let function =
        build_handcrafted_unary_leaf_function(prim, 0, 1, 2, 3, HandcraftedUnaryOp::AddOne);
    Ok(build_handcrafted_single_leaf_artifact(
        format!("handcrafted_matrix_unary_{}", prim_label(prim)),
        vec![WasmParamAbi::Bulk { prim }],
        prim,
        0,
        vec![0],
        vec![ValType::I32, ValType::I32, ValType::I32, ValType::I32],
        function,
        IntentClass::MapUnary,
    ))
}

fn handcrafted_matrix_square_rank2_artifact(prim: Prim) -> Result<WasmArtifact> {
    let function =
        build_handcrafted_unary_leaf_function(prim, 0, 1, 2, 3, HandcraftedUnaryOp::Square);
    Ok(build_handcrafted_single_leaf_artifact(
        format!("handcrafted_matrix_square_rank2_{}", prim_label(prim)),
        vec![WasmParamAbi::Bulk { prim }],
        prim,
        0,
        vec![0],
        vec![ValType::I32, ValType::I32, ValType::I32, ValType::I32],
        function,
        IntentClass::MapUnary,
    ))
}

fn handcrafted_matrix_axpy_artifact(prim: Prim) -> Result<WasmArtifact> {
    let function = build_handcrafted_axpy_like_leaf_function(
        prim,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        HandcraftedYMode::Unity,
    );
    Ok(build_handcrafted_single_leaf_artifact(
        format!("handcrafted_matrix_axpy_{}", prim_label(prim)),
        vec![
            WasmParamAbi::Scalar { prim },
            WasmParamAbi::Bulk { prim },
            WasmParamAbi::Bulk { prim },
        ],
        prim,
        1,
        vec![0, 1, 2],
        vec![
            handcrafted_val_type(prim),
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
        ],
        function,
        IntentClass::MapTernaryBroadcast,
    ))
}

fn handcrafted_matrix_affine_rank2_artifact(prim: Prim) -> Result<WasmArtifact> {
    let function = build_handcrafted_affine_scalar_bias_leaf_function(prim, 0, 1, 2, 3, 4, 5);
    Ok(build_handcrafted_single_leaf_artifact(
        format!("handcrafted_matrix_affine_rank2_{}", prim_label(prim)),
        vec![
            WasmParamAbi::Scalar { prim },
            WasmParamAbi::Scalar { prim },
            WasmParamAbi::Bulk { prim },
        ],
        prim,
        2,
        vec![0, 1, 2],
        vec![
            handcrafted_val_type(prim),
            handcrafted_val_type(prim),
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
        ],
        function,
        IntentClass::MapTernaryBroadcast,
    ))
}

fn handcrafted_matrix_axpy2_record_artifact(prim: Prim) -> Result<WasmArtifact> {
    let mut leaf_exports = Vec::new();
    let function_x = build_handcrafted_axpy_like_leaf_function(
        prim,
        0,
        1,
        2,
        5,
        6,
        9,
        10,
        HandcraftedYMode::Unity,
    );
    leaf_exports.push(WasmLeafExport {
        leaf_path: LeafPath(vec!["x".to_string()]),
        export_name: "main$x".to_string(),
        result: WasmLeafResultAbi::Bulk { prim },
        used_param_leaves: vec![0, 1, 3],
        reusable_param_leaf: None,
    });
    let function_y = build_handcrafted_axpy_like_leaf_function(
        prim,
        0,
        3,
        4,
        7,
        8,
        9,
        10,
        HandcraftedYMode::Unity,
    );
    leaf_exports.push(WasmLeafExport {
        leaf_path: LeafPath(vec!["y".to_string()]),
        export_name: "main$y".to_string(),
        result: WasmLeafResultAbi::Bulk { prim },
        used_param_leaves: vec![0, 2, 4],
        reusable_param_leaf: None,
    });

    let bytes = build_handcrafted_module_bytes(
        &[
            handcrafted_val_type(prim),
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
        ],
        vec![function_x, function_y],
        vec![
            ("main".to_string(), 0),
            ("main$x".to_string(), 0),
            ("main$y".to_string(), 1),
        ],
    );

    Ok(WasmArtifact {
        bytes,
        export_name: "main".to_string(),
        params: vec![
            WasmParamAbi::Scalar { prim },
            WasmParamAbi::Record {
                fields: vec![
                    ("x".to_string(), WasmParamAbi::Bulk { prim }),
                    ("y".to_string(), WasmParamAbi::Bulk { prim }),
                ],
            },
            WasmParamAbi::Record {
                fields: vec![
                    ("x".to_string(), WasmParamAbi::Bulk { prim }),
                    ("y".to_string(), WasmParamAbi::Bulk { prim }),
                ],
            },
        ],
        result: WasmResultAbi::Record {
            fields: vec![
                (
                    "x".to_string(),
                    WasmResultAbi::Bulk {
                        prim,
                        shape_param: 1,
                    },
                ),
                (
                    "y".to_string(),
                    WasmResultAbi::Bulk {
                        prim,
                        shape_param: 1,
                    },
                ),
            ],
        },
        grouped_export: None,
        leaf_exports,
        optimizer_reports: vec![WasmOptimizationReport {
            function: format!("handcrafted_matrix_axpy2_record_{}", prim_label(prim)),
            intent: IntentClass::GroupedMap,
            vectorizable: true,
            vector_unroll: 1,
            fallback_reason: None,
        }],
    })
}

fn handcrafted_matrix_image_tone_rgb_f32_artifact() -> Result<WasmArtifact> {
    let prim = Prim::F32;
    let function_r = build_handcrafted_affine_const_bias_leaf_function(prim, 0, 1, 2, 7, 8, 0.05);
    let function_g = build_handcrafted_affine_const_bias_leaf_function(prim, 0, 3, 4, 7, 8, 0.03);
    let function_b = build_handcrafted_affine_const_bias_leaf_function(prim, 0, 5, 6, 7, 8, 0.01);
    let bytes = build_handcrafted_module_bytes(
        &[
            ValType::F32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
        ],
        vec![function_r, function_g, function_b],
        vec![
            ("main".to_string(), 0),
            ("main$r".to_string(), 0),
            ("main$g".to_string(), 1),
            ("main$b".to_string(), 2),
        ],
    );

    Ok(WasmArtifact {
        bytes,
        export_name: "main".to_string(),
        params: vec![
            WasmParamAbi::Scalar { prim },
            WasmParamAbi::Record {
                fields: vec![
                    ("r".to_string(), WasmParamAbi::Bulk { prim }),
                    ("g".to_string(), WasmParamAbi::Bulk { prim }),
                    ("b".to_string(), WasmParamAbi::Bulk { prim }),
                ],
            },
        ],
        result: WasmResultAbi::Record {
            fields: vec![
                (
                    "r".to_string(),
                    WasmResultAbi::Bulk {
                        prim,
                        shape_param: 1,
                    },
                ),
                (
                    "g".to_string(),
                    WasmResultAbi::Bulk {
                        prim,
                        shape_param: 1,
                    },
                ),
                (
                    "b".to_string(),
                    WasmResultAbi::Bulk {
                        prim,
                        shape_param: 1,
                    },
                ),
            ],
        },
        grouped_export: None,
        leaf_exports: vec![
            WasmLeafExport {
                leaf_path: LeafPath(vec!["r".to_string()]),
                export_name: "main$r".to_string(),
                result: WasmLeafResultAbi::Bulk { prim },
                used_param_leaves: vec![0, 1],
                reusable_param_leaf: None,
            },
            WasmLeafExport {
                leaf_path: LeafPath(vec!["g".to_string()]),
                export_name: "main$g".to_string(),
                result: WasmLeafResultAbi::Bulk { prim },
                used_param_leaves: vec![0, 2],
                reusable_param_leaf: None,
            },
            WasmLeafExport {
                leaf_path: LeafPath(vec!["b".to_string()]),
                export_name: "main$b".to_string(),
                result: WasmLeafResultAbi::Bulk { prim },
                used_param_leaves: vec![0, 3],
                reusable_param_leaf: None,
            },
        ],
        optimizer_reports: vec![WasmOptimizationReport {
            function: "handcrafted_matrix_image_tone_rgb_f32".to_string(),
            intent: IntentClass::GroupedMap,
            vectorizable: true,
            vector_unroll: 1,
            fallback_reason: None,
        }],
    })
}

fn handcrafted_matrix_image_blend_rgb_f32_artifact() -> Result<WasmArtifact> {
    let prim = Prim::F32;
    let function_r = build_handcrafted_axpy_like_leaf_function(
        prim,
        0,
        1,
        2,
        7,
        8,
        13,
        14,
        HandcraftedYMode::OneMinusScale,
    );
    let function_g = build_handcrafted_axpy_like_leaf_function(
        prim,
        0,
        3,
        4,
        9,
        10,
        13,
        14,
        HandcraftedYMode::OneMinusScale,
    );
    let function_b = build_handcrafted_axpy_like_leaf_function(
        prim,
        0,
        5,
        6,
        11,
        12,
        13,
        14,
        HandcraftedYMode::OneMinusScale,
    );
    let bytes = build_handcrafted_module_bytes(
        &[
            ValType::F32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
        ],
        vec![function_r, function_g, function_b],
        vec![
            ("main".to_string(), 0),
            ("main$r".to_string(), 0),
            ("main$g".to_string(), 1),
            ("main$b".to_string(), 2),
        ],
    );
    Ok(WasmArtifact {
        bytes,
        export_name: "main".to_string(),
        params: vec![
            WasmParamAbi::Scalar { prim },
            WasmParamAbi::Record {
                fields: vec![
                    ("r".to_string(), WasmParamAbi::Bulk { prim }),
                    ("g".to_string(), WasmParamAbi::Bulk { prim }),
                    ("b".to_string(), WasmParamAbi::Bulk { prim }),
                ],
            },
            WasmParamAbi::Record {
                fields: vec![
                    ("r".to_string(), WasmParamAbi::Bulk { prim }),
                    ("g".to_string(), WasmParamAbi::Bulk { prim }),
                    ("b".to_string(), WasmParamAbi::Bulk { prim }),
                ],
            },
        ],
        result: WasmResultAbi::Record {
            fields: vec![
                (
                    "r".to_string(),
                    WasmResultAbi::Bulk {
                        prim,
                        shape_param: 1,
                    },
                ),
                (
                    "g".to_string(),
                    WasmResultAbi::Bulk {
                        prim,
                        shape_param: 1,
                    },
                ),
                (
                    "b".to_string(),
                    WasmResultAbi::Bulk {
                        prim,
                        shape_param: 1,
                    },
                ),
            ],
        },
        grouped_export: None,
        leaf_exports: vec![
            WasmLeafExport {
                leaf_path: LeafPath(vec!["r".to_string()]),
                export_name: "main$r".to_string(),
                result: WasmLeafResultAbi::Bulk { prim },
                used_param_leaves: vec![0, 1, 4],
                reusable_param_leaf: None,
            },
            WasmLeafExport {
                leaf_path: LeafPath(vec!["g".to_string()]),
                export_name: "main$g".to_string(),
                result: WasmLeafResultAbi::Bulk { prim },
                used_param_leaves: vec![0, 2, 5],
                reusable_param_leaf: None,
            },
            WasmLeafExport {
                leaf_path: LeafPath(vec!["b".to_string()]),
                export_name: "main$b".to_string(),
                result: WasmLeafResultAbi::Bulk { prim },
                used_param_leaves: vec![0, 3, 6],
                reusable_param_leaf: None,
            },
        ],
        optimizer_reports: vec![WasmOptimizationReport {
            function: "handcrafted_matrix_image_blend_rgb_f32".to_string(),
            intent: IntentClass::GroupedMap,
            vectorizable: true,
            vector_unroll: 1,
            fallback_reason: None,
        }],
    })
}

fn build_handcrafted_single_leaf_artifact(
    report_name: String,
    params: Vec<WasmParamAbi>,
    result_prim: Prim,
    shape_param: usize,
    used_param_leaves: Vec<usize>,
    param_types: Vec<ValType>,
    function: Function,
    intent: IntentClass,
) -> WasmArtifact {
    let bytes =
        build_handcrafted_module_bytes(&param_types, vec![function], vec![("main".to_string(), 0)]);
    WasmArtifact {
        bytes,
        export_name: "main".to_string(),
        params,
        result: WasmResultAbi::Bulk {
            prim: result_prim,
            shape_param,
        },
        grouped_export: None,
        leaf_exports: vec![WasmLeafExport {
            leaf_path: LeafPath::root(),
            export_name: "main".to_string(),
            result: WasmLeafResultAbi::Bulk { prim: result_prim },
            used_param_leaves,
            reusable_param_leaf: None,
        }],
        optimizer_reports: vec![WasmOptimizationReport {
            function: report_name,
            intent,
            vectorizable: true,
            vector_unroll: 1,
            fallback_reason: None,
        }],
    }
}

fn build_handcrafted_module_bytes(
    param_types: &[ValType],
    functions: Vec<Function>,
    exports: Vec<(String, u32)>,
) -> Vec<u8> {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function(param_types.to_vec(), []);
    let mut function_section = FunctionSection::new();
    for _ in &functions {
        function_section.function(0);
    }
    let mut memories = MemorySection::new();
    memories.memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    let mut export_section = ExportSection::new();
    export_section.export("memory", ExportKind::Memory, 0);
    for (name, index) in exports {
        export_section.export(&name, ExportKind::Func, index);
    }
    let mut code = CodeSection::new();
    for function in functions {
        code.function(&function);
    }
    module.section(&types);
    module.section(&function_section);
    module.section(&memories);
    module.section(&export_section);
    module.section(&code);
    module.finish()
}

fn build_handcrafted_unary_leaf_function(
    prim: Prim,
    input_ptr_index: u32,
    input_len_index: u32,
    output_ptr_index: u32,
    output_len_index: u32,
    op: HandcraftedUnaryOp,
) -> Function {
    let index_local = output_len_index + 1;
    let vector_end_local = output_len_index + 2;
    let input_ptr_local = output_len_index + 3;
    let output_ptr_local = output_len_index + 4;
    let scalar_local = output_len_index + 5;
    let vector_step = handcrafted_lane_count(prim) as i32;
    let scalar_step = handcrafted_element_bytes(prim);
    let vector_step_bytes = vector_step * scalar_step;

    let mut function = Function::new(vec![(4, ValType::I32), (1, handcrafted_val_type(prim))]);
    function.instruction(&Instruction::LocalGet(output_len_index));
    function.instruction(&Instruction::LocalGet(input_len_index));
    function.instruction(&Instruction::I32Ne);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);

    function.instruction(&Instruction::I32Const(0));
    function.instruction(&Instruction::LocalSet(index_local));
    function.instruction(&Instruction::LocalGet(input_ptr_index));
    function.instruction(&Instruction::LocalSet(input_ptr_local));
    function.instruction(&Instruction::LocalGet(output_ptr_index));
    function.instruction(&Instruction::LocalSet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(input_len_index));
    function.instruction(&Instruction::LocalGet(input_len_index));
    function.instruction(&Instruction::I32Const(vector_step));
    function.instruction(&Instruction::I32RemU);
    function.instruction(&Instruction::I32Sub);
    function.instruction(&Instruction::LocalSet(vector_end_local));

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::LocalGet(vector_end_local));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(input_ptr_local));
    function.instruction(&Instruction::V128Load(handcrafted_memarg(0, 4)));
    match op {
        HandcraftedUnaryOp::AddOne => {
            emit_vector_splat_const(&mut function, prim, 1.0);
            emit_vector_add(&mut function, prim);
        }
        HandcraftedUnaryOp::Square => {
            function.instruction(&Instruction::LocalGet(input_ptr_local));
            function.instruction(&Instruction::V128Load(handcrafted_memarg(0, 4)));
            emit_vector_mul(&mut function, prim);
        }
    }
    function.instruction(&Instruction::V128Store(handcrafted_memarg(0, 4)));
    function.instruction(&Instruction::LocalGet(input_ptr_local));
    function.instruction(&Instruction::I32Const(vector_step_bytes));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(input_ptr_local));
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    function.instruction(&Instruction::I32Const(vector_step_bytes));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::I32Const(vector_step));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(index_local));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::LocalGet(input_len_index));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(input_ptr_local));
    emit_scalar_load(&mut function, prim);
    function.instruction(&Instruction::LocalSet(scalar_local));
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(scalar_local));
    match op {
        HandcraftedUnaryOp::AddOne => {
            emit_scalar_const(&mut function, prim, 1.0);
            emit_scalar_add(&mut function, prim);
        }
        HandcraftedUnaryOp::Square => {
            function.instruction(&Instruction::LocalGet(scalar_local));
            emit_scalar_mul(&mut function, prim);
        }
    }
    emit_scalar_store(&mut function, prim);
    function.instruction(&Instruction::LocalGet(input_ptr_local));
    function.instruction(&Instruction::I32Const(scalar_step));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(input_ptr_local));
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    function.instruction(&Instruction::I32Const(scalar_step));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::I32Const(1));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(index_local));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function
}

fn build_handcrafted_affine_scalar_bias_leaf_function(
    prim: Prim,
    scale_index: u32,
    bias_index: u32,
    input_ptr_index: u32,
    input_len_index: u32,
    output_ptr_index: u32,
    output_len_index: u32,
) -> Function {
    let index_local = output_len_index + 1;
    let vector_end_local = output_len_index + 2;
    let input_ptr_local = output_len_index + 3;
    let output_ptr_local = output_len_index + 4;
    let x_local = output_len_index + 5;
    let vector_step = handcrafted_lane_count(prim) as i32;
    let scalar_step = handcrafted_element_bytes(prim);
    let vector_step_bytes = vector_step * scalar_step;

    let mut function = Function::new(vec![(4, ValType::I32), (1, handcrafted_val_type(prim))]);
    function.instruction(&Instruction::LocalGet(output_len_index));
    function.instruction(&Instruction::LocalGet(input_len_index));
    function.instruction(&Instruction::I32Ne);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);

    function.instruction(&Instruction::I32Const(0));
    function.instruction(&Instruction::LocalSet(index_local));
    function.instruction(&Instruction::LocalGet(input_ptr_index));
    function.instruction(&Instruction::LocalSet(input_ptr_local));
    function.instruction(&Instruction::LocalGet(output_ptr_index));
    function.instruction(&Instruction::LocalSet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(input_len_index));
    function.instruction(&Instruction::LocalGet(input_len_index));
    function.instruction(&Instruction::I32Const(vector_step));
    function.instruction(&Instruction::I32RemU);
    function.instruction(&Instruction::I32Sub);
    function.instruction(&Instruction::LocalSet(vector_end_local));

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::LocalGet(vector_end_local));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    emit_vector_splat_param(&mut function, prim, scale_index);
    function.instruction(&Instruction::LocalGet(input_ptr_local));
    function.instruction(&Instruction::V128Load(handcrafted_memarg(0, 4)));
    emit_vector_mul(&mut function, prim);
    emit_vector_splat_param(&mut function, prim, bias_index);
    emit_vector_add(&mut function, prim);
    function.instruction(&Instruction::V128Store(handcrafted_memarg(0, 4)));
    function.instruction(&Instruction::LocalGet(input_ptr_local));
    function.instruction(&Instruction::I32Const(vector_step_bytes));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(input_ptr_local));
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    function.instruction(&Instruction::I32Const(vector_step_bytes));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::I32Const(vector_step));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(index_local));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::LocalGet(input_len_index));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(input_ptr_local));
    emit_scalar_load(&mut function, prim);
    function.instruction(&Instruction::LocalSet(x_local));
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(scale_index));
    function.instruction(&Instruction::LocalGet(x_local));
    emit_scalar_mul(&mut function, prim);
    function.instruction(&Instruction::LocalGet(bias_index));
    emit_scalar_add(&mut function, prim);
    emit_scalar_store(&mut function, prim);
    function.instruction(&Instruction::LocalGet(input_ptr_local));
    function.instruction(&Instruction::I32Const(scalar_step));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(input_ptr_local));
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    function.instruction(&Instruction::I32Const(scalar_step));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::I32Const(1));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(index_local));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function
}

fn build_handcrafted_affine_const_bias_leaf_function(
    prim: Prim,
    scale_index: u32,
    input_ptr_index: u32,
    input_len_index: u32,
    output_ptr_index: u32,
    output_len_index: u32,
    bias: f64,
) -> Function {
    let index_local = output_len_index + 1;
    let vector_end_local = output_len_index + 2;
    let input_ptr_local = output_len_index + 3;
    let output_ptr_local = output_len_index + 4;
    let x_local = output_len_index + 5;
    let vector_step = handcrafted_lane_count(prim) as i32;
    let scalar_step = handcrafted_element_bytes(prim);
    let vector_step_bytes = vector_step * scalar_step;

    let mut function = Function::new(vec![(4, ValType::I32), (1, handcrafted_val_type(prim))]);
    function.instruction(&Instruction::LocalGet(output_len_index));
    function.instruction(&Instruction::LocalGet(input_len_index));
    function.instruction(&Instruction::I32Ne);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);

    function.instruction(&Instruction::I32Const(0));
    function.instruction(&Instruction::LocalSet(index_local));
    function.instruction(&Instruction::LocalGet(input_ptr_index));
    function.instruction(&Instruction::LocalSet(input_ptr_local));
    function.instruction(&Instruction::LocalGet(output_ptr_index));
    function.instruction(&Instruction::LocalSet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(input_len_index));
    function.instruction(&Instruction::LocalGet(input_len_index));
    function.instruction(&Instruction::I32Const(vector_step));
    function.instruction(&Instruction::I32RemU);
    function.instruction(&Instruction::I32Sub);
    function.instruction(&Instruction::LocalSet(vector_end_local));

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::LocalGet(vector_end_local));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    emit_vector_splat_param(&mut function, prim, scale_index);
    function.instruction(&Instruction::LocalGet(input_ptr_local));
    function.instruction(&Instruction::V128Load(handcrafted_memarg(0, 4)));
    emit_vector_mul(&mut function, prim);
    emit_vector_splat_const(&mut function, prim, bias);
    emit_vector_add(&mut function, prim);
    function.instruction(&Instruction::V128Store(handcrafted_memarg(0, 4)));
    function.instruction(&Instruction::LocalGet(input_ptr_local));
    function.instruction(&Instruction::I32Const(vector_step_bytes));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(input_ptr_local));
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    function.instruction(&Instruction::I32Const(vector_step_bytes));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::I32Const(vector_step));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(index_local));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::LocalGet(input_len_index));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(input_ptr_local));
    emit_scalar_load(&mut function, prim);
    function.instruction(&Instruction::LocalSet(x_local));
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(scale_index));
    function.instruction(&Instruction::LocalGet(x_local));
    emit_scalar_mul(&mut function, prim);
    emit_scalar_const(&mut function, prim, bias);
    emit_scalar_add(&mut function, prim);
    emit_scalar_store(&mut function, prim);
    function.instruction(&Instruction::LocalGet(input_ptr_local));
    function.instruction(&Instruction::I32Const(scalar_step));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(input_ptr_local));
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    function.instruction(&Instruction::I32Const(scalar_step));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::I32Const(1));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(index_local));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function
}

#[allow(clippy::too_many_arguments)]
fn build_handcrafted_axpy_like_leaf_function(
    prim: Prim,
    scale_index: u32,
    x_ptr_index: u32,
    x_len_index: u32,
    y_ptr_index: u32,
    y_len_index: u32,
    output_ptr_index: u32,
    output_len_index: u32,
    y_mode: HandcraftedYMode,
) -> Function {
    let index_local = output_len_index + 1;
    let vector_end_local = output_len_index + 2;
    let x_ptr_local = output_len_index + 3;
    let y_ptr_local = output_len_index + 4;
    let output_ptr_local = output_len_index + 5;
    let x_local = output_len_index + 6;
    let y_local = output_len_index + 7;
    let inv_local = output_len_index + 8;
    let scalar_locals = if matches!(y_mode, HandcraftedYMode::OneMinusScale) {
        3
    } else {
        2
    };
    let vector_step = handcrafted_lane_count(prim) as i32;
    let scalar_step = handcrafted_element_bytes(prim);
    let vector_step_bytes = vector_step * scalar_step;

    let mut function = Function::new(vec![
        (5, ValType::I32),
        (scalar_locals, handcrafted_val_type(prim)),
    ]);
    function.instruction(&Instruction::LocalGet(y_len_index));
    function.instruction(&Instruction::LocalGet(x_len_index));
    function.instruction(&Instruction::I32Ne);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::LocalGet(output_len_index));
    function.instruction(&Instruction::LocalGet(x_len_index));
    function.instruction(&Instruction::I32Ne);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);

    if matches!(y_mode, HandcraftedYMode::OneMinusScale) {
        emit_scalar_const(&mut function, prim, 1.0);
        function.instruction(&Instruction::LocalGet(scale_index));
        emit_scalar_sub(&mut function, prim);
        function.instruction(&Instruction::LocalSet(inv_local));
    }

    function.instruction(&Instruction::I32Const(0));
    function.instruction(&Instruction::LocalSet(index_local));
    function.instruction(&Instruction::LocalGet(x_ptr_index));
    function.instruction(&Instruction::LocalSet(x_ptr_local));
    function.instruction(&Instruction::LocalGet(y_ptr_index));
    function.instruction(&Instruction::LocalSet(y_ptr_local));
    function.instruction(&Instruction::LocalGet(output_ptr_index));
    function.instruction(&Instruction::LocalSet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(x_len_index));
    function.instruction(&Instruction::LocalGet(x_len_index));
    function.instruction(&Instruction::I32Const(vector_step));
    function.instruction(&Instruction::I32RemU);
    function.instruction(&Instruction::I32Sub);
    function.instruction(&Instruction::LocalSet(vector_end_local));

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::LocalGet(vector_end_local));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    emit_vector_splat_param(&mut function, prim, scale_index);
    function.instruction(&Instruction::LocalGet(x_ptr_local));
    function.instruction(&Instruction::V128Load(handcrafted_memarg(0, 4)));
    emit_vector_mul(&mut function, prim);
    function.instruction(&Instruction::LocalGet(y_ptr_local));
    function.instruction(&Instruction::V128Load(handcrafted_memarg(0, 4)));
    if matches!(y_mode, HandcraftedYMode::OneMinusScale) {
        emit_vector_splat_local(&mut function, prim, inv_local);
        emit_vector_mul(&mut function, prim);
    }
    emit_vector_add(&mut function, prim);
    function.instruction(&Instruction::V128Store(handcrafted_memarg(0, 4)));
    function.instruction(&Instruction::LocalGet(x_ptr_local));
    function.instruction(&Instruction::I32Const(vector_step_bytes));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(x_ptr_local));
    function.instruction(&Instruction::LocalGet(y_ptr_local));
    function.instruction(&Instruction::I32Const(vector_step_bytes));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(y_ptr_local));
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    function.instruction(&Instruction::I32Const(vector_step_bytes));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::I32Const(vector_step));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(index_local));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::LocalGet(x_len_index));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(x_ptr_local));
    emit_scalar_load(&mut function, prim);
    function.instruction(&Instruction::LocalSet(x_local));
    function.instruction(&Instruction::LocalGet(y_ptr_local));
    emit_scalar_load(&mut function, prim);
    function.instruction(&Instruction::LocalSet(y_local));
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(scale_index));
    function.instruction(&Instruction::LocalGet(x_local));
    emit_scalar_mul(&mut function, prim);
    match y_mode {
        HandcraftedYMode::Unity => {
            function.instruction(&Instruction::LocalGet(y_local));
        }
        HandcraftedYMode::OneMinusScale => {
            function.instruction(&Instruction::LocalGet(inv_local));
            function.instruction(&Instruction::LocalGet(y_local));
            emit_scalar_mul(&mut function, prim);
        }
    }
    emit_scalar_add(&mut function, prim);
    emit_scalar_store(&mut function, prim);
    function.instruction(&Instruction::LocalGet(x_ptr_local));
    function.instruction(&Instruction::I32Const(scalar_step));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(x_ptr_local));
    function.instruction(&Instruction::LocalGet(y_ptr_local));
    function.instruction(&Instruction::I32Const(scalar_step));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(y_ptr_local));
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    function.instruction(&Instruction::I32Const(scalar_step));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(output_ptr_local));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::I32Const(1));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(index_local));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function
}

fn handcrafted_val_type(prim: Prim) -> ValType {
    match prim {
        Prim::I32 => ValType::I32,
        Prim::I64 => ValType::I64,
        Prim::F32 => ValType::F32,
        Prim::F64 => ValType::F64,
    }
}

fn handcrafted_lane_count(prim: Prim) -> u32 {
    match prim {
        Prim::I32 | Prim::F32 => 4,
        Prim::I64 | Prim::F64 => 2,
    }
}

fn handcrafted_element_bytes(prim: Prim) -> i32 {
    match prim {
        Prim::I32 | Prim::F32 => 4,
        Prim::I64 | Prim::F64 => 8,
    }
}

fn handcrafted_scalar_align_log2(prim: Prim) -> u32 {
    match prim {
        Prim::I32 | Prim::F32 => 2,
        Prim::I64 | Prim::F64 => 3,
    }
}

fn emit_scalar_const(function: &mut Function, prim: Prim, value: f64) {
    match prim {
        Prim::I32 => {
            function.instruction(&Instruction::I32Const(value as i32));
        }
        Prim::I64 => {
            function.instruction(&Instruction::I64Const(value as i64));
        }
        Prim::F32 => {
            function.instruction(&Instruction::F32Const((value as f32).into()));
        }
        Prim::F64 => {
            function.instruction(&Instruction::F64Const(value.into()));
        }
    };
}

fn emit_vector_splat_const(function: &mut Function, prim: Prim, value: f64) {
    emit_scalar_const(function, prim, value);
    match prim {
        Prim::I32 => function.instruction(&Instruction::I32x4Splat),
        Prim::I64 => function.instruction(&Instruction::I64x2Splat),
        Prim::F32 => function.instruction(&Instruction::F32x4Splat),
        Prim::F64 => function.instruction(&Instruction::F64x2Splat),
    };
}

fn emit_vector_splat_param(function: &mut Function, prim: Prim, param_index: u32) {
    function.instruction(&Instruction::LocalGet(param_index));
    match prim {
        Prim::I32 => function.instruction(&Instruction::I32x4Splat),
        Prim::I64 => function.instruction(&Instruction::I64x2Splat),
        Prim::F32 => function.instruction(&Instruction::F32x4Splat),
        Prim::F64 => function.instruction(&Instruction::F64x2Splat),
    };
}

fn emit_vector_splat_local(function: &mut Function, prim: Prim, local_index: u32) {
    function.instruction(&Instruction::LocalGet(local_index));
    match prim {
        Prim::I32 => function.instruction(&Instruction::I32x4Splat),
        Prim::I64 => function.instruction(&Instruction::I64x2Splat),
        Prim::F32 => function.instruction(&Instruction::F32x4Splat),
        Prim::F64 => function.instruction(&Instruction::F64x2Splat),
    };
}

fn emit_vector_add(function: &mut Function, prim: Prim) {
    match prim {
        Prim::I32 => function.instruction(&Instruction::I32x4Add),
        Prim::I64 => function.instruction(&Instruction::I64x2Add),
        Prim::F32 => function.instruction(&Instruction::F32x4Add),
        Prim::F64 => function.instruction(&Instruction::F64x2Add),
    };
}

fn emit_vector_mul(function: &mut Function, prim: Prim) {
    match prim {
        Prim::I32 => function.instruction(&Instruction::I32x4Mul),
        Prim::I64 => function.instruction(&Instruction::I64x2Mul),
        Prim::F32 => function.instruction(&Instruction::F32x4Mul),
        Prim::F64 => function.instruction(&Instruction::F64x2Mul),
    };
}

fn emit_scalar_add(function: &mut Function, prim: Prim) {
    match prim {
        Prim::I32 => function.instruction(&Instruction::I32Add),
        Prim::I64 => function.instruction(&Instruction::I64Add),
        Prim::F32 => function.instruction(&Instruction::F32Add),
        Prim::F64 => function.instruction(&Instruction::F64Add),
    };
}

fn emit_scalar_sub(function: &mut Function, prim: Prim) {
    match prim {
        Prim::I32 => function.instruction(&Instruction::I32Sub),
        Prim::I64 => function.instruction(&Instruction::I64Sub),
        Prim::F32 => function.instruction(&Instruction::F32Sub),
        Prim::F64 => function.instruction(&Instruction::F64Sub),
    };
}

fn emit_scalar_mul(function: &mut Function, prim: Prim) {
    match prim {
        Prim::I32 => function.instruction(&Instruction::I32Mul),
        Prim::I64 => function.instruction(&Instruction::I64Mul),
        Prim::F32 => function.instruction(&Instruction::F32Mul),
        Prim::F64 => function.instruction(&Instruction::F64Mul),
    };
}

fn emit_scalar_load(function: &mut Function, prim: Prim) {
    let align = handcrafted_scalar_align_log2(prim);
    match prim {
        Prim::I32 => function.instruction(&Instruction::I32Load(handcrafted_memarg(0, align))),
        Prim::I64 => function.instruction(&Instruction::I64Load(handcrafted_memarg(0, align))),
        Prim::F32 => function.instruction(&Instruction::F32Load(handcrafted_memarg(0, align))),
        Prim::F64 => function.instruction(&Instruction::F64Load(handcrafted_memarg(0, align))),
    };
}

fn emit_scalar_store(function: &mut Function, prim: Prim) {
    let align = handcrafted_scalar_align_log2(prim);
    match prim {
        Prim::I32 => function.instruction(&Instruction::I32Store(handcrafted_memarg(0, align))),
        Prim::I64 => function.instruction(&Instruction::I64Store(handcrafted_memarg(0, align))),
        Prim::F32 => function.instruction(&Instruction::F32Store(handcrafted_memarg(0, align))),
        Prim::F64 => function.instruction(&Instruction::F64Store(handcrafted_memarg(0, align))),
    };
}

#[derive(Clone, Copy)]
enum HandcraftedI64UnaryOp {
    AddConst(i64),
    MulConst(i64),
}

fn handcrafted_inc_i64_artifact() -> Result<WasmArtifact> {
    handcrafted_unary_i64_artifact("handcrafted_inc_i64", HandcraftedI64UnaryOp::AddConst(1))
}

fn handcrafted_pow2_i64_artifact() -> Result<WasmArtifact> {
    handcrafted_unary_i64_artifact("handcrafted_pow2_i64", HandcraftedI64UnaryOp::MulConst(8))
}

fn handcrafted_unary_i64_artifact(
    report_name: &str,
    op: HandcraftedI64UnaryOp,
) -> Result<WasmArtifact> {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types
        .ty()
        .function([ValType::I32, ValType::I32, ValType::I32, ValType::I32], []);
    let mut functions = FunctionSection::new();
    functions.function(0);
    let mut memories = MemorySection::new();
    memories.memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("main", ExportKind::Func, 0);
    let mut code = CodeSection::new();
    let mut function = Function::new(vec![(5, ValType::I32), (1, ValType::I64)]);

    function.instruction(&Instruction::LocalGet(3));
    function.instruction(&Instruction::LocalGet(1));
    function.instruction(&Instruction::I32Ne);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);

    function.instruction(&Instruction::I32Const(0));
    function.instruction(&Instruction::LocalSet(4));
    function.instruction(&Instruction::LocalGet(0));
    function.instruction(&Instruction::LocalSet(6));
    function.instruction(&Instruction::LocalGet(2));
    function.instruction(&Instruction::LocalSet(7));
    function.instruction(&Instruction::LocalGet(1));
    function.instruction(&Instruction::LocalGet(1));
    function.instruction(&Instruction::I32Const(2));
    function.instruction(&Instruction::I32RemU);
    function.instruction(&Instruction::I32Sub);
    function.instruction(&Instruction::LocalSet(5));

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(4));
    function.instruction(&Instruction::LocalGet(5));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(7));
    function.instruction(&Instruction::LocalGet(6));
    function.instruction(&Instruction::V128Load(handcrafted_memarg(0, 4)));
    match op {
        HandcraftedI64UnaryOp::AddConst(value) => {
            function.instruction(&Instruction::I64Const(value));
            function.instruction(&Instruction::I64x2Splat);
            function.instruction(&Instruction::I64x2Add);
        }
        HandcraftedI64UnaryOp::MulConst(value) => {
            function.instruction(&Instruction::I64Const(value));
            function.instruction(&Instruction::I64x2Splat);
            function.instruction(&Instruction::I64x2Mul);
        }
    }
    function.instruction(&Instruction::V128Store(handcrafted_memarg(0, 4)));
    function.instruction(&Instruction::LocalGet(6));
    function.instruction(&Instruction::I32Const(16));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(6));
    function.instruction(&Instruction::LocalGet(7));
    function.instruction(&Instruction::I32Const(16));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(7));
    function.instruction(&Instruction::LocalGet(4));
    function.instruction(&Instruction::I32Const(2));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(4));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(4));
    function.instruction(&Instruction::LocalGet(1));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(6));
    function.instruction(&Instruction::I64Load(handcrafted_memarg(0, 3)));
    function.instruction(&Instruction::LocalSet(9));
    function.instruction(&Instruction::LocalGet(7));
    function.instruction(&Instruction::LocalGet(9));
    match op {
        HandcraftedI64UnaryOp::AddConst(value) => {
            function.instruction(&Instruction::I64Const(value));
            function.instruction(&Instruction::I64Add);
        }
        HandcraftedI64UnaryOp::MulConst(value) => {
            function.instruction(&Instruction::I64Const(value));
            function.instruction(&Instruction::I64Mul);
        }
    }
    function.instruction(&Instruction::I64Store(handcrafted_memarg(0, 3)));
    function.instruction(&Instruction::LocalGet(6));
    function.instruction(&Instruction::I32Const(8));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(6));
    function.instruction(&Instruction::LocalGet(7));
    function.instruction(&Instruction::I32Const(8));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(7));
    function.instruction(&Instruction::LocalGet(4));
    function.instruction(&Instruction::I32Const(1));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(4));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    code.function(&function);

    module.section(&types);
    module.section(&functions);
    module.section(&memories);
    module.section(&exports);
    module.section(&code);

    Ok(WasmArtifact {
        bytes: module.finish(),
        export_name: "main".to_string(),
        params: vec![WasmParamAbi::Bulk { prim: Prim::I64 }],
        result: WasmResultAbi::Bulk {
            prim: Prim::I64,
            shape_param: 0,
        },
        grouped_export: None,
        leaf_exports: vec![WasmLeafExport {
            leaf_path: LeafPath::root(),
            export_name: "main".to_string(),
            result: WasmLeafResultAbi::Bulk { prim: Prim::I64 },
            used_param_leaves: vec![0],
            reusable_param_leaf: None,
        }],
        optimizer_reports: vec![WasmOptimizationReport {
            function: report_name.to_string(),
            intent: IntentClass::MapUnary,
            vectorizable: true,
            vector_unroll: 1,
            fallback_reason: None,
        }],
    })
}

fn handcrafted_square_f32_artifact() -> Result<WasmArtifact> {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types
        .ty()
        .function([ValType::I32, ValType::I32, ValType::I32, ValType::I32], []);
    let mut functions = FunctionSection::new();
    functions.function(0);
    let mut memories = MemorySection::new();
    memories.memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("main", ExportKind::Func, 0);
    let mut code = CodeSection::new();
    let mut function = Function::new(vec![(4, ValType::I32), (1, ValType::F32)]);

    function.instruction(&Instruction::LocalGet(3));
    function.instruction(&Instruction::LocalGet(1));
    function.instruction(&Instruction::I32Ne);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);

    function.instruction(&Instruction::I32Const(0));
    function.instruction(&Instruction::LocalSet(4));
    function.instruction(&Instruction::LocalGet(0));
    function.instruction(&Instruction::LocalSet(6));
    function.instruction(&Instruction::LocalGet(2));
    function.instruction(&Instruction::LocalSet(7));
    function.instruction(&Instruction::LocalGet(1));
    function.instruction(&Instruction::LocalGet(1));
    function.instruction(&Instruction::I32Const(4));
    function.instruction(&Instruction::I32RemU);
    function.instruction(&Instruction::I32Sub);
    function.instruction(&Instruction::LocalSet(5));

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(4));
    function.instruction(&Instruction::LocalGet(5));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(7));
    function.instruction(&Instruction::LocalGet(6));
    function.instruction(&Instruction::V128Load(handcrafted_memarg(0, 4)));
    function.instruction(&Instruction::LocalGet(6));
    function.instruction(&Instruction::V128Load(handcrafted_memarg(0, 4)));
    function.instruction(&Instruction::F32x4Mul);
    function.instruction(&Instruction::V128Store(handcrafted_memarg(0, 4)));
    function.instruction(&Instruction::LocalGet(6));
    function.instruction(&Instruction::I32Const(16));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(6));
    function.instruction(&Instruction::LocalGet(7));
    function.instruction(&Instruction::I32Const(16));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(7));
    function.instruction(&Instruction::LocalGet(4));
    function.instruction(&Instruction::I32Const(4));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(4));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(4));
    function.instruction(&Instruction::LocalGet(1));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(6));
    function.instruction(&Instruction::F32Load(handcrafted_memarg(0, 2)));
    function.instruction(&Instruction::LocalTee(8));
    function.instruction(&Instruction::LocalGet(8));
    function.instruction(&Instruction::F32Mul);
    function.instruction(&Instruction::LocalSet(8));
    function.instruction(&Instruction::LocalGet(7));
    function.instruction(&Instruction::LocalGet(8));
    function.instruction(&Instruction::F32Store(handcrafted_memarg(0, 2)));
    function.instruction(&Instruction::LocalGet(6));
    function.instruction(&Instruction::I32Const(4));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(6));
    function.instruction(&Instruction::LocalGet(7));
    function.instruction(&Instruction::I32Const(4));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(7));
    function.instruction(&Instruction::LocalGet(4));
    function.instruction(&Instruction::I32Const(1));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(4));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    code.function(&function);

    module.section(&types);
    module.section(&functions);
    module.section(&memories);
    module.section(&exports);
    module.section(&code);

    Ok(WasmArtifact {
        bytes: module.finish(),
        export_name: "main".to_string(),
        params: vec![WasmParamAbi::Bulk { prim: Prim::F32 }],
        result: WasmResultAbi::Bulk {
            prim: Prim::F32,
            shape_param: 0,
        },
        grouped_export: None,
        leaf_exports: vec![WasmLeafExport {
            leaf_path: LeafPath::root(),
            export_name: "main".to_string(),
            result: WasmLeafResultAbi::Bulk { prim: Prim::F32 },
            used_param_leaves: vec![0],
            reusable_param_leaf: None,
        }],
        optimizer_reports: vec![WasmOptimizationReport {
            function: "handcrafted_square_f32".to_string(),
            intent: IntentClass::MapUnary,
            vectorizable: true,
            vector_unroll: 1,
            fallback_reason: None,
        }],
    })
}

fn handcrafted_axpy_i64_artifact() -> Result<WasmArtifact> {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    types.ty().function(
        [
            ValType::I64,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
        ],
        [],
    );

    let mut functions = FunctionSection::new();
    functions.function(0);

    let mut memories = MemorySection::new();
    memories.memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });

    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("main", ExportKind::Func, 0);

    let mut code = CodeSection::new();
    let mut function = Function::new(vec![(5, ValType::I32), (2, ValType::I64)]);

    // Validate lengths for xs/ys/output buffers.
    function.instruction(&Instruction::LocalGet(4));
    function.instruction(&Instruction::LocalGet(2));
    function.instruction(&Instruction::I32Ne);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);

    function.instruction(&Instruction::LocalGet(6));
    function.instruction(&Instruction::LocalGet(2));
    function.instruction(&Instruction::I32Ne);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);

    // index = 0; cur pointers = base pointers.
    function.instruction(&Instruction::I32Const(0));
    function.instruction(&Instruction::LocalSet(7));
    function.instruction(&Instruction::LocalGet(1));
    function.instruction(&Instruction::LocalSet(9));
    function.instruction(&Instruction::LocalGet(3));
    function.instruction(&Instruction::LocalSet(10));
    function.instruction(&Instruction::LocalGet(5));
    function.instruction(&Instruction::LocalSet(11));

    // vector_end = len - (len % 2)
    function.instruction(&Instruction::LocalGet(2));
    function.instruction(&Instruction::LocalGet(2));
    function.instruction(&Instruction::I32Const(2));
    function.instruction(&Instruction::I32RemU);
    function.instruction(&Instruction::I32Sub);
    function.instruction(&Instruction::LocalSet(8));

    // Vector loop (i64x2 lanes).
    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(7));
    function.instruction(&Instruction::LocalGet(8));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(11));
    function.instruction(&Instruction::LocalGet(0));
    function.instruction(&Instruction::I64x2Splat);
    function.instruction(&Instruction::LocalGet(9));
    function.instruction(&Instruction::V128Load(handcrafted_memarg(0, 4)));
    function.instruction(&Instruction::I64x2Mul);
    function.instruction(&Instruction::LocalGet(10));
    function.instruction(&Instruction::V128Load(handcrafted_memarg(0, 4)));
    function.instruction(&Instruction::I64x2Add);
    function.instruction(&Instruction::V128Store(handcrafted_memarg(0, 4)));
    function.instruction(&Instruction::LocalGet(9));
    function.instruction(&Instruction::I32Const(16));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(9));
    function.instruction(&Instruction::LocalGet(10));
    function.instruction(&Instruction::I32Const(16));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(10));
    function.instruction(&Instruction::LocalGet(11));
    function.instruction(&Instruction::I32Const(16));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(11));
    function.instruction(&Instruction::LocalGet(7));
    function.instruction(&Instruction::I32Const(2));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(7));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);

    // Scalar tail loop.
    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(7));
    function.instruction(&Instruction::LocalGet(2));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(9));
    function.instruction(&Instruction::I64Load(handcrafted_memarg(0, 3)));
    function.instruction(&Instruction::LocalSet(12));
    function.instruction(&Instruction::LocalGet(10));
    function.instruction(&Instruction::I64Load(handcrafted_memarg(0, 3)));
    function.instruction(&Instruction::LocalSet(13));
    function.instruction(&Instruction::LocalGet(11));
    function.instruction(&Instruction::LocalGet(0));
    function.instruction(&Instruction::LocalGet(12));
    function.instruction(&Instruction::I64Mul);
    function.instruction(&Instruction::LocalGet(13));
    function.instruction(&Instruction::I64Add);
    function.instruction(&Instruction::I64Store(handcrafted_memarg(0, 3)));
    function.instruction(&Instruction::LocalGet(9));
    function.instruction(&Instruction::I32Const(8));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(9));
    function.instruction(&Instruction::LocalGet(10));
    function.instruction(&Instruction::I32Const(8));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(10));
    function.instruction(&Instruction::LocalGet(11));
    function.instruction(&Instruction::I32Const(8));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(11));
    function.instruction(&Instruction::LocalGet(7));
    function.instruction(&Instruction::I32Const(1));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(7));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);

    code.function(&function);

    module.section(&types);
    module.section(&functions);
    module.section(&memories);
    module.section(&exports);
    module.section(&code);

    Ok(WasmArtifact {
        bytes: module.finish(),
        export_name: "main".to_string(),
        params: vec![
            WasmParamAbi::Scalar { prim: Prim::I64 },
            WasmParamAbi::Bulk { prim: Prim::I64 },
            WasmParamAbi::Bulk { prim: Prim::I64 },
        ],
        result: WasmResultAbi::Bulk {
            prim: Prim::I64,
            shape_param: 1,
        },
        grouped_export: None,
        leaf_exports: vec![WasmLeafExport {
            leaf_path: LeafPath::root(),
            export_name: "main".to_string(),
            result: WasmLeafResultAbi::Bulk { prim: Prim::I64 },
            used_param_leaves: vec![0, 1, 2],
            reusable_param_leaf: None,
        }],
        optimizer_reports: vec![WasmOptimizationReport {
            function: "handwritten_axpy_i64".to_string(),
            intent: IntentClass::MapTernaryBroadcast,
            vectorizable: true,
            vector_unroll: 1,
            fallback_reason: None,
        }],
    })
}

fn handcrafted_axpy2_record_i64_artifact() -> Result<WasmArtifact> {
    let mut module = Module::new();
    let mut types = TypeSection::new();
    // a, ux(ptr,len), uy(ptr,len), vx(ptr,len), vy(ptr,len), out(ptr,len)
    types.ty().function(
        [
            ValType::I64,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
            ValType::I32,
        ],
        [],
    );
    let mut functions = FunctionSection::new();
    functions.function(0);
    functions.function(0);
    let mut memories = MemorySection::new();
    memories.memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    let mut exports = ExportSection::new();
    exports.export("memory", ExportKind::Memory, 0);
    exports.export("main", ExportKind::Func, 0);
    exports.export("main$x", ExportKind::Func, 0);
    exports.export("main$y", ExportKind::Func, 1);
    let mut code = CodeSection::new();
    code.function(&build_handcrafted_axpy_leaf_i64_function(1, 2, 5, 6, 9, 10));
    code.function(&build_handcrafted_axpy_leaf_i64_function(3, 4, 7, 8, 9, 10));
    module.section(&types);
    module.section(&functions);
    module.section(&memories);
    module.section(&exports);
    module.section(&code);

    let leafs = vec![
        WasmLeafExport {
            leaf_path: LeafPath(vec!["x".to_string()]),
            export_name: "main$x".to_string(),
            result: WasmLeafResultAbi::Bulk { prim: Prim::I64 },
            used_param_leaves: vec![0, 1, 3],
            reusable_param_leaf: None,
        },
        WasmLeafExport {
            leaf_path: LeafPath(vec!["y".to_string()]),
            export_name: "main$y".to_string(),
            result: WasmLeafResultAbi::Bulk { prim: Prim::I64 },
            used_param_leaves: vec![0, 2, 4],
            reusable_param_leaf: None,
        },
    ];

    Ok(WasmArtifact {
        bytes: module.finish(),
        export_name: "main".to_string(),
        params: vec![
            WasmParamAbi::Scalar { prim: Prim::I64 },
            WasmParamAbi::Record {
                fields: vec![
                    ("x".to_string(), WasmParamAbi::Bulk { prim: Prim::I64 }),
                    ("y".to_string(), WasmParamAbi::Bulk { prim: Prim::I64 }),
                ],
            },
            WasmParamAbi::Record {
                fields: vec![
                    ("x".to_string(), WasmParamAbi::Bulk { prim: Prim::I64 }),
                    ("y".to_string(), WasmParamAbi::Bulk { prim: Prim::I64 }),
                ],
            },
        ],
        result: WasmResultAbi::Record {
            fields: vec![
                (
                    "x".to_string(),
                    WasmResultAbi::Bulk {
                        prim: Prim::I64,
                        shape_param: 1,
                    },
                ),
                (
                    "y".to_string(),
                    WasmResultAbi::Bulk {
                        prim: Prim::I64,
                        shape_param: 1,
                    },
                ),
            ],
        },
        grouped_export: None,
        leaf_exports: leafs,
        optimizer_reports: vec![WasmOptimizationReport {
            function: "handcrafted_axpy2_record_i64".to_string(),
            intent: IntentClass::GroupedMap,
            vectorizable: true,
            vector_unroll: 1,
            fallback_reason: None,
        }],
    })
}

fn build_handcrafted_axpy_leaf_i64_function(
    x_ptr_index: u32,
    x_len_index: u32,
    y_ptr_index: u32,
    y_len_index: u32,
    output_ptr_index: u32,
    output_len_index: u32,
) -> Function {
    let mut function = Function::new(vec![(5, ValType::I32), (2, ValType::I64)]);
    let index_local = 11u32;
    let vector_end_local = 12u32;
    let x_ptr_local = 13u32;
    let y_ptr_local = 14u32;
    let out_ptr_local = 15u32;
    let x_local = 16u32;
    let y_local = 17u32;

    function.instruction(&Instruction::LocalGet(y_len_index));
    function.instruction(&Instruction::LocalGet(x_len_index));
    function.instruction(&Instruction::I32Ne);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::LocalGet(output_len_index));
    function.instruction(&Instruction::LocalGet(x_len_index));
    function.instruction(&Instruction::I32Ne);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);

    function.instruction(&Instruction::I32Const(0));
    function.instruction(&Instruction::LocalSet(index_local));
    function.instruction(&Instruction::LocalGet(x_ptr_index));
    function.instruction(&Instruction::LocalSet(x_ptr_local));
    function.instruction(&Instruction::LocalGet(y_ptr_index));
    function.instruction(&Instruction::LocalSet(y_ptr_local));
    function.instruction(&Instruction::LocalGet(output_ptr_index));
    function.instruction(&Instruction::LocalSet(out_ptr_local));
    function.instruction(&Instruction::LocalGet(x_len_index));
    function.instruction(&Instruction::LocalGet(x_len_index));
    function.instruction(&Instruction::I32Const(2));
    function.instruction(&Instruction::I32RemU);
    function.instruction(&Instruction::I32Sub);
    function.instruction(&Instruction::LocalSet(vector_end_local));

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::LocalGet(vector_end_local));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(out_ptr_local));
    function.instruction(&Instruction::LocalGet(0));
    function.instruction(&Instruction::I64x2Splat);
    function.instruction(&Instruction::LocalGet(x_ptr_local));
    function.instruction(&Instruction::V128Load(handcrafted_memarg(0, 4)));
    function.instruction(&Instruction::I64x2Mul);
    function.instruction(&Instruction::LocalGet(y_ptr_local));
    function.instruction(&Instruction::V128Load(handcrafted_memarg(0, 4)));
    function.instruction(&Instruction::I64x2Add);
    function.instruction(&Instruction::V128Store(handcrafted_memarg(0, 4)));
    function.instruction(&Instruction::LocalGet(x_ptr_local));
    function.instruction(&Instruction::I32Const(16));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(x_ptr_local));
    function.instruction(&Instruction::LocalGet(y_ptr_local));
    function.instruction(&Instruction::I32Const(16));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(y_ptr_local));
    function.instruction(&Instruction::LocalGet(out_ptr_local));
    function.instruction(&Instruction::I32Const(16));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(out_ptr_local));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::I32Const(2));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(index_local));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::LocalGet(x_len_index));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(x_ptr_local));
    function.instruction(&Instruction::I64Load(handcrafted_memarg(0, 3)));
    function.instruction(&Instruction::LocalSet(x_local));
    function.instruction(&Instruction::LocalGet(y_ptr_local));
    function.instruction(&Instruction::I64Load(handcrafted_memarg(0, 3)));
    function.instruction(&Instruction::LocalSet(y_local));
    function.instruction(&Instruction::LocalGet(out_ptr_local));
    function.instruction(&Instruction::LocalGet(0));
    function.instruction(&Instruction::LocalGet(x_local));
    function.instruction(&Instruction::I64Mul);
    function.instruction(&Instruction::LocalGet(y_local));
    function.instruction(&Instruction::I64Add);
    function.instruction(&Instruction::I64Store(handcrafted_memarg(0, 3)));
    function.instruction(&Instruction::LocalGet(x_ptr_local));
    function.instruction(&Instruction::I32Const(8));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(x_ptr_local));
    function.instruction(&Instruction::LocalGet(y_ptr_local));
    function.instruction(&Instruction::I32Const(8));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(y_ptr_local));
    function.instruction(&Instruction::LocalGet(out_ptr_local));
    function.instruction(&Instruction::I32Const(8));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(out_ptr_local));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::I32Const(1));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(index_local));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function
}

fn handcrafted_memarg(offset: u64, align: u32) -> MemArg {
    MemArg {
        offset,
        align,
        memory_index: 0,
    }
}

fn default_iterations(size: usize) -> usize {
    match size {
        0..=4_096 => 400,
        4_097..=65_536 => 160,
        65_537..=262_144 => 50,
        _ => 20,
    }
}

fn end_to_end_iterations(iterations: usize) -> usize {
    iterations.clamp(1, 5)
}

fn measure<F>(iterations: usize, mut f: F) -> Result<BenchStats>
where
    F: FnMut() -> Result<Value>,
{
    black_box(f()?);
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(f()?);
    }
    Ok(BenchStats {
        iterations,
        total: start.elapsed(),
    })
}

fn measure_void<F>(iterations: usize, mut f: F) -> Result<BenchStats>
where
    F: FnMut() -> Result<()>,
{
    black_box(f()?);
    let start = Instant::now();
    for _ in 0..iterations {
        black_box(f()?);
    }
    Ok(BenchStats {
        iterations,
        total: start.elapsed(),
    })
}

enum PreparedBulkScratch {
    I32(Vec<i32>),
    I64(Vec<i64>),
    F32(Vec<f32>),
    F64(Vec<f64>),
}

struct PreparedReadSlotScratch {
    slot: usize,
    scratch: PreparedBulkScratch,
}

fn build_prepared_bulk_read_scratch(bound: &BoundPreparedRun) -> Vec<PreparedReadSlotScratch> {
    let mut scratch = Vec::new();
    for slot in bound.slots() {
        if slot.role != PreparedSlotRole::Output || slot.kind != PreparedSlotKind::Bulk {
            continue;
        }
        let len = slot.shape.iter().copied().product::<usize>();
        let item = match slot.prim {
            Prim::I32 => PreparedBulkScratch::I32(vec![0; len]),
            Prim::I64 => PreparedBulkScratch::I64(vec![0; len]),
            Prim::F32 => PreparedBulkScratch::F32(vec![0.0; len]),
            Prim::F64 => PreparedBulkScratch::F64(vec![0.0; len]),
        };
        scratch.push(PreparedReadSlotScratch {
            slot: slot.index,
            scratch: item,
        });
    }
    scratch
}

fn prepared_typed_readback(
    bound: &BoundPreparedRun,
    scratch: &mut [PreparedReadSlotScratch],
) -> Result<()> {
    for item in scratch {
        match &mut item.scratch {
            PreparedBulkScratch::I32(buffer) => {
                bound.read_bulk_i32(item.slot, buffer)?;
                black_box(buffer.first().copied());
            }
            PreparedBulkScratch::I64(buffer) => {
                bound.read_bulk_i64(item.slot, buffer)?;
                black_box(buffer.first().copied());
            }
            PreparedBulkScratch::F32(buffer) => {
                bound.read_bulk_f32(item.slot, buffer)?;
                black_box(buffer.first().copied());
            }
            PreparedBulkScratch::F64(buffer) => {
                bound.read_bulk_f64(item.slot, buffer)?;
                black_box(buffer.first().copied());
            }
        }
    }
    Ok(())
}

fn avg_duration(stats: &BenchStats) -> Duration {
    Duration::from_secs_f64(stats.total.as_secs_f64() / stats.iterations as f64)
}

fn nanos_per_element(stats: &BenchStats, elements: usize) -> f64 {
    if elements == 0 {
        0.0
    } else {
        stats.total.as_secs_f64() * 1_000_000_000.0 / (stats.iterations * elements) as f64
    }
}

fn melems_per_second(stats: &BenchStats, elements: usize) -> f64 {
    if stats.total.is_zero() {
        0.0
    } else {
        (stats.iterations * elements) as f64 / stats.total.as_secs_f64() / 1_000_000.0
    }
}

fn slowdown(rust: &BenchStats, wasm: &BenchStats) -> f64 {
    avg_duration(wasm).as_secs_f64() / avg_duration(rust).as_secs_f64()
}

fn format_duration(duration: Duration) -> String {
    let secs = duration.as_secs_f64();
    if secs >= 1.0 {
        format!("{secs:.2}s")
    } else if secs >= 0.001 {
        format!("{:.2}ms", secs * 1_000.0)
    } else if secs >= 0.000_001 {
        format!("{:.2}us", secs * 1_000_000.0)
    } else {
        format!("{:.0}ns", secs * 1_000_000_000.0)
    }
}

fn format_leaf_metrics(
    rust: &BenchStats,
    wasm_e2e: &BenchStats,
    wasm_hot: &BenchStats,
    wasm_prepared_hot: &BenchStats,
    wasm_prepared_hot_read: &BenchStats,
    input: &BenchInput,
) -> String {
    if input.leaf_elements == input.logical_elements {
        String::new()
    } else {
        format!(
            concat!(
                "\n",
                "  normalized-per-leaf: rust ns/leaf-elem={:.2}, wasm-e2e ns/leaf-elem={:.2}, wasm-hot ns/leaf-elem={:.2}, wasm-prepared-hot ns/leaf-elem={:.2}, wasm-prepared-hot+read ns/leaf-elem={:.2}"
            ),
            nanos_per_element(rust, input.leaf_elements),
            nanos_per_element(wasm_e2e, input.leaf_elements),
            nanos_per_element(wasm_hot, input.leaf_elements),
            nanos_per_element(wasm_prepared_hot, input.leaf_elements),
            nanos_per_element(wasm_prepared_hot_read, input.leaf_elements),
        )
    }
}

fn build_inc_i64_input(size: usize) -> BenchInput {
    let elements = (0..size)
        .map(|index| ScalarValue::I64((index % 1024) as i64))
        .collect::<Vec<_>>();
    BenchInput {
        args: vec![Value::Bulk(BulkValue {
            prim: Prim::I64,
            shape: vec![size],
            elements,
        })],
        logical_elements: size,
        leaf_elements: size,
    }
}

fn build_square_f32_input(size: usize) -> BenchInput {
    let elements = (0..size)
        .map(|index| ScalarValue::F32(((index % 97) as u32 + 1) as f32))
        .collect::<Vec<_>>();
    BenchInput {
        args: vec![Value::Bulk(BulkValue {
            prim: Prim::F32,
            shape: vec![size],
            elements,
        })],
        logical_elements: size,
        leaf_elements: size,
    }
}

fn build_axpy_i64_input(size: usize) -> BenchInput {
    let xs = (0..size)
        .map(|index| ScalarValue::I64((index % 257) as i64))
        .collect::<Vec<_>>();
    let ys = (0..size)
        .map(|index| ScalarValue::I64(((index * 3) % 257) as i64))
        .collect::<Vec<_>>();
    BenchInput {
        args: vec![
            Value::Scalar(ScalarValue::I64(7)),
            Value::Bulk(BulkValue {
                prim: Prim::I64,
                shape: vec![size],
                elements: xs,
            }),
            Value::Bulk(BulkValue {
                prim: Prim::I64,
                shape: vec![size],
                elements: ys,
            }),
        ],
        logical_elements: size,
        leaf_elements: size,
    }
}

fn build_axpy2_record_i64_input(size: usize) -> BenchInput {
    let ux = (0..size)
        .map(|index| ScalarValue::I64((index % 257) as i64))
        .collect::<Vec<_>>();
    let uy = (0..size)
        .map(|index| ScalarValue::I64(((index * 3) % 257) as i64))
        .collect::<Vec<_>>();
    let vx = (0..size)
        .map(|index| ScalarValue::I64(((index * 5) % 257) as i64))
        .collect::<Vec<_>>();
    let vy = (0..size)
        .map(|index| ScalarValue::I64(((index * 7) % 257) as i64))
        .collect::<Vec<_>>();
    BenchInput {
        args: vec![
            Value::Scalar(ScalarValue::I64(7)),
            Value::Record(std::collections::BTreeMap::from([
                (
                    "x".to_string(),
                    Value::Bulk(BulkValue {
                        prim: Prim::I64,
                        shape: vec![size],
                        elements: ux,
                    }),
                ),
                (
                    "y".to_string(),
                    Value::Bulk(BulkValue {
                        prim: Prim::I64,
                        shape: vec![size],
                        elements: uy,
                    }),
                ),
            ])),
            Value::Record(std::collections::BTreeMap::from([
                (
                    "x".to_string(),
                    Value::Bulk(BulkValue {
                        prim: Prim::I64,
                        shape: vec![size],
                        elements: vx,
                    }),
                ),
                (
                    "y".to_string(),
                    Value::Bulk(BulkValue {
                        prim: Prim::I64,
                        shape: vec![size],
                        elements: vy,
                    }),
                ),
            ])),
        ],
        logical_elements: size,
        leaf_elements: size * 2,
    }
}

fn build_pow2_i64_input(size: usize) -> BenchInput {
    let elements = (0..size)
        .map(|index| ScalarValue::I64((index % 4096) as i64))
        .collect::<Vec<_>>();
    BenchInput {
        args: vec![Value::Bulk(BulkValue {
            prim: Prim::I64,
            shape: vec![size],
            elements,
        })],
        logical_elements: size,
        leaf_elements: size,
    }
}

fn rust_inc_i64(args: &[Value]) -> Result<Value> {
    let xs = expect_bulk_i64(args, 0)?;
    Ok(Value::Bulk(BulkValue {
        prim: Prim::I64,
        shape: xs.shape.clone(),
        elements: xs
            .elements
            .iter()
            .map(|value| match value {
                ScalarValue::I64(value) => Ok(ScalarValue::I64(value + 1)),
                other => Err(SimdError::new(format!(
                    "expected i64 element, found {:?}",
                    other
                ))),
            })
            .collect::<Result<Vec<_>>>()?,
    }))
}

fn rust_square_f32(args: &[Value]) -> Result<Value> {
    let xs = expect_bulk_f32(args, 0)?;
    Ok(Value::Bulk(BulkValue {
        prim: Prim::F32,
        shape: xs.shape.clone(),
        elements: xs
            .elements
            .iter()
            .map(|value| match value {
                ScalarValue::F32(value) => Ok(ScalarValue::F32(value * value)),
                other => Err(SimdError::new(format!(
                    "expected f32 element, found {:?}",
                    other
                ))),
            })
            .collect::<Result<Vec<_>>>()?,
    }))
}

fn rust_axpy_i64(args: &[Value]) -> Result<Value> {
    let a = expect_scalar_i64(args, 0)?;
    let xs = expect_bulk_i64(args, 1)?;
    let ys = expect_bulk_i64(args, 2)?;
    if xs.shape != ys.shape || xs.elements.len() != ys.elements.len() {
        return Err(SimdError::new(
            "axpy benchmark expected matching bulk shapes",
        ));
    }
    let mut elements = Vec::with_capacity(xs.elements.len());
    for (x, y) in xs.elements.iter().zip(&ys.elements) {
        let ScalarValue::I64(x) = x else {
            return Err(SimdError::new("axpy benchmark expected i64 x lanes"));
        };
        let ScalarValue::I64(y) = y else {
            return Err(SimdError::new("axpy benchmark expected i64 y lanes"));
        };
        elements.push(ScalarValue::I64((a * x) + y));
    }
    Ok(Value::Bulk(BulkValue {
        prim: Prim::I64,
        shape: xs.shape.clone(),
        elements,
    }))
}

fn rust_axpy2_record_i64(args: &[Value]) -> Result<Value> {
    let a = expect_scalar_i64(args, 0)?;
    let u = expect_record(args, 1)?;
    let v = expect_record(args, 2)?;
    let ux = expect_record_bulk_i64_field(u, "x")?;
    let uy = expect_record_bulk_i64_field(u, "y")?;
    let vx = expect_record_bulk_i64_field(v, "x")?;
    let vy = expect_record_bulk_i64_field(v, "y")?;
    if ux.shape != uy.shape
        || vx.shape != vy.shape
        || ux.shape != vx.shape
        || ux.elements.len() != uy.elements.len()
        || vx.elements.len() != vy.elements.len()
        || ux.elements.len() != vx.elements.len()
    {
        return Err(SimdError::new(
            "axpy2 record benchmark expected matching bulk shapes",
        ));
    }
    let mut x_elements = Vec::with_capacity(ux.elements.len());
    let mut y_elements = Vec::with_capacity(uy.elements.len());
    for ((ux, uy), (vx, vy)) in ux
        .elements
        .iter()
        .zip(&uy.elements)
        .zip(vx.elements.iter().zip(&vy.elements))
    {
        let ScalarValue::I64(ux) = ux else {
            return Err(SimdError::new(
                "axpy2 record benchmark expected i64 x lanes",
            ));
        };
        let ScalarValue::I64(uy) = uy else {
            return Err(SimdError::new(
                "axpy2 record benchmark expected i64 y lanes",
            ));
        };
        let ScalarValue::I64(vx) = vx else {
            return Err(SimdError::new(
                "axpy2 record benchmark expected i64 x lanes",
            ));
        };
        let ScalarValue::I64(vy) = vy else {
            return Err(SimdError::new(
                "axpy2 record benchmark expected i64 y lanes",
            ));
        };
        x_elements.push(ScalarValue::I64((a * ux) + vx));
        y_elements.push(ScalarValue::I64((a * uy) + vy));
    }
    Ok(Value::Record(std::collections::BTreeMap::from([
        (
            "x".to_string(),
            Value::Bulk(BulkValue {
                prim: Prim::I64,
                shape: ux.shape.clone(),
                elements: x_elements,
            }),
        ),
        (
            "y".to_string(),
            Value::Bulk(BulkValue {
                prim: Prim::I64,
                shape: uy.shape.clone(),
                elements: y_elements,
            }),
        ),
    ])))
}

fn rust_pow2_i64(args: &[Value]) -> Result<Value> {
    let xs = expect_bulk_i64(args, 0)?;
    Ok(Value::Bulk(BulkValue {
        prim: Prim::I64,
        shape: xs.shape.clone(),
        elements: xs
            .elements
            .iter()
            .map(|value| match value {
                ScalarValue::I64(value) => Ok(ScalarValue::I64(value * 8)),
                other => Err(SimdError::new(format!(
                    "expected i64 element, found {:?}",
                    other
                ))),
            })
            .collect::<Result<Vec<_>>>()?,
    }))
}

fn expect_scalar_i64(args: &[Value], index: usize) -> Result<i64> {
    match args.get(index) {
        Some(Value::Scalar(ScalarValue::I64(value))) => Ok(*value),
        Some(other) => Err(SimdError::new(format!(
            "expected i64 scalar at argument {}, found {:?}",
            index, other
        ))),
        None => Err(SimdError::new(format!("missing argument {}", index))),
    }
}

fn expect_bulk_i64(args: &[Value], index: usize) -> Result<&BulkValue> {
    match args.get(index) {
        Some(Value::Bulk(value)) if value.prim == Prim::I64 => Ok(value),
        Some(other) => Err(SimdError::new(format!(
            "expected i64 bulk at argument {}, found {:?}",
            index, other
        ))),
        None => Err(SimdError::new(format!("missing argument {}", index))),
    }
}

fn expect_record(
    args: &[Value],
    index: usize,
) -> Result<&std::collections::BTreeMap<String, Value>> {
    match args.get(index) {
        Some(Value::Record(value)) => Ok(value),
        Some(other) => Err(SimdError::new(format!(
            "expected record at argument {}, found {:?}",
            index, other
        ))),
        None => Err(SimdError::new(format!("missing argument {}", index))),
    }
}

fn expect_record_bulk_i64_field<'a>(
    record: &'a std::collections::BTreeMap<String, Value>,
    field: &str,
) -> Result<&'a BulkValue> {
    match record.get(field) {
        Some(Value::Bulk(value)) if value.prim == Prim::I64 => Ok(value),
        Some(other) => Err(SimdError::new(format!(
            "expected i64 bulk field '{}' in record, found {:?}",
            field, other
        ))),
        None => Err(SimdError::new(format!("missing record field '{}'", field))),
    }
}

fn expect_bulk_f32(args: &[Value], index: usize) -> Result<&BulkValue> {
    match args.get(index) {
        Some(Value::Bulk(value)) if value.prim == Prim::F32 => Ok(value),
        Some(other) => Err(SimdError::new(format!(
            "expected f32 bulk at argument {}, found {:?}",
            index, other
        ))),
        None => Err(SimdError::new(format!("missing argument {}", index))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_examples_match_rust_reference() {
        for spec in benchmark_specs() {
            let input = (spec.build_input)(32);
            let expected = (spec.rust_impl)(&input.args).unwrap();
            let compiled = compile_source(spec.source).unwrap();
            let eval = run_compiled_main(&compiled, spec.main, &input.args).unwrap();
            assert_eq!(eval, expected, "evaluator mismatch for {}", spec.name);

            let artifact = compile_wasm_main(spec.source, spec.main).unwrap();
            let executable = prepare_wasm_artifact(artifact).unwrap();
            let wasm = executable.run(&input.args).unwrap();
            assert_eq!(wasm, expected, "wasm mismatch for {}", spec.name);
        }
    }

    #[test]
    fn benchmark_command_reports_wasm_e2e_and_hot_paths() {
        let output = bench_command("axpy2-record-i64", 4, Some(1)).unwrap();
        assert!(output.contains("axpy2-record-i64"));
        assert!(output.contains("wasm-e2e: iterations=1"));
        assert!(output.contains("wasm-hot: iterations=1"));
        assert!(output.contains("normalized-per-leaf"));
    }

    #[test]
    fn benchmark_command_reports_core_contract_summary() {
        let output = bench_command("all", 4, Some(1)).unwrap();
        assert!(output.contains("contract summary:"));
        assert!(output.contains("core-large gate:"));
        assert!(output.contains("axpy2 per-leaf within 10% of axpy per-leaf at large:"));
        assert!(output.contains("overall:"));
    }

    #[test]
    fn matrix_benchmark_command_reports_contract_summary() {
        let output = bench_command("matrix", 0, Some(1)).unwrap();
        assert!(output.contains("contract summary:"));
        assert!(output.contains("large slowdown<1.0:"));
        assert!(output.contains("medium slowdown<=1.05:"));
        assert!(output.contains("small baseline-regression gate: SKIP"));
        assert!(output.contains("core-suite large gate:"));
        assert!(output.contains("axpy2 per-leaf within 10% of axpy per-leaf at large:"));
        assert!(output.contains("matrix-unary-i32-small"));
        assert!(output.contains("matrix-axpy2-record-f64-large"));
        assert!(output.contains("matrix-affine-rank2-f32-large"));
        assert!(output.contains("matrix-image-tone-rgb-f32-medium"));
        assert!(output.contains("matrix-image-blend-rgb-f32-small"));
    }

    #[test]
    fn handcrafted_matrix_baselines_compile() {
        let names = [
            "matrix-unary-i32-small",
            "matrix-unary-i64-small",
            "matrix-unary-f32-small",
            "matrix-unary-f64-small",
            "matrix-square-rank2-i32-small",
            "matrix-square-rank2-i64-small",
            "matrix-square-rank2-f32-small",
            "matrix-square-rank2-f64-small",
            "matrix-axpy-i32-small",
            "matrix-axpy-i64-small",
            "matrix-axpy-f32-small",
            "matrix-axpy-f64-small",
            "matrix-affine-rank2-i32-small",
            "matrix-affine-rank2-i64-small",
            "matrix-affine-rank2-f32-small",
            "matrix-affine-rank2-f64-small",
            "matrix-axpy2-record-i32-small",
            "matrix-axpy2-record-i64-small",
            "matrix-axpy2-record-f32-small",
            "matrix-axpy2-record-f64-small",
            "matrix-image-tone-rgb-f32-small",
            "matrix-image-blend-rgb-f32-small",
        ];
        for name in names {
            let artifact = handcrafted_baseline_artifact(name)
                .unwrap_or_else(|error| panic!("{} baseline lookup failed: {}", name, error))
                .unwrap_or_else(|| panic!("{} missing handwritten baseline", name));
            prepare_wasm_artifact(artifact)
                .unwrap_or_else(|error| panic!("{} compile failed: {}", name, error));
        }
    }
}
