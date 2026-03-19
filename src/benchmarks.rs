use super::*;
use std::hint::black_box;
use std::time::{Duration, Instant};

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

    Ok(format!(
        concat!(
            "{}\n",
            "  description: {}\n",
            "  source: {}\n",
            "  setup: frontend={} compile-wasm={} prepare-module={} prepare-bound={} wasm-bytes={}\n",
            "  rust: avg/run={} ns/logical-elem={:.2} mlogical/s={:.2}\n",
            "  wasm-e2e: iterations={} avg/run={} ns/logical-elem={:.2} mlogical/s={:.2} slowdown={:.2}x\n",
            "  wasm-hot: iterations={} avg/run={} ns/logical-elem={:.2} mlogical/s={:.2} slowdown={:.2}x\n",
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
                "  rust: avg/run={} ns/logical-elem={:.2} mlogical/s={:.2}\n",
                "  wasm-e2e: iterations={} avg/run={} ns/logical-elem={:.2} mlogical/s={:.2} slowdown={:.2}x\n",
                "  wasm-hot: iterations={} avg/run={} ns/logical-elem={:.2} mlogical/s={:.2} slowdown={:.2}x\n",
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
        }
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
    }
}
