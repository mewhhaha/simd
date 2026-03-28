use super::*;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fs;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use wasm_encoder::{
    BlockType, CodeSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    ImportSection, Instruction, MemArg, MemorySection, MemoryType, Module, TypeSection, ValType,
};
use wasmtime::{
    Cache, Config, Engine, Func, Instance, Memory, Module as WasmtimeModule, OptLevel, Store, Val,
};

#[derive(Debug, Clone, PartialEq)]
pub struct WasmArtifact {
    pub bytes: Vec<u8>,
    pub export_name: String,
    pub params: Vec<WasmParamAbi>,
    pub result: WasmResultAbi,
    pub result_type: Type,
    pub enum_ctors: BTreeMap<String, EnumCtorInfo>,
    pub wasm_enum_layouts: BTreeMap<String, WasmEnumLayout>,
    pub grouped_export: Option<WasmGroupedExport>,
    pub leaf_exports: Vec<WasmLeafExport>,
    pub optimizer_reports: Vec<WasmOptimizationReport>,
    pub higher_order_reports: Vec<WasmHigherOrderReport>,
    pub function_profile_names: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmOptimizationReport {
    pub function: String,
    pub intent: IntentClass,
    pub structural_exec: StructuralExecMode,
    pub vectorizable: bool,
    pub vector_unroll: usize,
    pub fallback_reason: Option<String>,
    pub structural_scc: Option<usize>,
    pub structural_state_count: usize,
    pub structural_transition_count: usize,
    pub structural_span_ops: usize,
    pub structural_enum_ops: usize,
    pub structural_region_count: usize,
    pub structural_char_prefix_regions: usize,
    pub structural_separated_item_regions: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum StructuralExecMode {
    Scalar,
    StructuralLoop,
    StructuralBatched,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmHigherOrderReport {
    pub function: String,
    pub specialization_origin: String,
    pub lambda_mode: LambdaLoweringMode,
    pub capture_count: usize,
    pub env_bytes: usize,
    pub known_fn_values: usize,
    pub known_lambda_values: usize,
    pub escaping_unknown_values: usize,
    pub rejection_reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LambdaLoweringMode {
    DirectFirstOrder,
    ClosureConverted,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmGroupedExport {
    pub export_name: String,
    pub leaf_exports: Vec<WasmLeafExport>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmLeafExport {
    pub leaf_path: LeafPath,
    pub export_name: String,
    pub result: WasmLeafResultAbi,
    pub used_param_leaves: Vec<usize>,
    pub reusable_param_leaf: Option<usize>,
}

#[derive(Debug)]
pub struct WasmExecutable {
    artifact: WasmArtifact,
    runtime: RefCell<WasmRuntime>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmRunProfile {
    pub result_json: String,
    pub artifact_bytes: usize,
    pub frontend_ms: u128,
    pub frontend_us: u128,
    pub specialize_ms: u128,
    pub specialize_us: u128,
    pub lambda_lower_ms: u128,
    pub lambda_lower_us: u128,
    pub canonicalize_ms: u128,
    pub canonicalize_us: u128,
    pub parse_args_ms: u128,
    pub parse_args_us: u128,
    pub compile_artifact_ms: u128,
    pub compile_artifact_us: u128,
    pub compile_enum_layouts_ms: u128,
    pub compile_enum_layouts_us: u128,
    pub compile_plan_ms: u128,
    pub compile_plan_us: u128,
    pub compile_normalize_ms: u128,
    pub compile_normalize_us: u128,
    pub compile_lower_ms: u128,
    pub compile_lower_us: u128,
    pub compile_group_ms: u128,
    pub compile_group_us: u128,
    pub compile_intents_ms: u128,
    pub compile_intents_us: u128,
    pub engine_ms: u128,
    pub engine_us: u128,
    pub module_compile_ms: u128,
    pub module_compile_us: u128,
    pub runtime_build_ms: u128,
    pub runtime_build_us: u128,
    pub execute_ms: u128,
    pub execute_us: u128,
    pub structural_functions: usize,
    pub fallback_functions: usize,
    pub parser_fallback_functions: usize,
    pub structural_sccs: usize,
    pub structural_state_count: usize,
    pub structural_transition_count: usize,
    pub structural_span_ops: usize,
    pub structural_enum_ops: usize,
    pub total_ms: u128,
    pub total_us: u128,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmFunctionProfile {
    pub result_json: String,
    pub total_us: u128,
    pub functions: Vec<FunctionProfileRow>,
}

impl WasmFunctionProfile {
    pub fn to_table_string(&self) -> String {
        render_function_profile_table(&self.result_json, self.total_us, &self.functions)
    }

    pub fn to_json_string(&self) -> String {
        let functions = self
            .functions
            .iter()
            .map(|row| {
                serde_json::json!({
                    "name": row.name,
                    "calls": row.calls,
                    "total_us": row.total_us.to_string(),
                    "avg_us": row.avg_us.to_string(),
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "result_json": self.result_json,
            "total_us": self.total_us.to_string(),
            "function_count": self.functions.len(),
            "functions": functions,
        })
        .to_string()
    }
}

impl WasmRunProfile {
    fn to_kv_string(&self) -> String {
        [
            format!("result_json={}", self.result_json),
            format!("artifact_bytes={}", self.artifact_bytes),
            format!("frontend_ms={}", self.frontend_ms),
            format!("frontend_us={}", self.frontend_us),
            format!("specialize_ms={}", self.specialize_ms),
            format!("specialize_us={}", self.specialize_us),
            format!("lambda_lower_ms={}", self.lambda_lower_ms),
            format!("lambda_lower_us={}", self.lambda_lower_us),
            format!("canonicalize_ms={}", self.canonicalize_ms),
            format!("canonicalize_us={}", self.canonicalize_us),
            format!("parse_args_ms={}", self.parse_args_ms),
            format!("parse_args_us={}", self.parse_args_us),
            format!("compile_artifact_ms={}", self.compile_artifact_ms),
            format!("compile_artifact_us={}", self.compile_artifact_us),
            format!("compile_enum_layouts_ms={}", self.compile_enum_layouts_ms),
            format!("compile_enum_layouts_us={}", self.compile_enum_layouts_us),
            format!("compile_plan_ms={}", self.compile_plan_ms),
            format!("compile_plan_us={}", self.compile_plan_us),
            format!("compile_normalize_ms={}", self.compile_normalize_ms),
            format!("compile_normalize_us={}", self.compile_normalize_us),
            format!("compile_lower_ms={}", self.compile_lower_ms),
            format!("compile_lower_us={}", self.compile_lower_us),
            format!("compile_group_ms={}", self.compile_group_ms),
            format!("compile_group_us={}", self.compile_group_us),
            format!("compile_intents_ms={}", self.compile_intents_ms),
            format!("compile_intents_us={}", self.compile_intents_us),
            format!("engine_ms={}", self.engine_ms),
            format!("engine_us={}", self.engine_us),
            format!("module_compile_ms={}", self.module_compile_ms),
            format!("module_compile_us={}", self.module_compile_us),
            format!("runtime_build_ms={}", self.runtime_build_ms),
            format!("runtime_build_us={}", self.runtime_build_us),
            format!("execute_ms={}", self.execute_ms),
            format!("execute_us={}", self.execute_us),
            format!("structural_functions={}", self.structural_functions),
            format!("fallback_functions={}", self.fallback_functions),
            format!(
                "parser_fallback_functions={}",
                self.parser_fallback_functions
            ),
            format!("structural_sccs={}", self.structural_sccs),
            format!("structural_state_count={}", self.structural_state_count),
            format!(
                "structural_transition_count={}",
                self.structural_transition_count
            ),
            format!("structural_span_ops={}", self.structural_span_ops),
            format!("structural_enum_ops={}", self.structural_enum_ops),
            format!("total_ms={}", self.total_ms),
            format!("total_us={}", self.total_us),
        ]
        .join("\n")
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
struct StructuralOptimizerSummary {
    structural_functions: usize,
    fallback_functions: usize,
    parser_fallback_functions: usize,
    structural_sccs: usize,
    structural_state_count: usize,
    structural_transition_count: usize,
    structural_span_ops: usize,
    structural_enum_ops: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WasmCompileOptions {
    function_profiler: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct WasmFunctionInstrumentation {
    func_id: u32,
    enter_import_index: u32,
    exit_import_index: u32,
    allow_return_call: bool,
}

const WASM_PROFILE_ENTER_IMPORT_INDEX: u32 = 0;
const WASM_PROFILE_EXIT_IMPORT_INDEX: u32 = 1;
const WASM_PROFILE_IMPORT_COUNT: u32 = 2;

#[derive(Debug, Clone)]
struct WasmFunctionProfilerState {
    rows: BTreeMap<String, FunctionProfileStats>,
    stack: Vec<(usize, Instant)>,
    names: Vec<String>,
}

impl WasmFunctionProfilerState {
    fn new(names: Vec<String>) -> Self {
        let mut rows = BTreeMap::new();
        for name in &names {
            rows.insert(name.clone(), FunctionProfileStats::default());
        }
        Self {
            rows,
            stack: Vec::new(),
            names,
        }
    }

    fn enter(&mut self, func_id: usize) {
        if let Some(name) = self.names.get(func_id)
            && let Some(row) = self.rows.get_mut(name)
        {
            row.calls += 1;
            self.stack.push((func_id, Instant::now()));
        }
    }

    fn exit(&mut self, func_id: usize) {
        if let Some((active_id, start)) = self.stack.pop()
            && active_id == func_id
            && let Some(name) = self.names.get(func_id)
            && let Some(row) = self.rows.get_mut(name)
        {
            row.total_us += start.elapsed().as_micros();
        }
    }

    fn record_top_level_call_if_missing(&mut self, name: &str, total_us: u128) {
        if let Some(row) = self.rows.get_mut(name)
            && row.calls == 0
        {
            row.calls = 1;
            row.total_us = total_us;
        }
    }

    fn finish(&self) -> Vec<FunctionProfileRow> {
        let mut rows = self
            .rows
            .iter()
            .map(|(name, stats)| FunctionProfileRow {
                name: name.clone(),
                calls: stats.calls,
                total_us: stats.total_us,
                avg_us: if stats.calls == 0 {
                    0
                } else {
                    stats.total_us / u128::from(stats.calls)
                },
            })
            .collect::<Vec<_>>();
        rows.sort_by(|left, right| {
            right
                .total_us
                .cmp(&left.total_us)
                .then_with(|| left.name.cmp(&right.name))
        });
        rows
    }
}

fn summarize_structural_optimizer_reports(
    reports: &[WasmOptimizationReport],
) -> StructuralOptimizerSummary {
    let structural_sccs = reports
        .iter()
        .filter_map(|report| report.structural_scc)
        .collect::<BTreeSet<_>>()
        .len();
    StructuralOptimizerSummary {
        structural_functions: reports
            .iter()
            .filter(|report| report.intent == IntentClass::Structural)
            .count(),
        fallback_functions: reports
            .iter()
            .filter(|report| report.intent == IntentClass::Fallback)
            .count(),
        parser_fallback_functions: reports
            .iter()
            .filter(|report| {
                report.intent == IntentClass::Fallback && report.function.starts_with("parse_")
            })
            .count(),
        structural_sccs,
        structural_state_count: reports
            .iter()
            .map(|report| report.structural_state_count)
            .sum(),
        structural_transition_count: reports
            .iter()
            .map(|report| report.structural_transition_count)
            .sum(),
        structural_span_ops: reports
            .iter()
            .map(|report| report.structural_span_ops)
            .sum(),
        structural_enum_ops: reports
            .iter()
            .map(|report| report.structural_enum_ops)
            .sum(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedLayout {
    pub bulk_input_shapes: Vec<Vec<usize>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedSlotRole {
    Input,
    Output,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PreparedSlotKind {
    Scalar,
    Bulk,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PreparedSlotMetadata {
    pub index: usize,
    pub role: PreparedSlotRole,
    pub kind: PreparedSlotKind,
    pub logical_path: LeafPath,
    pub prim: Prim,
    pub shape: Vec<usize>,
    pub offset_elems: usize,
    pub strides_elems: Vec<usize>,
}

#[derive(Debug)]
pub struct PreparedWasmMain {
    artifact: WasmArtifact,
    params: Vec<WasmParamAbi>,
    flat_param_abis: Vec<WasmParamAbi>,
    input_templates: Vec<PreparedInputTemplate>,
    output_templates: Vec<PreparedOutputTemplate>,
    runtime: Rc<RefCell<WasmRuntime>>,
}

#[derive(Debug)]
pub struct BoundPreparedRun {
    artifact: WasmArtifact,
    flat_param_abis: Vec<WasmParamAbi>,
    slots: Vec<PreparedSlotState>,
    input_scalar_slot_by_flat: BTreeMap<usize, usize>,
    input_bulk_slot_by_flat: BTreeMap<usize, usize>,
    call_plan: PreparedCallPlan,
    last_dynamic_result: Option<Value>,
    runtime: Rc<RefCell<WasmRuntime>>,
}

#[derive(Debug)]
struct WasmRuntime {
    store: Store<()>,
    memory: Memory,
    funcs: BTreeMap<String, Func>,
    input_buffers: BTreeMap<InputBufferKey, usize>,
    output_buffers: BTreeMap<OutputBufferKey, usize>,
    arena_pinned_end: usize,
    arena_cursor: usize,
    function_profiler: Option<Arc<Mutex<WasmFunctionProfilerState>>>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct InputBufferKey {
    param_leaf: usize,
    prim: Prim,
    shape: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum OutputBufferKind {
    Scalar,
    Bulk,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct OutputBufferKey {
    leaf_path: LeafPath,
    prim: Prim,
    kind: OutputBufferKind,
    shape: Vec<usize>,
}

#[derive(Debug, Clone)]
struct PreparedInputTemplate {
    flat_index: usize,
    prim: Prim,
    kind: PreparedSlotKind,
    logical_path: LeafPath,
}

#[derive(Debug, Clone)]
struct PreparedOutputTemplate {
    prim: Prim,
    kind: PreparedSlotKind,
    logical_path: LeafPath,
    leaf_path: LeafPath,
    scalar_via_pointer: bool,
    shape_param_top_level: Option<usize>,
}

#[derive(Debug, Clone)]
enum PreparedSlotState {
    InputScalar {
        meta: PreparedSlotMetadata,
        value: ScalarValue,
    },
    InputBulk {
        meta: PreparedSlotMetadata,
        ptr: usize,
        len: usize,
    },
    OutputScalar {
        meta: PreparedSlotMetadata,
        leaf_path: LeafPath,
        ptr: Option<usize>,
        value: ScalarValue,
    },
    OutputBulk {
        meta: PreparedSlotMetadata,
        leaf_path: LeafPath,
        ptr: usize,
        len: usize,
    },
}

#[derive(Debug, Clone)]
enum PreparedCallPlan {
    DirectScalar {
        func_name: String,
        output_slot: usize,
        prim: Prim,
    },
    DirectStarSeq {
        func_name: String,
        prim: Prim,
    },
    DirectBulk {
        func_name: String,
        output_slot: usize,
    },
    GroupedExport {
        func_name: String,
        outputs: Vec<PreparedLeafCall>,
    },
    LeafExports {
        leaves: Vec<PreparedLeafCall>,
    },
}

#[derive(Debug, Clone)]
struct PreparedLeafCall {
    func_name: String,
    output_slot: usize,
    result: WasmLeafResultAbi,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmParamAbi {
    Scalar { prim: Prim },
    Bulk { prim: Prim },
    Record { fields: Vec<(String, WasmParamAbi)> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmResultAbi {
    Scalar {
        prim: Prim,
    },
    StarSeq {
        prim: Prim,
    },
    Bulk {
        prim: Prim,
        shape_param: usize,
    },
    Record {
        fields: Vec<(String, WasmResultAbi)>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmLeafResultAbi {
    Scalar { prim: Prim },
    StarSeq { prim: Prim },
    Bulk { prim: Prim },
}

#[derive(Debug, Clone, PartialEq)]
pub struct WasmEnumLayout {
    pub enum_name: String,
    pub enum_id: u32,
    pub max_tag: u16,
    pub ctors: BTreeMap<String, WasmEnumCtorLayout>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WasmEnumCtorLayout {
    pub ctor_name: String,
    pub tag: u16,
    pub row_stride: u32,
    pub recursive_field_indices: Vec<usize>,
    pub non_recursive_fields: Vec<WasmEnumFieldLayout>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WasmEnumFieldLayout {
    pub source_index: usize,
    pub row_index: usize,
    pub offset: u32,
    pub ty: Type,
}

const WASM_ARTIFACT_BYTE_CACHE_VERSION: &str = "wasm-artifact-bytes-v1";

fn build_wasm_execution_artifact_checked(
    checked_program: &CheckedProgram,
    main: &str,
    origins: &BTreeMap<String, String>,
    bytes: Vec<u8>,
) -> Result<WasmArtifact> {
    let higher_order_reports = collect_higher_order_reports(checked_program, origins);
    ensure_backend_first_order_compatibility(&higher_order_reports)?;
    let wasm_enum_layouts = build_precomputed_wasm_enum_layouts(checked_program)?;
    let entry_result_type = checked_program
        .functions
        .iter()
        .find(|function| function.name == main)
        .map(|function| function.signature.ty.fun_parts().1)
        .ok_or_else(|| SimdError::new(format!("unknown entry function '{}'", main)))?;
    let plan = build_wasm_plan(checked_program, main)?;
    let grouped_export =
        if matches!(plan.result, WasmResultAbi::Record { .. }) && !plan.leaf_exports.is_empty() {
            Some(WasmGroupedExport {
                export_name: main.to_string(),
                leaf_exports: plan.leaf_exports.clone(),
            })
        } else {
            None
        };
    Ok(WasmArtifact {
        bytes,
        export_name: main.to_string(),
        params: plan.params,
        result: plan.result,
        result_type: entry_result_type,
        enum_ctors: checked_program.enum_ctors.clone(),
        wasm_enum_layouts,
        grouped_export,
        leaf_exports: plan.leaf_exports,
        optimizer_reports: Vec::new(),
        higher_order_reports,
        function_profile_names: Vec::new(),
    })
}

fn ensure_wasm_artifact_byte_cache_dir() -> Result<PathBuf> {
    let cwd = std::env::current_dir()
        .map_err(|error| SimdError::new(format!("failed to read current directory: {}", error)))?;
    let cache_dir = cwd.join(".tmp/wasm-artifact-cache");
    fs::create_dir_all(&cache_dir).map_err(|error| {
        SimdError::new(format!(
            "failed to create Wasm artifact cache directory '{}': {}",
            cache_dir.display(),
            error
        ))
    })?;
    Ok(cache_dir)
}

fn wasm_artifact_byte_cache_path(source: &str, main: &str) -> Result<PathBuf> {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    WASM_ARTIFACT_BYTE_CACHE_VERSION.hash(&mut hasher);
    main.hash(&mut hasher);
    source.hash(&mut hasher);
    let key = hasher.finish();
    Ok(ensure_wasm_artifact_byte_cache_dir()?.join(format!("{key:016x}.wasm")))
}

fn load_or_compile_wasm_command_artifact(
    path: &str,
    main: &str,
) -> Result<(CheckedProgram, WasmArtifact, bool)> {
    let source = read_source_file(path)?;
    let (_surface, _module, checked) = compile_frontend(&source)?;
    let specialized = specialize_checked_program_for_main(&checked, main)?;
    let lowered_ready = eliminate_non_escaping_lambdas_program(&specialized.checked)?;
    let canonical = canonicalize_backend_higher_order_program(&lowered_ready)?;
    let cache_path = wasm_artifact_byte_cache_path(&source, main)?;
    if let Ok(bytes) = fs::read(&cache_path) {
        let artifact =
            build_wasm_execution_artifact_checked(&canonical, main, &specialized.origins, bytes)?;
        return Ok((canonical, artifact, true));
    }
    let artifact = compile_wasm_artifact_checked(&canonical, main, &specialized.origins)?;
    fs::write(&cache_path, &artifact.bytes).map_err(|error| {
        SimdError::new(format!(
            "failed to write Wasm artifact cache '{}': {}",
            cache_path.display(),
            error
        ))
    })?;
    Ok((canonical, artifact, false))
}

pub fn compile_wasm_main(source: &str, main: &str) -> Result<WasmArtifact> {
    let (_surface, _module, checked) = compile_frontend(source)?;
    let specialized = specialize_checked_program_for_main(&checked, main)?;
    let lowered_ready = eliminate_non_escaping_lambdas_program(&specialized.checked)?;
    let canonical = canonicalize_backend_higher_order_program(&lowered_ready)?;
    compile_wasm_artifact_checked(&canonical, main, &specialized.origins)
}

pub fn prepare_wasm_main(source: &str, main: &str) -> Result<PreparedWasmMain> {
    let (_surface, _module, checked) = compile_frontend(source)?;
    let specialized = specialize_checked_program_for_main(&checked, main)?;
    let lowered_ready = eliminate_non_escaping_lambdas_program(&specialized.checked)?;
    let canonical = canonicalize_backend_higher_order_program(&lowered_ready)?;
    let plan = build_wasm_plan(&canonical, main)?;
    let artifact = compile_wasm_artifact_checked(&canonical, main, &specialized.origins)?;
    let flat_param_abis = flatten_wasm_param_abis(&plan.params)?;
    let input_templates = prepared_input_templates(&plan.params)?;
    let output_templates = prepared_output_templates(&plan.result, &artifact, &input_templates)?;
    let engine = build_engine()?;
    let module = WasmtimeModule::from_binary(&engine, &artifact.bytes)
        .map_err(|error| SimdError::new(format!("failed to compile Wasm module: {error:#}")))?;
    let runtime = build_runtime(&engine, &module, &artifact)?;
    Ok(PreparedWasmMain {
        artifact,
        params: plan.params,
        flat_param_abis,
        input_templates,
        output_templates,
        runtime: Rc::new(RefCell::new(runtime)),
    })
}

pub fn prepare_wasm_artifact(artifact: WasmArtifact) -> Result<WasmExecutable> {
    let engine = build_engine()?;
    let module = WasmtimeModule::from_binary(&engine, &artifact.bytes)
        .map_err(|error| SimdError::new(format!("failed to compile Wasm module: {error:#}")))?;
    let runtime = build_runtime(&engine, &module, &artifact)?;
    Ok(WasmExecutable {
        artifact,
        runtime: RefCell::new(runtime),
    })
}

pub fn run_wasm_artifact(artifact: &WasmArtifact, args: &[Value]) -> Result<Value> {
    prepare_wasm_artifact(artifact.clone())?.run(args)
}

pub fn run_wasm_main(source: &str, main: &str, args_json: &str) -> Result<Value> {
    let (_surface, _module, checked) = compile_frontend(source)?;
    let specialized = specialize_checked_program_for_main(&checked, main)?;
    let lowered_ready = eliminate_non_escaping_lambdas_program(&specialized.checked)?;
    let canonical = canonicalize_backend_higher_order_program(&lowered_ready)?;
    let args = parse_host_args(args_json, &canonical, main)?;
    let artifact = compile_wasm_artifact_checked(&canonical, main, &specialized.origins)?;
    prepare_wasm_artifact(artifact)?.run(&args)
}

pub fn run_wasm_prepared_main(
    source: &str,
    main: &str,
    args_json: &str,
    iters: usize,
) -> Result<Value> {
    if iters == 0 {
        return Err(SimdError::new("run-wasm-prepared requires --iters >= 1"));
    }
    let (_surface, _module, checked) = compile_frontend(source)?;
    let specialized = specialize_checked_program_for_main(&checked, main)?;
    let lowered_ready = eliminate_non_escaping_lambdas_program(&specialized.checked)?;
    let canonical = canonicalize_backend_higher_order_program(&lowered_ready)?;
    let args = parse_host_args(args_json, &canonical, main)?;
    let plan = build_wasm_plan(&canonical, main)?;
    let artifact = compile_wasm_artifact_checked(&canonical, main, &specialized.origins)?;
    let flat_param_abis = flatten_wasm_param_abis(&plan.params)?;
    let input_templates = prepared_input_templates(&plan.params)?;
    let output_templates = prepared_output_templates(&plan.result, &artifact, &input_templates)?;
    let engine = build_engine()?;
    let module = WasmtimeModule::from_binary(&engine, &artifact.bytes)
        .map_err(|error| SimdError::new(format!("failed to compile Wasm module: {error:#}")))?;
    let runtime = build_runtime(&engine, &module, &artifact)?;
    let prepared = PreparedWasmMain {
        artifact,
        params: plan.params,
        flat_param_abis,
        input_templates,
        output_templates,
        runtime: Rc::new(RefCell::new(runtime)),
    };
    let layout = prepared.layout_from_args(&args)?;
    let mut bound = prepared.bind(layout)?;
    for _ in 0..iters {
        bound.load_args(&args)?;
        bound.run()?;
    }
    bound.read_result()
}

pub fn wasm_command(path: &str, main: &str, out: Option<&str>) -> Result<String> {
    let (_canonical, artifact, _cache_hit) = load_or_compile_wasm_command_artifact(path, main)?;
    if let Some(path) = out {
        fs::write(path, &artifact.bytes)
            .map_err(|error| SimdError::new(format!("failed to write '{path}': {error}")))?;
    }
    Ok(format!("{:#?}", artifact))
}

pub fn wat_main(source: &str, main: &str) -> Result<String> {
    let artifact = compile_wasm_main(source, main)?;
    let wat = wasmprinter::print_bytes(&artifact.bytes)
        .map_err(|error| SimdError::new(format!("failed to print WAT: {error}")))?;
    Ok(format!(
        "{}\n{}\n{}",
        render_optimizer_report_comments(&artifact.optimizer_reports),
        render_higher_order_report_comments(&artifact.higher_order_reports),
        wat
    ))
}

pub fn wat_command(path: &str, main: &str) -> Result<String> {
    let source = read_source_file(path)?;
    wat_main(&source, main)
}

pub fn run_wasm_command(path: &str, main: &str, args_json: &str) -> Result<String> {
    let (canonical, artifact, _cache_hit) = load_or_compile_wasm_command_artifact(path, main)?;
    let args = parse_host_args(args_json, &canonical, main)?;
    let value = prepare_wasm_artifact(artifact)?.run(&args)?;
    Ok(value.to_json_string())
}

pub fn run_wasm_prepared_command(
    path: &str,
    main: &str,
    args_json: &str,
    iters: usize,
) -> Result<String> {
    if iters == 0 {
        return Err(SimdError::new("run-wasm-prepared requires --iters >= 1"));
    }
    let (canonical, artifact, _cache_hit) = load_or_compile_wasm_command_artifact(path, main)?;
    let args = parse_host_args(args_json, &canonical, main)?;
    let executable = prepare_wasm_artifact(artifact)?;
    let mut last = None;
    for _ in 0..iters {
        last = Some(executable.run(&args)?);
    }
    let value = last.ok_or_else(|| SimdError::new("missing prepared Wasm result"))?;
    Ok(value.to_json_string())
}

pub fn run_wasm_profile_fns_main(
    source: &str,
    main: &str,
    args_json: &str,
) -> Result<WasmFunctionProfile> {
    let total_start = Instant::now();
    let (_surface, _module, checked) = compile_frontend(source)?;
    let specialized = specialize_checked_program_for_main(&checked, main)?;
    let lowered_ready = eliminate_non_escaping_lambdas_program(&specialized.checked)?;
    let canonical = canonicalize_backend_higher_order_program(&lowered_ready)?;
    let args = parse_host_args(args_json, &canonical, main)?;
    let artifact = compile_wasm_artifact_checked_with_options(
        &canonical,
        main,
        &specialized.origins,
        WasmCompileOptions {
            function_profiler: true,
        },
    )?;
    let engine = build_engine()?;
    let module = WasmtimeModule::from_binary(&engine, &artifact.bytes)
        .map_err(|error| SimdError::new(format!("failed to compile Wasm module: {error:#}")))?;
    let mut runtime = build_runtime(&engine, &module, &artifact)?;
    let start = Instant::now();
    let value = execute_wasm_artifact_in_runtime(&mut runtime, &artifact, &args)?;
    let execute_us = start.elapsed().as_micros();
    let functions = {
        let profiler = runtime
            .function_profiler
            .as_ref()
            .ok_or_else(|| SimdError::new("missing Wasm function profiler state"))?;
        let mut profiler = profiler
            .lock()
            .map_err(|_| SimdError::new("Wasm function profiler mutex was poisoned"))?;
        profiler.record_top_level_call_if_missing(main, execute_us);
        profiler.finish()
    };
    let _total_us = total_start.elapsed().as_micros();
    Ok(WasmFunctionProfile {
        result_json: value.to_json_string(),
        total_us: execute_us,
        functions,
    })
}

pub fn run_wasm_profile_fns_command(
    path: &str,
    main: &str,
    args_json: &str,
    json_output: bool,
) -> Result<String> {
    let source = read_source_file(path)?;
    let profile = run_wasm_profile_fns_main(&source, main, args_json)?;
    Ok(if json_output {
        profile.to_json_string()
    } else {
        profile.to_table_string()
    })
}

pub fn run_wasm_profile_main(source: &str, main: &str, args_json: &str) -> Result<WasmRunProfile> {
    let total_start = Instant::now();

    let start = Instant::now();
    let (_surface, _module, checked) = compile_frontend(source)?;
    let frontend_elapsed = start.elapsed();
    let frontend_ms = frontend_elapsed.as_millis();
    let frontend_us = frontend_elapsed.as_micros();

    let start = Instant::now();
    let specialized = specialize_checked_program_for_main(&checked, main)?;
    let specialize_elapsed = start.elapsed();
    let specialize_ms = specialize_elapsed.as_millis();
    let specialize_us = specialize_elapsed.as_micros();

    let start = Instant::now();
    let lowered_ready = eliminate_non_escaping_lambdas_program(&specialized.checked)?;
    let lambda_lower_elapsed = start.elapsed();
    let lambda_lower_ms = lambda_lower_elapsed.as_millis();
    let lambda_lower_us = lambda_lower_elapsed.as_micros();

    let start = Instant::now();
    let canonical = canonicalize_backend_higher_order_program(&lowered_ready)?;
    let canonicalize_elapsed = start.elapsed();
    let canonicalize_ms = canonicalize_elapsed.as_millis();
    let canonicalize_us = canonicalize_elapsed.as_micros();

    let start = Instant::now();
    let args = parse_host_args(args_json, &canonical, main)?;
    let parse_args_elapsed = start.elapsed();
    let parse_args_ms = parse_args_elapsed.as_millis();
    let parse_args_us = parse_args_elapsed.as_micros();

    let start = Instant::now();
    let _profiled_wasm_enum_layouts = build_precomputed_wasm_enum_layouts(&canonical)?;
    let compile_enum_layouts_elapsed = start.elapsed();
    let compile_enum_layouts_ms = compile_enum_layouts_elapsed.as_millis();
    let compile_enum_layouts_us = compile_enum_layouts_elapsed.as_micros();

    let start = Instant::now();
    let profiled_plan = build_wasm_plan(&canonical, main)?;
    let compile_plan_elapsed = start.elapsed();
    let compile_plan_ms = compile_plan_elapsed.as_millis();
    let compile_plan_us = compile_plan_elapsed.as_micros();

    let start = Instant::now();
    let profiled_normalized = normalize_records(&profiled_plan.checked)?;
    let compile_normalize_elapsed = start.elapsed();
    let compile_normalize_ms = compile_normalize_elapsed.as_millis();
    let compile_normalize_us = compile_normalize_elapsed.as_micros();

    let start = Instant::now();
    let profiled_lowered =
        optimize_lowered_program(&prepare_lowered_program(&profiled_normalized)?);
    let compile_lower_elapsed = start.elapsed();
    let compile_lower_ms = compile_lower_elapsed.as_millis();
    let compile_lower_us = compile_lower_elapsed.as_micros();

    let start = Instant::now();
    let profiled_grouped = group_lowered_program(&profiled_normalized, &profiled_lowered)?;
    let compile_group_elapsed = start.elapsed();
    let compile_group_ms = compile_group_elapsed.as_millis();
    let compile_group_us = compile_group_elapsed.as_micros();

    let start = Instant::now();
    let _profiled_intents = analyze_intents(&profiled_grouped);
    let compile_intents_elapsed = start.elapsed();
    let compile_intents_ms = compile_intents_elapsed.as_millis();
    let compile_intents_us = compile_intents_elapsed.as_micros();

    let start = Instant::now();
    let artifact = compile_wasm_artifact_checked(&canonical, main, &specialized.origins)?;
    let compile_artifact_elapsed = start.elapsed();
    let compile_artifact_ms = compile_artifact_elapsed.as_millis();
    let compile_artifact_us = compile_artifact_elapsed.as_micros();
    let artifact_bytes = artifact.bytes.len();

    let start = Instant::now();
    let engine = build_engine()?;
    let engine_elapsed = start.elapsed();
    let engine_ms = engine_elapsed.as_millis();
    let engine_us = engine_elapsed.as_micros();

    let start = Instant::now();
    let module = WasmtimeModule::from_binary(&engine, &artifact.bytes)
        .map_err(|error| SimdError::new(format!("failed to compile Wasm module: {error:#}")))?;
    let module_compile_elapsed = start.elapsed();
    let module_compile_ms = module_compile_elapsed.as_millis();
    let module_compile_us = module_compile_elapsed.as_micros();

    let start = Instant::now();
    let runtime = build_runtime(&engine, &module, &artifact)?;
    let runtime_build_elapsed = start.elapsed();
    let runtime_build_ms = runtime_build_elapsed.as_millis();
    let runtime_build_us = runtime_build_elapsed.as_micros();

    let executable = WasmExecutable {
        artifact,
        runtime: RefCell::new(runtime),
    };
    let structural_summary =
        summarize_structural_optimizer_reports(&executable.artifact.optimizer_reports);

    let start = Instant::now();
    let value = executable.run(&args)?;
    let execute_elapsed = start.elapsed();
    let execute_ms = execute_elapsed.as_millis();
    let execute_us = execute_elapsed.as_micros();

    let total_elapsed = total_start.elapsed();

    Ok(WasmRunProfile {
        result_json: value.to_json_string(),
        artifact_bytes,
        frontend_ms,
        frontend_us,
        specialize_ms,
        specialize_us,
        lambda_lower_ms,
        lambda_lower_us,
        canonicalize_ms,
        canonicalize_us,
        parse_args_ms,
        parse_args_us,
        compile_artifact_ms,
        compile_artifact_us,
        compile_enum_layouts_ms,
        compile_enum_layouts_us,
        compile_plan_ms,
        compile_plan_us,
        compile_normalize_ms,
        compile_normalize_us,
        compile_lower_ms,
        compile_lower_us,
        compile_group_ms,
        compile_group_us,
        compile_intents_ms,
        compile_intents_us,
        engine_ms,
        engine_us,
        module_compile_ms,
        module_compile_us,
        runtime_build_ms,
        runtime_build_us,
        execute_ms,
        execute_us,
        structural_functions: structural_summary.structural_functions,
        fallback_functions: structural_summary.fallback_functions,
        parser_fallback_functions: structural_summary.parser_fallback_functions,
        structural_sccs: structural_summary.structural_sccs,
        structural_state_count: structural_summary.structural_state_count,
        structural_transition_count: structural_summary.structural_transition_count,
        structural_span_ops: structural_summary.structural_span_ops,
        structural_enum_ops: structural_summary.structural_enum_ops,
        total_ms: total_elapsed.as_millis(),
        total_us: total_elapsed.as_micros(),
    })
}

pub fn run_wasm_profile_command(path: &str, main: &str, args_json: &str) -> Result<String> {
    let source = read_source_file(path)?;
    Ok(run_wasm_profile_main(&source, main, args_json)?.to_kv_string())
}

fn render_optimizer_report_comments(reports: &[WasmOptimizationReport]) -> String {
    if reports.is_empty() {
        return ";; optimizer: none".to_string();
    }
    let mut lines = vec![";; optimizer reports:".to_string()];
    for report in reports {
        let plan = match report.vector_unroll {
            0 => "scalar",
            1 => "vec1",
            2 => "vec2",
            4 => "vec4",
            _ => "vecN",
        };
        let exec = match report.structural_exec {
            StructuralExecMode::Scalar => "scalar",
            StructuralExecMode::StructuralLoop => "structural-loop",
            StructuralExecMode::StructuralBatched => "structural-batched",
        };
        let fallback = report
            .fallback_reason
            .as_ref()
            .map(|reason| format!(" fallback={reason}"))
            .unwrap_or_default();
        let structural = if report.structural_state_count == 0
            && report.structural_transition_count == 0
            && report.structural_span_ops == 0
            && report.structural_enum_ops == 0
            && report.structural_region_count == 0
            && report.structural_scc.is_none()
        {
            String::new()
        } else {
            format!(
                " scc={:?} structural=states:{} transitions:{} span:{} enum:{} regions:{} char-runs:{} separated:{}",
                report.structural_scc,
                report.structural_state_count,
                report.structural_transition_count,
                report.structural_span_ops,
                report.structural_enum_ops,
                report.structural_region_count,
                report.structural_char_prefix_regions,
                report.structural_separated_item_regions
            )
        };
        lines.push(format!(
            ";; - fn={} intent={:?} exec={} plan={}{}{}",
            report.function, report.intent, exec, plan, structural, fallback
        ));
    }
    lines.join("\n")
}

fn render_higher_order_report_comments(reports: &[WasmHigherOrderReport]) -> String {
    if reports.is_empty() {
        return ";; higher-order reports: none".to_string();
    }
    let mut lines = vec![";; higher-order reports:".to_string()];
    for report in reports {
        let rejection = report
            .rejection_reason
            .as_ref()
            .map(|reason| format!(" reject={reason}"))
            .unwrap_or_default();
        lines.push(format!(
            ";; - fn={} origin={} mode={:?} captures={} env_bytes={} known_fn={} known_lambda={} escaping_unknown={}{}",
            report.function,
            report.specialization_origin,
            report.lambda_mode,
            report.capture_count,
            report.env_bytes,
            report.known_fn_values,
            report.known_lambda_values,
            report.escaping_unknown_values,
            rejection
        ));
    }
    lines.join("\n")
}

fn structural_cluster_membership(
    analysis: &StructuralAnalysis,
) -> BTreeMap<String, StructuralClusterReport> {
    let mut membership = BTreeMap::new();
    for cluster in &analysis.clusters {
        for function in &cluster.functions {
            membership.insert(function.clone(), cluster.clone());
        }
    }
    membership
}

fn structural_report_fields(
    lowered: &LoweredFunction,
    cluster: Option<&StructuralClusterReport>,
) -> (
    Option<usize>,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
    usize,
) {
    let mut span_ops = 0usize;
    let mut enum_ops = 0usize;
    let (region_count, char_prefix_regions, separated_item_regions) = match &lowered.kind {
        LoweredKind::Structural { program, .. } => {
            let mut char_prefix = 0usize;
            let mut separated = 0usize;
            for region in &program.sequence_regions {
                match region.mode {
                    StructuralSequenceMode::CharPrefixRun => char_prefix += 1,
                    StructuralSequenceMode::SeparatedItems => separated += 1,
                }
            }
            (program.sequence_regions.len(), char_prefix, separated)
        }
        _ => (0, 0, 0),
    };
    match &lowered.kind {
        LoweredKind::Scalar { clauses } | LoweredKind::Structural { clauses, .. } => {
            for clause in clauses {
                span_ops += clause
                    .patterns
                    .iter()
                    .filter(|pattern| matches!(pattern.pattern, Pattern::Slice { .. }))
                    .count();
                enum_ops += clause
                    .patterns
                    .iter()
                    .filter(|pattern| matches!(pattern.pattern, Pattern::Ctor(_, _)))
                    .count();
                span_ops += count_ir_structural_span_ops(&clause.body);
                enum_ops += count_ir_structural_enum_ops(&clause.body);
            }
        }
        LoweredKind::Kernel { .. } => {}
    }
    if let Some(tail_loop) = &lowered.tail_loop {
        for clause in &tail_loop.clauses {
            span_ops += clause
                .patterns
                .iter()
                .filter(|pattern| matches!(pattern.pattern, Pattern::Slice { .. }))
                .count();
            enum_ops += clause
                .patterns
                .iter()
                .filter(|pattern| matches!(pattern.pattern, Pattern::Ctor(_, _)))
                .count();
            match &clause.action {
                TailAction::Continue { args } => {
                    for arg in args {
                        span_ops += count_ir_structural_span_ops(arg);
                        enum_ops += count_ir_structural_enum_ops(arg);
                    }
                }
                TailAction::Return { expr } => {
                    span_ops += count_ir_structural_span_ops(expr);
                    enum_ops += count_ir_structural_enum_ops(expr);
                }
            }
        }
    }
    (
        cluster.map(|cluster| cluster.id),
        cluster.map(|cluster| cluster.state_count).unwrap_or(0),
        cluster.map(|cluster| cluster.transition_count).unwrap_or(0),
        span_ops,
        enum_ops,
        region_count,
        char_prefix_regions,
        separated_item_regions,
    )
}

fn lowered_calls_structural_candidate(
    lowered: &LoweredFunction,
    candidates: &BTreeSet<String>,
) -> bool {
    (match &lowered.kind {
        LoweredKind::Scalar { clauses } | LoweredKind::Structural { clauses, .. } => clauses
            .iter()
            .any(|clause| ir_calls_structural_candidate(&clause.body, candidates)),
        LoweredKind::Kernel { .. } => false,
    }) || lowered.tail_loop.as_ref().is_some_and(|tail_loop| {
        tail_loop.clauses.iter().any(|clause| match &clause.action {
            TailAction::Continue { args } => args
                .iter()
                .any(|arg| ir_calls_structural_candidate(arg, candidates)),
            TailAction::Return { expr } => ir_calls_structural_candidate(expr, candidates),
        })
    })
}

fn ir_calls_structural_candidate(expr: &IrExpr, candidates: &BTreeSet<String>) -> bool {
    match &expr.kind {
        IrExprKind::Call {
            callee: Callee::Function(name),
            args,
        } => {
            candidates.contains(name)
                || args
                    .iter()
                    .any(|arg| ir_calls_structural_candidate(arg, candidates))
        }
        IrExprKind::Call { args, .. } => args
            .iter()
            .any(|arg| ir_calls_structural_candidate(arg, candidates)),
        IrExprKind::Record(fields) => fields
            .values()
            .any(|field| ir_calls_structural_candidate(field, candidates)),
        IrExprKind::EnumCtor { args, .. } => args
            .iter()
            .any(|arg| ir_calls_structural_candidate(arg, candidates)),
        IrExprKind::EnumTag { value }
        | IrExprKind::EnumChildBySlot { value, .. }
        | IrExprKind::EnumNonRecField { value, .. } => {
            ir_calls_structural_candidate(value, candidates)
        }
        IrExprKind::Seq(items) => items
            .iter()
            .any(|item| ir_calls_structural_candidate(item, candidates)),
        IrExprKind::SeqSplice { prefix, tail } => {
            prefix
                .iter()
                .any(|item| ir_calls_structural_candidate(item, candidates))
                || ir_calls_structural_candidate(tail, candidates)
        }
        IrExprKind::Let { bindings, body } => {
            bindings
                .iter()
                .any(|binding| ir_calls_structural_candidate(&binding.expr, candidates))
                || ir_calls_structural_candidate(body, candidates)
        }
        IrExprKind::Local(_) | IrExprKind::Int(_, _) | IrExprKind::Float(_, _) => false,
    }
}

fn count_ir_structural_span_ops(expr: &IrExpr) -> usize {
    match &expr.kind {
        IrExprKind::Call {
            callee:
                Callee::Builtin(
                    BuiltinFamilyCallee::ConcatString
                    | BuiltinFamilyCallee::LenString
                    | BuiltinFamilyCallee::SliceString
                    | BuiltinFamilyCallee::ContainsString
                    | BuiltinFamilyCallee::EqString,
                ),
            args,
        } => 1 + args.iter().map(count_ir_structural_span_ops).sum::<usize>(),
        IrExprKind::Call { args, .. } => args.iter().map(count_ir_structural_span_ops).sum(),
        IrExprKind::Record(fields) => fields.values().map(count_ir_structural_span_ops).sum(),
        IrExprKind::EnumCtor { args, .. } => args.iter().map(count_ir_structural_span_ops).sum(),
        IrExprKind::EnumTag { value }
        | IrExprKind::EnumChildBySlot { value, .. }
        | IrExprKind::EnumNonRecField { value, .. } => count_ir_structural_span_ops(value),
        IrExprKind::Seq(items) => items.iter().map(count_ir_structural_span_ops).sum(),
        IrExprKind::SeqSplice { prefix, tail } => {
            prefix
                .iter()
                .map(count_ir_structural_span_ops)
                .sum::<usize>()
                + count_ir_structural_span_ops(tail)
        }
        IrExprKind::Let { bindings, body } => {
            bindings
                .iter()
                .map(|binding| count_ir_structural_span_ops(&binding.expr))
                .sum::<usize>()
                + count_ir_structural_span_ops(body)
        }
        IrExprKind::Local(_) | IrExprKind::Int(_, _) | IrExprKind::Float(_, _) => 0,
    }
}

fn count_ir_structural_enum_ops(expr: &IrExpr) -> usize {
    match &expr.kind {
        IrExprKind::EnumCtor { args, .. } => {
            1 + args.iter().map(count_ir_structural_enum_ops).sum::<usize>()
        }
        IrExprKind::EnumTag { value }
        | IrExprKind::EnumChildBySlot { value, .. }
        | IrExprKind::EnumNonRecField { value, .. } => 1 + count_ir_structural_enum_ops(value),
        IrExprKind::Seq(items) => items.iter().map(count_ir_structural_enum_ops).sum(),
        IrExprKind::SeqSplice { prefix, tail } => {
            prefix
                .iter()
                .map(count_ir_structural_enum_ops)
                .sum::<usize>()
                + count_ir_structural_enum_ops(tail)
        }
        IrExprKind::Call { args, .. } => args.iter().map(count_ir_structural_enum_ops).sum(),
        IrExprKind::Record(fields) => fields.values().map(count_ir_structural_enum_ops).sum(),
        IrExprKind::Let { bindings, body } => {
            bindings
                .iter()
                .map(|binding| count_ir_structural_enum_ops(&binding.expr))
                .sum::<usize>()
                + count_ir_structural_enum_ops(body)
        }
        IrExprKind::Local(_) | IrExprKind::Int(_, _) | IrExprKind::Float(_, _) => 0,
    }
}

impl WasmExecutable {
    pub fn artifact(&self) -> &WasmArtifact {
        &self.artifact
    }

    pub fn run(&self, args: &[Value]) -> Result<Value> {
        let mut runtime = self.runtime.borrow_mut();
        execute_wasm_artifact_in_runtime(&mut runtime, &self.artifact, args)
    }

    pub(crate) fn run_hot(&self, args: &[Value]) -> Result<Value> {
        self.run(args)
    }
}

impl PreparedWasmMain {
    pub fn artifact(&self) -> &WasmArtifact {
        &self.artifact
    }

    pub fn layout_from_args(&self, args: &[Value]) -> Result<PreparedLayout> {
        if args.len() != self.params.len() {
            return Err(SimdError::new(format!(
                "prepared entry expects {} arguments, found {}",
                self.params.len(),
                args.len()
            )));
        }
        let (flat_args, _) = flatten_wasm_args(&self.params, args)?;
        let mut bulk_input_shapes = Vec::new();
        for (abi, arg) in self.flat_param_abis.iter().zip(flat_args.iter()) {
            if matches!(abi, WasmParamAbi::Bulk { .. }) {
                let Value::Bulk(bulk) = arg else {
                    return Err(SimdError::new(
                        "flattened argument mismatch: expected bulk for prepared layout",
                    ));
                };
                bulk_input_shapes.push(bulk.shape.clone());
            }
        }
        Ok(PreparedLayout { bulk_input_shapes })
    }

    pub fn bind(&self, layout: PreparedLayout) -> Result<BoundPreparedRun> {
        bind_prepared_run(self, layout)
    }
}

impl BoundPreparedRun {
    pub fn slots(&self) -> Vec<PreparedSlotMetadata> {
        self.slots.iter().map(|slot| slot.meta().clone()).collect()
    }

    pub fn set_scalar_i32(&mut self, slot: usize, value: i32) -> Result<()> {
        self.set_input_scalar(slot, Prim::I32, ScalarValue::I32(value))
    }

    pub fn set_scalar_i64(&mut self, slot: usize, value: i64) -> Result<()> {
        self.set_input_scalar(slot, Prim::I64, ScalarValue::I64(value))
    }

    pub fn set_scalar_f32(&mut self, slot: usize, value: f32) -> Result<()> {
        self.set_input_scalar(slot, Prim::F32, ScalarValue::F32(value))
    }

    pub fn set_scalar_f64(&mut self, slot: usize, value: f64) -> Result<()> {
        self.set_input_scalar(slot, Prim::F64, ScalarValue::F64(value))
    }

    pub fn write_bulk_i32(&mut self, slot: usize, values: &[i32]) -> Result<()> {
        self.write_input_bulk(
            slot,
            Prim::I32,
            values
                .iter()
                .copied()
                .map(ScalarValue::I32)
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    pub fn write_bulk_i64(&mut self, slot: usize, values: &[i64]) -> Result<()> {
        self.write_input_bulk(
            slot,
            Prim::I64,
            values
                .iter()
                .copied()
                .map(ScalarValue::I64)
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    pub fn write_bulk_f32(&mut self, slot: usize, values: &[f32]) -> Result<()> {
        self.write_input_bulk(
            slot,
            Prim::F32,
            values
                .iter()
                .copied()
                .map(ScalarValue::F32)
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    pub fn write_bulk_f64(&mut self, slot: usize, values: &[f64]) -> Result<()> {
        self.write_input_bulk(
            slot,
            Prim::F64,
            values
                .iter()
                .copied()
                .map(ScalarValue::F64)
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    pub fn read_bulk_i32(&self, slot: usize, out: &mut [i32]) -> Result<()> {
        read_output_bulk_i32(self, slot, out)
    }

    pub fn read_bulk_i64(&self, slot: usize, out: &mut [i64]) -> Result<()> {
        read_output_bulk_i64(self, slot, out)
    }

    pub fn read_bulk_f32(&self, slot: usize, out: &mut [f32]) -> Result<()> {
        read_output_bulk_f32(self, slot, out)
    }

    pub fn read_bulk_f64(&self, slot: usize, out: &mut [f64]) -> Result<()> {
        read_output_bulk_f64(self, slot, out)
    }

    pub fn load_args(&mut self, args: &[Value]) -> Result<()> {
        load_prepared_args(self, args)
    }

    pub fn run(&mut self) -> Result<()> {
        run_prepared_bound(self)
    }

    pub fn read_result(&self) -> Result<Value> {
        read_prepared_result(self)
    }
}

impl PreparedSlotState {
    fn meta(&self) -> &PreparedSlotMetadata {
        match self {
            PreparedSlotState::InputScalar { meta, .. }
            | PreparedSlotState::InputBulk { meta, .. }
            | PreparedSlotState::OutputScalar { meta, .. }
            | PreparedSlotState::OutputBulk { meta, .. } => meta,
        }
    }
}

fn prepared_input_templates(params: &[WasmParamAbi]) -> Result<Vec<PreparedInputTemplate>> {
    let mut templates = Vec::new();
    let mut next_flat = 0usize;
    for (index, param) in params.iter().enumerate() {
        collect_prepared_input_templates(
            param,
            &LeafPath(vec![format!("arg{}", index)]),
            &mut next_flat,
            &mut templates,
        )?;
    }
    Ok(templates)
}

fn collect_prepared_input_templates(
    abi: &WasmParamAbi,
    logical_path: &LeafPath,
    next_flat: &mut usize,
    out: &mut Vec<PreparedInputTemplate>,
) -> Result<()> {
    match abi {
        WasmParamAbi::Scalar { prim } => {
            out.push(PreparedInputTemplate {
                flat_index: *next_flat,
                prim: *prim,
                kind: PreparedSlotKind::Scalar,
                logical_path: logical_path.clone(),
            });
            *next_flat += 1;
        }
        WasmParamAbi::Bulk { prim } => {
            out.push(PreparedInputTemplate {
                flat_index: *next_flat,
                prim: *prim,
                kind: PreparedSlotKind::Bulk,
                logical_path: logical_path.clone(),
            });
            *next_flat += 1;
        }
        WasmParamAbi::Record { fields } => {
            for (name, field) in fields {
                collect_prepared_input_templates(field, &logical_path.child(name), next_flat, out)?;
            }
        }
    }
    Ok(())
}

fn prepared_output_templates(
    result: &WasmResultAbi,
    artifact: &WasmArtifact,
    input_templates: &[PreparedInputTemplate],
) -> Result<Vec<PreparedOutputTemplate>> {
    let first_bulk_flat = input_templates
        .iter()
        .find(|slot| slot.kind == PreparedSlotKind::Bulk)
        .map(|slot| slot.flat_index);
    if can_use_direct_wasm_path(artifact) {
        return match result {
            WasmResultAbi::Scalar { prim } => Ok(vec![PreparedOutputTemplate {
                prim: *prim,
                kind: PreparedSlotKind::Scalar,
                logical_path: LeafPath(vec!["result".to_string()]),
                leaf_path: LeafPath::root(),
                scalar_via_pointer: false,
                shape_param_top_level: None,
            }]),
            WasmResultAbi::Bulk { prim, shape_param } => Ok(vec![PreparedOutputTemplate {
                prim: *prim,
                kind: PreparedSlotKind::Bulk,
                logical_path: LeafPath(vec!["result".to_string()]),
                leaf_path: LeafPath::root(),
                scalar_via_pointer: false,
                shape_param_top_level: Some(*shape_param),
            }]),
            WasmResultAbi::StarSeq { .. } => Ok(Vec::new()),
            WasmResultAbi::Record { .. } => Err(SimdError::new(
                "direct prepared path does not support record results",
            )),
        };
    }

    let leafs = if let Some(grouped) = &artifact.grouped_export {
        grouped.leaf_exports.as_slice()
    } else {
        artifact.leaf_exports.as_slice()
    };
    let mut templates = Vec::with_capacity(leafs.len());
    for leaf in leafs {
        let (prim, kind, scalar_via_pointer) = match leaf.result {
            WasmLeafResultAbi::Scalar { prim } => (
                prim,
                PreparedSlotKind::Scalar,
                artifact.grouped_export.is_some(),
            ),
            WasmLeafResultAbi::Bulk { prim } => (prim, PreparedSlotKind::Bulk, false),
            WasmLeafResultAbi::StarSeq { .. } => {
                return Err(SimdError::new(
                    "prepared execution does not support T[*] outputs yet",
                ));
            }
        };
        if kind == PreparedSlotKind::Bulk && first_bulk_flat.is_none() {
            return Err(SimdError::new(
                "prepared output bulk slot requires at least one bulk input slot",
            ));
        }
        templates.push(PreparedOutputTemplate {
            prim,
            kind,
            logical_path: prefix_result_path(&leaf.leaf_path),
            leaf_path: leaf.leaf_path.clone(),
            scalar_via_pointer,
            shape_param_top_level: None,
        });
    }
    Ok(templates)
}

fn bind_prepared_run(main: &PreparedWasmMain, layout: PreparedLayout) -> Result<BoundPreparedRun> {
    let bulk_input_templates = main
        .input_templates
        .iter()
        .filter(|slot| slot.kind == PreparedSlotKind::Bulk)
        .collect::<Vec<_>>();
    if layout.bulk_input_shapes.len() != bulk_input_templates.len() {
        return Err(SimdError::new(format!(
            "prepared layout expected {} bulk input shapes, found {}",
            bulk_input_templates.len(),
            layout.bulk_input_shapes.len()
        )));
    }
    let bulk_shapes = layout.bulk_input_shapes;
    for shape in bulk_shapes.iter().skip(1) {
        if shape != &bulk_shapes[0] {
            return Err(SimdError::new(
                "prepared layout currently requires all bulk input shapes to match",
            ));
        }
    }

    let top_level_flat_bulk = top_level_bulk_flat_indices(&main.params)?;
    let mut runtime = main.runtime.borrow_mut();
    let mut slots = Vec::<PreparedSlotState>::new();
    let mut input_scalar_slot_by_flat = BTreeMap::<usize, usize>::new();
    let mut input_bulk_slot_by_flat = BTreeMap::<usize, usize>::new();
    let mut output_slot_by_leaf = BTreeMap::<LeafPath, usize>::new();

    let mut bulk_shape_by_flat = BTreeMap::<usize, Vec<usize>>::new();
    for (template, shape) in bulk_input_templates.iter().zip(&bulk_shapes) {
        bulk_shape_by_flat.insert(template.flat_index, shape.clone());
    }

    for template in &main.input_templates {
        let slot_index = slots.len();
        match template.kind {
            PreparedSlotKind::Scalar => {
                let meta = PreparedSlotMetadata {
                    index: slot_index,
                    role: PreparedSlotRole::Input,
                    kind: PreparedSlotKind::Scalar,
                    logical_path: template.logical_path.clone(),
                    prim: template.prim,
                    shape: Vec::new(),
                    offset_elems: 0,
                    strides_elems: Vec::new(),
                };
                slots.push(PreparedSlotState::InputScalar {
                    meta,
                    value: default_scalar_value(template.prim),
                });
                input_scalar_slot_by_flat.insert(template.flat_index, slot_index);
            }
            PreparedSlotKind::Bulk => {
                let shape = bulk_shape_by_flat
                    .get(&template.flat_index)
                    .cloned()
                    .ok_or_else(|| {
                        SimdError::new(format!(
                            "missing prepared bulk shape for input flat index {}",
                            template.flat_index
                        ))
                    })?;
                let len = element_count(&shape)?;
                let ptr = runtime.alloc_pinned(len * byte_width(template.prim) as usize)?;
                let meta = PreparedSlotMetadata {
                    index: slot_index,
                    role: PreparedSlotRole::Input,
                    kind: PreparedSlotKind::Bulk,
                    logical_path: template.logical_path.clone(),
                    prim: template.prim,
                    shape: shape.clone(),
                    offset_elems: 0,
                    strides_elems: row_major_strides(&shape),
                };
                slots.push(PreparedSlotState::InputBulk { meta, ptr, len });
                input_bulk_slot_by_flat.insert(template.flat_index, slot_index);
            }
        }
    }

    let default_bulk_shape = bulk_shapes.first().cloned();
    for template in &main.output_templates {
        let slot_index = slots.len();
        match template.kind {
            PreparedSlotKind::Scalar => {
                let ptr = if template.scalar_via_pointer {
                    Some(runtime.alloc_pinned(byte_width(template.prim) as usize)?)
                } else {
                    None
                };
                let meta = PreparedSlotMetadata {
                    index: slot_index,
                    role: PreparedSlotRole::Output,
                    kind: PreparedSlotKind::Scalar,
                    logical_path: template.logical_path.clone(),
                    prim: template.prim,
                    shape: Vec::new(),
                    offset_elems: 0,
                    strides_elems: Vec::new(),
                };
                slots.push(PreparedSlotState::OutputScalar {
                    meta,
                    leaf_path: template.leaf_path.clone(),
                    ptr,
                    value: default_scalar_value(template.prim),
                });
                output_slot_by_leaf.insert(template.leaf_path.clone(), slot_index);
            }
            PreparedSlotKind::Bulk => {
                let shape = if let Some(shape_param) = template.shape_param_top_level {
                    let flat_index = top_level_flat_bulk
                        .get(&shape_param)
                        .and_then(|indices| indices.first())
                        .copied()
                        .ok_or_else(|| {
                            SimdError::new(format!(
                                "prepared bulk output shape source argument {} did not flatten to a bulk slot",
                                shape_param
                            ))
                        })?;
                    bulk_shape_by_flat.get(&flat_index).cloned().ok_or_else(|| {
                        SimdError::new(format!(
                            "prepared bulk output could not resolve shape from flat input slot {}",
                            flat_index
                        ))
                    })?
                } else {
                    default_bulk_shape.clone().ok_or_else(|| {
                        SimdError::new(
                            "prepared bulk output requires at least one bulk input shape",
                        )
                    })?
                };
                let len = element_count(&shape)?;
                let ptr = runtime.alloc_pinned(len * byte_width(template.prim) as usize)?;
                let meta = PreparedSlotMetadata {
                    index: slot_index,
                    role: PreparedSlotRole::Output,
                    kind: PreparedSlotKind::Bulk,
                    logical_path: template.logical_path.clone(),
                    prim: template.prim,
                    shape: shape.clone(),
                    offset_elems: 0,
                    strides_elems: row_major_strides(&shape),
                };
                slots.push(PreparedSlotState::OutputBulk {
                    meta,
                    leaf_path: template.leaf_path.clone(),
                    ptr,
                    len,
                });
                output_slot_by_leaf.insert(template.leaf_path.clone(), slot_index);
            }
        }
    }

    let call_plan = build_prepared_call_plan(&main.artifact, &output_slot_by_leaf)?;
    drop(runtime);
    Ok(BoundPreparedRun {
        artifact: main.artifact.clone(),
        flat_param_abis: main.flat_param_abis.clone(),
        slots,
        input_scalar_slot_by_flat,
        input_bulk_slot_by_flat,
        call_plan,
        last_dynamic_result: None,
        runtime: Rc::clone(&main.runtime),
    })
}

fn build_prepared_call_plan(
    artifact: &WasmArtifact,
    output_slot_by_leaf: &BTreeMap<LeafPath, usize>,
) -> Result<PreparedCallPlan> {
    if can_use_direct_wasm_path(artifact) {
        return match artifact.result {
            WasmResultAbi::Scalar { prim } => Ok(PreparedCallPlan::DirectScalar {
                func_name: artifact.export_name.clone(),
                output_slot: *output_slot_by_leaf
                    .get(&LeafPath::root())
                    .ok_or_else(|| SimdError::new("missing prepared output slot for root"))?,
                prim,
            }),
            WasmResultAbi::StarSeq { prim } => Ok(PreparedCallPlan::DirectStarSeq {
                func_name: artifact.export_name.clone(),
                prim,
            }),
            WasmResultAbi::Bulk { .. } => Ok(PreparedCallPlan::DirectBulk {
                func_name: artifact.export_name.clone(),
                output_slot: *output_slot_by_leaf
                    .get(&LeafPath::root())
                    .ok_or_else(|| SimdError::new("missing prepared output slot for root"))?,
            }),
            WasmResultAbi::Record { .. } => Err(SimdError::new(
                "direct prepared call plan does not support record results",
            )),
        };
    }

    if let Some(grouped) = &artifact.grouped_export {
        let outputs = grouped
            .leaf_exports
            .iter()
            .map(|leaf| {
                Ok(PreparedLeafCall {
                    func_name: grouped.export_name.clone(),
                    output_slot: *output_slot_by_leaf.get(&leaf.leaf_path).ok_or_else(|| {
                        SimdError::new(format!(
                            "missing prepared output slot for leaf {:?}",
                            leaf.leaf_path
                        ))
                    })?,
                    result: leaf.result.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(PreparedCallPlan::GroupedExport {
            func_name: grouped.export_name.clone(),
            outputs,
        });
    }

    let leaves = artifact
        .leaf_exports
        .iter()
        .map(|leaf| {
            Ok(PreparedLeafCall {
                func_name: leaf.export_name.clone(),
                output_slot: *output_slot_by_leaf.get(&leaf.leaf_path).ok_or_else(|| {
                    SimdError::new(format!(
                        "missing prepared output slot for leaf {:?}",
                        leaf.leaf_path
                    ))
                })?,
                result: leaf.result.clone(),
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(PreparedCallPlan::LeafExports { leaves })
}

fn top_level_bulk_flat_indices(params: &[WasmParamAbi]) -> Result<BTreeMap<usize, Vec<usize>>> {
    let mut next_flat = 0usize;
    let mut map = BTreeMap::<usize, Vec<usize>>::new();
    for (index, param) in params.iter().enumerate() {
        collect_top_level_bulk_flat_indices(param, index, &mut next_flat, &mut map)?;
    }
    Ok(map)
}

fn collect_top_level_bulk_flat_indices(
    abi: &WasmParamAbi,
    top_level_index: usize,
    next_flat: &mut usize,
    out: &mut BTreeMap<usize, Vec<usize>>,
) -> Result<()> {
    match abi {
        WasmParamAbi::Scalar { .. } => *next_flat += 1,
        WasmParamAbi::Bulk { .. } => {
            out.entry(top_level_index).or_default().push(*next_flat);
            *next_flat += 1;
        }
        WasmParamAbi::Record { fields } => {
            for (_, field) in fields {
                collect_top_level_bulk_flat_indices(field, top_level_index, next_flat, out)?;
            }
        }
    }
    Ok(())
}

fn prefix_result_path(path: &LeafPath) -> LeafPath {
    let mut prefixed = Vec::with_capacity(path.0.len() + 1);
    prefixed.push("result".to_string());
    prefixed.extend(path.0.iter().cloned());
    LeafPath(prefixed)
}

fn element_count(shape: &[usize]) -> Result<usize> {
    shape.iter().try_fold(1usize, |acc, dim| {
        acc.checked_mul(*dim)
            .ok_or_else(|| SimdError::new("shape element count overflowed"))
    })
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }
    let mut strides = vec![1usize; shape.len()];
    for index in (0..shape.len().saturating_sub(1)).rev() {
        strides[index] = strides[index + 1].saturating_mul(shape[index + 1]);
    }
    strides
}

fn default_scalar_value(prim: Prim) -> ScalarValue {
    match prim {
        Prim::I32 => ScalarValue::I32(0),
        Prim::Char => ScalarValue::Char('\0'),
        Prim::I64 => ScalarValue::I64(0),
        Prim::F32 => ScalarValue::F32(0.0),
        Prim::F64 => ScalarValue::F64(0.0),
    }
}

impl BoundPreparedRun {
    fn slot_mut(&mut self, slot: usize) -> Result<&mut PreparedSlotState> {
        let total = self.slots.len();
        self.slots.get_mut(slot).ok_or_else(|| {
            SimdError::new(format!(
                "prepared slot {} was out of bounds ({} slots)",
                slot, total
            ))
        })
    }

    fn slot_ref(&self, slot: usize) -> Result<&PreparedSlotState> {
        self.slots.get(slot).ok_or_else(|| {
            SimdError::new(format!(
                "prepared slot {} was out of bounds ({} slots)",
                slot,
                self.slots.len()
            ))
        })
    }

    fn set_input_scalar(&mut self, slot: usize, prim: Prim, value: ScalarValue) -> Result<()> {
        match self.slot_mut(slot)? {
            PreparedSlotState::InputScalar {
                meta,
                value: slot_value,
                ..
            } if meta.prim == prim => {
                *slot_value = value;
                Ok(())
            }
            PreparedSlotState::InputScalar { meta, .. } => Err(SimdError::new(format!(
                "prepared slot {} expected {:?} scalar, found {:?}",
                slot, meta.prim, prim
            ))),
            PreparedSlotState::InputBulk { .. } => Err(SimdError::new(format!(
                "prepared slot {} is bulk, not scalar",
                slot
            ))),
            PreparedSlotState::OutputScalar { .. } | PreparedSlotState::OutputBulk { .. } => Err(
                SimdError::new(format!("prepared slot {} is an output slot", slot)),
            ),
        }
    }

    fn write_input_bulk(&mut self, slot: usize, prim: Prim, values: &[ScalarValue]) -> Result<()> {
        let (ptr, len, meta_prim) = match self.slot_ref(slot)? {
            PreparedSlotState::InputBulk { meta, ptr, len, .. } => (*ptr, *len, meta.prim),
            PreparedSlotState::InputScalar { .. } => {
                return Err(SimdError::new(format!(
                    "prepared slot {} is scalar, not bulk",
                    slot
                )));
            }
            PreparedSlotState::OutputScalar { .. } | PreparedSlotState::OutputBulk { .. } => {
                return Err(SimdError::new(format!(
                    "prepared slot {} is an output slot",
                    slot
                )));
            }
        };
        if meta_prim != prim {
            return Err(SimdError::new(format!(
                "prepared slot {} expected {:?} bulk values, found {:?}",
                slot, meta_prim, prim
            )));
        }
        if values.len() != len {
            return Err(SimdError::new(format!(
                "prepared slot {} expected {} elements, found {}",
                slot,
                len,
                values.len()
            )));
        }
        let mut runtime = self.runtime.borrow_mut();
        let memory = runtime.memory.clone();
        let store = &mut runtime.store;
        let data = memory.data_mut(store);
        let mut cursor = ptr;
        for value in values {
            write_scalar_bytes(data, &mut cursor, value)?;
        }
        Ok(())
    }
}

fn load_prepared_args(bound: &mut BoundPreparedRun, args: &[Value]) -> Result<()> {
    if args.len() != bound.artifact.params.len() {
        return Err(SimdError::new(format!(
            "prepared entry expects {} arguments, found {}",
            bound.artifact.params.len(),
            args.len()
        )));
    }
    let (flat_args, _) = flatten_wasm_args(&bound.artifact.params, args)?;
    for (flat_index, arg) in flat_args.iter().enumerate() {
        match arg {
            Value::Scalar(value) => {
                let slot = *bound
                    .input_scalar_slot_by_flat
                    .get(&flat_index)
                    .ok_or_else(|| {
                        SimdError::new(format!(
                            "prepared input scalar slot for flat index {} is missing",
                            flat_index
                        ))
                    })?;
                match value {
                    ScalarValue::I32(value) => bound.set_scalar_i32(slot, *value)?,
                    ScalarValue::I64(value) => bound.set_scalar_i64(slot, *value)?,
                    ScalarValue::F32(value) => bound.set_scalar_f32(slot, *value)?,
                    ScalarValue::F64(value) => bound.set_scalar_f64(slot, *value)?,
                    ScalarValue::Char(value) => {
                        bound.set_input_scalar(slot, Prim::Char, ScalarValue::Char(*value))?
                    }
                }
            }
            Value::Bulk(bulk) => {
                let slot = *bound
                    .input_bulk_slot_by_flat
                    .get(&flat_index)
                    .ok_or_else(|| {
                        SimdError::new(format!(
                            "prepared input bulk slot for flat index {} is missing",
                            flat_index
                        ))
                    })?;
                bound.write_input_bulk(slot, bulk.prim, &bulk.elements)?;
            }
            Value::Record(_) => {
                return Err(SimdError::new(
                    "prepared flattened input should not contain record values",
                ));
            }
            Value::Tuple(_) => {
                return Err(SimdError::new(
                    "prepared flattened input should not contain tuple values",
                ));
            }
            Value::Bool(_) => {
                return Err(SimdError::new(
                    "prepared execution does not support bool arguments",
                ));
            }
            Value::String(_) => {
                return Err(SimdError::new(
                    "prepared execution does not support string arguments",
                ));
            }
            Value::StarSeq(_) => {
                return Err(SimdError::new(
                    "prepared execution does not support T[*] arguments yet",
                ));
            }
            Value::TypeToken(_) => {
                return Err(SimdError::new(
                    "prepared execution does not support type witness runtime values",
                ));
            }
            Value::Enum(_) => {
                return Err(SimdError::new(
                    "prepared execution does not support enum host arguments yet",
                ));
            }
        }
    }
    Ok(())
}

fn run_prepared_bound(bound: &mut BoundPreparedRun) -> Result<()> {
    let input_args = build_prepared_input_wasm_args(bound)?;
    let mut scalar_updates = Vec::<(usize, ScalarValue)>::new();
    bound.last_dynamic_result = None;
    let mut runtime = bound.runtime.borrow_mut();
    let mut scratch_results = vec![Val::I64(0)];
    match &bound.call_plan {
        PreparedCallPlan::DirectScalar {
            func_name,
            output_slot,
            prim,
        } => {
            let func = runtime_func(&runtime, func_name)?;
            runtime.sync_enum_heap_ptr()?;
            func.call(&mut runtime.store, &input_args, &mut scratch_results)
                .map_err(|error| {
                    SimdError::new(format!("Wasm prepared execution failed: {error}"))
                })?;
            let value = scratch_results.first().ok_or_else(|| {
                SimdError::new("Wasm prepared scalar call did not return a value")
            })?;
            if let Type::Named(enum_name, args) = &bound.artifact.result_type {
                if is_wasm_enum_named_type(enum_name, args) {
                    let ptr = match value {
                        Val::I32(ptr) => *ptr,
                        other => {
                            return Err(SimdError::new(format!(
                                "Wasm prepared enum entry '{}' returned non-i32 handle: {:?}",
                                func_name, other
                            )));
                        }
                    };
                    let wasm_layout = lookup_specialized_wasm_enum_layout(
                        &bound.artifact.wasm_enum_layouts,
                        enum_name,
                        args,
                    )?;
                    let decoded = decode_wasm_enum_value_from_ptr(
                        &runtime.memory,
                        &runtime.store,
                        ptr,
                        enum_name,
                        &bound.artifact.enum_ctors,
                        &bound.artifact.wasm_enum_layouts,
                        wasm_layout,
                    )?;
                    bound.last_dynamic_result = Some(decoded);
                } else {
                    let scalar = wasmtime_to_scalar(value.clone(), *prim)?;
                    scalar_updates.push((*output_slot, scalar));
                }
            } else {
                let scalar = wasmtime_to_scalar(value.clone(), *prim)?;
                scalar_updates.push((*output_slot, scalar));
            }
        }
        PreparedCallPlan::DirectStarSeq { func_name, prim } => {
            let func = runtime_func(&runtime, func_name)?;
            let mut results = vec![Val::I32(0)];
            runtime.sync_enum_heap_ptr()?;
            func.call(&mut runtime.store, &input_args, &mut results)
                .map_err(|error| {
                    SimdError::new(format!("Wasm prepared execution failed: {error}"))
                })?;
            let value = results
                .first()
                .ok_or_else(|| SimdError::new("Wasm prepared T[*] call did not return a value"))?;
            let Val::I32(handle) = value else {
                return Err(SimdError::new(
                    "Wasm prepared T[*] call did not return an i32 handle",
                ));
            };
            let decoded =
                read_wasm_star_seq_from_handle(&runtime.memory, &runtime.store, *handle, *prim)?;
            bound.last_dynamic_result = Some(decoded);
        }
        PreparedCallPlan::DirectBulk {
            func_name,
            output_slot,
        } => {
            let mut args = input_args;
            let PreparedSlotState::OutputBulk { ptr, len, .. } = bound.slot_ref(*output_slot)?
            else {
                return Err(SimdError::new(
                    "prepared direct bulk output slot metadata mismatch",
                ));
            };
            args.push(Val::I32(i32::try_from(*ptr).map_err(|_| {
                SimdError::new("prepared output pointer does not fit in i32")
            })?));
            args.push(Val::I32(i32::try_from(*len).map_err(|_| {
                SimdError::new("prepared output length does not fit in i32")
            })?));
            let func = runtime_func(&runtime, func_name)?;
            runtime.sync_enum_heap_ptr()?;
            func.call(&mut runtime.store, &args, &mut [])
                .map_err(|error| {
                    SimdError::new(format!("Wasm prepared execution failed: {error}"))
                })?;
        }
        PreparedCallPlan::GroupedExport { func_name, outputs } => {
            let mut args = input_args;
            for output in outputs {
                match bound.slot_ref(output.output_slot)? {
                    PreparedSlotState::OutputScalar { ptr: Some(ptr), .. } => {
                        args.push(Val::I32(i32::try_from(*ptr).map_err(|_| {
                            SimdError::new("prepared scalar output pointer does not fit in i32")
                        })?));
                    }
                    PreparedSlotState::OutputBulk { ptr, len, .. } => {
                        args.push(Val::I32(i32::try_from(*ptr).map_err(|_| {
                            SimdError::new("prepared bulk output pointer does not fit in i32")
                        })?));
                        args.push(Val::I32(i32::try_from(*len).map_err(|_| {
                            SimdError::new("prepared bulk output length does not fit in i32")
                        })?));
                    }
                    _ => {
                        return Err(SimdError::new(
                            "prepared grouped export output slot metadata mismatch",
                        ));
                    }
                }
            }
            let func = runtime_func(&runtime, func_name)?;
            runtime.sync_enum_heap_ptr()?;
            func.call(&mut runtime.store, &args, &mut [])
                .map_err(|error| {
                    SimdError::new(format!("Wasm prepared execution failed: {error}"))
                })?;
        }
        PreparedCallPlan::LeafExports { leaves } => {
            for leaf in leaves {
                let mut args = input_args.clone();
                let mut results = match leaf.result {
                    WasmLeafResultAbi::Scalar { .. } => vec![Val::I64(0)],
                    WasmLeafResultAbi::Bulk { .. } => Vec::new(),
                    WasmLeafResultAbi::StarSeq { .. } => {
                        return Err(SimdError::new(
                            "prepared execution does not support T[*] outputs yet",
                        ));
                    }
                };
                if let PreparedSlotState::OutputBulk { ptr, len, .. } =
                    bound.slot_ref(leaf.output_slot)?
                {
                    args.push(Val::I32(i32::try_from(*ptr).map_err(|_| {
                        SimdError::new("prepared bulk output pointer does not fit in i32")
                    })?));
                    args.push(Val::I32(i32::try_from(*len).map_err(|_| {
                        SimdError::new("prepared bulk output length does not fit in i32")
                    })?));
                }
                let func = runtime_func(&runtime, &leaf.func_name)?;
                runtime.sync_enum_heap_ptr()?;
                func.call(&mut runtime.store, &args, &mut results)
                    .map_err(|error| {
                        SimdError::new(format!("Wasm prepared execution failed: {error}"))
                    })?;
                if let (WasmLeafResultAbi::Scalar { prim }, Some(result)) =
                    (&leaf.result, results.into_iter().next())
                {
                    let scalar = wasmtime_to_scalar(result, *prim)?;
                    scalar_updates.push((leaf.output_slot, scalar));
                }
            }
        }
    }
    drop(runtime);
    for (slot, value) in scalar_updates {
        if let PreparedSlotState::OutputScalar {
            value: slot_value, ..
        } = bound.slot_mut(slot)?
        {
            *slot_value = value;
        }
    }
    Ok(())
}

fn build_prepared_input_wasm_args(bound: &BoundPreparedRun) -> Result<Vec<Val>> {
    let mut args = Vec::<Val>::new();
    for (flat_index, abi) in bound.flat_param_abis.iter().enumerate() {
        match abi {
            WasmParamAbi::Scalar { prim } => {
                let slot = bound
                    .input_scalar_slot_by_flat
                    .get(&flat_index)
                    .ok_or_else(|| {
                        SimdError::new(format!(
                            "prepared scalar input slot for flat index {} is missing",
                            flat_index
                        ))
                    })?;
                let PreparedSlotState::InputScalar { value, .. } = bound.slot_ref(*slot)? else {
                    return Err(SimdError::new(
                        "prepared scalar input slot metadata mismatch",
                    ));
                };
                args.push(scalar_to_wasmtime(value, *prim)?);
            }
            WasmParamAbi::Bulk { .. } => {
                let slot = bound
                    .input_bulk_slot_by_flat
                    .get(&flat_index)
                    .ok_or_else(|| {
                        SimdError::new(format!(
                            "prepared bulk input slot for flat index {} is missing",
                            flat_index
                        ))
                    })?;
                let PreparedSlotState::InputBulk { ptr, len, .. } = bound.slot_ref(*slot)? else {
                    return Err(SimdError::new("prepared bulk input slot metadata mismatch"));
                };
                args.push(Val::I32(i32::try_from(*ptr).map_err(|_| {
                    SimdError::new("prepared bulk input pointer does not fit in i32")
                })?));
                args.push(Val::I32(i32::try_from(*len).map_err(|_| {
                    SimdError::new("prepared bulk input length does not fit in i32")
                })?));
            }
            WasmParamAbi::Record { .. } => {
                return Err(SimdError::new(
                    "flattened prepared input ABI should not contain records",
                ));
            }
        }
    }
    Ok(args)
}

fn read_prepared_result(bound: &BoundPreparedRun) -> Result<Value> {
    if matches!(bound.artifact.result, WasmResultAbi::StarSeq { .. })
        || matches!(
            &bound.artifact.result_type,
            Type::Named(enum_name, args) if is_wasm_enum_named_type(enum_name, args)
        )
    {
        return bound.last_dynamic_result.clone().ok_or_else(|| {
            SimdError::new("prepared dynamic result is not available before the run completes")
        });
    }
    let runtime = bound.runtime.borrow();
    let memory = runtime.memory.clone();
    let mut leaf_values = BTreeMap::<LeafPath, Value>::new();
    for slot in &bound.slots {
        match slot {
            PreparedSlotState::OutputScalar {
                leaf_path,
                ptr,
                value,
                meta,
            } => {
                let scalar = if let Some(ptr) = ptr {
                    read_scalar_from_memory(&memory, &runtime.store, *ptr, meta.prim)?
                } else {
                    value.clone()
                };
                leaf_values.insert(leaf_path.clone(), Value::Scalar(scalar));
            }
            PreparedSlotState::OutputBulk {
                leaf_path,
                ptr,
                len,
                meta,
            } => {
                leaf_values.insert(
                    leaf_path.clone(),
                    Value::Bulk(read_bulk_from_memory(
                        &memory,
                        &runtime.store,
                        *ptr,
                        *len,
                        meta.prim,
                        meta.shape.clone(),
                    )?),
                );
            }
            PreparedSlotState::InputScalar { .. } | PreparedSlotState::InputBulk { .. } => {}
        }
    }
    rebuild_wasm_value_from_abi(&bound.artifact.result, &leaf_values, &LeafPath::root())
}

fn read_output_bulk_i32(bound: &BoundPreparedRun, slot: usize, out: &mut [i32]) -> Result<()> {
    let (ptr, len, prim) = match bound.slot_ref(slot)? {
        PreparedSlotState::OutputBulk { ptr, len, meta, .. } => (*ptr, *len, meta.prim),
        _ => {
            return Err(SimdError::new(format!(
                "prepared slot {} is not an output bulk slot",
                slot
            )));
        }
    };
    if prim != Prim::I32 {
        return Err(SimdError::new(format!(
            "prepared slot {} expected {:?} output, requested i32",
            slot, prim
        )));
    }
    if out.len() != len {
        return Err(SimdError::new(format!(
            "prepared slot {} expected {} output lanes, found {}",
            slot,
            len,
            out.len()
        )));
    }
    let runtime = bound.runtime.borrow();
    let data = runtime.memory.data(&runtime.store);
    let byte_len = len * 4;
    let start = ptr;
    let end = start + byte_len;
    if cfg!(target_endian = "little") {
        let out_bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), byte_len) };
        out_bytes.copy_from_slice(&data[start..end]);
        return Ok(());
    }
    for (index, item) in out.iter_mut().enumerate() {
        let lane_start = ptr + (index * 4);
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&data[lane_start..lane_start + 4]);
        *item = i32::from_le_bytes(bytes);
    }
    Ok(())
}

fn read_output_bulk_i64(bound: &BoundPreparedRun, slot: usize, out: &mut [i64]) -> Result<()> {
    let (ptr, len, prim) = match bound.slot_ref(slot)? {
        PreparedSlotState::OutputBulk { ptr, len, meta, .. } => (*ptr, *len, meta.prim),
        _ => {
            return Err(SimdError::new(format!(
                "prepared slot {} is not an output bulk slot",
                slot
            )));
        }
    };
    if prim != Prim::I64 {
        return Err(SimdError::new(format!(
            "prepared slot {} expected {:?} output, requested i64",
            slot, prim
        )));
    }
    if out.len() != len {
        return Err(SimdError::new(format!(
            "prepared slot {} expected {} output lanes, found {}",
            slot,
            len,
            out.len()
        )));
    }
    let runtime = bound.runtime.borrow();
    let data = runtime.memory.data(&runtime.store);
    let byte_len = len * 8;
    let start = ptr;
    let end = start + byte_len;
    if cfg!(target_endian = "little") {
        let out_bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), byte_len) };
        out_bytes.copy_from_slice(&data[start..end]);
        return Ok(());
    }
    for (index, item) in out.iter_mut().enumerate() {
        let lane_start = ptr + (index * 8);
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&data[lane_start..lane_start + 8]);
        *item = i64::from_le_bytes(bytes);
    }
    Ok(())
}

fn read_output_bulk_f32(bound: &BoundPreparedRun, slot: usize, out: &mut [f32]) -> Result<()> {
    let (ptr, len, prim) = match bound.slot_ref(slot)? {
        PreparedSlotState::OutputBulk { ptr, len, meta, .. } => (*ptr, *len, meta.prim),
        _ => {
            return Err(SimdError::new(format!(
                "prepared slot {} is not an output bulk slot",
                slot
            )));
        }
    };
    if prim != Prim::F32 {
        return Err(SimdError::new(format!(
            "prepared slot {} expected {:?} output, requested f32",
            slot, prim
        )));
    }
    if out.len() != len {
        return Err(SimdError::new(format!(
            "prepared slot {} expected {} output lanes, found {}",
            slot,
            len,
            out.len()
        )));
    }
    let runtime = bound.runtime.borrow();
    let data = runtime.memory.data(&runtime.store);
    let byte_len = len * 4;
    let start = ptr;
    let end = start + byte_len;
    if cfg!(target_endian = "little") {
        let out_bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), byte_len) };
        out_bytes.copy_from_slice(&data[start..end]);
        return Ok(());
    }
    for (index, item) in out.iter_mut().enumerate() {
        let lane_start = ptr + (index * 4);
        let mut bytes = [0u8; 4];
        bytes.copy_from_slice(&data[lane_start..lane_start + 4]);
        *item = f32::from_bits(u32::from_le_bytes(bytes));
    }
    Ok(())
}

fn read_output_bulk_f64(bound: &BoundPreparedRun, slot: usize, out: &mut [f64]) -> Result<()> {
    let (ptr, len, prim) = match bound.slot_ref(slot)? {
        PreparedSlotState::OutputBulk { ptr, len, meta, .. } => (*ptr, *len, meta.prim),
        _ => {
            return Err(SimdError::new(format!(
                "prepared slot {} is not an output bulk slot",
                slot
            )));
        }
    };
    if prim != Prim::F64 {
        return Err(SimdError::new(format!(
            "prepared slot {} expected {:?} output, requested f64",
            slot, prim
        )));
    }
    if out.len() != len {
        return Err(SimdError::new(format!(
            "prepared slot {} expected {} output lanes, found {}",
            slot,
            len,
            out.len()
        )));
    }
    let runtime = bound.runtime.borrow();
    let data = runtime.memory.data(&runtime.store);
    let byte_len = len * 8;
    let start = ptr;
    let end = start + byte_len;
    if cfg!(target_endian = "little") {
        let out_bytes =
            unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), byte_len) };
        out_bytes.copy_from_slice(&data[start..end]);
        return Ok(());
    }
    for (index, item) in out.iter_mut().enumerate() {
        let lane_start = ptr + (index * 8);
        let mut bytes = [0u8; 8];
        bytes.copy_from_slice(&data[lane_start..lane_start + 8]);
        *item = f64::from_bits(u64::from_le_bytes(bytes));
    }
    Ok(())
}

impl WasmRuntime {
    fn begin_run(&mut self) {
        self.arena_cursor = self.arena_pinned_end;
    }

    fn alloc_pinned(&mut self, bytes: usize) -> Result<usize> {
        let pinned_floor = usize::try_from(ENUM_HEAP_START)
            .map_err(|_| SimdError::new("enum heap start did not fit in usize"))?;
        let ptr = align_to(self.arena_pinned_end.max(pinned_floor), 16);
        let end = ptr
            .checked_add(bytes)
            .ok_or_else(|| SimdError::new("Wasm arena pinned allocation overflowed"))?;
        ensure_memory_size(&self.memory, &mut self.store, end)?;
        self.arena_pinned_end = end;
        self.arena_cursor = self.arena_cursor.max(end);
        Ok(ptr)
    }

    fn sync_enum_heap_ptr(&mut self) -> Result<()> {
        let pinned_floor = usize::try_from(ENUM_HEAP_START)
            .map_err(|_| SimdError::new("enum heap start did not fit in usize"))?;
        let heap_start = align_to(self.arena_pinned_end.max(pinned_floor), 16);
        ensure_memory_size(
            &self.memory,
            &mut self.store,
            (ENUM_SAVE_STACK_PTR_ADDR as usize) + 4,
        )?;
        let heap_start_i32 = i32::try_from(heap_start)
            .map_err(|_| SimdError::new("enum heap pointer did not fit in i32"))?;
        let data = self.memory.data_mut(&mut self.store);
        data[(ENUM_HEAP_PTR_ADDR as usize)..((ENUM_HEAP_PTR_ADDR as usize) + 4)]
            .copy_from_slice(&heap_start_i32.to_le_bytes());
        data[(ENUM_SAVE_STACK_PTR_ADDR as usize)..((ENUM_SAVE_STACK_PTR_ADDR as usize) + 4)]
            .copy_from_slice(&ENUM_SAVE_STACK_START.to_le_bytes());
        self.arena_cursor = heap_start;
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct PendingSpecialization {
    base_name: String,
    output_name: String,
    type_subst: BTreeMap<String, Type>,
    known_fun_args: BTreeMap<usize, KnownFunctionArgSpec>,
    known_type_witness_args: BTreeMap<usize, Prim>,
    origin: String,
}

#[derive(Debug, Clone)]
struct SpecializedProgram {
    checked: CheckedProgram,
    origins: BTreeMap<String, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct KnownFunctionArgSpec {
    target_name: String,
    capture_types: Vec<Type>,
}

#[derive(Debug, Clone)]
struct KnownFunctionBinding {
    target_name: String,
    bound_args: Vec<TypedExpr>,
}

#[derive(Debug, Clone)]
struct ResolvedSpecializedCall {
    callee_name: String,
    args: Vec<TypedArg>,
    closure_info: Vec<ClosureBindingSummary>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ClosureBindingSummary {
    capture_count: usize,
    env_bytes: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FunctionValueClass {
    KnownFn,
    KnownLambda,
    EscapingUnknown,
}

#[derive(Debug, Clone)]
struct LambdaLiftState {
    next_id: usize,
    generated: BTreeMap<String, CheckedFunction>,
    origins: BTreeMap<String, String>,
    closure_info: BTreeMap<String, Vec<ClosureBindingSummary>>,
}

fn specialize_checked_program_for_main(
    checked: &CheckedProgram,
    main: &str,
) -> Result<SpecializedProgram> {
    let originals = checked
        .functions
        .iter()
        .map(|function| (function.name.clone(), function))
        .collect::<BTreeMap<_, _>>();
    let entry = originals
        .get(main)
        .copied()
        .ok_or_else(|| SimdError::new(format!("unknown entry function '{}'", main)))?;
    if type_contains_unresolved_vars(&entry.signature.ty) {
        return Err(SimdError::new(format!(
            "entry function '{}' must be monomorphic for Wasm lowering",
            main
        )));
    }
    let (entry_params, _) = entry.signature.ty.fun_parts();
    if entry_params
        .iter()
        .any(|param| matches!(param, Type::TypeToken(_)))
    {
        return Err(SimdError::new(format!(
            "entry function '{}' cannot expose Type witness parameters to Wasm runtime; specialize them at compile-time before main",
            main
        )));
    }

    let mut queue = VecDeque::<PendingSpecialization>::new();
    let mut scheduled_names = BTreeSet::<String>::new();
    let mut built = BTreeMap::<String, CheckedFunction>::new();
    let mut generic_instances = BTreeMap::<String, String>::new();
    let mut origins = BTreeMap::<String, String>::new();
    let mut lambda_lifts = LambdaLiftState {
        next_id: 0,
        generated: BTreeMap::new(),
        origins: BTreeMap::new(),
        closure_info: BTreeMap::new(),
    };

    enqueue_specialization(
        &mut queue,
        &mut scheduled_names,
        PendingSpecialization {
            base_name: main.to_string(),
            output_name: main.to_string(),
            type_subst: BTreeMap::new(),
            known_fun_args: BTreeMap::new(),
            known_type_witness_args: BTreeMap::new(),
            origin: format!("entry:{main}"),
        },
    );

    while let Some(pending) = queue.pop_front() {
        if built.contains_key(&pending.output_name) {
            continue;
        }
        origins
            .entry(pending.output_name.clone())
            .or_insert_with(|| pending.origin.clone());
        let specialized = specialize_function_instance(
            &pending,
            &originals,
            &mut lambda_lifts,
            &mut queue,
            &mut scheduled_names,
            &mut generic_instances,
        )?;
        built.insert(specialized.name.clone(), specialized);
    }

    for (name, function) in lambda_lifts.generated {
        built.insert(name, function);
    }
    for (name, origin) in lambda_lifts.origins {
        origins.entry(name).or_insert(origin);
    }
    for (name, summaries) in lambda_lifts.closure_info {
        let capture_count = summaries
            .iter()
            .map(|item| item.capture_count)
            .sum::<usize>();
        let env_bytes = summaries.iter().map(|item| item.env_bytes).sum::<usize>();
        let detail = format!("closure-converted captures={capture_count} env-bytes={env_bytes}");
        origins
            .entry(name)
            .and_modify(|existing| {
                existing.push_str(" | ");
                existing.push_str(&detail);
            })
            .or_insert(detail);
    }

    Ok(SpecializedProgram {
        checked: CheckedProgram {
            enum_names: checked.enum_names.clone(),
            enum_ctors: checked.enum_ctors.clone(),
            enum_layouts: checked.enum_layouts.clone(),
            functions: built.into_values().collect(),
        },
        origins,
    })
}

fn specialize_function_instance(
    pending: &PendingSpecialization,
    originals: &BTreeMap<String, &CheckedFunction>,
    lambda_lifts: &mut LambdaLiftState,
    queue: &mut VecDeque<PendingSpecialization>,
    scheduled_names: &mut BTreeSet<String>,
    generic_instances: &mut BTreeMap<String, String>,
) -> Result<CheckedFunction> {
    let original = originals.get(&pending.base_name).copied().ok_or_else(|| {
        SimdError::new(format!(
            "missing function '{}' during Wasm specialization",
            pending.base_name
        ))
    })?;
    let (base_param_types, base_result_ty) = original.signature.ty.fun_parts();
    let applied_param_types = base_param_types
        .iter()
        .map(|ty| apply_type_subst(ty, &pending.type_subst))
        .collect::<Vec<_>>();
    let applied_result_ty = apply_type_subst(&base_result_ty, &pending.type_subst);
    for (index, binding) in &pending.known_fun_args {
        let Some(param_ty) = applied_param_types.get(*index) else {
            return Err(SimdError::new(format!(
                "specialization for '{}' referenced missing function-arg index {}",
                pending.output_name, index
            )));
        };
        if !matches!(param_ty, Type::Fun(_, _)) {
            return Err(SimdError::new(format!(
                "specialization for '{}' tried to bind non-function param {} to '{}'",
                pending.output_name, index, binding.target_name
            )));
        }
    }
    for (index, witness_prim) in &pending.known_type_witness_args {
        let Some(param_ty) = applied_param_types.get(*index) else {
            return Err(SimdError::new(format!(
                "specialization for '{}' referenced missing type-witness arg index {}",
                pending.output_name, index
            )));
        };
        let Type::TypeToken(inner) = param_ty else {
            return Err(SimdError::new(format!(
                "specialization for '{}' tried to bind non-Type parameter {} to witness '{}'",
                pending.output_name,
                index,
                format_prim(*witness_prim)
            )));
        };
        let mut check_subst = BTreeMap::<String, Type>::new();
        collect_type_var_bindings(inner, &Type::Scalar(*witness_prim), &mut check_subst)?;
    }
    let mut kept_param_types = Vec::<Type>::new();
    for (index, ty) in applied_param_types.iter().enumerate() {
        if let Some(binding) = pending.known_fun_args.get(&index) {
            kept_param_types.extend(binding.capture_types.clone());
        } else if pending.known_type_witness_args.contains_key(&index) {
            // Type witnesses are compile-time dispatch values and are erased
            // from specialized runtime signatures.
        } else {
            kept_param_types.push(ty.clone());
        }
    }
    let specialized_signature = Signature {
        name: pending.output_name.clone(),
        ty: Type::Fun(kept_param_types, Box::new(applied_result_ty)),
        operator_instance: None,
        family_instance: None,
    };
    if type_contains_unresolved_vars(&specialized_signature.ty) {
        return Err(SimdError::new(format!(
            "specialized function '{}' still has unresolved polymorphic types",
            pending.output_name
        )));
    }
    let clauses = original
        .clauses
        .iter()
        .map(|clause| {
            let mut fun_subst = BTreeMap::<String, KnownFunctionBinding>::new();
            let mut scope_types = BTreeMap::<String, Type>::new();
            let mut patterns = Vec::<TypedPattern>::new();
            let mut clause_matches_witnesses = true;
            for (index, pattern) in clause.patterns.iter().enumerate() {
                let typed_pattern = TypedPattern {
                    pattern: pattern.pattern.clone(),
                    ty: apply_type_subst(&pattern.ty, &pending.type_subst),
                };
                if let Some(binding) = pending.known_fun_args.get(&index) {
                    let capture_locals = binding
                        .capture_types
                        .iter()
                        .enumerate()
                        .map(|(capture_index, capture_ty)| {
                            let local = format!("__fn{}_cap{}", index, capture_index);
                            patterns.push(TypedPattern {
                                pattern: Pattern::Name(local.clone()),
                                ty: capture_ty.clone(),
                            });
                            scope_types.insert(local.clone(), capture_ty.clone());
                            TypedExpr {
                                ty: capture_ty.clone(),
                                kind: TypedExprKind::Local(local),
                            }
                        })
                        .collect::<Vec<_>>();
                    match &typed_pattern.pattern {
                        Pattern::Name(name) => {
                            fun_subst.insert(
                                name.clone(),
                                KnownFunctionBinding {
                                    target_name: binding.target_name.clone(),
                                    bound_args: capture_locals,
                                },
                            );
                        }
                        Pattern::Wildcard => {}
                        Pattern::Int(_)
                        | Pattern::Float(_)
                        | Pattern::Char(_)
                        | Pattern::Bool(_)
                        | Pattern::Type(_)
                        | Pattern::Tuple(_)
                        | Pattern::Ctor(_, _)
                        | Pattern::Slice { .. } => {
                            return Err(SimdError::new(format!(
                                "function-typed parameter {} in '{}' used a literal pattern",
                                index, pending.base_name
                            )));
                        }
                    }
                } else if let Some(witness_prim) = pending.known_type_witness_args.get(&index) {
                    match &typed_pattern.pattern {
                        Pattern::Type(pattern_prim) => {
                            if pattern_prim != witness_prim {
                                clause_matches_witnesses = false;
                                break;
                            }
                        }
                        Pattern::Wildcard | Pattern::Name(_) => {}
                        Pattern::Int(_)
                        | Pattern::Float(_)
                        | Pattern::Char(_)
                        | Pattern::Bool(_)
                        | Pattern::Tuple(_)
                        | Pattern::Ctor(_, _)
                        | Pattern::Slice { .. } => {
                            return Err(SimdError::new(format!(
                                "type witness parameter {} in '{}' used a non-Type literal pattern",
                                index, pending.base_name
                            )));
                        }
                    }
                    // Do not push this witness pattern: it was consumed at compile-time.
                } else {
                    if let Pattern::Name(name) = &typed_pattern.pattern {
                        scope_types.insert(name.clone(), typed_pattern.ty.clone());
                    }
                    patterns.push(typed_pattern);
                }
            }
            if !clause_matches_witnesses {
                return Ok(None);
            }
            Ok(Some(TypedClause {
                patterns,
                body: specialize_expr(
                    &clause.body,
                    &pending.type_subst,
                    &fun_subst,
                    &scope_types,
                    &pending.output_name,
                    originals,
                    lambda_lifts,
                    queue,
                    scheduled_names,
                    generic_instances,
                )?,
            }))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    if clauses.is_empty() {
        return Err(SimdError::new(format!(
            "specialization removed all clauses for '{}' with the selected type witness arguments",
            pending.output_name
        )));
    }

    Ok(CheckedFunction {
        name: pending.output_name.clone(),
        signature: specialized_signature,
        clauses,
        pointwise: original.pointwise,
        tailrec: original.tailrec.clone(),
    })
}

fn specialize_expr(
    expr: &TypedExpr,
    type_subst: &BTreeMap<String, Type>,
    fun_subst: &BTreeMap<String, KnownFunctionBinding>,
    scope_types: &BTreeMap<String, Type>,
    current_function: &str,
    originals: &BTreeMap<String, &CheckedFunction>,
    lambda_lifts: &mut LambdaLiftState,
    queue: &mut VecDeque<PendingSpecialization>,
    scheduled_names: &mut BTreeSet<String>,
    generic_instances: &mut BTreeMap<String, String>,
) -> Result<TypedExpr> {
    let ty = apply_type_subst(&expr.ty, type_subst);
    let kind = match &expr.kind {
        TypedExprKind::Local(name) => {
            if let Some(binding) = fun_subst.get(name) {
                build_known_binding_expr(binding, &ty)?.kind
            } else {
                TypedExprKind::Local(name.clone())
            }
        }
        TypedExprKind::FunctionRef { name } => TypedExprKind::FunctionRef {
            name: resolve_function_ref_name_for_type(
                name,
                &ty,
                originals,
                lambda_lifts,
                queue,
                scheduled_names,
                generic_instances,
                &format!("fnref:{current_function}"),
            )?,
        },
        TypedExprKind::ConstructorRef { name } => {
            TypedExprKind::ConstructorRef { name: name.clone() }
        }
        TypedExprKind::Int(value, prim) => match &ty {
            Type::Scalar(target_prim) if target_prim.is_int() => {
                TypedExprKind::Int(*value, *target_prim)
            }
            Type::Scalar(target_prim) if target_prim.is_float() => {
                TypedExprKind::Float(*value as f64, *target_prim)
            }
            _ => TypedExprKind::Int(*value, *prim),
        },
        TypedExprKind::Float(value, prim) => match &ty {
            Type::Scalar(target_prim) if target_prim.is_float() => {
                TypedExprKind::Float(*value, *target_prim)
            }
            _ => TypedExprKind::Float(*value, *prim),
        },
        TypedExprKind::Bool(value) => TypedExprKind::Bool(*value),
        TypedExprKind::Char(value) => TypedExprKind::Char(*value),
        TypedExprKind::TypeToken(prim) => TypedExprKind::TypeToken(*prim),
        TypedExprKind::String(value) => TypedExprKind::String(value.clone()),
        TypedExprKind::Lambda { param, body } => TypedExprKind::Lambda {
            param: param.clone(),
            body: {
                let mut nested_fun_subst = fun_subst.clone();
                nested_fun_subst.remove(param);
                Box::new(specialize_expr(
                    body,
                    type_subst,
                    &nested_fun_subst,
                    scope_types,
                    current_function,
                    originals,
                    lambda_lifts,
                    queue,
                    scheduled_names,
                    generic_instances,
                )?)
            },
        },
        TypedExprKind::Let { bindings, body } => {
            let mut next_fun_subst = fun_subst.clone();
            let mut next_scope_types = scope_types.clone();
            let rewritten = bindings
                .iter()
                .map(|binding| {
                    let rewritten_expr = specialize_expr(
                        &binding.expr,
                        type_subst,
                        &next_fun_subst,
                        &next_scope_types,
                        current_function,
                        originals,
                        lambda_lifts,
                        queue,
                        scheduled_names,
                        generic_instances,
                    )?;
                    if binding.name != "_" {
                        if matches!(rewritten_expr.ty, Type::Fun(_, _)) {
                            if let Some(template) = extract_known_function_binding(&rewritten_expr)
                            {
                                next_fun_subst.insert(binding.name.clone(), template);
                            } else {
                                next_fun_subst.remove(&binding.name);
                            }
                        } else {
                            next_fun_subst.remove(&binding.name);
                        }
                        next_scope_types.insert(binding.name.clone(), rewritten_expr.ty.clone());
                    }
                    Ok(TypedLetBinding {
                        name: binding.name.clone(),
                        expr: rewritten_expr,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            TypedExprKind::Let {
                bindings: rewritten,
                body: Box::new(specialize_expr(
                    body,
                    type_subst,
                    &next_fun_subst,
                    &next_scope_types,
                    current_function,
                    originals,
                    lambda_lifts,
                    queue,
                    scheduled_names,
                    generic_instances,
                )?),
            }
        }
        TypedExprKind::Record(fields) => TypedExprKind::Record(
            fields
                .iter()
                .map(|(name, value)| {
                    Ok((
                        name.clone(),
                        specialize_expr(
                            value,
                            type_subst,
                            fun_subst,
                            scope_types,
                            current_function,
                            originals,
                            lambda_lifts,
                            queue,
                            scheduled_names,
                            generic_instances,
                        )?,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?,
        ),
        TypedExprKind::Seq(items) => TypedExprKind::Seq(
            items
                .iter()
                .map(|item| {
                    specialize_expr(
                        item,
                        type_subst,
                        fun_subst,
                        scope_types,
                        current_function,
                        originals,
                        lambda_lifts,
                        queue,
                        scheduled_names,
                        generic_instances,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        TypedExprKind::SeqSplice { prefix, tail } => TypedExprKind::SeqSplice {
            prefix: prefix
                .iter()
                .map(|item| {
                    specialize_expr(
                        item,
                        type_subst,
                        fun_subst,
                        scope_types,
                        current_function,
                        originals,
                        lambda_lifts,
                        queue,
                        scheduled_names,
                        generic_instances,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
            tail: Box::new(specialize_expr(
                tail,
                type_subst,
                fun_subst,
                scope_types,
                current_function,
                originals,
                lambda_lifts,
                queue,
                scheduled_names,
                generic_instances,
            )?),
        },
        TypedExprKind::Tuple(items) => TypedExprKind::Tuple(
            items
                .iter()
                .map(|item| {
                    specialize_expr(
                        item,
                        type_subst,
                        fun_subst,
                        scope_types,
                        current_function,
                        originals,
                        lambda_lifts,
                        queue,
                        scheduled_names,
                        generic_instances,
                    )
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        TypedExprKind::Project { base, field } => TypedExprKind::Project {
            base: Box::new(specialize_expr(
                base,
                type_subst,
                fun_subst,
                scope_types,
                current_function,
                originals,
                lambda_lifts,
                queue,
                scheduled_names,
                generic_instances,
            )?),
            field: field.clone(),
        },
        TypedExprKind::TupleProject { base, index } => TypedExprKind::TupleProject {
            base: Box::new(specialize_expr(
                base,
                type_subst,
                fun_subst,
                scope_types,
                current_function,
                originals,
                lambda_lifts,
                queue,
                scheduled_names,
                generic_instances,
            )?),
            index: *index,
        },
        TypedExprKind::Index {
            base,
            index,
            checked,
        } => TypedExprKind::Index {
            base: Box::new(specialize_expr(
                base,
                type_subst,
                fun_subst,
                scope_types,
                current_function,
                originals,
                lambda_lifts,
                queue,
                scheduled_names,
                generic_instances,
            )?),
            index: Box::new(specialize_expr(
                index,
                type_subst,
                fun_subst,
                scope_types,
                current_function,
                originals,
                lambda_lifts,
                queue,
                scheduled_names,
                generic_instances,
            )?),
            checked: *checked,
        },
        TypedExprKind::RecordUpdate { base, fields } => TypedExprKind::RecordUpdate {
            base: Box::new(specialize_expr(
                base,
                type_subst,
                fun_subst,
                scope_types,
                current_function,
                originals,
                lambda_lifts,
                queue,
                scheduled_names,
                generic_instances,
            )?),
            fields: fields
                .iter()
                .map(|(name, value)| {
                    Ok((
                        name.clone(),
                        specialize_expr(
                            value,
                            type_subst,
                            fun_subst,
                            scope_types,
                            current_function,
                            originals,
                            lambda_lifts,
                            queue,
                            scheduled_names,
                            generic_instances,
                        )?,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?,
        },
        TypedExprKind::Call {
            callee,
            args,
            lifted_shape,
        } => {
            let args = args
                .iter()
                .map(|arg| {
                    Ok(TypedArg {
                        mode: arg.mode,
                        expr: Box::new(specialize_expr(
                            &arg.expr,
                            type_subst,
                            fun_subst,
                            scope_types,
                            current_function,
                            originals,
                            lambda_lifts,
                            queue,
                            scheduled_names,
                            generic_instances,
                        )?),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let callee = match callee {
                Callee::Prim(op) => Callee::Prim(*op),
                Callee::Builtin(builtin) => Callee::Builtin(builtin.clone()),
                Callee::Function(name) => {
                    if let Some(inlined) = try_inline_lambda_result_call(
                        name,
                        &args,
                        &ty,
                        originals,
                        current_function,
                        scope_types,
                        lambda_lifts,
                        queue,
                        scheduled_names,
                        generic_instances,
                    )? {
                        return Ok(inlined);
                    }
                    let resolved = resolve_specialized_callee_call(
                        name,
                        &args,
                        originals,
                        lambda_lifts,
                        scope_types,
                        queue,
                        scheduled_names,
                        generic_instances,
                        &format!("call:{current_function}->{name}"),
                    )?;
                    if !resolved.closure_info.is_empty() {
                        lambda_lifts
                            .closure_info
                            .entry(current_function.to_string())
                            .or_default()
                            .extend(resolved.closure_info.clone());
                    }
                    return Ok(TypedExpr {
                        ty,
                        kind: TypedExprKind::Call {
                            callee: Callee::Function(resolved.callee_name),
                            args: resolved.args,
                            lifted_shape: lifted_shape.clone(),
                        },
                    });
                }
            };
            TypedExprKind::Call {
                callee,
                args,
                lifted_shape: lifted_shape.clone(),
            }
        }
        TypedExprKind::Apply { callee, arg } => TypedExprKind::Apply {
            callee: Box::new(specialize_expr(
                callee,
                type_subst,
                fun_subst,
                scope_types,
                current_function,
                originals,
                lambda_lifts,
                queue,
                scheduled_names,
                generic_instances,
            )?),
            arg: Box::new(specialize_expr(
                arg,
                type_subst,
                fun_subst,
                scope_types,
                current_function,
                originals,
                lambda_lifts,
                queue,
                scheduled_names,
                generic_instances,
            )?),
        },
    };
    Ok(TypedExpr { ty, kind })
}

fn extract_known_function_binding(expr: &TypedExpr) -> Option<KnownFunctionBinding> {
    match &expr.kind {
        TypedExprKind::FunctionRef { name } => Some(KnownFunctionBinding {
            target_name: name.clone(),
            bound_args: Vec::new(),
        }),
        TypedExprKind::Apply { callee, arg } => {
            let mut binding = extract_known_function_binding(callee)?;
            binding.bound_args.push((**arg).clone());
            Some(binding)
        }
        TypedExprKind::Local(_)
        | TypedExprKind::ConstructorRef { .. }
        | TypedExprKind::Lambda { .. }
        | TypedExprKind::Let { .. }
        | TypedExprKind::Seq(_)
        | TypedExprKind::SeqSplice { .. }
        | TypedExprKind::Tuple(_)
        | TypedExprKind::Record(_)
        | TypedExprKind::Project { .. }
        | TypedExprKind::TupleProject { .. }
        | TypedExprKind::Index { .. }
        | TypedExprKind::RecordUpdate { .. }
        | TypedExprKind::Call { .. }
        | TypedExprKind::Int(_, _)
        | TypedExprKind::Float(_, _)
        | TypedExprKind::Bool(_)
        | TypedExprKind::Char(_)
        | TypedExprKind::String(_)
        | TypedExprKind::TypeToken(_) => None,
    }
}

fn build_known_binding_expr(
    binding: &KnownFunctionBinding,
    expected_ty: &Type,
) -> Result<TypedExpr> {
    let (remaining_args, remaining_ret) = match expected_ty {
        Type::Fun(args, ret) => (args.clone(), ret.clone()),
        other => {
            return Err(SimdError::new(format!(
                "known function binding for '{}' requires function type, found {:?}",
                binding.target_name, other
            )));
        }
    };
    let mut base_args = binding
        .bound_args
        .iter()
        .map(|arg| arg.ty.clone())
        .collect::<Vec<_>>();
    base_args.extend(remaining_args.clone());
    let mut expr = TypedExpr {
        ty: Type::Fun(base_args, remaining_ret.clone()),
        kind: TypedExprKind::FunctionRef {
            name: binding.target_name.clone(),
        },
    };
    for arg in &binding.bound_args {
        let next_ty = match &expr.ty {
            Type::Fun(args, ret) if !args.is_empty() => {
                if arg.ty != args[0] {
                    return Err(SimdError::new(format!(
                        "known function binding argument type mismatch for '{}': {:?} vs {:?}",
                        binding.target_name, arg.ty, args[0]
                    )));
                }
                if args.len() == 1 {
                    ret.as_ref().clone()
                } else {
                    Type::Fun(args[1..].to_vec(), ret.clone())
                }
            }
            other => {
                return Err(SimdError::new(format!(
                    "known function binding for '{}' over-applied non-function {:?}",
                    binding.target_name, other
                )));
            }
        };
        expr = TypedExpr {
            ty: next_ty.clone(),
            kind: TypedExprKind::Apply {
                callee: Box::new(expr),
                arg: Box::new(arg.clone()),
            },
        };
    }
    if expr.ty != *expected_ty {
        return Err(SimdError::new(format!(
            "known function binding for '{}' produced {:?}, expected {:?}",
            binding.target_name, expr.ty, expected_ty
        )));
    }
    Ok(expr)
}

fn try_inline_lambda_result_call(
    callee_name: &str,
    args: &[TypedArg],
    call_ty: &Type,
    originals: &BTreeMap<String, &CheckedFunction>,
    current_function: &str,
    caller_scope_types: &BTreeMap<String, Type>,
    lambda_lifts: &mut LambdaLiftState,
    queue: &mut VecDeque<PendingSpecialization>,
    scheduled_names: &mut BTreeSet<String>,
    generic_instances: &mut BTreeMap<String, String>,
) -> Result<Option<TypedExpr>> {
    if !matches!(call_ty, Type::Fun(_, _)) {
        return Ok(None);
    }
    if callee_name == current_function {
        return Ok(None);
    }
    if args.iter().any(|arg| arg.mode != AccessKind::Same) {
        return Ok(None);
    }
    let Some(callee) = originals.get(callee_name).copied() else {
        return Ok(None);
    };
    if callee.clauses.len() != 1 {
        return Ok(None);
    }
    let clause = &callee.clauses[0];
    if clause.patterns.len() != args.len() {
        return Ok(None);
    }
    if clause.patterns.iter().any(|pattern| {
        matches!(
            pattern.pattern,
            Pattern::Int(_) | Pattern::Float(_) | Pattern::Char(_) | Pattern::Slice { .. }
        )
    }) {
        return Ok(None);
    }
    let (param_types, _) = callee.signature.ty.fun_parts();
    if param_types.len() != args.len() {
        return Ok(None);
    }
    let mut callee_subst = BTreeMap::<String, Type>::new();
    for (param_ty, arg) in param_types.iter().zip(args.iter()) {
        let actual_ty = specialization_arg_type(arg)?;
        collect_type_var_bindings(param_ty, &actual_ty, &mut callee_subst)?;
    }
    let instantiated_param_types = param_types
        .iter()
        .map(|ty| apply_type_subst(ty, &callee_subst))
        .collect::<Vec<_>>();
    let mut callee_fun_subst = BTreeMap::<String, KnownFunctionBinding>::new();
    let mut callee_scope_types = BTreeMap::<String, Type>::new();
    let mut let_bindings = Vec::<TypedLetBinding>::new();
    for (index, pattern) in clause.patterns.iter().enumerate() {
        let pattern_ty = apply_type_subst(&pattern.ty, &callee_subst);
        let arg_expr = args[index].expr.as_ref().clone();
        match &pattern.pattern {
            Pattern::Wildcard => {}
            Pattern::Name(name) => {
                callee_scope_types.insert(name.clone(), pattern_ty.clone());
                if matches!(instantiated_param_types[index], Type::Fun(_, _)) {
                    let (binding, summary) = extract_specializable_function_value(
                        &arg_expr,
                        originals,
                        lambda_lifts,
                        caller_scope_types,
                        queue,
                        scheduled_names,
                        generic_instances,
                        &format!("inline:{current_function}->{callee_name}:arg{index}"),
                    )?;
                    callee_fun_subst.insert(name.clone(), binding);
                    lambda_lifts
                        .closure_info
                        .entry(current_function.to_string())
                        .or_default()
                        .push(summary);
                } else {
                    let_bindings.push(TypedLetBinding {
                        name: name.clone(),
                        expr: arg_expr,
                    });
                }
            }
            Pattern::Int(_)
            | Pattern::Float(_)
            | Pattern::Char(_)
            | Pattern::Bool(_)
            | Pattern::Type(_)
            | Pattern::Tuple(_)
            | Pattern::Ctor(_, _)
            | Pattern::Slice { .. } => {
                return Ok(None);
            }
        }
    }
    let rewritten_body = specialize_expr(
        &clause.body,
        &callee_subst,
        &callee_fun_subst,
        &callee_scope_types,
        current_function,
        originals,
        lambda_lifts,
        queue,
        scheduled_names,
        generic_instances,
    )?;
    if let_bindings.is_empty() {
        return Ok(Some(rewritten_body));
    }
    Ok(Some(TypedExpr {
        ty: call_ty.clone(),
        kind: TypedExprKind::Let {
            bindings: let_bindings,
            body: Box::new(rewritten_body),
        },
    }))
}

fn resolve_specialized_callee_call(
    callee_name: &str,
    args: &[TypedArg],
    originals: &BTreeMap<String, &CheckedFunction>,
    lambda_lifts: &mut LambdaLiftState,
    scope_types: &BTreeMap<String, Type>,
    queue: &mut VecDeque<PendingSpecialization>,
    scheduled_names: &mut BTreeSet<String>,
    generic_instances: &mut BTreeMap<String, String>,
    origin: &str,
) -> Result<ResolvedSpecializedCall> {
    let Some(callee) = originals.get(callee_name).copied() else {
        return Ok(ResolvedSpecializedCall {
            callee_name: callee_name.to_string(),
            args: args.to_vec(),
            closure_info: Vec::new(),
        });
    };
    let (param_types, _) = callee.signature.ty.fun_parts();
    if args.len() != param_types.len() {
        return Err(SimdError::new(format!(
            "specialization expected {} arguments for '{}', found {}",
            param_types.len(),
            callee_name,
            args.len()
        )));
    }
    let mut callee_subst = BTreeMap::<String, Type>::new();
    for (param, arg) in param_types.iter().zip(args) {
        let actual_ty = specialization_arg_type(arg)?;
        collect_type_var_bindings(param, &actual_ty, &mut callee_subst)?;
    }
    let instantiated_params = param_types
        .iter()
        .map(|ty| apply_type_subst(ty, &callee_subst))
        .collect::<Vec<_>>();
    let mut known_fun_args = BTreeMap::<usize, KnownFunctionArgSpec>::new();
    let mut known_type_witness_args = BTreeMap::<usize, Prim>::new();
    let mut retained_args = Vec::<TypedArg>::new();
    let mut closure_info = Vec::<ClosureBindingSummary>::new();
    for (index, (param_ty, arg)) in instantiated_params.iter().zip(args.iter()).enumerate() {
        if matches!(param_ty, Type::Fun(_, _)) {
            let (binding, summary) = extract_specializable_function_value(
                &arg.expr,
                originals,
                lambda_lifts,
                scope_types,
                queue,
                scheduled_names,
                generic_instances,
                &format!("{origin}:arg{index}"),
            )?;
            known_fun_args.insert(
                index,
                KnownFunctionArgSpec {
                    target_name: binding.target_name.clone(),
                    capture_types: binding
                        .bound_args
                        .iter()
                        .map(|expr| expr.ty.clone())
                        .collect(),
                },
            );
            for bound_arg in binding.bound_args {
                retained_args.push(TypedArg {
                    mode: AccessKind::Same,
                    expr: Box::new(bound_arg),
                });
            }
            closure_info.push(summary);
            continue;
        }
        if matches!(param_ty, Type::TypeToken(_)) {
            let witness_prim = match &arg.expr.kind {
                TypedExprKind::TypeToken(prim) => *prim,
                _ => {
                    return Err(SimdError::new(format!(
                        "Wasm specialization requires type witness argument {} in call to '{}' to be a literal primitive token",
                        index, callee_name
                    )));
                }
            };
            known_type_witness_args.insert(index, witness_prim);
            continue;
        }
        retained_args.push(arg.clone());
    }
    let output_name = resolve_specialized_function_name(
        callee_name,
        &callee_subst,
        &known_fun_args,
        &known_type_witness_args,
        originals,
        queue,
        scheduled_names,
        generic_instances,
        origin,
    )?;
    Ok(ResolvedSpecializedCall {
        callee_name: output_name,
        args: retained_args,
        closure_info,
    })
}

fn extract_specializable_function_value(
    expr: &TypedExpr,
    originals: &BTreeMap<String, &CheckedFunction>,
    lambda_lifts: &mut LambdaLiftState,
    scope_types: &BTreeMap<String, Type>,
    queue: &mut VecDeque<PendingSpecialization>,
    scheduled_names: &mut BTreeSet<String>,
    generic_instances: &mut BTreeMap<String, String>,
    origin: &str,
) -> Result<(KnownFunctionBinding, ClosureBindingSummary)> {
    if let Some(binding) = extract_known_function_binding(expr) {
        let target_name = resolve_function_ref_name_for_type(
            &binding.target_name,
            &expr.ty,
            originals,
            lambda_lifts,
            queue,
            scheduled_names,
            generic_instances,
            origin,
        )?;
        let summary = ClosureBindingSummary {
            capture_count: binding.bound_args.len(),
            env_bytes: binding
                .bound_args
                .iter()
                .map(|arg| estimate_closure_env_bytes(&arg.ty))
                .sum::<Result<usize>>()?,
        };
        return Ok((
            KnownFunctionBinding {
                target_name,
                bound_args: binding.bound_args,
            },
            summary,
        ));
    }
    if let TypedExprKind::Lambda { param, body } = &expr.kind {
        let (target_name, bound_args, summary) = lift_lambda_for_specialization(
            param,
            body,
            &expr.ty,
            scope_types,
            lambda_lifts,
            origin,
        )?;
        return Ok((
            KnownFunctionBinding {
                target_name,
                bound_args,
            },
            summary,
        ));
    }
    let class = classify_function_value_expr_for_backend(expr);
    Err(SimdError::new(format!(
        "cannot specialize higher-order value at '{}'; expected known function or liftable lambda, found {:?}",
        origin, class
    )))
}

fn lift_lambda_for_specialization(
    param: &str,
    body: &TypedExpr,
    lambda_ty: &Type,
    scope_types: &BTreeMap<String, Type>,
    lambda_lifts: &mut LambdaLiftState,
    origin: &str,
) -> Result<(String, Vec<TypedExpr>, ClosureBindingSummary)> {
    let Type::Fun(args, ret) = lambda_ty else {
        return Err(SimdError::new("lambda lift expected a function type"));
    };
    if args.len() != 1 {
        return Err(SimdError::new(
            "lambda lift currently supports unary lambdas only",
        ));
    }
    let param_ty = args[0].clone();
    if matches!(param_ty, Type::Fun(_, _)) {
        return Err(SimdError::new(
            "lambda lift does not support function-typed lambda parameters yet",
        ));
    }
    let in_scope = scope_types.keys().cloned().collect::<BTreeSet<_>>();
    let captures = free_lambda_capture_names(body, param, &in_scope);
    let mut capture_map = BTreeMap::<String, String>::new();
    let mut bound_args = Vec::<TypedExpr>::new();
    let mut capture_types = Vec::<Type>::new();
    for (index, capture) in captures.iter().enumerate() {
        let capture_ty = scope_types.get(capture).cloned().ok_or_else(|| {
            SimdError::new(format!(
                "lambda lift capture '{}' is missing from scope",
                capture
            ))
        })?;
        if matches!(capture_ty, Type::Fun(_, _)) {
            return Err(SimdError::new(format!(
                "lambda lift does not support function capture '{}' yet",
                capture
            )));
        }
        let alias = format!("__cap{index}");
        capture_map.insert(capture.clone(), alias.clone());
        capture_types.push(capture_ty.clone());
        bound_args.push(TypedExpr {
            ty: capture_ty,
            kind: TypedExprKind::Local(capture.clone()),
        });
    }
    let renamed_body = rename_free_locals(body, &capture_map, &BTreeSet::new())?;
    let lift_id = lambda_lifts.next_id;
    lambda_lifts.next_id += 1;
    let lift_name = format!("__lambda_lift${lift_id}");
    let mut patterns = Vec::<TypedPattern>::new();
    for (index, capture_ty) in capture_types.iter().enumerate() {
        patterns.push(TypedPattern {
            pattern: Pattern::Name(format!("__cap{index}")),
            ty: capture_ty.clone(),
        });
    }
    patterns.push(TypedPattern {
        pattern: Pattern::Name(param.to_string()),
        ty: param_ty.clone(),
    });
    let mut signature_params = capture_types.clone();
    signature_params.push(param_ty);
    let env_bytes = capture_types
        .iter()
        .map(estimate_closure_env_bytes)
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .sum::<usize>();
    lambda_lifts.generated.insert(
        lift_name.clone(),
        CheckedFunction {
            name: lift_name.clone(),
            signature: Signature {
                name: lift_name.clone(),
                ty: Type::Fun(signature_params, ret.clone()),
                operator_instance: None,
                family_instance: None,
            },
            clauses: vec![TypedClause {
                patterns,
                body: renamed_body,
            }],
            pointwise: true,
            tailrec: TailRecInfo {
                recursive: false,
                loop_lowerable: false,
            },
        },
    );
    lambda_lifts
        .origins
        .insert(lift_name.clone(), format!("lambda-lift:{origin}"));
    let summary = ClosureBindingSummary {
        capture_count: capture_types.len(),
        env_bytes,
    };
    lambda_lifts
        .closure_info
        .entry(lift_name.clone())
        .or_default()
        .push(summary.clone());
    Ok((lift_name, bound_args, summary))
}

fn estimate_closure_env_bytes(ty: &Type) -> Result<usize> {
    match ty {
        Type::Scalar(prim) => Ok(byte_width(*prim) as usize),
        Type::Bulk(_, _) => Ok(8),
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _) => Ok(4),
        Type::Index(_) => Ok(8),
        Type::TypeToken(_) => Err(SimdError::new(
            "closure env does not support type witness captures",
        )),
        Type::Tuple(items) => items
            .iter()
            .map(estimate_closure_env_bytes)
            .collect::<Result<Vec<_>>>()
            .map(|items| items.into_iter().sum()),
        Type::Record(fields) => fields
            .values()
            .map(estimate_closure_env_bytes)
            .collect::<Result<Vec<_>>>()
            .map(|items| items.into_iter().sum()),
        Type::Named(_, _) | Type::Var(_) | Type::Infer(_) => Err(SimdError::new(
            "closure env size cannot be computed for unresolved polymorphic capture",
        )),
        Type::Fun(_, _) => Err(SimdError::new(
            "closure env does not support function-typed captures",
        )),
    }
}

fn resolve_function_ref_name_for_type(
    function_name: &str,
    concrete_ty: &Type,
    originals: &BTreeMap<String, &CheckedFunction>,
    _lambda_lifts: &mut LambdaLiftState,
    queue: &mut VecDeque<PendingSpecialization>,
    scheduled_names: &mut BTreeSet<String>,
    generic_instances: &mut BTreeMap<String, String>,
    origin: &str,
) -> Result<String> {
    let Some(function) = originals.get(function_name).copied() else {
        return Ok(function_name.to_string());
    };
    let mut subst = BTreeMap::<String, Type>::new();
    collect_type_var_bindings(&function.signature.ty, concrete_ty, &mut subst)?;
    resolve_specialized_function_name(
        function_name,
        &subst,
        &BTreeMap::new(),
        &BTreeMap::new(),
        originals,
        queue,
        scheduled_names,
        generic_instances,
        origin,
    )
}

fn resolve_specialized_function_name(
    base_name: &str,
    type_subst: &BTreeMap<String, Type>,
    known_fun_args: &BTreeMap<usize, KnownFunctionArgSpec>,
    known_type_witness_args: &BTreeMap<usize, Prim>,
    originals: &BTreeMap<String, &CheckedFunction>,
    queue: &mut VecDeque<PendingSpecialization>,
    scheduled_names: &mut BTreeSet<String>,
    generic_instances: &mut BTreeMap<String, String>,
    origin: &str,
) -> Result<String> {
    let Some(original) = originals.get(base_name).copied() else {
        return Ok(base_name.to_string());
    };
    let (base_params, base_result) = original.signature.ty.fun_parts();
    let instantiated_params = base_params
        .iter()
        .map(|ty| apply_type_subst(ty, type_subst))
        .collect::<Vec<_>>();
    let instantiated_result = apply_type_subst(&base_result, type_subst);
    for (index, _) in known_fun_args {
        let Some(param_ty) = instantiated_params.get(*index) else {
            return Err(SimdError::new(format!(
                "higher-order specialization for '{}' referenced missing arg index {}",
                base_name, index
            )));
        };
        if !matches!(param_ty, Type::Fun(_, _)) {
            return Err(SimdError::new(format!(
                "higher-order specialization for '{}' tried to bind non-function arg {}",
                base_name, index
            )));
        }
    }
    for (index, prim) in known_type_witness_args {
        let Some(param_ty) = instantiated_params.get(*index) else {
            return Err(SimdError::new(format!(
                "type-witness specialization for '{}' referenced missing arg index {}",
                base_name, index
            )));
        };
        let Type::TypeToken(inner) = param_ty else {
            return Err(SimdError::new(format!(
                "type-witness specialization for '{}' tried to bind non-Type arg {} to '{}'",
                base_name,
                index,
                format_prim(*prim)
            )));
        };
        let mut check_subst = BTreeMap::<String, Type>::new();
        collect_type_var_bindings(inner, &Type::Scalar(*prim), &mut check_subst)?;
    }
    let mut kept_params = Vec::<Type>::new();
    for (index, ty) in instantiated_params.iter().enumerate() {
        if let Some(binding) = known_fun_args.get(&index) {
            kept_params.extend(binding.capture_types.clone());
        } else if known_type_witness_args.contains_key(&index) {
            // Type witness arg consumed by monomorphized dispatch.
        } else {
            kept_params.push(ty.clone());
        }
    }
    if kept_params.iter().any(type_contains_unresolved_vars)
        || type_contains_unresolved_vars(&instantiated_result)
    {
        return Err(SimdError::new(format!(
            "cannot fully specialize '{}' at Wasm lowering time",
            base_name
        )));
    }
    let needs_specialized_clone = !known_fun_args.is_empty()
        || !known_type_witness_args.is_empty()
        || !type_subst.is_empty()
        || type_contains_unresolved_vars(&original.signature.ty);
    if !needs_specialized_clone {
        enqueue_specialization(
            queue,
            scheduled_names,
            PendingSpecialization {
                base_name: base_name.to_string(),
                output_name: base_name.to_string(),
                type_subst: BTreeMap::new(),
                known_fun_args: BTreeMap::new(),
                known_type_witness_args: BTreeMap::new(),
                origin: origin.to_string(),
            },
        );
        return Ok(base_name.to_string());
    }
    let instance_key = specialization_key(
        base_name,
        &instantiated_params,
        &instantiated_result,
        known_fun_args,
        known_type_witness_args,
    );
    let output_name = if let Some(existing) = generic_instances.get(&instance_key) {
        existing.clone()
    } else {
        let known_suffix = if known_fun_args.is_empty() {
            String::new()
        } else {
            format!(
                "$fn${}",
                known_fun_args
                    .iter()
                    .map(|(index, binding)| {
                        format!(
                            "{index}_{}_{}",
                            sanitize_specialized_name(&binding.target_name),
                            binding
                                .capture_types
                                .iter()
                                .map(mangle_type)
                                .collect::<Vec<_>>()
                                .join("_")
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("_")
            )
        };
        let witness_suffix = if known_type_witness_args.is_empty() {
            String::new()
        } else {
            format!(
                "$tw${}",
                known_type_witness_args
                    .iter()
                    .map(|(index, prim)| format!("{index}_{}", format_prim(*prim)))
                    .collect::<Vec<_>>()
                    .join("_")
            )
        };
        let generated = format!(
            "{}$mono${}{}{}",
            sanitize_specialized_name(base_name),
            instantiated_params
                .iter()
                .map(mangle_type)
                .collect::<Vec<_>>()
                .join("_"),
            known_suffix,
            witness_suffix
        );
        generic_instances.insert(instance_key, generated.clone());
        enqueue_specialization(
            queue,
            scheduled_names,
            PendingSpecialization {
                base_name: base_name.to_string(),
                output_name: generated.clone(),
                type_subst: type_subst.clone(),
                known_fun_args: known_fun_args.clone(),
                known_type_witness_args: known_type_witness_args.clone(),
                origin: origin.to_string(),
            },
        );
        generated
    };
    Ok(output_name)
}

fn enqueue_specialization(
    queue: &mut VecDeque<PendingSpecialization>,
    scheduled_names: &mut BTreeSet<String>,
    pending: PendingSpecialization,
) {
    if scheduled_names.insert(pending.output_name.clone()) {
        queue.push_back(pending);
    }
}

fn specialization_key(
    base_name: &str,
    params: &[Type],
    result: &Type,
    known_fun_args: &BTreeMap<usize, KnownFunctionArgSpec>,
    known_type_witness_args: &BTreeMap<usize, Prim>,
) -> String {
    let known = if known_fun_args.is_empty() {
        "none".to_string()
    } else {
        known_fun_args
            .iter()
            .map(|(index, value)| {
                format!(
                    "{index}:{}:{}",
                    sanitize_specialized_name(&value.target_name),
                    value
                        .capture_types
                        .iter()
                        .map(mangle_type)
                        .collect::<Vec<_>>()
                        .join("_")
                )
            })
            .collect::<Vec<_>>()
            .join("|")
    };
    let type_witness = if known_type_witness_args.is_empty() {
        "none".to_string()
    } else {
        known_type_witness_args
            .iter()
            .map(|(index, prim)| format!("{index}:{}", format_prim(*prim)))
            .collect::<Vec<_>>()
            .join("|")
    };
    format!(
        "{}::{}::ret={}::fn={}::tw={}",
        base_name,
        params.iter().map(mangle_type).collect::<Vec<_>>().join("|"),
        mangle_type(result),
        known,
        type_witness
    )
}

fn sanitize_specialized_name(name: &str) -> String {
    name.chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                ch
            } else {
                '$'
            }
        })
        .collect()
}

fn mangle_type(ty: &Type) -> String {
    match ty {
        Type::Scalar(prim) => match prim {
            Prim::I32 => "i32".to_string(),
            Prim::I64 => "i64".to_string(),
            Prim::Char => "char".to_string(),
            Prim::F32 => "f32".to_string(),
            Prim::F64 => "f64".to_string(),
        },
        Type::Bulk(prim, shape) => format!(
            "b{}{}",
            mangle_type(&Type::Scalar(*prim)),
            shape.0.iter().map(mangle_dim).collect::<Vec<_>>().join("")
        ),
        Type::StarSeq(item) | Type::StarSeqWitnessed(item, _) => {
            format!("s{}", mangle_type(item))
        }
        Type::Index(_) => "i64".to_string(),
        Type::TypeToken(inner) => format!("t{}", mangle_type(inner)),
        Type::Tuple(items) => format!(
            "p{}",
            items.iter().map(mangle_type).collect::<Vec<_>>().join("_")
        ),
        Type::Record(fields) => format!(
            "r{}",
            fields
                .iter()
                .map(|(name, ty)| format!("{}{}", name, mangle_type(ty)))
                .collect::<Vec<_>>()
                .join("")
        ),
        Type::Named(name, args) => format!(
            "n{}{}",
            name,
            args.iter().map(mangle_type).collect::<Vec<_>>().join("_")
        ),
        Type::Var(name) => format!("v{}", name),
        Type::Infer(index) => format!("u{}", index),
        Type::Fun(args, ret) => format!(
            "f{}r{}",
            args.iter().map(mangle_type).collect::<Vec<_>>().join("_"),
            mangle_type(ret)
        ),
    }
}

fn mangle_dim(dim: &Dim) -> String {
    match dim {
        Dim::Const(value) => format!("c{}", value),
        Dim::Var(name) => format!("s{}", name),
    }
}

fn type_contains_unresolved_vars(ty: &Type) -> bool {
    match ty {
        Type::Var(_) | Type::Infer(_) => true,
        Type::Scalar(_) | Type::Bulk(_, _) | Type::Index(_) => false,
        Type::StarSeq(item) | Type::StarSeqWitnessed(item, _) => {
            type_contains_unresolved_vars(item)
        }
        Type::Named(_, args) => args.iter().any(type_contains_unresolved_vars),
        Type::TypeToken(inner) => type_contains_unresolved_vars(inner),
        Type::Tuple(items) => items.iter().any(type_contains_unresolved_vars),
        Type::Record(fields) => fields.values().any(type_contains_unresolved_vars),
        Type::Fun(args, ret) => {
            args.iter().any(type_contains_unresolved_vars) || type_contains_unresolved_vars(ret)
        }
    }
}

#[derive(Clone)]
struct LambdaTemplate {
    param: String,
    body: TypedExpr,
}

fn eliminate_non_escaping_lambdas_program(checked: &CheckedProgram) -> Result<CheckedProgram> {
    let mut functions = Vec::with_capacity(checked.functions.len());
    for function in &checked.functions {
        let mut clauses = Vec::with_capacity(function.clauses.len());
        for clause in &function.clauses {
            let mut scope_types = BTreeMap::<String, Type>::new();
            for pattern in &clause.patterns {
                if let Pattern::Name(name) = &pattern.pattern {
                    scope_types.insert(name.clone(), pattern.ty.clone());
                }
            }
            clauses.push(TypedClause {
                patterns: clause.patterns.clone(),
                body: eliminate_non_escaping_lambdas_expr(
                    &clause.body,
                    &BTreeMap::new(),
                    &scope_types,
                )?,
            });
        }
        functions.push(CheckedFunction {
            name: function.name.clone(),
            signature: function.signature.clone(),
            clauses,
            pointwise: function.pointwise,
            tailrec: function.tailrec.clone(),
        });
    }
    Ok(CheckedProgram {
        enum_names: checked.enum_names.clone(),
        enum_ctors: checked.enum_ctors.clone(),
        enum_layouts: checked.enum_layouts.clone(),
        functions,
    })
}

fn eliminate_non_escaping_lambdas_expr(
    expr: &TypedExpr,
    lambda_env: &BTreeMap<String, LambdaTemplate>,
    scope_types: &BTreeMap<String, Type>,
) -> Result<TypedExpr> {
    let ty = expr.ty.clone();
    let kind = match &expr.kind {
        TypedExprKind::Local(name) => TypedExprKind::Local(name.clone()),
        TypedExprKind::FunctionRef { name } => TypedExprKind::FunctionRef { name: name.clone() },
        TypedExprKind::ConstructorRef { name } => {
            TypedExprKind::ConstructorRef { name: name.clone() }
        }
        TypedExprKind::Int(value, prim) => TypedExprKind::Int(*value, *prim),
        TypedExprKind::Float(value, prim) => TypedExprKind::Float(*value, *prim),
        TypedExprKind::Bool(value) => TypedExprKind::Bool(*value),
        TypedExprKind::Char(value) => TypedExprKind::Char(*value),
        TypedExprKind::String(value) => TypedExprKind::String(value.clone()),
        TypedExprKind::Seq(items) => TypedExprKind::Seq(
            items
                .iter()
                .map(|item| eliminate_non_escaping_lambdas_expr(item, lambda_env, scope_types))
                .collect::<Result<Vec<_>>>()?,
        ),
        TypedExprKind::SeqSplice { prefix, tail } => TypedExprKind::SeqSplice {
            prefix: prefix
                .iter()
                .map(|item| eliminate_non_escaping_lambdas_expr(item, lambda_env, scope_types))
                .collect::<Result<Vec<_>>>()?,
            tail: Box::new(eliminate_non_escaping_lambdas_expr(
                tail,
                lambda_env,
                scope_types,
            )?),
        },
        TypedExprKind::TypeToken(prim) => TypedExprKind::TypeToken(*prim),
        TypedExprKind::Lambda { param, body } => TypedExprKind::Lambda {
            param: param.clone(),
            body: Box::new(eliminate_non_escaping_lambdas_expr(
                body,
                lambda_env,
                scope_types,
            )?),
        },
        TypedExprKind::Let { bindings, body } => {
            let mut next_scope = scope_types.clone();
            let mut next_env = lambda_env.clone();
            let mut rebuilt = Vec::<TypedLetBinding>::new();
            for binding in bindings {
                let mut rewritten_expr =
                    eliminate_non_escaping_lambdas_expr(&binding.expr, &next_env, &next_scope)?;
                let lambda_shape = match &rewritten_expr.kind {
                    TypedExprKind::Lambda { param, body } if binding.name != "_" => {
                        Some((param.clone(), (**body).clone()))
                    }
                    _ => None,
                };
                if let Some((param_name, body_expr)) = lambda_shape {
                    let captures = free_lambda_capture_names(
                        &body_expr,
                        &param_name,
                        &next_scope.keys().cloned().collect(),
                    );
                    if !captures.is_empty() {
                        let mut capture_map = BTreeMap::<String, String>::new();
                        for (index, capture) in captures.iter().enumerate() {
                            if let Some(capture_ty) = next_scope.get(capture) {
                                let alias = format!("{}$cap${}", binding.name, index);
                                rebuilt.push(TypedLetBinding {
                                    name: alias.clone(),
                                    expr: TypedExpr {
                                        ty: capture_ty.clone(),
                                        kind: TypedExprKind::Local(capture.clone()),
                                    },
                                });
                                next_scope.insert(alias.clone(), capture_ty.clone());
                                capture_map.insert(capture.clone(), alias);
                            }
                        }
                        let renamed_body =
                            rename_free_locals(&body_expr, &capture_map, &BTreeSet::new())?;
                        rewritten_expr = TypedExpr {
                            ty: rewritten_expr.ty.clone(),
                            kind: TypedExprKind::Lambda {
                                param: param_name.clone(),
                                body: Box::new(renamed_body.clone()),
                            },
                        };
                        next_env.insert(
                            binding.name.clone(),
                            LambdaTemplate {
                                param: param_name,
                                body: renamed_body,
                            },
                        );
                    } else {
                        next_env.insert(
                            binding.name.clone(),
                            LambdaTemplate {
                                param: param_name,
                                body: body_expr,
                            },
                        );
                    }
                }
                rebuilt.push(TypedLetBinding {
                    name: binding.name.clone(),
                    expr: rewritten_expr.clone(),
                });
                if binding.name != "_" {
                    next_scope.insert(binding.name.clone(), rewritten_expr.ty);
                }
            }
            let rewritten_body = eliminate_non_escaping_lambdas_expr(body, &next_env, &next_scope)?;
            let pruned = prune_typed_let_bindings_for_backend(rebuilt, &rewritten_body);
            if pruned.is_empty() {
                return Ok(rewritten_body);
            }
            TypedExprKind::Let {
                bindings: pruned,
                body: Box::new(rewritten_body),
            }
        }
        TypedExprKind::Record(fields) => TypedExprKind::Record(
            fields
                .iter()
                .map(|(name, value)| {
                    Ok((
                        name.clone(),
                        eliminate_non_escaping_lambdas_expr(value, lambda_env, scope_types)?,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?,
        ),
        TypedExprKind::Tuple(items) => TypedExprKind::Tuple(
            items
                .iter()
                .map(|item| eliminate_non_escaping_lambdas_expr(item, lambda_env, scope_types))
                .collect::<Result<Vec<_>>>()?,
        ),
        TypedExprKind::Project { base, field } => TypedExprKind::Project {
            base: Box::new(eliminate_non_escaping_lambdas_expr(
                base,
                lambda_env,
                scope_types,
            )?),
            field: field.clone(),
        },
        TypedExprKind::TupleProject { base, index } => TypedExprKind::TupleProject {
            base: Box::new(eliminate_non_escaping_lambdas_expr(
                base,
                lambda_env,
                scope_types,
            )?),
            index: *index,
        },
        TypedExprKind::Index {
            base,
            index,
            checked,
        } => TypedExprKind::Index {
            base: Box::new(eliminate_non_escaping_lambdas_expr(
                base,
                lambda_env,
                scope_types,
            )?),
            index: Box::new(eliminate_non_escaping_lambdas_expr(
                index,
                lambda_env,
                scope_types,
            )?),
            checked: *checked,
        },
        TypedExprKind::RecordUpdate { base, fields } => TypedExprKind::RecordUpdate {
            base: Box::new(eliminate_non_escaping_lambdas_expr(
                base,
                lambda_env,
                scope_types,
            )?),
            fields: fields
                .iter()
                .map(|(name, value)| {
                    Ok((
                        name.clone(),
                        eliminate_non_escaping_lambdas_expr(value, lambda_env, scope_types)?,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?,
        },
        TypedExprKind::Call {
            callee,
            args,
            lifted_shape,
        } => TypedExprKind::Call {
            callee: callee.clone(),
            args: args
                .iter()
                .map(|arg| {
                    Ok(TypedArg {
                        mode: arg.mode,
                        expr: Box::new(eliminate_non_escaping_lambdas_expr(
                            &arg.expr,
                            lambda_env,
                            scope_types,
                        )?),
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            lifted_shape: lifted_shape.clone(),
        },
        TypedExprKind::Apply { callee, arg } => {
            let rewritten_callee =
                eliminate_non_escaping_lambdas_expr(callee, lambda_env, scope_types)?;
            let rewritten_arg = eliminate_non_escaping_lambdas_expr(arg, lambda_env, scope_types)?;
            if let TypedExprKind::Lambda { param, body } = &rewritten_callee.kind {
                let reduced = TypedExpr {
                    ty: ty.clone(),
                    kind: TypedExprKind::Let {
                        bindings: vec![TypedLetBinding {
                            name: param.clone(),
                            expr: rewritten_arg,
                        }],
                        body: body.clone(),
                    },
                };
                return eliminate_non_escaping_lambdas_expr(&reduced, lambda_env, scope_types);
            }
            if let TypedExprKind::Local(local_name) = &rewritten_callee.kind {
                if let Some(template) = lambda_env.get(local_name) {
                    let reduced = TypedExpr {
                        ty: ty.clone(),
                        kind: TypedExprKind::Let {
                            bindings: vec![TypedLetBinding {
                                name: template.param.clone(),
                                expr: rewritten_arg,
                            }],
                            body: Box::new(template.body.clone()),
                        },
                    };
                    return eliminate_non_escaping_lambdas_expr(&reduced, lambda_env, scope_types);
                }
            }
            TypedExprKind::Apply {
                callee: Box::new(rewritten_callee),
                arg: Box::new(rewritten_arg),
            }
        }
    };
    Ok(TypedExpr { ty, kind })
}

fn free_lambda_capture_names(
    body: &TypedExpr,
    param: &str,
    in_scope: &BTreeSet<String>,
) -> Vec<String> {
    let mut bound = BTreeSet::<String>::new();
    bound.insert(param.to_string());
    let mut free = BTreeSet::<String>::new();
    collect_free_locals(body, &mut bound, &mut free);
    free.into_iter()
        .filter(|name| in_scope.contains(name))
        .collect()
}

fn collect_free_locals(
    expr: &TypedExpr,
    bound: &mut BTreeSet<String>,
    free: &mut BTreeSet<String>,
) {
    match &expr.kind {
        TypedExprKind::Local(name) => {
            if !bound.contains(name) {
                free.insert(name.clone());
            }
        }
        TypedExprKind::FunctionRef { .. }
        | TypedExprKind::ConstructorRef { .. }
        | TypedExprKind::Int(_, _)
        | TypedExprKind::Float(_, _)
        | TypedExprKind::Bool(_)
        | TypedExprKind::Char(_)
        | TypedExprKind::String(_)
        | TypedExprKind::TypeToken(_) => {}
        TypedExprKind::Lambda { param, body } => {
            let mut nested = bound.clone();
            nested.insert(param.clone());
            collect_free_locals(body, &mut nested, free);
        }
        TypedExprKind::Let { bindings, body } => {
            let mut nested = bound.clone();
            for binding in bindings {
                collect_free_locals(&binding.expr, &mut nested, free);
                if binding.name != "_" {
                    nested.insert(binding.name.clone());
                }
            }
            collect_free_locals(body, &mut nested, free);
        }
        TypedExprKind::Record(fields) => {
            for value in fields.values() {
                collect_free_locals(value, bound, free);
            }
        }
        TypedExprKind::Seq(items) => {
            for item in items {
                collect_free_locals(item, bound, free);
            }
        }
        TypedExprKind::SeqSplice { prefix, tail } => {
            for item in prefix {
                collect_free_locals(item, bound, free);
            }
            collect_free_locals(tail, bound, free);
        }
        TypedExprKind::Index { base, index, .. } => {
            collect_free_locals(base, bound, free);
            collect_free_locals(index, bound, free);
        }
        TypedExprKind::Tuple(items) => {
            for item in items {
                collect_free_locals(item, bound, free);
            }
        }
        TypedExprKind::Project { base, .. } | TypedExprKind::TupleProject { base, .. } => {
            collect_free_locals(base, bound, free)
        }
        TypedExprKind::RecordUpdate { base, fields } => {
            collect_free_locals(base, bound, free);
            for value in fields.values() {
                collect_free_locals(value, bound, free);
            }
        }
        TypedExprKind::Call { args, .. } => {
            for arg in args {
                collect_free_locals(&arg.expr, bound, free);
            }
        }
        TypedExprKind::Apply { callee, arg } => {
            collect_free_locals(callee, bound, free);
            collect_free_locals(arg, bound, free);
        }
    }
}

fn rename_free_locals(
    expr: &TypedExpr,
    mapping: &BTreeMap<String, String>,
    bound: &BTreeSet<String>,
) -> Result<TypedExpr> {
    let ty = expr.ty.clone();
    let kind = match &expr.kind {
        TypedExprKind::Local(name) => {
            if !bound.contains(name) {
                if let Some(renamed) = mapping.get(name) {
                    TypedExprKind::Local(renamed.clone())
                } else {
                    TypedExprKind::Local(name.clone())
                }
            } else {
                TypedExprKind::Local(name.clone())
            }
        }
        TypedExprKind::FunctionRef { name } => TypedExprKind::FunctionRef { name: name.clone() },
        TypedExprKind::ConstructorRef { name } => {
            TypedExprKind::ConstructorRef { name: name.clone() }
        }
        TypedExprKind::Int(value, prim) => TypedExprKind::Int(*value, *prim),
        TypedExprKind::Float(value, prim) => TypedExprKind::Float(*value, *prim),
        TypedExprKind::Bool(value) => TypedExprKind::Bool(*value),
        TypedExprKind::Char(value) => TypedExprKind::Char(*value),
        TypedExprKind::String(value) => TypedExprKind::String(value.clone()),
        TypedExprKind::Seq(items) => TypedExprKind::Seq(
            items
                .iter()
                .map(|item| rename_free_locals(item, mapping, bound))
                .collect::<Result<Vec<_>>>()?,
        ),
        TypedExprKind::SeqSplice { prefix, tail } => TypedExprKind::SeqSplice {
            prefix: prefix
                .iter()
                .map(|item| rename_free_locals(item, mapping, bound))
                .collect::<Result<Vec<_>>>()?,
            tail: Box::new(rename_free_locals(tail, mapping, bound)?),
        },
        TypedExprKind::TypeToken(prim) => TypedExprKind::TypeToken(*prim),
        TypedExprKind::Lambda { param, body } => {
            let mut nested = bound.clone();
            nested.insert(param.clone());
            TypedExprKind::Lambda {
                param: param.clone(),
                body: Box::new(rename_free_locals(body, mapping, &nested)?),
            }
        }
        TypedExprKind::Let { bindings, body } => {
            let mut nested = bound.clone();
            let mut rewritten = Vec::with_capacity(bindings.len());
            for binding in bindings {
                rewritten.push(TypedLetBinding {
                    name: binding.name.clone(),
                    expr: rename_free_locals(&binding.expr, mapping, &nested)?,
                });
                if binding.name != "_" {
                    nested.insert(binding.name.clone());
                }
            }
            TypedExprKind::Let {
                bindings: rewritten,
                body: Box::new(rename_free_locals(body, mapping, &nested)?),
            }
        }
        TypedExprKind::Record(fields) => TypedExprKind::Record(
            fields
                .iter()
                .map(|(name, value)| Ok((name.clone(), rename_free_locals(value, mapping, bound)?)))
                .collect::<Result<BTreeMap<_, _>>>()?,
        ),
        TypedExprKind::Tuple(items) => TypedExprKind::Tuple(
            items
                .iter()
                .map(|item| rename_free_locals(item, mapping, bound))
                .collect::<Result<Vec<_>>>()?,
        ),
        TypedExprKind::Project { base, field } => TypedExprKind::Project {
            base: Box::new(rename_free_locals(base, mapping, bound)?),
            field: field.clone(),
        },
        TypedExprKind::TupleProject { base, index } => TypedExprKind::TupleProject {
            base: Box::new(rename_free_locals(base, mapping, bound)?),
            index: *index,
        },
        TypedExprKind::Index {
            base,
            index,
            checked,
        } => TypedExprKind::Index {
            base: Box::new(rename_free_locals(base, mapping, bound)?),
            index: Box::new(rename_free_locals(index, mapping, bound)?),
            checked: *checked,
        },
        TypedExprKind::RecordUpdate { base, fields } => TypedExprKind::RecordUpdate {
            base: Box::new(rename_free_locals(base, mapping, bound)?),
            fields: fields
                .iter()
                .map(|(name, value)| Ok((name.clone(), rename_free_locals(value, mapping, bound)?)))
                .collect::<Result<BTreeMap<_, _>>>()?,
        },
        TypedExprKind::Call {
            callee,
            args,
            lifted_shape,
        } => TypedExprKind::Call {
            callee: callee.clone(),
            args: args
                .iter()
                .map(|arg| {
                    Ok(TypedArg {
                        mode: arg.mode,
                        expr: Box::new(rename_free_locals(&arg.expr, mapping, bound)?),
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            lifted_shape: lifted_shape.clone(),
        },
        TypedExprKind::Apply { callee, arg } => TypedExprKind::Apply {
            callee: Box::new(rename_free_locals(callee, mapping, bound)?),
            arg: Box::new(rename_free_locals(arg, mapping, bound)?),
        },
    };
    Ok(TypedExpr { ty, kind })
}

fn prune_typed_let_bindings_for_backend(
    bindings: Vec<TypedLetBinding>,
    body: &TypedExpr,
) -> Vec<TypedLetBinding> {
    let mut demanded = BTreeSet::<String>::new();
    collect_used_locals(body, &mut demanded);
    let mut kept = Vec::<TypedLetBinding>::new();
    for binding in bindings.into_iter().rev() {
        if binding.name != "_" && demanded.remove(&binding.name) {
            collect_used_locals(&binding.expr, &mut demanded);
            kept.push(binding);
        }
    }
    kept.reverse();
    kept
}

fn collect_used_locals(expr: &TypedExpr, names: &mut BTreeSet<String>) {
    match &expr.kind {
        TypedExprKind::Local(name) => {
            names.insert(name.clone());
        }
        TypedExprKind::FunctionRef { .. }
        | TypedExprKind::ConstructorRef { .. }
        | TypedExprKind::Int(_, _)
        | TypedExprKind::Float(_, _)
        | TypedExprKind::Bool(_)
        | TypedExprKind::Char(_)
        | TypedExprKind::String(_)
        | TypedExprKind::TypeToken(_) => {}
        TypedExprKind::Lambda { body, .. } => collect_used_locals(body, names),
        TypedExprKind::Let { bindings, body } => {
            for binding in bindings {
                collect_used_locals(&binding.expr, names);
            }
            collect_used_locals(body, names);
        }
        TypedExprKind::Record(fields) => {
            for value in fields.values() {
                collect_used_locals(value, names);
            }
        }
        TypedExprKind::Seq(items) => {
            for item in items {
                collect_used_locals(item, names);
            }
        }
        TypedExprKind::SeqSplice { prefix, tail } => {
            for item in prefix {
                collect_used_locals(item, names);
            }
            collect_used_locals(tail, names);
        }
        TypedExprKind::Index { base, index, .. } => {
            collect_used_locals(base, names);
            collect_used_locals(index, names);
        }
        TypedExprKind::Tuple(items) => {
            for item in items {
                collect_used_locals(item, names);
            }
        }
        TypedExprKind::Project { base, .. } | TypedExprKind::TupleProject { base, .. } => {
            collect_used_locals(base, names)
        }
        TypedExprKind::RecordUpdate { base, fields } => {
            collect_used_locals(base, names);
            for value in fields.values() {
                collect_used_locals(value, names);
            }
        }
        TypedExprKind::Call { args, .. } => {
            for arg in args {
                collect_used_locals(&arg.expr, names);
            }
        }
        TypedExprKind::Apply { callee, arg } => {
            collect_used_locals(callee, names);
            collect_used_locals(arg, names);
        }
    }
}

#[derive(Debug, Clone)]
struct KnownFunctionTemplate {
    name: String,
    bound_args: Vec<TypedExpr>,
}

fn canonicalize_backend_higher_order_program(checked: &CheckedProgram) -> Result<CheckedProgram> {
    let signatures = checked
        .functions
        .iter()
        .map(|function| (function.name.clone(), function))
        .collect::<BTreeMap<_, _>>();
    let mut functions = Vec::with_capacity(checked.functions.len());
    for function in &checked.functions {
        let clauses = function
            .clauses
            .iter()
            .map(|clause| {
                let body = canonicalize_backend_higher_order_expr(
                    &clause.body,
                    &BTreeMap::new(),
                    &signatures,
                )?;
                Ok(TypedClause {
                    patterns: clause.patterns.clone(),
                    body,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        functions.push(CheckedFunction {
            name: function.name.clone(),
            signature: function.signature.clone(),
            clauses,
            pointwise: function.pointwise,
            tailrec: function.tailrec.clone(),
        });
    }
    Ok(CheckedProgram {
        enum_names: checked.enum_names.clone(),
        enum_ctors: checked.enum_ctors.clone(),
        enum_layouts: checked.enum_layouts.clone(),
        functions,
    })
}

fn canonicalize_backend_higher_order_expr(
    expr: &TypedExpr,
    known_functions: &BTreeMap<String, KnownFunctionTemplate>,
    signatures: &BTreeMap<String, &CheckedFunction>,
) -> Result<TypedExpr> {
    let ty = expr.ty.clone();
    let kind = match &expr.kind {
        TypedExprKind::Local(name) => TypedExprKind::Local(name.clone()),
        TypedExprKind::FunctionRef { name } => TypedExprKind::FunctionRef { name: name.clone() },
        TypedExprKind::ConstructorRef { name } => {
            TypedExprKind::ConstructorRef { name: name.clone() }
        }
        TypedExprKind::Int(value, prim) => TypedExprKind::Int(*value, *prim),
        TypedExprKind::Float(value, prim) => TypedExprKind::Float(*value, *prim),
        TypedExprKind::Bool(value) => TypedExprKind::Bool(*value),
        TypedExprKind::Char(value) => TypedExprKind::Char(*value),
        TypedExprKind::String(value) => TypedExprKind::String(value.clone()),
        TypedExprKind::Seq(items) => TypedExprKind::Seq(
            items
                .iter()
                .map(|item| {
                    canonicalize_backend_higher_order_expr(item, known_functions, signatures)
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        TypedExprKind::SeqSplice { prefix, tail } => TypedExprKind::SeqSplice {
            prefix: prefix
                .iter()
                .map(|item| {
                    canonicalize_backend_higher_order_expr(item, known_functions, signatures)
                })
                .collect::<Result<Vec<_>>>()?,
            tail: Box::new(canonicalize_backend_higher_order_expr(
                tail,
                known_functions,
                signatures,
            )?),
        },
        TypedExprKind::TypeToken(prim) => TypedExprKind::TypeToken(*prim),
        TypedExprKind::Lambda { param, body } => TypedExprKind::Lambda {
            param: param.clone(),
            body: Box::new(canonicalize_backend_higher_order_expr(
                body,
                known_functions,
                signatures,
            )?),
        },
        TypedExprKind::Let { bindings, body } => {
            let mut next_known = known_functions.clone();
            let mut rewritten = Vec::with_capacity(bindings.len());
            for binding in bindings {
                let rewritten_expr =
                    canonicalize_backend_higher_order_expr(&binding.expr, &next_known, signatures)?;
                if binding.name != "_" {
                    if let Some(template) =
                        extract_known_function_template(&rewritten_expr, &next_known)
                    {
                        next_known.insert(binding.name.clone(), template);
                    } else {
                        next_known.remove(&binding.name);
                    }
                }
                rewritten.push(TypedLetBinding {
                    name: binding.name.clone(),
                    expr: rewritten_expr,
                });
            }
            let rewritten_body =
                canonicalize_backend_higher_order_expr(body, &next_known, signatures)?;
            let pruned = prune_typed_let_bindings_for_backend(rewritten, &rewritten_body);
            if pruned.is_empty() {
                return Ok(rewritten_body);
            }
            TypedExprKind::Let {
                bindings: pruned,
                body: Box::new(rewritten_body),
            }
        }
        TypedExprKind::Record(fields) => TypedExprKind::Record(
            fields
                .iter()
                .map(|(name, value)| {
                    Ok((
                        name.clone(),
                        canonicalize_backend_higher_order_expr(value, known_functions, signatures)?,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?,
        ),
        TypedExprKind::Tuple(items) => TypedExprKind::Tuple(
            items
                .iter()
                .map(|item| {
                    canonicalize_backend_higher_order_expr(item, known_functions, signatures)
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        TypedExprKind::Project { base, field } => TypedExprKind::Project {
            base: Box::new(canonicalize_backend_higher_order_expr(
                base,
                known_functions,
                signatures,
            )?),
            field: field.clone(),
        },
        TypedExprKind::TupleProject { base, index } => TypedExprKind::TupleProject {
            base: Box::new(canonicalize_backend_higher_order_expr(
                base,
                known_functions,
                signatures,
            )?),
            index: *index,
        },
        TypedExprKind::Index {
            base,
            index,
            checked,
        } => TypedExprKind::Index {
            base: Box::new(canonicalize_backend_higher_order_expr(
                base,
                known_functions,
                signatures,
            )?),
            index: Box::new(canonicalize_backend_higher_order_expr(
                index,
                known_functions,
                signatures,
            )?),
            checked: *checked,
        },
        TypedExprKind::RecordUpdate { base, fields } => TypedExprKind::RecordUpdate {
            base: Box::new(canonicalize_backend_higher_order_expr(
                base,
                known_functions,
                signatures,
            )?),
            fields: fields
                .iter()
                .map(|(name, value)| {
                    Ok((
                        name.clone(),
                        canonicalize_backend_higher_order_expr(value, known_functions, signatures)?,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?,
        },
        TypedExprKind::Call {
            callee,
            args,
            lifted_shape,
        } => TypedExprKind::Call {
            callee: callee.clone(),
            args: args
                .iter()
                .map(|arg| {
                    Ok(TypedArg {
                        mode: arg.mode,
                        expr: Box::new(canonicalize_backend_higher_order_expr(
                            &arg.expr,
                            known_functions,
                            signatures,
                        )?),
                    })
                })
                .collect::<Result<Vec<_>>>()?,
            lifted_shape: lifted_shape.clone(),
        },
        TypedExprKind::Apply { callee, arg } => {
            let rewritten_callee =
                canonicalize_backend_higher_order_expr(callee, known_functions, signatures)?;
            let rewritten_arg =
                canonicalize_backend_higher_order_expr(arg, known_functions, signatures)?;
            if let Some(mut template) =
                extract_known_function_template(&rewritten_callee, known_functions)
            {
                template.bound_args.push(rewritten_arg);
                return canonicalize_known_function_application(template, &ty, signatures);
            }
            TypedExprKind::Apply {
                callee: Box::new(rewritten_callee),
                arg: Box::new(rewritten_arg),
            }
        }
    };
    Ok(TypedExpr { ty, kind })
}

fn extract_known_function_template(
    expr: &TypedExpr,
    known_functions: &BTreeMap<String, KnownFunctionTemplate>,
) -> Option<KnownFunctionTemplate> {
    match &expr.kind {
        TypedExprKind::FunctionRef { name } => Some(KnownFunctionTemplate {
            name: name.clone(),
            bound_args: Vec::new(),
        }),
        TypedExprKind::ConstructorRef { .. } => None,
        TypedExprKind::Local(name) => known_functions.get(name).cloned(),
        TypedExprKind::Apply { callee, arg } => {
            let mut template = extract_known_function_template(callee, known_functions)?;
            template.bound_args.push((**arg).clone());
            Some(template)
        }
        _ => None,
    }
}

fn classify_function_value_expr_for_backend(expr: &TypedExpr) -> FunctionValueClass {
    match &expr.kind {
        TypedExprKind::FunctionRef { .. } => FunctionValueClass::KnownFn,
        TypedExprKind::ConstructorRef { .. } => FunctionValueClass::EscapingUnknown,
        TypedExprKind::Lambda { .. } => FunctionValueClass::KnownLambda,
        TypedExprKind::Local(_)
        | TypedExprKind::Apply { .. }
        | TypedExprKind::Let { .. }
        | TypedExprKind::Call { .. } => FunctionValueClass::EscapingUnknown,
        TypedExprKind::Int(_, _)
        | TypedExprKind::Float(_, _)
        | TypedExprKind::Bool(_)
        | TypedExprKind::Char(_)
        | TypedExprKind::String(_)
        | TypedExprKind::TypeToken(_)
        | TypedExprKind::Seq(_)
        | TypedExprKind::SeqSplice { .. }
        | TypedExprKind::Tuple(_)
        | TypedExprKind::Record(_)
        | TypedExprKind::Project { .. }
        | TypedExprKind::TupleProject { .. }
        | TypedExprKind::Index { .. }
        | TypedExprKind::RecordUpdate { .. } => FunctionValueClass::EscapingUnknown,
    }
}

fn canonicalize_known_function_application(
    template: KnownFunctionTemplate,
    expected_ty: &Type,
    signatures: &BTreeMap<String, &CheckedFunction>,
) -> Result<TypedExpr> {
    let signature = signatures.get(&template.name).copied().ok_or_else(|| {
        SimdError::new(format!(
            "unknown function '{}' during backend apply canonicalization",
            template.name
        ))
    })?;
    let (param_types, result_ty) = signature.signature.ty.fun_parts();
    if template.bound_args.len() > param_types.len() {
        return Err(SimdError::new(format!(
            "backend apply canonicalization over-applied '{}' with {} args (arity {})",
            template.name,
            template.bound_args.len(),
            param_types.len()
        )));
    }
    if template.bound_args.len() < param_types.len() {
        let mut expr = TypedExpr {
            ty: signature.signature.ty.clone(),
            kind: TypedExprKind::FunctionRef {
                name: template.name.clone(),
            },
        };
        for arg in template.bound_args {
            let next_ty = match &expr.ty {
                Type::Fun(args, ret) if !args.is_empty() => {
                    if arg.ty != args[0] {
                        return Err(SimdError::new(format!(
                            "backend apply canonicalization argument type mismatch for '{}': {:?} vs {:?}",
                            template.name, arg.ty, args[0]
                        )));
                    }
                    if args.len() == 1 {
                        ret.as_ref().clone()
                    } else {
                        Type::Fun(args[1..].to_vec(), ret.clone())
                    }
                }
                other => {
                    return Err(SimdError::new(format!(
                        "backend apply canonicalization attempted to apply non-function {:?}",
                        other
                    )));
                }
            };
            expr = TypedExpr {
                ty: next_ty.clone(),
                kind: TypedExprKind::Apply {
                    callee: Box::new(expr),
                    arg: Box::new(arg),
                },
            };
        }
        if expr.ty != *expected_ty {
            return Err(SimdError::new(format!(
                "backend apply canonicalization produced type {:?}, expected {:?}",
                expr.ty, expected_ty
            )));
        }
        return Ok(expr);
    }
    let mut dim_subst = BTreeMap::<String, Dim>::new();
    let mut lifted_shape: Option<Shape> = None;
    let mut typed_args = Vec::<TypedArg>::new();
    for (arg, param_ty) in template.bound_args.into_iter().zip(param_types.iter()) {
        let mut exact_subst = dim_subst.clone();
        let mode = if unify_param_type(param_ty, &arg.ty, &mut exact_subst).is_ok() {
            dim_subst = exact_subst;
            AccessKind::Same
        } else {
            let mut trial_shape = lifted_shape.clone();
            unify_lifted_type(&mut trial_shape, param_ty, &arg.ty)?;
            lifted_shape = trial_shape;
            AccessKind::Lane
        };
        typed_args.push(TypedArg {
            mode,
            expr: Box::new(arg),
        });
    }
    let concrete_ret = apply_dim_subst(&result_ty, &dim_subst);
    let call_ty = if let Some(shape) = &lifted_shape {
        lift_type_over_shape(&concrete_ret, shape)?
    } else {
        concrete_ret
    };
    if call_ty != *expected_ty {
        return Err(SimdError::new(format!(
            "backend apply canonicalization call type mismatch for '{}': {:?} vs {:?}",
            template.name, call_ty, expected_ty
        )));
    }
    Ok(TypedExpr {
        ty: call_ty,
        kind: TypedExprKind::Call {
            callee: Callee::Function(template.name),
            args: typed_args,
            lifted_shape,
        },
    })
}

fn apply_type_subst(ty: &Type, subst: &BTreeMap<String, Type>) -> Type {
    match ty {
        Type::Var(name) => subst.get(name).cloned().unwrap_or_else(|| ty.clone()),
        Type::Scalar(_) | Type::Bulk(_, _) | Type::Infer(_) | Type::Index(_) => ty.clone(),
        Type::StarSeq(item) => Type::StarSeq(Box::new(apply_type_subst(item, subst))),
        Type::StarSeqWitnessed(item, witness) => {
            Type::StarSeqWitnessed(Box::new(apply_type_subst(item, subst)), witness.clone())
        }
        Type::Tuple(items) => Type::Tuple(
            items
                .iter()
                .map(|item| apply_type_subst(item, subst))
                .collect(),
        ),
        Type::Named(name, args) => Type::Named(
            name.clone(),
            args.iter()
                .map(|arg| apply_type_subst(arg, subst))
                .collect(),
        ),
        Type::TypeToken(inner) => Type::TypeToken(Box::new(apply_type_subst(inner, subst))),
        Type::Record(fields) => Type::Record(
            fields
                .iter()
                .map(|(name, field_ty)| (name.clone(), apply_type_subst(field_ty, subst)))
                .collect(),
        ),
        Type::Fun(args, ret) => Type::Fun(
            args.iter()
                .map(|arg| apply_type_subst(arg, subst))
                .collect(),
            Box::new(apply_type_subst(ret, subst)),
        ),
    }
}

fn collect_type_var_bindings(
    template: &Type,
    actual: &Type,
    subst: &mut BTreeMap<String, Type>,
) -> Result<()> {
    match template {
        Type::Var(name) => match subst.get(name) {
            Some(existing) if existing == actual => Ok(()),
            Some(existing) => Err(SimdError::new(format!(
                "type variable '{}' was inferred as {:?}, not {:?}",
                name, existing, actual
            ))),
            None => {
                subst.insert(name.clone(), actual.clone());
                Ok(())
            }
        },
        Type::Scalar(left) => match actual {
            Type::Scalar(right) if left == right => Ok(()),
            _ => Err(SimdError::new(format!(
                "specialization expected scalar {:?}, found {:?}",
                left, actual
            ))),
        },
        Type::Bulk(left_prim, left_shape) => match actual {
            Type::Bulk(right_prim, right_shape)
                if left_prim == right_prim && left_shape.0.len() == right_shape.0.len() =>
            {
                Ok(())
            }
            _ => Err(SimdError::new(format!(
                "specialization expected bulk {:?}, found {:?}",
                template, actual
            ))),
        },
        Type::Index(left_witness) => match actual {
            Type::Index(right_witness) if left_witness == right_witness => Ok(()),
            Type::Scalar(Prim::I64) => Ok(()),
            _ => Err(SimdError::new(format!(
                "specialization expected Index {}, found {:?}",
                left_witness, actual
            ))),
        },
        Type::StarSeq(left_item) | Type::StarSeqWitnessed(left_item, _) => match actual {
            Type::StarSeq(right_item) | Type::StarSeqWitnessed(right_item, _) => {
                collect_type_var_bindings(left_item, right_item, subst)
            }
            _ => Err(SimdError::new(format!(
                "specialization expected T[*] {:?}, found {:?}",
                template, actual
            ))),
        },
        Type::Tuple(left_items) => match actual {
            Type::Tuple(right_items) if left_items.len() == right_items.len() => {
                for (left_item, right_item) in left_items.iter().zip(right_items.iter()) {
                    collect_type_var_bindings(left_item, right_item, subst)?;
                }
                Ok(())
            }
            _ => Err(SimdError::new(format!(
                "specialization expected tuple {:?}, found {:?}",
                template, actual
            ))),
        },
        Type::Named(left_name, left_args) => match actual {
            Type::Named(right_name, right_args)
                if left_name == right_name && left_args.len() == right_args.len() =>
            {
                for (left_arg, right_arg) in left_args.iter().zip(right_args) {
                    collect_type_var_bindings(left_arg, right_arg, subst)?;
                }
                Ok(())
            }
            _ => Err(SimdError::new(format!(
                "specialization expected type constructor {:?}, found {:?}",
                template, actual
            ))),
        },
        Type::TypeToken(left) => match actual {
            Type::TypeToken(right) => collect_type_var_bindings(left, right, subst),
            _ => Err(SimdError::new(format!(
                "specialization expected type witness {:?}, found {:?}",
                template, actual
            ))),
        },
        Type::Record(left_fields) => match actual {
            Type::Record(right_fields) if left_fields.len() == right_fields.len() => {
                for (name, left_ty) in left_fields {
                    let right_ty = right_fields.get(name).ok_or_else(|| {
                        SimdError::new(format!(
                            "record field '{}' is missing in specialization",
                            name
                        ))
                    })?;
                    collect_type_var_bindings(left_ty, right_ty, subst)?;
                }
                Ok(())
            }
            _ => Err(SimdError::new(format!(
                "specialization expected record {:?}, found {:?}",
                template, actual
            ))),
        },
        Type::Fun(left_args, left_ret) => match actual {
            Type::Fun(right_args, right_ret) if left_args.len() == right_args.len() => {
                for (left_arg, right_arg) in left_args.iter().zip(right_args) {
                    collect_type_var_bindings(left_arg, right_arg, subst)?;
                }
                collect_type_var_bindings(left_ret, right_ret, subst)
            }
            _ => Err(SimdError::new(format!(
                "specialization expected function {:?}, found {:?}",
                template, actual
            ))),
        },
        Type::Infer(_) => Ok(()),
    }
}

fn specialization_arg_type(arg: &TypedArg) -> Result<Type> {
    match arg.mode {
        AccessKind::Same => Ok(arg.expr.ty.clone()),
        AccessKind::Lane => lane_scalar_type(&arg.expr.ty),
    }
}

fn lane_scalar_type(ty: &Type) -> Result<Type> {
    match ty {
        Type::Bulk(prim, _) => Ok(Type::Scalar(*prim)),
        Type::Index(_) => Ok(Type::Scalar(Prim::I64)),
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _) => Err(SimdError::new(
            "lane argument expected bulk-compatible type, found T[*]",
        )),
        Type::Tuple(items) => Ok(Type::Tuple(
            items
                .iter()
                .map(lane_scalar_type)
                .collect::<Result<Vec<_>>>()?,
        )),
        Type::Record(fields) => Ok(Type::Record(
            fields
                .iter()
                .map(|(name, field_ty)| Ok((name.clone(), lane_scalar_type(field_ty)?)))
                .collect::<Result<BTreeMap<_, _>>>()?,
        )),
        Type::TypeToken(_) => Err(SimdError::new(
            "lane argument expected bulk-compatible type, found type witness",
        )),
        Type::Scalar(_) | Type::Named(_, _) | Type::Var(_) | Type::Infer(_) | Type::Fun(_, _) => {
            Err(SimdError::new(format!(
                "lane argument expected bulk-compatible type, found {:?}",
                ty
            )))
        }
    }
}

fn collect_higher_order_reports(
    program: &CheckedProgram,
    origins: &BTreeMap<String, String>,
) -> Vec<WasmHigherOrderReport> {
    program
        .functions
        .iter()
        .map(|function| {
            let mut known_fn_values = 0usize;
            let mut known_lambda_values = 0usize;
            let mut escaping_unknown_values = 0usize;
            for clause in &function.clauses {
                collect_higher_order_expr_counts(
                    &clause.body,
                    &mut known_fn_values,
                    &mut known_lambda_values,
                    &mut escaping_unknown_values,
                );
            }
            let signature_has_higher_order = function
                .signature
                .ty
                .fun_parts()
                .0
                .iter()
                .any(|ty| matches!(ty, Type::Fun(_, _)))
                || matches!(function.signature.ty.fun_parts().1, Type::Fun(_, _));
            let expr_has_higher_order = function.clauses.iter().any(|clause| {
                typed_expr_contains_lambda_or_apply(&clause.body)
                    || typed_expr_contains_fun_local(&clause.body)
            });
            let rejection_reason = if signature_has_higher_order {
                Some("function signature remains higher-order after specialization".to_string())
            } else if expr_has_higher_order {
                Some("function body still contains higher-order forms".to_string())
            } else {
                None
            };
            let origin_text = origins
                .get(&function.name)
                .cloned()
                .unwrap_or_else(|| "source".to_string());
            let capture_count = parse_origin_metric(&origin_text, "captures=").unwrap_or(0);
            let env_bytes = parse_origin_metric(&origin_text, "env-bytes=").unwrap_or(0);
            let lambda_mode = if rejection_reason.is_some() {
                LambdaLoweringMode::Rejected
            } else if capture_count > 0
                || origin_text.contains("closure-converted")
                || origin_text.contains("lambda-lift")
            {
                LambdaLoweringMode::ClosureConverted
            } else {
                LambdaLoweringMode::DirectFirstOrder
            };
            WasmHigherOrderReport {
                function: function.name.clone(),
                specialization_origin: origin_text,
                lambda_mode,
                capture_count,
                env_bytes,
                known_fn_values,
                known_lambda_values,
                escaping_unknown_values,
                rejection_reason,
            }
        })
        .collect()
}

fn parse_origin_metric(origin: &str, key: &str) -> Option<usize> {
    let index = origin.find(key)?;
    let tail = &origin[index + key.len()..];
    let digits = tail
        .chars()
        .take_while(|ch| ch.is_ascii_digit())
        .collect::<String>();
    if digits.is_empty() {
        None
    } else {
        digits.parse::<usize>().ok()
    }
}

fn collect_higher_order_expr_counts(
    expr: &TypedExpr,
    known_fn_values: &mut usize,
    known_lambda_values: &mut usize,
    escaping_unknown_values: &mut usize,
) {
    match &expr.kind {
        TypedExprKind::FunctionRef { .. } => *known_fn_values += 1,
        TypedExprKind::ConstructorRef { .. } => *escaping_unknown_values += 1,
        TypedExprKind::Lambda { body, .. } => {
            *known_lambda_values += 1;
            collect_higher_order_expr_counts(
                body,
                known_fn_values,
                known_lambda_values,
                escaping_unknown_values,
            );
        }
        TypedExprKind::Apply { callee, arg } => {
            *escaping_unknown_values += 1;
            collect_higher_order_expr_counts(
                callee,
                known_fn_values,
                known_lambda_values,
                escaping_unknown_values,
            );
            collect_higher_order_expr_counts(
                arg,
                known_fn_values,
                known_lambda_values,
                escaping_unknown_values,
            );
        }
        TypedExprKind::SeqSplice { prefix, tail } => {
            for item in prefix {
                collect_higher_order_expr_counts(
                    item,
                    known_fn_values,
                    known_lambda_values,
                    escaping_unknown_values,
                );
            }
            collect_higher_order_expr_counts(
                tail,
                known_fn_values,
                known_lambda_values,
                escaping_unknown_values,
            );
        }
        TypedExprKind::Let { bindings, body } => {
            for binding in bindings {
                collect_higher_order_expr_counts(
                    &binding.expr,
                    known_fn_values,
                    known_lambda_values,
                    escaping_unknown_values,
                );
            }
            collect_higher_order_expr_counts(
                body,
                known_fn_values,
                known_lambda_values,
                escaping_unknown_values,
            );
        }
        TypedExprKind::Record(fields) => {
            for value in fields.values() {
                collect_higher_order_expr_counts(
                    value,
                    known_fn_values,
                    known_lambda_values,
                    escaping_unknown_values,
                );
            }
        }
        TypedExprKind::Seq(items) => {
            for item in items {
                collect_higher_order_expr_counts(
                    item,
                    known_fn_values,
                    known_lambda_values,
                    escaping_unknown_values,
                );
            }
        }
        TypedExprKind::Tuple(items) => {
            for item in items {
                collect_higher_order_expr_counts(
                    item,
                    known_fn_values,
                    known_lambda_values,
                    escaping_unknown_values,
                );
            }
        }
        TypedExprKind::Project { base, .. } | TypedExprKind::TupleProject { base, .. } => {
            collect_higher_order_expr_counts(
                base,
                known_fn_values,
                known_lambda_values,
                escaping_unknown_values,
            )
        }
        TypedExprKind::Index { base, index, .. } => {
            collect_higher_order_expr_counts(
                base,
                known_fn_values,
                known_lambda_values,
                escaping_unknown_values,
            );
            collect_higher_order_expr_counts(
                index,
                known_fn_values,
                known_lambda_values,
                escaping_unknown_values,
            );
        }
        TypedExprKind::RecordUpdate { base, fields } => {
            collect_higher_order_expr_counts(
                base,
                known_fn_values,
                known_lambda_values,
                escaping_unknown_values,
            );
            for value in fields.values() {
                collect_higher_order_expr_counts(
                    value,
                    known_fn_values,
                    known_lambda_values,
                    escaping_unknown_values,
                );
            }
        }
        TypedExprKind::Call { args, .. } => {
            for arg in args {
                collect_higher_order_expr_counts(
                    &arg.expr,
                    known_fn_values,
                    known_lambda_values,
                    escaping_unknown_values,
                );
            }
        }
        TypedExprKind::Local(_)
        | TypedExprKind::Int(_, _)
        | TypedExprKind::Float(_, _)
        | TypedExprKind::Bool(_)
        | TypedExprKind::Char(_)
        | TypedExprKind::String(_)
        | TypedExprKind::TypeToken(_) => {}
    }
}

fn typed_expr_apply_chain_root<'a>(expr: &'a TypedExpr) -> (&'a TypedExpr, Vec<&'a TypedExpr>) {
    let mut head = expr;
    let mut args = Vec::<&TypedExpr>::new();
    while let TypedExprKind::Apply { callee, arg } = &head.kind {
        args.push(arg.as_ref());
        head = callee.as_ref();
    }
    args.reverse();
    (head, args)
}

fn typed_expr_contains_lambda_or_apply(expr: &TypedExpr) -> bool {
    match &expr.kind {
        TypedExprKind::Lambda { .. } => true,
        TypedExprKind::Apply { .. } => {
            let (head, args) = typed_expr_apply_chain_root(expr);
            if matches!(head.kind, TypedExprKind::ConstructorRef { .. }) {
                args.iter()
                    .any(|arg| typed_expr_contains_lambda_or_apply(arg))
            } else {
                true
            }
        }
        TypedExprKind::Let { bindings, body } => {
            bindings
                .iter()
                .any(|binding| typed_expr_contains_lambda_or_apply(&binding.expr))
                || typed_expr_contains_lambda_or_apply(body)
        }
        TypedExprKind::Seq(items) | TypedExprKind::Tuple(items) => {
            items.iter().any(typed_expr_contains_lambda_or_apply)
        }
        TypedExprKind::SeqSplice { prefix, tail } => {
            prefix.iter().any(typed_expr_contains_lambda_or_apply)
                || typed_expr_contains_lambda_or_apply(tail)
        }
        TypedExprKind::Record(fields) => fields.values().any(typed_expr_contains_lambda_or_apply),
        TypedExprKind::Project { base, .. } | TypedExprKind::TupleProject { base, .. } => {
            typed_expr_contains_lambda_or_apply(base)
        }
        TypedExprKind::Index { base, index, .. } => {
            typed_expr_contains_lambda_or_apply(base) || typed_expr_contains_lambda_or_apply(index)
        }
        TypedExprKind::RecordUpdate { base, fields } => {
            typed_expr_contains_lambda_or_apply(base)
                || fields.values().any(typed_expr_contains_lambda_or_apply)
        }
        TypedExprKind::Call { args, .. } => args
            .iter()
            .any(|arg| typed_expr_contains_lambda_or_apply(&arg.expr)),
        TypedExprKind::Local(_)
        | TypedExprKind::FunctionRef { .. }
        | TypedExprKind::ConstructorRef { .. }
        | TypedExprKind::Int(_, _)
        | TypedExprKind::Float(_, _)
        | TypedExprKind::Bool(_)
        | TypedExprKind::Char(_)
        | TypedExprKind::String(_)
        | TypedExprKind::TypeToken(_) => false,
    }
}

fn typed_expr_contains_fun_local(expr: &TypedExpr) -> bool {
    match &expr.kind {
        TypedExprKind::Local(_) if matches!(expr.ty, Type::Fun(_, _)) => true,
        TypedExprKind::Let { bindings, body } => {
            bindings
                .iter()
                .any(|binding| typed_expr_contains_fun_local(&binding.expr))
                || typed_expr_contains_fun_local(body)
        }
        TypedExprKind::Seq(items) | TypedExprKind::Tuple(items) => {
            items.iter().any(typed_expr_contains_fun_local)
        }
        TypedExprKind::SeqSplice { prefix, tail } => {
            prefix.iter().any(typed_expr_contains_fun_local) || typed_expr_contains_fun_local(tail)
        }
        TypedExprKind::Record(fields) => fields.values().any(typed_expr_contains_fun_local),
        TypedExprKind::Project { base, .. } | TypedExprKind::TupleProject { base, .. } => {
            typed_expr_contains_fun_local(base)
        }
        TypedExprKind::Index { base, index, .. } => {
            typed_expr_contains_fun_local(base) || typed_expr_contains_fun_local(index)
        }
        TypedExprKind::RecordUpdate { base, fields } => {
            typed_expr_contains_fun_local(base)
                || fields.values().any(typed_expr_contains_fun_local)
        }
        TypedExprKind::Call { args, .. } => args
            .iter()
            .any(|arg| typed_expr_contains_fun_local(&arg.expr)),
        TypedExprKind::Apply { callee, arg } => {
            typed_expr_contains_fun_local(callee) || typed_expr_contains_fun_local(arg)
        }
        TypedExprKind::Lambda { body, .. } => typed_expr_contains_fun_local(body),
        TypedExprKind::Local(_)
        | TypedExprKind::FunctionRef { .. }
        | TypedExprKind::ConstructorRef { .. }
        | TypedExprKind::Int(_, _)
        | TypedExprKind::Float(_, _)
        | TypedExprKind::Bool(_)
        | TypedExprKind::Char(_)
        | TypedExprKind::String(_)
        | TypedExprKind::TypeToken(_) => false,
    }
}

fn ensure_backend_first_order_compatibility(reports: &[WasmHigherOrderReport]) -> Result<()> {
    if let Some(report) = reports
        .iter()
        .find(|report| report.rejection_reason.is_some())
    {
        return Err(SimdError::new(format!(
            "Wasm backend supports only fully specialized closure-free functions; '{}' rejected: {}",
            report.function,
            report
                .rejection_reason
                .as_deref()
                .unwrap_or("higher-order forms remain")
        )));
    }
    Ok(())
}

fn compile_wasm_artifact_checked(
    checked_program: &CheckedProgram,
    main: &str,
    origins: &BTreeMap<String, String>,
) -> Result<WasmArtifact> {
    compile_wasm_artifact_checked_with_options(
        checked_program,
        main,
        origins,
        WasmCompileOptions {
            function_profiler: false,
        },
    )
}

fn compile_wasm_artifact_checked_with_options(
    checked_program: &CheckedProgram,
    main: &str,
    origins: &BTreeMap<String, String>,
    options: WasmCompileOptions,
) -> Result<WasmArtifact> {
    let higher_order_reports = collect_higher_order_reports(checked_program, origins);
    ensure_backend_first_order_compatibility(&higher_order_reports)?;
    let wasm_enum_layouts = build_precomputed_wasm_enum_layouts(checked_program)?;
    let entry_result_type = checked_program
        .functions
        .iter()
        .find(|function| function.name == main)
        .map(|function| function.signature.ty.fun_parts().1)
        .ok_or_else(|| SimdError::new(format!("unknown entry function '{}'", main)))?;
    let mut plan = build_wasm_plan(checked_program, main)?;
    let normalized = normalize_records(&plan.checked)?;
    let lowered_program = optimize_lowered_program(&prepare_lowered_program(&normalized)?);
    let grouped_program = group_lowered_program(&normalized, &lowered_program)?;
    let intent_analysis = analyze_intents(&grouped_program);
    let structural_analysis = analyze_structural_clusters(&lowered_program);
    let structural_membership = structural_cluster_membership(&structural_analysis);
    let structural_candidates = lowered_program
        .functions
        .iter()
        .filter(|function| {
            matches!(function.kind, LoweredKind::Structural { .. }) || function.tail_loop.is_some()
        })
        .map(|function| function.name.clone())
        .collect::<BTreeSet<_>>();
    let lowered_map: BTreeMap<_, _> = lowered_program
        .functions
        .iter()
        .map(|function| (function.name.clone(), function))
        .collect();
    let mut intent_by_leaf = BTreeMap::<String, KernelIntentReport>::new();
    for group in &grouped_program.functions {
        if let Some(report) = intent_analysis
            .reports
            .iter()
            .find(|report| {
                report.source_name == group.source_name && report.leaf_paths == group.leaf_paths
            })
            .cloned()
        {
            for leaf in &group.leaves {
                intent_by_leaf.insert(leaf.name.clone(), report.clone());
            }
        }
    }
    let flat_param_abis = flatten_wasm_param_abis(&plan.params)?;
    let param_use_summaries = summarize_param_uses(&lowered_program);
    for leaf in &mut plan.leaf_exports {
        let summary = param_use_summaries.get(&leaf.export_name).ok_or_else(|| {
            SimdError::new(format!(
                "missing parameter-usage summary for leaf '{}'",
                leaf.export_name
            ))
        })?;
        let (used_param_leaves, reusable_param_leaf) =
            analyze_leaf_buffer_reuse(&flat_param_abis, &leaf.result, summary)?;
        leaf.used_param_leaves = used_param_leaves;
        leaf.reusable_param_leaf = reusable_param_leaf;
    }
    let imported_func_count = if options.function_profiler {
        WASM_PROFILE_IMPORT_COUNT
    } else {
        0
    };
    let function_profile_names = if options.function_profiler {
        plan.checked
            .functions
            .iter()
            .map(|function| function.name.clone())
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let mut function_indices = BTreeMap::<String, u32>::new();
    let mut scalar_indices = BTreeMap::<String, u32>::new();
    for (index, function) in plan.checked.functions.iter().enumerate() {
        let wasm_index = index as u32 + imported_func_count;
        function_indices.insert(function.name.clone(), wasm_index);
        if matches!(
            lowered_map
                .get(&function.name)
                .map(|function| &function.kind),
            Some(LoweredKind::Scalar { .. } | LoweredKind::Structural { .. })
        ) {
            scalar_indices.insert(function.name.clone(), wasm_index);
        }
    }

    let signatures = plan
        .checked
        .functions
        .iter()
        .map(|function| (function.name.clone(), function))
        .collect::<BTreeMap<_, _>>();

    let mut module = Module::new();
    let mut types = TypeSection::new();
    let mut import_section = ImportSection::new();
    let mut function_section = FunctionSection::new();
    let mut memory_section = MemorySection::new();
    let mut export_section = ExportSection::new();
    let mut code_section = CodeSection::new();
    let mut optimizer_reports = Vec::<WasmOptimizationReport>::new();

    if options.function_profiler {
        let hook_type = add_function_type(&mut types, &[ValType::I32], None);
        import_section.import("simd_profile", "enter", EntityType::Function(hook_type));
        import_section.import("simd_profile", "exit", EntityType::Function(hook_type));
    }

    for function in &plan.checked.functions {
        let lowered = lowered_map.get(&function.name).copied().ok_or_else(|| {
            SimdError::new(format!("missing lowered function '{}'", function.name))
        })?;
        let (params, result) = match lowered.kind {
            LoweredKind::Scalar { .. } | LoweredKind::Structural { .. } => {
                scalar_signature(function)?
            }
            LoweredKind::Kernel { .. } => entry_signature(function)?,
        };
        let type_index = add_function_type(&mut types, &params, result.as_ref());
        function_section.function(type_index);
    }

    memory_section.memory(MemoryType {
        minimum: 1,
        maximum: None,
        memory64: false,
        shared: false,
        page_size_log2: None,
    });
    export_section.export("memory", ExportKind::Memory, 0);
    for leaf in &plan.leaf_exports {
        let index = *function_indices.get(&leaf.export_name).ok_or_else(|| {
            SimdError::new(format!("missing function index for '{}'", leaf.export_name))
        })?;
        export_section.export(&leaf.export_name, ExportKind::Func, index);
    }

    for function in &plan.checked.functions {
        let lowered = lowered_map.get(&function.name).copied().ok_or_else(|| {
            SimdError::new(format!("missing lowered function '{}'", function.name))
        })?;
        let function_profile = options
            .function_profiler
            .then_some(WasmFunctionInstrumentation {
                func_id: plan
                    .checked
                    .functions
                    .iter()
                    .position(|candidate| candidate.name == function.name)
                    .unwrap_or(0) as u32,
                enter_import_index: WASM_PROFILE_ENTER_IMPORT_INDEX,
                exit_import_index: WASM_PROFILE_EXIT_IMPORT_INDEX,
                allow_return_call: false,
            });
        let structural_cluster = structural_membership.get(&function.name);
        let intent = intent_by_leaf
            .get(&function.name)
            .map(|report| report.intent.clone())
            .unwrap_or_else(|| {
                if matches!(lowered.kind, LoweredKind::Structural { .. })
                    || structural_cluster.is_some()
                    || lowered_calls_structural_candidate(lowered, &structural_candidates)
                {
                    IntentClass::Structural
                } else if lowered.tail_loop.is_some() {
                    IntentClass::ScalarTailRec
                } else {
                    IntentClass::Fallback
                }
            });
        let (
            structural_scc,
            structural_state_count,
            structural_transition_count,
            structural_span_ops,
            structural_enum_ops,
            structural_region_count,
            structural_char_prefix_regions,
            structural_separated_item_regions,
        ) = structural_report_fields(lowered, structural_cluster);
        let wasm_function = match lowered.kind {
            LoweredKind::Scalar { .. } => {
                optimizer_reports.push(WasmOptimizationReport {
                    function: function.name.clone(),
                    intent: if intent == IntentClass::Structural {
                        IntentClass::Structural
                    } else if lowered.tail_loop.is_some() {
                        IntentClass::ScalarTailRec
                    } else {
                        intent
                    },
                    structural_exec: StructuralExecMode::Scalar,
                    vectorizable: false,
                    vector_unroll: 0,
                    fallback_reason: None,
                    structural_scc,
                    structural_state_count,
                    structural_transition_count,
                    structural_span_ops,
                    structural_enum_ops,
                    structural_region_count,
                    structural_char_prefix_regions,
                    structural_separated_item_regions,
                });
                compile_scalar_function(
                    lowered,
                    function,
                    &plan.checked.enum_ctors,
                    &wasm_enum_layouts,
                    &scalar_indices,
                    &signatures,
                    function_profile,
                )?
            }
            LoweredKind::Structural { .. } => {
                optimizer_reports.push(WasmOptimizationReport {
                    function: function.name.clone(),
                    intent: IntentClass::Structural,
                    structural_exec: structural_exec_mode_for(lowered, function),
                    vectorizable: false,
                    vector_unroll: 0,
                    fallback_reason: None,
                    structural_scc,
                    structural_state_count,
                    structural_transition_count,
                    structural_span_ops,
                    structural_enum_ops,
                    structural_region_count,
                    structural_char_prefix_regions,
                    structural_separated_item_regions,
                });
                compile_structural_function(
                    lowered,
                    function,
                    &lowered_map,
                    &plan.checked.enum_ctors,
                    &wasm_enum_layouts,
                    &scalar_indices,
                    &signatures,
                    function_profile,
                )?
            }
            LoweredKind::Kernel { .. } => {
                let compiled = compile_kernel_entry(
                    lowered,
                    function,
                    &lowered_map,
                    &scalar_indices,
                    &signatures,
                    intent,
                    function_profile,
                )?;
                optimizer_reports.push(compiled.report);
                compiled.function
            }
        };
        code_section.function(&wasm_function);
    }

    let mut next_function_index = plan.checked.functions.len() as u32 + imported_func_count;
    let mut grouped_kernel_calls = Vec::<GroupedKernelCall>::new();
    for group in grouped_program
        .functions
        .iter()
        .filter(|group| group.source_name == main)
    {
        let Some(compiled_group) = compile_grouped_kernel_function(
            group,
            &lowered_map,
            &scalar_indices,
            &signatures,
            intent_analysis
                .reports
                .iter()
                .find(|report| {
                    report.source_name == group.source_name && report.leaf_paths == group.leaf_paths
                })
                .map(|report| report.intent.clone())
                .unwrap_or(IntentClass::GroupedMap),
        )?
        else {
            continue;
        };
        let type_index = add_function_type(
            &mut types,
            &compiled_group.signature.0,
            compiled_group.signature.1.as_ref(),
        );
        function_section.function(type_index);
        let func_index = next_function_index;
        next_function_index += 1;
        code_section.function(&compiled_group.function);
        optimizer_reports.push(compiled_group.report);
        grouped_kernel_calls.push(GroupedKernelCall {
            leaf_paths: compiled_group.leaf_paths,
            func_index,
        });
    }

    let grouped_export = if matches!(plan.result, WasmResultAbi::Record { .. })
        && !plan.leaf_exports.is_empty()
    {
        let (grouped_function, grouped_signature) =
            compile_grouped_record_export(&plan, main, &function_indices, &grouped_kernel_calls)?;
        let type_index = add_function_type(
            &mut types,
            &grouped_signature.0,
            grouped_signature.1.as_ref(),
        );
        function_section.function(type_index);
        let grouped_index = next_function_index;
        function_indices.insert(main.to_string(), grouped_index);
        export_section.export(main, ExportKind::Func, grouped_index);
        code_section.function(&grouped_function);
        Some(WasmGroupedExport {
            export_name: main.to_string(),
            leaf_exports: plan.leaf_exports.clone(),
        })
    } else {
        None
    };

    module.section(&types);
    if options.function_profiler {
        module.section(&import_section);
    }
    module.section(&function_section);
    module.section(&memory_section);
    module.section(&export_section);
    module.section(&code_section);

    Ok(WasmArtifact {
        bytes: module.finish(),
        export_name: main.to_string(),
        params: plan.params,
        result: plan.result,
        result_type: entry_result_type,
        enum_ctors: checked_program.enum_ctors.clone(),
        wasm_enum_layouts,
        grouped_export,
        leaf_exports: plan.leaf_exports,
        optimizer_reports,
        higher_order_reports,
        function_profile_names,
    })
}

fn compile_grouped_record_export(
    plan: &WasmPlan,
    main: &str,
    function_indices: &BTreeMap<String, u32>,
    grouped_kernel_calls: &[GroupedKernelCall],
) -> Result<(Function, (Vec<ValType>, Option<ValType>))> {
    let flat_input_abis = flatten_wasm_param_abis(&plan.params)?;
    let mut params = Vec::<ValType>::new();
    for abi in &flat_input_abis {
        match abi {
            WasmParamAbi::Scalar { prim } => params.push(wasm_val_type(*prim)),
            WasmParamAbi::Bulk { .. } => {
                params.push(ValType::I32);
                params.push(ValType::I32);
            }
            WasmParamAbi::Record { .. } => {
                return Err(SimdError::new(
                    "grouped Wasm export should not contain record ABIs",
                ));
            }
        }
    }
    let input_param_count = params.len() as u32;
    let mut output_param_indices = Vec::<GroupOutputParam>::new();
    for leaf in &plan.leaf_exports {
        match &leaf.result {
            WasmLeafResultAbi::Scalar { prim } => {
                output_param_indices.push(GroupOutputParam::Scalar {
                    leaf_path: leaf.leaf_path.clone(),
                    prim: *prim,
                    ptr_index: params.len() as u32,
                    ptr_local: 0,
                    result_local: 0,
                });
                params.push(ValType::I32);
            }
            WasmLeafResultAbi::Bulk { .. } => {
                output_param_indices.push(GroupOutputParam::Bulk {
                    leaf_path: leaf.leaf_path.clone(),
                    ptr_index: params.len() as u32,
                    len_index: params.len() as u32 + 1,
                    ptr_local: 0,
                    len_local: 0,
                });
                params.push(ValType::I32);
                params.push(ValType::I32);
            }
            WasmLeafResultAbi::StarSeq { .. } => {
                return Err(SimdError::new(
                    "grouped Wasm export does not yet support T[*] leaf outputs",
                ));
            }
        }
    }

    let mut output_local_decls = Vec::<(u32, ValType)>::new();
    let mut next_output_local = params.len() as u32;
    for output in &mut output_param_indices {
        match output {
            GroupOutputParam::Scalar {
                prim,
                ptr_local,
                result_local,
                ..
            } => {
                *ptr_local = next_output_local;
                next_output_local += 1;
                *result_local = next_output_local;
                next_output_local += 1;
                output_local_decls.push((1, ValType::I32));
                output_local_decls.push((1, wasm_val_type(*prim)));
            }
            GroupOutputParam::Bulk {
                ptr_local,
                len_local,
                ..
            } => {
                *ptr_local = next_output_local;
                next_output_local += 1;
                *len_local = next_output_local;
                next_output_local += 1;
                output_local_decls.push((1, ValType::I32));
                output_local_decls.push((1, ValType::I32));
            }
        }
    }

    let mut function = Function::new(output_local_decls);
    let mut covered_leaves = BTreeSet::<LeafPath>::new();
    for grouped_call in grouped_kernel_calls {
        for local_index in 0..input_param_count {
            function.instruction(&Instruction::LocalGet(local_index));
        }
        for leaf_path in &grouped_call.leaf_paths {
            let output = output_param_indices
                .iter()
                .find(|output| output.leaf_path() == leaf_path)
                .ok_or_else(|| {
                    SimdError::new(format!(
                        "missing grouped output metadata for leaf {:?} in '{}'",
                        leaf_path, main
                    ))
                })?;
            match output {
                GroupOutputParam::Scalar { ptr_index, .. } => {
                    function.instruction(&Instruction::LocalGet(*ptr_index));
                }
                GroupOutputParam::Bulk {
                    ptr_index,
                    len_index,
                    ..
                } => {
                    function.instruction(&Instruction::LocalGet(*ptr_index));
                    function.instruction(&Instruction::LocalGet(*len_index));
                }
            }
            covered_leaves.insert(leaf_path.clone());
        }
        function.instruction(&Instruction::Call(grouped_call.func_index));
    }

    for leaf in &plan.leaf_exports {
        if covered_leaves.contains(&leaf.leaf_path) {
            continue;
        }
        let index = *function_indices.get(&leaf.export_name).ok_or_else(|| {
            SimdError::new(format!(
                "missing leaf function index for grouped export '{}'",
                leaf.export_name
            ))
        })?;
        let output = output_param_indices
            .iter()
            .find(|output| output.leaf_path() == &leaf.leaf_path)
            .ok_or_else(|| {
                SimdError::new(format!(
                    "missing grouped output metadata for leaf {:?} in '{}'",
                    leaf.leaf_path, main
                ))
            })?;
        match output {
            GroupOutputParam::Scalar {
                ptr_index,
                prim,
                ptr_local,
                result_local,
                ..
            } => {
                function.instruction(&Instruction::LocalGet(*ptr_index));
                function.instruction(&Instruction::LocalSet(*ptr_local));
                for local_index in 0..input_param_count {
                    function.instruction(&Instruction::LocalGet(local_index));
                }
                function.instruction(&Instruction::Call(index));
                function.instruction(&Instruction::LocalSet(*result_local));
                function.instruction(&Instruction::LocalGet(*ptr_local));
                function.instruction(&Instruction::LocalGet(*result_local));
                emit_scalar_store(&mut function, *prim);
            }
            GroupOutputParam::Bulk {
                ptr_index,
                len_index,
                ptr_local,
                len_local,
                ..
            } => {
                function.instruction(&Instruction::LocalGet(*ptr_index));
                function.instruction(&Instruction::LocalSet(*ptr_local));
                function.instruction(&Instruction::LocalGet(*len_index));
                function.instruction(&Instruction::LocalSet(*len_local));
                for local_index in 0..input_param_count {
                    function.instruction(&Instruction::LocalGet(local_index));
                }
                function.instruction(&Instruction::LocalGet(*ptr_local));
                function.instruction(&Instruction::LocalGet(*len_local));
                function.instruction(&Instruction::Call(index));
            }
        }
    }

    function.instruction(&Instruction::End);
    Ok((function, (params, None)))
}

#[derive(Debug, Clone)]
struct GroupedKernelCall {
    leaf_paths: Vec<LeafPath>,
    func_index: u32,
}

#[derive(Debug)]
struct CompiledKernelEntry {
    function: Function,
    report: WasmOptimizationReport,
}

#[derive(Debug)]
struct CompiledGroupedKernel {
    leaf_paths: Vec<LeafPath>,
    function: Function,
    signature: (Vec<ValType>, Option<ValType>),
    report: WasmOptimizationReport,
}

#[derive(Debug, Clone)]
struct PreparedGroupedLeaf {
    leaf_path: LeafPath,
    result_prim: Prim,
    clauses: Vec<LoweredClause>,
}

#[derive(Debug, Clone, Copy)]
struct GroupedKernelOutput {
    prim: Prim,
    ptr_local: u32,
    len_local: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum VectorPlan {
    ScalarOnly,
    VectorSingle,
    VectorUnroll2,
    VectorUnroll4,
}

#[derive(Debug, Clone, Copy)]
enum VectorAddressing {
    Pointer { lane_offset_elems: u32 },
}

fn vector_plan_unroll(plan: VectorPlan) -> usize {
    match plan {
        VectorPlan::ScalarOnly => 0,
        VectorPlan::VectorSingle => 1,
        VectorPlan::VectorUnroll2 => 2,
        VectorPlan::VectorUnroll4 => 4,
    }
}

fn choose_vector_plan(
    has_vector_clause: bool,
    vector_width: usize,
    result_prim: Prim,
    op_count: usize,
    load_streams: usize,
    store_streams: usize,
) -> (VectorPlan, Option<String>) {
    if !has_vector_clause {
        return (
            VectorPlan::ScalarOnly,
            Some("non-vectorizable clause".to_string()),
        );
    }
    let total_streams = load_streams.saturating_add(store_streams);
    let stream_weight = total_streams.max(1);
    let weighted_ops = op_count.saturating_mul(stream_weight);
    let vec4_ready = match result_prim {
        Prim::I32 | Prim::Char | Prim::F32 => weighted_ops >= 18 && total_streams <= 8,
        Prim::I64 | Prim::F64 => weighted_ops >= 28 && total_streams <= 5,
    };
    let vec2_ready = match result_prim {
        Prim::I32 | Prim::Char | Prim::F32 => weighted_ops >= 8 || op_count >= 6,
        Prim::I64 | Prim::F64 => weighted_ops >= 10 || op_count >= 7,
    };

    if vector_width >= 2 && vec4_ready {
        (VectorPlan::VectorUnroll4, None)
    } else if vector_width >= 2 && vec2_ready {
        (VectorPlan::VectorUnroll2, None)
    } else {
        (VectorPlan::VectorSingle, None)
    }
}

#[derive(Debug, Clone)]
enum GroupOutputParam {
    Scalar {
        leaf_path: LeafPath,
        prim: Prim,
        ptr_index: u32,
        ptr_local: u32,
        result_local: u32,
    },
    Bulk {
        leaf_path: LeafPath,
        ptr_index: u32,
        len_index: u32,
        ptr_local: u32,
        len_local: u32,
    },
}

impl GroupOutputParam {
    fn leaf_path(&self) -> &LeafPath {
        match self {
            Self::Scalar { leaf_path, .. } | Self::Bulk { leaf_path, .. } => leaf_path,
        }
    }
}

fn compile_grouped_kernel_function(
    group: &GroupedLoweredFunction,
    lowered_map: &BTreeMap<String, &LoweredFunction>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    intent: IntentClass,
) -> Result<Option<CompiledGroupedKernel>> {
    let GroupedLoweredKind::Kernel { vector_width, .. } = &group.kind else {
        return Ok(None);
    };
    if group.leaves.len() <= 1 || group.tail_loop.is_some() {
        return Ok(None);
    }

    let mut prepared_leaves = Vec::with_capacity(group.leaves.len());
    for (leaf_path, leaf) in group.leaf_paths.iter().zip(&group.leaves) {
        let prim = leaf.result.prim().ok_or_else(|| {
            SimdError::new(format!(
                "grouped kernel leaf '{}' does not have a primitive result",
                leaf.name
            ))
        })?;
        let clauses = match &leaf.kind {
            LoweredKind::Kernel { clauses, .. } => clauses
                .iter()
                .map(|clause| {
                    let body = inline_grouped_kernel_expr(
                        &clause.body,
                        lowered_map,
                        &mut BTreeSet::new(),
                    )?;
                    if expr_contains_non_scalar_calls(&body, scalar_indices) {
                        return Ok(None);
                    }
                    Ok(Some(LoweredClause {
                        patterns: clause.patterns.clone(),
                        body,
                    }))
                })
                .collect::<Result<Option<Vec<_>>>>()?,
            LoweredKind::Scalar { .. } | LoweredKind::Structural { .. } => None,
        };
        let Some(clauses) = clauses else {
            return Ok(None);
        };
        prepared_leaves.push(PreparedGroupedLeaf {
            leaf_path: leaf_path.clone(),
            result_prim: prim,
            clauses,
        });
    }

    let reference = prepared_leaves
        .first()
        .ok_or_else(|| SimdError::new("grouped kernel had no leaves"))?;
    let reference_clause = reference
        .clauses
        .first()
        .ok_or_else(|| SimdError::new("grouped kernel had no clauses"))?;
    for leaf in prepared_leaves.iter().skip(1) {
        if leaf.clauses.len() != reference.clauses.len() {
            return Ok(None);
        }
        for (left, right) in reference.clauses.iter().zip(&leaf.clauses) {
            if left.patterns != right.patterns {
                return Ok(None);
            }
        }
    }

    let mut abi_params = Vec::<KernelParam>::new();
    let mut wasm_params = Vec::<ValType>::new();
    let mut wasm_param_index = 0u32;
    for (access, pattern) in group.param_access.iter().zip(&reference_clause.patterns) {
        match access {
            AccessKind::Same => {
                if let Some(prim) = wasm_star_seq_storage_prim(&pattern.ty) {
                    abi_params.push(KernelParam::SameSeq {
                        prim,
                        ptr_local: wasm_param_index,
                        len_local: wasm_param_index + 1,
                    });
                    wasm_params.push(ValType::I32);
                    wasm_params.push(ValType::I32);
                    wasm_param_index += 2;
                } else {
                    let prim = pattern.ty.prim().ok_or_else(|| {
                        SimdError::new(
                            "grouped kernel parameter pattern did not have a primitive type",
                        )
                    })?;
                    abi_params.push(KernelParam::Same {
                        prim,
                        value_local: wasm_param_index,
                    });
                    wasm_params.push(wasm_val_type(prim));
                    wasm_param_index += 1;
                }
            }
            AccessKind::Lane => {
                let prim = pattern.ty.prim().ok_or_else(|| {
                    SimdError::new("grouped kernel lane parameter did not have a primitive type")
                })?;
                abi_params.push(KernelParam::Lane {
                    prim,
                    ptr_local: wasm_param_index,
                    len_local: wasm_param_index + 1,
                });
                wasm_params.push(ValType::I32);
                wasm_params.push(ValType::I32);
                wasm_param_index += 2;
            }
        }
    }

    let mut outputs = Vec::with_capacity(prepared_leaves.len());
    for leaf in &prepared_leaves {
        outputs.push(GroupedKernelOutput {
            prim: leaf.result_prim,
            ptr_local: wasm_param_index,
            len_local: wasm_param_index + 1,
        });
        wasm_params.push(ValType::I32);
        wasm_params.push(ValType::I32);
        wasm_param_index += 2;
    }

    let mut local_decls = vec![
        (1, ValType::I32),
        (1, ValType::I32),
        (1, ValType::I32),
        (1, ValType::I32),
    ];
    let index_local = wasm_param_index;
    let vector_end_local = wasm_param_index + 1;
    let vector_unrolled_end_local = wasm_param_index + 2;
    let vector_unrolled4_end_local = wasm_param_index + 3;
    let mut next_local = wasm_param_index + 4;

    let mut lane_locals = Vec::<Option<u32>>::new();
    let mut lane_ptr_locals = Vec::<Option<u32>>::new();
    for param in &abi_params {
        match param {
            KernelParam::Same { .. } | KernelParam::SameSeq { .. } => {
                lane_locals.push(None);
                lane_ptr_locals.push(None);
            }
            KernelParam::Lane { prim, .. } => {
                local_decls.push((1, wasm_val_type(*prim)));
                lane_locals.push(Some(next_local));
                next_local += 1;
                local_decls.push((1, ValType::I32));
                lane_ptr_locals.push(Some(next_local));
                next_local += 1;
            }
        }
    }
    let mut output_ptr_loop_locals = Vec::with_capacity(outputs.len());
    for _ in &outputs {
        local_decls.push((1, ValType::I32));
        output_ptr_loop_locals.push(next_local);
        next_local += 1;
    }
    let enum_state = append_enum_scratch_locals(&mut local_decls, &mut next_local);

    let first_len_local = abi_params
        .iter()
        .find_map(|param| match param {
            KernelParam::Lane { len_local, .. } => Some(*len_local),
            KernelParam::Same { .. } | KernelParam::SameSeq { .. } => None,
        })
        .ok_or_else(|| SimdError::new("grouped kernel requires at least one bulk parameter"))?;

    let vector_clause = vectorizable_grouped_kernel_clause(&prepared_leaves, &abi_params);
    let op_count = prepared_leaves
        .iter()
        .flat_map(|leaf| leaf.clauses.iter())
        .map(|clause| count_primitive_ops(&clause.body))
        .sum::<usize>();
    let load_streams = abi_params
        .iter()
        .filter(|param| matches!(param, KernelParam::Lane { .. }))
        .count();
    let (vector_plan, fallback_reason) = choose_vector_plan(
        vector_clause.is_some(),
        *vector_width,
        reference.result_prim,
        op_count,
        load_streams,
        outputs.len(),
    );
    let vector_acc_local = if vector_clause.is_some() {
        let local = next_local;
        local_decls.push((1, ValType::V128));
        next_local += 1;
        Some(local)
    } else {
        None
    };
    let variant_locals = kernel_variant_locals(&reference.clauses, &abi_params);
    let available_locals = kernel_available_locals(&reference.clauses, &abi_params);
    let vector_hoists = vector_clause
        .as_ref()
        .map(|vector_clause| {
            if vector_clause.clauses.len() == 1 {
                collect_hoisted_exprs(
                    vector_clause.clauses[0].bodies.iter().copied(),
                    &available_locals,
                    &variant_locals,
                    HoistMode::Vector,
                )
            } else {
                Vec::new()
            }
        })
        .unwrap_or_default();
    for _ in &vector_hoists {
        local_decls.push((1, ValType::V128));
    }
    let cleanup_hoists = collect_hoisted_exprs(
        prepared_leaves
            .iter()
            .flat_map(|leaf| leaf.clauses.iter().map(|clause| &clause.body)),
        &available_locals,
        &variant_locals,
        HoistMode::ScalarCleanup,
    );
    for hoisted in &cleanup_hoists {
        let prim = hoisted.expr.ty.prim().ok_or_else(|| {
            SimdError::new("cleanup hoisted expression did not have a scalar type")
        })?;
        local_decls.push((1, wasm_val_type(prim)));
    }

    let mut function = Function::new(local_decls);
    let mut next_hoisted_local = next_local;
    let vector_hoisted_locals = vector_hoists
        .iter()
        .map(|hoisted| {
            let local = next_hoisted_local;
            next_hoisted_local += 1;
            (hoisted.key.clone(), local)
        })
        .collect::<BTreeMap<_, _>>();
    let cleanup_hoisted_locals = cleanup_hoists
        .iter()
        .map(|hoisted| {
            let local = next_hoisted_local;
            next_hoisted_local += 1;
            (hoisted.key.clone(), local)
        })
        .collect::<BTreeMap<_, _>>();

    for param in &abi_params {
        if let KernelParam::Lane { len_local, .. } = param {
            if *len_local == first_len_local {
                continue;
            }
            function.instruction(&Instruction::LocalGet(*len_local));
            function.instruction(&Instruction::LocalGet(first_len_local));
            function.instruction(&Instruction::I32Ne);
            function.instruction(&Instruction::If(BlockType::Empty));
            function.instruction(&Instruction::Unreachable);
            function.instruction(&Instruction::End);
        }
    }
    for output in &outputs {
        function.instruction(&Instruction::LocalGet(output.len_local));
        function.instruction(&Instruction::LocalGet(first_len_local));
        function.instruction(&Instruction::I32Ne);
        function.instruction(&Instruction::If(BlockType::Empty));
        function.instruction(&Instruction::Unreachable);
        function.instruction(&Instruction::End);
    }

    function.instruction(&Instruction::I32Const(0));
    function.instruction(&Instruction::LocalSet(index_local));

    for (param, loop_ptr_local) in abi_params.iter().zip(&lane_ptr_locals) {
        if let (KernelParam::Lane { ptr_local, .. }, Some(loop_ptr_local)) = (param, loop_ptr_local)
        {
            function.instruction(&Instruction::LocalGet(*ptr_local));
            function.instruction(&Instruction::LocalSet(*loop_ptr_local));
        }
    }
    for (output, output_ptr_loop_local) in outputs.iter().zip(&output_ptr_loop_locals) {
        function.instruction(&Instruction::LocalGet(output.ptr_local));
        function.instruction(&Instruction::LocalSet(*output_ptr_loop_local));
    }

    let vector_params = abi_params
        .iter()
        .zip(&lane_ptr_locals)
        .map(|(param, ptr_local)| match (param, ptr_local) {
            (KernelParam::Same { prim, value_local }, _) => KernelParam::Same {
                prim: *prim,
                value_local: *value_local,
            },
            (
                KernelParam::SameSeq {
                    prim,
                    ptr_local,
                    len_local,
                },
                _,
            ) => KernelParam::SameSeq {
                prim: *prim,
                ptr_local: *ptr_local,
                len_local: *len_local,
            },
            (
                KernelParam::Lane {
                    prim, len_local, ..
                },
                Some(loop_ptr_local),
            ) => KernelParam::Lane {
                prim: *prim,
                ptr_local: *loop_ptr_local,
                len_local: *len_local,
            },
            (KernelParam::Lane { .. }, None) => *param,
        })
        .collect::<Vec<_>>();

    if let Some(vector_clause) = &vector_clause {
        emit_vector_hoists(
            &mut function,
            &vector_hoists,
            &vector_hoisted_locals,
            &vector_params,
            VectorAddressing::Pointer {
                lane_offset_elems: 0,
            },
            scalar_indices,
            &vector_clause
                .clauses
                .first()
                .ok_or_else(|| SimdError::new("vectorized grouped kernel had no clauses"))?
                .locals,
        )?;
        function.instruction(&Instruction::LocalGet(first_len_local));
        function.instruction(&Instruction::LocalGet(first_len_local));
        function.instruction(&Instruction::I32Const(*vector_width as i32));
        function.instruction(&Instruction::I32RemU);
        function.instruction(&Instruction::I32Sub);
        function.instruction(&Instruction::LocalSet(vector_end_local));

        if matches!(vector_plan, VectorPlan::VectorUnroll4) {
            function.instruction(&Instruction::LocalGet(vector_end_local));
            function.instruction(&Instruction::LocalGet(vector_end_local));
            function.instruction(&Instruction::I32Const((*vector_width as i32) * 4));
            function.instruction(&Instruction::I32RemU);
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::LocalSet(vector_unrolled4_end_local));

            function.instruction(&Instruction::Block(BlockType::Empty));
            function.instruction(&Instruction::Loop(BlockType::Empty));
            function.instruction(&Instruction::LocalGet(index_local));
            function.instruction(&Instruction::LocalGet(vector_unrolled4_end_local));
            function.instruction(&Instruction::I32GeU);
            function.instruction(&Instruction::BrIf(1));

            for chunk in 0..4u32 {
                for (output_index, (output, output_ptr_loop_local)) in
                    outputs.iter().zip(&output_ptr_loop_locals).enumerate()
                {
                    function.instruction(&Instruction::LocalGet(*output_ptr_loop_local));
                    compile_grouped_vectorized_clause_chain(
                        &mut function,
                        &vector_clause.clauses,
                        output_index,
                        &vector_params,
                        VectorAddressing::Pointer {
                            lane_offset_elems: *vector_width as u32 * chunk,
                        },
                        scalar_indices,
                        Some(&vector_hoisted_locals),
                        vector_acc_local.ok_or_else(|| {
                            SimdError::new("vectorized grouped kernel missing accumulator local")
                        })?,
                    )?;
                    function.instruction(&Instruction::V128Store(memarg(
                        u64::from(byte_width(output.prim)) * (*vector_width as u64) * chunk as u64,
                        4,
                    )));
                }
            }

            for param in &vector_params {
                if let KernelParam::Lane {
                    prim, ptr_local, ..
                } = param
                {
                    emit_pointer_bump(
                        &mut function,
                        *ptr_local,
                        byte_width(*prim) * (*vector_width as u32) * 4,
                    );
                }
            }
            for (output, output_ptr_loop_local) in outputs.iter().zip(&output_ptr_loop_locals) {
                emit_pointer_bump(
                    &mut function,
                    *output_ptr_loop_local,
                    byte_width(output.prim) * (*vector_width as u32) * 4,
                );
            }

            function.instruction(&Instruction::LocalGet(index_local));
            function.instruction(&Instruction::I32Const((*vector_width as i32) * 4));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(index_local));
            function.instruction(&Instruction::Br(0));
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);
        }

        if matches!(
            vector_plan,
            VectorPlan::VectorUnroll2 | VectorPlan::VectorUnroll4
        ) {
            function.instruction(&Instruction::LocalGet(vector_end_local));
            function.instruction(&Instruction::LocalGet(vector_end_local));
            function.instruction(&Instruction::I32Const((*vector_width as i32) * 2));
            function.instruction(&Instruction::I32RemU);
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::LocalSet(vector_unrolled_end_local));

            function.instruction(&Instruction::Block(BlockType::Empty));
            function.instruction(&Instruction::Loop(BlockType::Empty));
            function.instruction(&Instruction::LocalGet(index_local));
            function.instruction(&Instruction::LocalGet(vector_unrolled_end_local));
            function.instruction(&Instruction::I32GeU);
            function.instruction(&Instruction::BrIf(1));

            for chunk in 0..2u32 {
                for (output_index, (output, output_ptr_loop_local)) in
                    outputs.iter().zip(&output_ptr_loop_locals).enumerate()
                {
                    function.instruction(&Instruction::LocalGet(*output_ptr_loop_local));
                    compile_grouped_vectorized_clause_chain(
                        &mut function,
                        &vector_clause.clauses,
                        output_index,
                        &vector_params,
                        VectorAddressing::Pointer {
                            lane_offset_elems: *vector_width as u32 * chunk,
                        },
                        scalar_indices,
                        Some(&vector_hoisted_locals),
                        vector_acc_local.ok_or_else(|| {
                            SimdError::new("vectorized grouped kernel missing accumulator local")
                        })?,
                    )?;
                    function.instruction(&Instruction::V128Store(memarg(
                        u64::from(byte_width(output.prim)) * (*vector_width as u64) * chunk as u64,
                        4,
                    )));
                }
            }

            for param in &vector_params {
                if let KernelParam::Lane {
                    prim, ptr_local, ..
                } = param
                {
                    emit_pointer_bump(
                        &mut function,
                        *ptr_local,
                        byte_width(*prim) * (*vector_width as u32) * 2,
                    );
                }
            }
            for (output, output_ptr_loop_local) in outputs.iter().zip(&output_ptr_loop_locals) {
                emit_pointer_bump(
                    &mut function,
                    *output_ptr_loop_local,
                    byte_width(output.prim) * (*vector_width as u32) * 2,
                );
            }

            function.instruction(&Instruction::LocalGet(index_local));
            function.instruction(&Instruction::I32Const((*vector_width as i32) * 2));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(index_local));
            function.instruction(&Instruction::Br(0));
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);
        }

        if !matches!(vector_plan, VectorPlan::ScalarOnly) {
            function.instruction(&Instruction::Block(BlockType::Empty));
            function.instruction(&Instruction::Loop(BlockType::Empty));
            function.instruction(&Instruction::LocalGet(index_local));
            function.instruction(&Instruction::LocalGet(vector_end_local));
            function.instruction(&Instruction::I32GeU);
            function.instruction(&Instruction::BrIf(1));

            for (output_index, output_ptr_loop_local) in output_ptr_loop_locals.iter().enumerate() {
                function.instruction(&Instruction::LocalGet(*output_ptr_loop_local));
                compile_grouped_vectorized_clause_chain(
                    &mut function,
                    &vector_clause.clauses,
                    output_index,
                    &vector_params,
                    VectorAddressing::Pointer {
                        lane_offset_elems: 0,
                    },
                    scalar_indices,
                    Some(&vector_hoisted_locals),
                    vector_acc_local.ok_or_else(|| {
                        SimdError::new("vectorized grouped kernel missing accumulator local")
                    })?,
                )?;
                function.instruction(&Instruction::V128Store(memarg(0, 4)));
            }

            for param in &vector_params {
                if let KernelParam::Lane {
                    prim, ptr_local, ..
                } = param
                {
                    emit_pointer_bump(
                        &mut function,
                        *ptr_local,
                        byte_width(*prim) * (*vector_width as u32),
                    );
                }
            }
            for (output, output_ptr_loop_local) in outputs.iter().zip(&output_ptr_loop_locals) {
                emit_pointer_bump(
                    &mut function,
                    *output_ptr_loop_local,
                    byte_width(output.prim) * (*vector_width as u32),
                );
            }

            function.instruction(&Instruction::LocalGet(index_local));
            function.instruction(&Instruction::I32Const(*vector_width as i32));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(index_local));
            function.instruction(&Instruction::Br(0));
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);
        }
    }

    let scalar_invariant_locals = kernel_same_locals(&reference.clauses, &abi_params);
    emit_scalar_hoists(
        &mut function,
        &cleanup_hoists,
        &cleanup_hoisted_locals,
        &scalar_invariant_locals,
        scalar_indices,
        signatures,
    )?;
    let cleanup_inline_bindings = BTreeMap::<String, IrExpr>::new();

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::LocalGet(first_len_local));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));

    for ((param, lane_local), loop_ptr_local) in
        abi_params.iter().zip(&lane_locals).zip(&lane_ptr_locals)
    {
        if let (KernelParam::Lane { prim, .. }, Some(local), Some(loop_ptr_local)) =
            (param, lane_local, loop_ptr_local)
        {
            emit_lane_load_scalar_at_ptr(&mut function, *loop_ptr_local, *prim);
            function.instruction(&Instruction::LocalSet(*local));
        }
    }

    let scalar_locals = kernel_scalar_local_map(&reference.clauses, &abi_params, &lane_locals);
    let empty_enum_ctors = BTreeMap::<String, EnumCtorInfo>::new();
    let empty_wasm_enum_layouts = BTreeMap::<String, WasmEnumLayout>::new();

    for clause_index in 0..reference.clauses.len() {
        let reference_clause = &reference.clauses[clause_index];
        let clause_key = reference_clause as *const _ as usize;
        let locals = scalar_locals.get(&clause_key).cloned().unwrap_or_default();
        if clause_has_condition(&reference_clause.patterns, &empty_enum_ctors) {
            emit_matching_if(
                &mut function,
                &reference_clause.patterns,
                &empty_enum_ctors,
                &empty_wasm_enum_layouts,
                None,
            )?;
            for ((leaf, output), output_ptr_loop_local) in prepared_leaves
                .iter()
                .zip(&outputs)
                .zip(&output_ptr_loop_locals)
            {
                function.instruction(&Instruction::LocalGet(*output_ptr_loop_local));
                compile_scalar_ir_expr_with_hoists(
                    &mut function,
                    &leaf.clauses[clause_index].body,
                    &locals,
                    scalar_indices,
                    signatures,
                    &empty_enum_ctors,
                    &empty_wasm_enum_layouts,
                    Some(enum_state),
                    &cleanup_hoisted_locals,
                    &cleanup_inline_bindings,
                )?;
                emit_scalar_store(&mut function, output.prim);
            }
            for param in &vector_params {
                if let KernelParam::Lane {
                    prim, ptr_local, ..
                } = param
                {
                    emit_pointer_bump(&mut function, *ptr_local, byte_width(*prim));
                }
            }
            for (output, output_ptr_loop_local) in outputs.iter().zip(&output_ptr_loop_locals) {
                emit_pointer_bump(
                    &mut function,
                    *output_ptr_loop_local,
                    byte_width(output.prim),
                );
            }
            emit_scalar_index_bump(&mut function, index_local);
            function.instruction(&Instruction::Br(1));
            function.instruction(&Instruction::End);
        } else {
            for ((leaf, output), output_ptr_loop_local) in prepared_leaves
                .iter()
                .zip(&outputs)
                .zip(&output_ptr_loop_locals)
            {
                function.instruction(&Instruction::LocalGet(*output_ptr_loop_local));
                compile_scalar_ir_expr_with_hoists(
                    &mut function,
                    &leaf.clauses[clause_index].body,
                    &locals,
                    scalar_indices,
                    signatures,
                    &empty_enum_ctors,
                    &empty_wasm_enum_layouts,
                    Some(enum_state),
                    &cleanup_hoisted_locals,
                    &cleanup_inline_bindings,
                )?;
                emit_scalar_store(&mut function, output.prim);
            }
            for param in &vector_params {
                if let KernelParam::Lane {
                    prim, ptr_local, ..
                } = param
                {
                    emit_pointer_bump(&mut function, *ptr_local, byte_width(*prim));
                }
            }
            for (output, output_ptr_loop_local) in outputs.iter().zip(&output_ptr_loop_locals) {
                emit_pointer_bump(
                    &mut function,
                    *output_ptr_loop_local,
                    byte_width(output.prim),
                );
            }
            emit_scalar_index_bump(&mut function, index_local);
            function.instruction(&Instruction::Br(0));
        }
    }
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);

    Ok(Some(CompiledGroupedKernel {
        leaf_paths: prepared_leaves
            .iter()
            .map(|leaf| leaf.leaf_path.clone())
            .collect(),
        function,
        signature: (wasm_params, None),
        report: WasmOptimizationReport {
            function: format!("{}{:?}", group.source_name, group.leaf_paths),
            intent,
            structural_exec: StructuralExecMode::Scalar,
            vectorizable: !matches!(vector_plan, VectorPlan::ScalarOnly),
            vector_unroll: vector_plan_unroll(vector_plan),
            fallback_reason,
            structural_scc: None,
            structural_state_count: 0,
            structural_transition_count: 0,
            structural_span_ops: 0,
            structural_enum_ops: 0,
            structural_region_count: 0,
            structural_char_prefix_regions: 0,
            structural_separated_item_regions: 0,
        },
    }))
}

struct VectorizableGroupedClause<'a> {
    patterns: &'a [TypedPattern],
    bodies: Vec<&'a IrExpr>,
    locals: BTreeMap<String, usize>,
}

struct VectorizableGroupedKernel<'a> {
    clauses: Vec<VectorizableGroupedClause<'a>>,
}

fn vectorizable_grouped_kernel_clause<'a>(
    leaves: &'a [PreparedGroupedLeaf],
    params: &'a [KernelParam],
) -> Option<VectorizableGroupedKernel<'a>> {
    let empty_enum_ctors = BTreeMap::<String, EnumCtorInfo>::new();
    let reference = leaves.first()?;
    if reference.clauses.is_empty() {
        return None;
    }
    if leaves
        .iter()
        .any(|leaf| leaf.result_prim.lane_width() != reference.result_prim.lane_width())
    {
        return None;
    }
    if clause_has_condition(&reference.clauses.last()?.patterns, &empty_enum_ctors) {
        return None;
    }

    let mut grouped_clauses = Vec::with_capacity(reference.clauses.len());
    for clause_index in 0..reference.clauses.len() {
        let reference_clause = &reference.clauses[clause_index];
        let mut locals = BTreeMap::new();
        for (index, (pattern, param)) in reference_clause.patterns.iter().zip(params).enumerate() {
            if let Pattern::Name(name) = &pattern.pattern
                && matches!(param, KernelParam::Same { .. } | KernelParam::Lane { .. })
            {
                locals.insert(name.clone(), index);
            }
        }

        if !patterns_are_vectorizable(&reference_clause.patterns, params, reference.result_prim) {
            return None;
        }
        let mut bodies = Vec::with_capacity(leaves.len());
        for leaf in leaves {
            let clause = leaf.clauses.get(clause_index)?;
            if clause.patterns != reference_clause.patterns {
                return None;
            }
            if !is_vectorizable_expr(&clause.body, &locals, leaf.result_prim) {
                return None;
            }
            bodies.push(&clause.body);
        }
        grouped_clauses.push(VectorizableGroupedClause {
            patterns: &reference_clause.patterns,
            bodies,
            locals,
        });
    }

    Some(VectorizableGroupedKernel {
        clauses: grouped_clauses,
    })
}

fn inline_grouped_kernel_expr(
    expr: &IrExpr,
    lowered_map: &BTreeMap<String, &LoweredFunction>,
    visiting: &mut BTreeSet<String>,
) -> Result<IrExpr> {
    match &expr.kind {
        IrExprKind::Local(..) | IrExprKind::Int(..) | IrExprKind::Float(..) => Ok(expr.clone()),
        IrExprKind::Record(..) => Err(SimdError::new(
            "Wasm grouped-kernel inlining does not yet support record IR",
        )),
        IrExprKind::EnumCtor { .. }
        | IrExprKind::EnumTag { .. }
        | IrExprKind::EnumChildBySlot { .. }
        | IrExprKind::EnumNonRecField { .. } => Err(SimdError::new(
            "Wasm grouped-kernel inlining does not yet support enum IR",
        )),
        IrExprKind::Seq(_) | IrExprKind::SeqSplice { .. } => Err(SimdError::new(
            "Wasm grouped-kernel inlining does not yet support sequence IR",
        )),
        IrExprKind::Let { bindings, body } => Ok(IrExpr {
            ty: expr.ty.clone(),
            kind: IrExprKind::Let {
                bindings: bindings
                    .iter()
                    .map(|binding| {
                        Ok(IrLetBinding {
                            name: binding.name.clone(),
                            expr: inline_grouped_kernel_expr(&binding.expr, lowered_map, visiting)?,
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                body: Box::new(inline_grouped_kernel_expr(body, lowered_map, visiting)?),
            },
        }),
        IrExprKind::Call { callee, args } => {
            let args = args
                .iter()
                .map(|arg| inline_grouped_kernel_expr(arg, lowered_map, visiting))
                .collect::<Result<Vec<_>>>()?;
            match callee {
                Callee::Prim(op) => Ok(IrExpr {
                    ty: expr.ty.clone(),
                    kind: IrExprKind::Call {
                        callee: Callee::Prim(*op),
                        args,
                    },
                }),
                Callee::Function(name) => {
                    if let Some(inlined) =
                        try_inline_grouped_leaf_call(name, &args, lowered_map, visiting)?
                    {
                        Ok(inlined)
                    } else {
                        Ok(IrExpr {
                            ty: expr.ty.clone(),
                            kind: IrExprKind::Call {
                                callee: Callee::Function(name.clone()),
                                args,
                            },
                        })
                    }
                }
                Callee::Builtin(builtin) => Ok(IrExpr {
                    ty: expr.ty.clone(),
                    kind: IrExprKind::Call {
                        callee: Callee::Builtin(builtin.clone()),
                        args,
                    },
                }),
            }
        }
    }
}

fn try_inline_grouped_leaf_call(
    name: &str,
    args: &[IrExpr],
    lowered_map: &BTreeMap<String, &LoweredFunction>,
    visiting: &mut BTreeSet<String>,
) -> Result<Option<IrExpr>> {
    const INLINE_HELPER_NODE_BUDGET: usize = 48;
    if visiting.contains(name) {
        return Ok(None);
    }
    let Some(lowered) = lowered_map.get(name).copied() else {
        return Ok(None);
    };
    if lowered.tail_loop.is_some() {
        return Ok(None);
    }
    let clauses = match &lowered.kind {
        LoweredKind::Scalar { clauses }
        | LoweredKind::Structural { clauses, .. }
        | LoweredKind::Kernel { clauses, .. } => clauses,
    };
    if clauses.len() != 1 {
        return Ok(None);
    }
    let clause = &clauses[0];
    if clause.patterns.len() != args.len()
        || clause.patterns.iter().any(|pattern| {
            matches!(
                pattern.pattern,
                Pattern::Int(_) | Pattern::Float(_) | Pattern::Char(_) | Pattern::Slice { .. }
            )
        })
    {
        return Ok(None);
    }

    visiting.insert(name.to_string());
    let body = inline_grouped_kernel_expr(&clause.body, lowered_map, visiting)?;
    visiting.remove(name);
    if ir_expr_node_count(&body) > INLINE_HELPER_NODE_BUDGET {
        return Ok(None);
    }

    let mut substitutions = BTreeMap::new();
    for (pattern, arg) in clause.patterns.iter().zip(args) {
        if let Pattern::Name(local) = &pattern.pattern {
            substitutions.insert(local.clone(), arg.clone());
        }
    }
    Ok(Some(substitute_ir_expr(&body, &substitutions)))
}

fn ir_expr_node_count(expr: &IrExpr) -> usize {
    match &expr.kind {
        IrExprKind::Local(_) | IrExprKind::Int(_, _) | IrExprKind::Float(_, _) => 1,
        IrExprKind::Record(fields) => 1 + fields.values().map(ir_expr_node_count).sum::<usize>(),
        IrExprKind::EnumCtor { args, .. } => 1 + args.iter().map(ir_expr_node_count).sum::<usize>(),
        IrExprKind::EnumTag { value }
        | IrExprKind::EnumChildBySlot { value, .. }
        | IrExprKind::EnumNonRecField { value, .. } => 1 + ir_expr_node_count(value),
        IrExprKind::Seq(items) => 1 + items.iter().map(ir_expr_node_count).sum::<usize>(),
        IrExprKind::SeqSplice { prefix, tail } => {
            1 + prefix.iter().map(ir_expr_node_count).sum::<usize>() + ir_expr_node_count(tail)
        }
        IrExprKind::Let { bindings, body } => {
            1 + bindings
                .iter()
                .map(|binding| ir_expr_node_count(&binding.expr))
                .sum::<usize>()
                + ir_expr_node_count(body)
        }
        IrExprKind::Call { args, .. } => 1 + args.iter().map(ir_expr_node_count).sum::<usize>(),
    }
}

fn expr_contains_non_scalar_calls(expr: &IrExpr, scalar_indices: &BTreeMap<String, u32>) -> bool {
    match &expr.kind {
        IrExprKind::Local(_) | IrExprKind::Int(_, _) | IrExprKind::Float(_, _) => false,
        IrExprKind::Record(_) => true,
        IrExprKind::EnumCtor { .. }
        | IrExprKind::EnumTag { .. }
        | IrExprKind::EnumChildBySlot { .. }
        | IrExprKind::EnumNonRecField { .. } => true,
        IrExprKind::Seq(items) => items
            .iter()
            .any(|item| expr_contains_non_scalar_calls(item, scalar_indices)),
        IrExprKind::SeqSplice { prefix, tail } => {
            prefix
                .iter()
                .any(|item| expr_contains_non_scalar_calls(item, scalar_indices))
                || expr_contains_non_scalar_calls(tail, scalar_indices)
        }
        IrExprKind::Let { bindings, body } => {
            bindings
                .iter()
                .any(|binding| expr_contains_non_scalar_calls(&binding.expr, scalar_indices))
                || expr_contains_non_scalar_calls(body, scalar_indices)
        }
        IrExprKind::Call { callee, args } => match callee {
            Callee::Prim(_) => args
                .iter()
                .any(|arg| expr_contains_non_scalar_calls(arg, scalar_indices)),
            Callee::Function(name) => {
                !scalar_indices.contains_key(name)
                    || args
                        .iter()
                        .any(|arg| expr_contains_non_scalar_calls(arg, scalar_indices))
            }
            Callee::Builtin(_) => true,
        },
    }
}

fn substitute_ir_expr(expr: &IrExpr, substitutions: &BTreeMap<String, IrExpr>) -> IrExpr {
    match &expr.kind {
        IrExprKind::Local(name) => substitutions
            .get(name)
            .cloned()
            .unwrap_or_else(|| expr.clone()),
        IrExprKind::Int(_, _) | IrExprKind::Float(_, _) => expr.clone(),
        IrExprKind::Seq(items) => IrExpr {
            ty: expr.ty.clone(),
            kind: IrExprKind::Seq(
                items
                    .iter()
                    .map(|item| substitute_ir_expr(item, substitutions))
                    .collect(),
            ),
        },
        IrExprKind::SeqSplice { prefix, tail } => IrExpr {
            ty: expr.ty.clone(),
            kind: IrExprKind::SeqSplice {
                prefix: prefix
                    .iter()
                    .map(|item| substitute_ir_expr(item, substitutions))
                    .collect(),
                tail: Box::new(substitute_ir_expr(tail, substitutions)),
            },
        },
        IrExprKind::Record(fields) => IrExpr {
            ty: expr.ty.clone(),
            kind: IrExprKind::Record(
                fields
                    .iter()
                    .map(|(name, field)| (name.clone(), substitute_ir_expr(field, substitutions)))
                    .collect(),
            ),
        },
        IrExprKind::EnumCtor { ctor, args } => IrExpr {
            ty: expr.ty.clone(),
            kind: IrExprKind::EnumCtor {
                ctor: ctor.clone(),
                args: args
                    .iter()
                    .map(|arg| substitute_ir_expr(arg, substitutions))
                    .collect(),
            },
        },
        IrExprKind::EnumTag { value } => IrExpr {
            ty: expr.ty.clone(),
            kind: IrExprKind::EnumTag {
                value: Box::new(substitute_ir_expr(value, substitutions)),
            },
        },
        IrExprKind::EnumChildBySlot { value, ctor, slot } => IrExpr {
            ty: expr.ty.clone(),
            kind: IrExprKind::EnumChildBySlot {
                value: Box::new(substitute_ir_expr(value, substitutions)),
                ctor: ctor.clone(),
                slot: *slot,
            },
        },
        IrExprKind::EnumNonRecField { value, ctor, field } => IrExpr {
            ty: expr.ty.clone(),
            kind: IrExprKind::EnumNonRecField {
                value: Box::new(substitute_ir_expr(value, substitutions)),
                ctor: ctor.clone(),
                field: *field,
            },
        },
        IrExprKind::Let { bindings, body } => {
            let mut extended = substitutions.clone();
            for binding in bindings {
                let substituted = substitute_ir_expr(&binding.expr, &extended);
                extended.insert(binding.name.clone(), substituted);
            }
            substitute_ir_expr(body, &extended)
        }
        IrExprKind::Call { callee, args } => IrExpr {
            ty: expr.ty.clone(),
            kind: IrExprKind::Call {
                callee: callee.clone(),
                args: args
                    .iter()
                    .map(|arg| substitute_ir_expr(arg, substitutions))
                    .collect(),
            },
        },
    }
}

fn add_function_type(types: &mut TypeSection, params: &[ValType], result: Option<&ValType>) -> u32 {
    let index = types.len();
    let results = result.into_iter().copied();
    types.ty().function(params.iter().copied(), results);
    index
}

fn scalar_signature(function: &CheckedFunction) -> Result<(Vec<ValType>, Option<ValType>)> {
    let (params, result) = function.signature.ty.fun_parts();
    let _validated_params = params
        .into_iter()
        .map(|ty| match ty {
            Type::Scalar(prim) => Ok(wasm_val_type(prim)),
            Type::Index(_) => Ok(ValType::I64),
            Type::Bulk(_, _) => Ok(ValType::I32),
            Type::StarSeq(_) | Type::StarSeqWitnessed(_, _)
                if wasm_star_seq_storage_prim(&ty).is_some() =>
            {
                Ok(ValType::I32)
            }
            Type::Named(name, args) if is_wasm_string_named_type(&name, &args) => Ok(ValType::I32),
            Type::Named(name, args) if is_wasm_enum_named_type(&name, &args) => Ok(ValType::I32),
            Type::Record(_) => Err(SimdError::new(format!(
                "scalar Wasm function '{}' cannot accept record parameters",
                function.name
            ))),
            Type::Var(_) | Type::Infer(_) => Err(SimdError::new(format!(
                "scalar Wasm function '{}' cannot accept unresolved polymorphic parameters",
                function.name
            ))),
            other => Err(SimdError::new(format!(
                "scalar Wasm function '{}' cannot accept non-scalar parameter {:?}",
                function.name, other
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    let mut expanded_params = Vec::<ValType>::new();
    for ty in function.signature.ty.fun_parts().0 {
        match ty {
            Type::Bulk(_, _) => {
                expanded_params.push(ValType::I32);
                expanded_params.push(ValType::I32);
            }
            Type::Index(_) => expanded_params.push(ValType::I64),
            Type::StarSeq(_) | Type::StarSeqWitnessed(_, _)
                if wasm_star_seq_storage_prim(&ty).is_some() =>
            {
                expanded_params.push(ValType::I32);
                expanded_params.push(ValType::I32);
            }
            Type::Named(name, args) if is_wasm_string_named_type(&name, &args) => {
                expanded_params.push(ValType::I32);
                expanded_params.push(ValType::I32);
            }
            Type::Scalar(prim) => expanded_params.push(wasm_val_type(prim)),
            Type::Named(name, args) if is_wasm_enum_named_type(&name, &args) => {
                expanded_params.push(ValType::I32);
            }
            Type::StarSeq(_) | Type::StarSeqWitnessed(_, _) => {
                return Err(SimdError::new(format!(
                    "scalar Wasm function '{}' only supports supported T[*] parameters",
                    function.name
                )));
            }
            Type::Tuple(_)
            | Type::Record(_)
            | Type::Var(_)
            | Type::Infer(_)
            | Type::TypeToken(_)
            | Type::Named(_, _)
            | Type::Fun(_, _) => {
                // Already validated above; keep original error path.
                return Err(SimdError::new(format!(
                    "scalar Wasm function '{}' has unsupported parameter form after validation",
                    function.name
                )));
            }
        }
    }
    let result = match result {
        Type::Scalar(prim) => Some(wasm_val_type(prim)),
        Type::Index(_) => Some(ValType::I64),
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _) => Some(
            wasm_star_seq_storage_prim(&result)
                .map(|_| ValType::I32)
                .ok_or_else(|| {
                    SimdError::new(format!(
                        "scalar Wasm function '{}' only supports scalar/enum-element T[*] results",
                        function.name
                    ))
                })?,
        ),
        Type::Named(name, args) if is_wasm_enum_named_type(&name, &args) => Some(ValType::I32),
        Type::Tuple(_) | Type::Record(_) => {
            return Err(SimdError::new(format!(
                "scalar Wasm function '{}' cannot return tuple/record products",
                function.name
            )));
        }
        Type::Var(_) | Type::Infer(_) => {
            return Err(SimdError::new(format!(
                "scalar Wasm function '{}' cannot return unresolved polymorphic types",
                function.name
            )));
        }
        other => {
            return Err(SimdError::new(format!(
                "scalar Wasm function '{}' cannot return {:?}",
                function.name, other
            )));
        }
    };
    Ok((expanded_params, result))
}

fn entry_signature(function: &CheckedFunction) -> Result<(Vec<ValType>, Option<ValType>)> {
    let (params, result) = function.signature.ty.fun_parts();
    let mut lowered = Vec::new();
    let mut saw_bulk = None::<usize>;
    for (index, ty) in params.into_iter().enumerate() {
        match ty {
            Type::Scalar(prim) => lowered.push(wasm_val_type(prim)),
            Type::Index(_) => lowered.push(ValType::I64),
            Type::Bulk(_, _) => {
                saw_bulk.get_or_insert(index);
                lowered.push(ValType::I32);
                lowered.push(ValType::I32);
            }
            Type::StarSeq(_) | Type::StarSeqWitnessed(_, _)
                if wasm_star_seq_storage_prim(&ty).is_some() =>
            {
                saw_bulk.get_or_insert(index);
                lowered.push(ValType::I32);
                lowered.push(ValType::I32);
            }
            Type::Named(name, args) if is_wasm_string_named_type(&name, &args) => {
                saw_bulk.get_or_insert(index);
                lowered.push(ValType::I32);
                lowered.push(ValType::I32);
            }
            Type::Named(name, args) if is_wasm_enum_named_type(&name, &args) => {
                lowered.push(ValType::I32);
            }
            Type::StarSeq(_) | Type::StarSeqWitnessed(_, _) => {
                return Err(SimdError::new(
                    "Wasm backend only supports scalar-element T[*] entry parameters",
                ));
            }
            Type::TypeToken(_) => {
                return Err(SimdError::new(
                    "Wasm backend does not support Type witness entry parameters",
                ));
            }
            Type::Tuple(_) | Type::Record(_) => {
                return Err(SimdError::new(
                    "Wasm backend does not yet support tuple/record entry parameters",
                ));
            }
            Type::Named(_, _) | Type::Var(_) | Type::Infer(_) => {
                return Err(SimdError::new(
                    "Wasm backend does not support unresolved polymorphic entry parameters",
                ));
            }
            Type::Fun(_, _) => {
                return Err(SimdError::new(
                    "Wasm backend does not support higher-order entry parameters",
                ));
            }
        }
    }
    let result_ty = match result {
        Type::Scalar(prim) => Some(wasm_val_type(prim)),
        Type::Named(name, args) if is_wasm_enum_named_type(&name, &args) => Some(ValType::I32),
        Type::Index(_) => Some(ValType::I64),
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _) => Some(
            wasm_star_seq_scalar_prim(&result)
                .map(|_| ValType::I32)
                .ok_or_else(|| {
                    SimdError::new("Wasm backend only supports scalar-element T[*] entry results")
                })?,
        ),
        Type::Bulk(_, _) => {
            if saw_bulk.is_none() {
                return Err(SimdError::new(
                    "Wasm bulk entry requires a bulk parameter to determine output shape",
                ));
            }
            lowered.push(ValType::I32);
            lowered.push(ValType::I32);
            None
        }
        Type::TypeToken(_) => {
            return Err(SimdError::new(
                "Wasm backend does not support Type witness entry results",
            ));
        }
        Type::Tuple(_) | Type::Record(_) => {
            return Err(SimdError::new(
                "Wasm backend does not yet support tuple/record entry results",
            ));
        }
        Type::Named(_, _) | Type::Var(_) | Type::Infer(_) => {
            return Err(SimdError::new(
                "Wasm backend does not support unresolved polymorphic entry results",
            ));
        }
        Type::Fun(_, _) => {
            return Err(SimdError::new(
                "Wasm backend does not support higher-order entry results",
            ));
        }
    };
    Ok((lowered, result_ty))
}

fn wasm_val_type(prim: Prim) -> ValType {
    match prim {
        Prim::I32 => ValType::I32,
        Prim::Char => ValType::I32,
        Prim::I64 => ValType::I64,
        Prim::F32 => ValType::F32,
        Prim::F64 => ValType::F64,
    }
}

fn compile_scalar_function(
    lowered: &LoweredFunction,
    checked: &CheckedFunction,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    function_profile: Option<WasmFunctionInstrumentation>,
) -> Result<Function> {
    let (param_types, result_ty) = checked.signature.ty.fun_parts();
    let tail_loop = lowered.tail_loop.as_ref();
    let result_val_type = match result_ty {
        Type::Scalar(prim) => wasm_val_type(prim),
        Type::Index(_) => ValType::I64,
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _) => wasm_star_seq_storage_prim(&result_ty)
            .map(wasm_val_type)
            .map(|_| ValType::I32)
            .ok_or_else(|| {
                SimdError::new(format!(
                    "scalar function '{}' only supports scalar/enum-element T[*] results",
                    checked.name
                ))
            })?,
        Type::Named(name, args) if is_wasm_enum_named_type(&name, &args) => ValType::I32,
        other => {
            return Err(SimdError::new(format!(
                "scalar function '{}' has unsupported result {:?}",
                checked.name, other
            )));
        }
    };
    let mut locals = Vec::<(u32, ValType)>::new();
    let mut next_local = 0u32;
    let mut param_local_ranges = Vec::<(u32, u32)>::new();
    for ty in &param_types {
        let width = if is_wasm_bulk_like_type(ty) { 2 } else { 1 };
        param_local_ranges.push((next_local, width));
        next_local += width;
    }
    let temp_locals = if tail_loop.is_some() {
        let mut temp = Vec::new();
        for ty in &param_types {
            let start_local = next_local;
            if is_wasm_bulk_like_type(ty) {
                next_local += 2;
                locals.push((2, ValType::I32));
                temp.push((start_local, 2));
                continue;
            }
            let val_type = match ty {
                Type::Scalar(prim) => Ok(wasm_val_type(*prim)),
                Type::Named(name, args) if is_wasm_enum_named_type(name, args) => Ok(ValType::I32),
                Type::Record(_) => Err(SimdError::new(format!(
                    "scalar function '{}' has unsupported record parameters",
                    checked.name
                ))),
                other => Err(SimdError::new(format!(
                    "scalar function '{}' has unsupported parameter {:?}",
                    checked.name, other
                ))),
            }?;
            next_local += 1;
            locals.push((1, val_type));
            temp.push((start_local, 1));
        }
        temp
    } else {
        Vec::new()
    };
    let result_local = if tail_loop.is_some() {
        let local = next_local;
        next_local += 1;
        locals.push((1, result_val_type));
        Some(local)
    } else {
        None
    };
    let enum_ptr_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_base_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_save_sp_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_base_tmp_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_index_i64_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I64));
    let enum_scratch_sp_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_alloc_end_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_alloc_capacity_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_alloc_delta_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux0_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux1_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux2_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux3_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux4_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux5_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux6_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux7_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux8_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let mut enum_child_locals = [0u32; 8];
    for local in &mut enum_child_locals {
        *local = next_local;
        next_local += 1;
        locals.push((1, ValType::I32));
    }
    let enum_state = EnumWasmState {
        ptr_local: enum_ptr_local,
        base_local: enum_base_local,
        save_sp_local: enum_save_sp_local,
        base_tmp_local: enum_base_tmp_local,
        index_i64_local: enum_index_i64_local,
        scratch_sp_local: enum_scratch_sp_local,
        alloc_end_local: enum_alloc_end_local,
        alloc_capacity_local: enum_alloc_capacity_local,
        alloc_delta_local: enum_alloc_delta_local,
        aux0_local: enum_aux0_local,
        aux1_local: enum_aux1_local,
        aux2_local: enum_aux2_local,
        aux3_local: enum_aux3_local,
        aux4_local: enum_aux4_local,
        aux5_local: enum_aux5_local,
        aux6_local: enum_aux6_local,
        aux7_local: enum_aux7_local,
        aux8_local: enum_aux8_local,
        child_locals: enum_child_locals,
    };

    let mut clause_locals = BTreeMap::<usize, BTreeMap<String, u32>>::new();
    let mut shared_named_locals = BTreeMap::<String, (u32, Type)>::new();
    match (&lowered.kind, tail_loop) {
        (LoweredKind::Scalar { clauses: _ }, Some(tail_loop))
        | (LoweredKind::Structural { clauses: _, .. }, Some(tail_loop)) => {
            for clause in &tail_loop.clauses {
                let map = build_clause_local_map(
                    &clause.patterns,
                    &mut locals,
                    &mut next_local,
                    &mut shared_named_locals,
                    enum_ctors,
                )?;
                clause_locals.insert(clause as *const _ as usize, map);
            }
        }
        (LoweredKind::Scalar { clauses }, None)
        | (LoweredKind::Structural { clauses, .. }, None) => {
            for clause in clauses {
                let map = build_clause_local_map(
                    &clause.patterns,
                    &mut locals,
                    &mut next_local,
                    &mut shared_named_locals,
                    enum_ctors,
                )?;
                clause_locals.insert(clause as *const _ as usize, map);
            }
        }
        (LoweredKind::Kernel { .. }, _) => {
            return Err(SimdError::new(format!(
                "compile_scalar_function received kernel '{}'",
                checked.name
            )));
        }
    }
    let mut function = Function::new(locals);
    emit_wasm_function_profile_enter(&mut function, function_profile);
    function.instruction(&Instruction::I32Const(ENUM_SAVE_STACK_PTR_ADDR as i32));
    function.instruction(&Instruction::I32Load(memarg(0, 2)));
    function.instruction(&Instruction::LocalTee(enum_state.scratch_sp_local));
    function.instruction(&Instruction::I32Eqz);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::I32Const(ENUM_SAVE_STACK_START));
    function.instruction(&Instruction::LocalSet(enum_state.scratch_sp_local));
    function.instruction(&Instruction::End);

    match (&lowered.kind, tail_loop) {
        (LoweredKind::Scalar { clauses: _ }, Some(tail_loop))
        | (LoweredKind::Structural { clauses: _, .. }, Some(tail_loop)) => {
            let result_local =
                result_local.ok_or_else(|| SimdError::new("missing tail-loop result local"))?;
            function.instruction(&Instruction::Block(BlockType::Empty));
            function.instruction(&Instruction::Loop(BlockType::Empty));
            for clause in &tail_loop.clauses {
                let locals = clause_locals
                    .get(&(clause as *const _ as usize))
                    .cloned()
                    .unwrap_or_default();
                emit_matching_if(
                    &mut function,
                    &clause.patterns,
                    enum_ctors,
                    wasm_enum_layouts,
                    Some(enum_state),
                )?;
                emit_clause_bindings(
                    &mut function,
                    &clause.patterns,
                    &locals,
                    enum_ctors,
                    wasm_enum_layouts,
                    Some(enum_state),
                )?;
                match &clause.action {
                    TailAction::Continue { args } => {
                        for ((temp_local, width), arg) in temp_locals.iter().zip(args) {
                            compile_scalar_ir_expr(
                                &mut function,
                                arg,
                                &locals,
                                scalar_indices,
                                signatures,
                                enum_ctors,
                                wasm_enum_layouts,
                                Some(enum_state),
                            )?;
                            if *width == 2 {
                                function.instruction(&Instruction::LocalSet(*temp_local + 1));
                            }
                            function.instruction(&Instruction::LocalSet(*temp_local));
                        }
                        for ((temp_local, width), (param_local, _)) in
                            temp_locals.iter().zip(param_local_ranges.iter())
                        {
                            function.instruction(&Instruction::LocalGet(*temp_local));
                            function.instruction(&Instruction::LocalSet(*param_local));
                            if *width == 2 {
                                function.instruction(&Instruction::LocalGet(*temp_local + 1));
                                function.instruction(&Instruction::LocalSet(*param_local + 1));
                            }
                        }
                        function.instruction(&Instruction::Br(1));
                    }
                    TailAction::Return { expr } => {
                        emit_tail_position_scalar_return(
                            &mut function,
                            expr,
                            &locals,
                            scalar_indices,
                            signatures,
                            enum_ctors,
                            wasm_enum_layouts,
                            Some(enum_state),
                            function_profile,
                        )?;
                    }
                }
                function.instruction(&Instruction::End);
            }
            function.instruction(&Instruction::Unreachable);
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::LocalGet(result_local));
        }
        (LoweredKind::Scalar { clauses }, None)
        | (LoweredKind::Structural { clauses, .. }, None) => {
            for clause in clauses {
                let locals = clause_locals
                    .get(&(clause as *const _ as usize))
                    .cloned()
                    .unwrap_or_default();
                if clause_has_condition(&clause.patterns, enum_ctors) {
                    emit_matching_if(
                        &mut function,
                        &clause.patterns,
                        enum_ctors,
                        wasm_enum_layouts,
                        Some(enum_state),
                    )?;
                    emit_clause_bindings(
                        &mut function,
                        &clause.patterns,
                        &locals,
                        enum_ctors,
                        wasm_enum_layouts,
                        Some(enum_state),
                    )?;
                    emit_tail_position_scalar_return(
                        &mut function,
                        &clause.body,
                        &locals,
                        scalar_indices,
                        signatures,
                        enum_ctors,
                        wasm_enum_layouts,
                        Some(enum_state),
                        function_profile,
                    )?;
                    function.instruction(&Instruction::End);
                } else {
                    emit_clause_bindings(
                        &mut function,
                        &clause.patterns,
                        &locals,
                        enum_ctors,
                        wasm_enum_layouts,
                        Some(enum_state),
                    )?;
                    emit_tail_position_scalar_return(
                        &mut function,
                        &clause.body,
                        &locals,
                        scalar_indices,
                        signatures,
                        enum_ctors,
                        wasm_enum_layouts,
                        Some(enum_state),
                        function_profile,
                    )?;
                    break;
                }
            }
            function.instruction(&Instruction::Unreachable);
        }
        (LoweredKind::Kernel { .. }, _) => {
            return Err(SimdError::new(format!(
                "compile_scalar_function received kernel '{}'",
                checked.name
            )));
        }
    }

    emit_wasm_function_profile_exit(&mut function, function_profile);
    function.instruction(&Instruction::End);
    Ok(function)
}

fn emit_structural_state_action(
    function: &mut Function,
    action: &StructuralAction,
    locals: &BTreeMap<String, u32>,
    temp_locals: &[(u32, u32)],
    param_local_ranges: &[(u32, u32)],
    loop_branch_depth: u32,
    state_local: u32,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: EnumWasmState,
    function_profile: Option<WasmFunctionInstrumentation>,
) -> Result<()> {
    match action {
        StructuralAction::Transition { state, args } => {
            emit_structural_transition_assignment(
                function,
                args,
                locals,
                temp_locals,
                param_local_ranges,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                enum_state,
            )?;
            function.instruction(&Instruction::I32Const(*state as i32));
            function.instruction(&Instruction::LocalSet(state_local));
            function.instruction(&Instruction::Br(loop_branch_depth));
        }
        StructuralAction::Return { expr } => {
            emit_tail_position_scalar_return(
                function,
                expr,
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                function_profile,
            )?;
        }
    }
    Ok(())
}

fn emit_structural_transition_assignment(
    function: &mut Function,
    args: &[IrExpr],
    locals: &BTreeMap<String, u32>,
    temp_locals: &[(u32, u32)],
    param_local_ranges: &[(u32, u32)],
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: EnumWasmState,
) -> Result<()> {
    for ((temp_local, width), arg) in temp_locals.iter().zip(args) {
        compile_scalar_ir_expr(
            function,
            arg,
            locals,
            scalar_indices,
            signatures,
            enum_ctors,
            wasm_enum_layouts,
            Some(enum_state),
        )?;
        if *width == 2 {
            function.instruction(&Instruction::LocalSet(*temp_local + 1));
        }
        function.instruction(&Instruction::LocalSet(*temp_local));
    }
    for ((temp_local, width), (param_local, _)) in temp_locals.iter().zip(param_local_ranges.iter())
    {
        function.instruction(&Instruction::LocalGet(*temp_local));
        function.instruction(&Instruction::LocalSet(*param_local));
        if *width == 2 {
            function.instruction(&Instruction::LocalGet(*temp_local + 1));
            function.instruction(&Instruction::LocalSet(*param_local + 1));
        }
    }
    Ok(())
}

fn structural_state_is_leading_ascii_whitespace_self_transition(
    state: &StructuralState,
    index: usize,
    param_count: usize,
    function_name: &str,
) -> bool {
    if state.on_miss != Some((index + 1) as u32) || state.patterns.len() != param_count {
        return false;
    }
    let Pattern::Slice {
        prefix,
        suffix,
        rest: Some(SliceRest::Bind(rest_name)),
    } = &state.patterns[0].pattern
    else {
        return false;
    };
    if suffix.len() != 0 || prefix.len() != 1 {
        return false;
    }
    let Pattern::Char(ch) = prefix[0] else {
        return false;
    };
    if !matches!(ch, ' ' | '\n' | '\t' | '\r') {
        return false;
    }
    let Some(args) = structural_action_self_reentry_args(&state.on_match, function_name) else {
        return false;
    };
    if args.len() != param_count {
        return false;
    }
    matches!(&args[0].kind, IrExprKind::Local(name) if *name == *rest_name)
        && state
            .patterns
            .iter()
            .zip(args.iter())
            .skip(1)
            .all(|(pattern, arg)| {
                matches!(
                    (&pattern.pattern, &arg.kind),
                    (Pattern::Name(name), IrExprKind::Local(local)) if local == name
                )
            })
}

fn structural_expr_self_reentry_args(expr: &IrExpr, function_name: &str) -> Option<Vec<IrExpr>> {
    match &expr.kind {
        IrExprKind::Call {
            callee: Callee::Function(name),
            args,
        } if name == function_name => Some(args.clone()),
        IrExprKind::Let { bindings, body } => {
            let args = structural_expr_self_reentry_args(body, function_name)?;
            Some(
                args.into_iter()
                    .map(|arg| IrExpr {
                        ty: arg.ty.clone(),
                        kind: IrExprKind::Let {
                            bindings: bindings.clone(),
                            body: Box::new(arg),
                        },
                    })
                    .collect(),
            )
        }
        _ => None,
    }
}

fn structural_action_self_reentry_args(
    action: &StructuralAction,
    function_name: &str,
) -> Option<Vec<IrExpr>> {
    match action {
        StructuralAction::Transition { state: 0, args } => Some(args.clone()),
        StructuralAction::Return { expr } => structural_expr_self_reentry_args(expr, function_name),
        _ => None,
    }
}

fn structural_leading_ascii_whitespace_state_count(
    program: &StructuralProgram,
    param_types: &[Type],
    function_name: &str,
) -> usize {
    if param_types.is_empty() || !is_wasm_bulk_like_type(&param_types[0]) {
        return 0;
    }
    program
        .states
        .iter()
        .enumerate()
        .take_while(|(index, state)| {
            structural_state_is_leading_ascii_whitespace_self_transition(
                state,
                *index,
                param_types.len(),
                function_name,
            )
        })
        .count()
}

fn structural_entry_digit_scan_info(
    program: &StructuralProgram,
    param_types: &[Type],
    function_name: &str,
) -> Option<(usize, usize)> {
    let mut slice_param_index = None::<usize>;
    let mut count = 0usize;
    for (index, state) in program.states.iter().enumerate() {
        if state.on_miss != Some((index + 1) as u32) || state.patterns.len() != param_types.len() {
            break;
        }
        let mut current_slice_index = None::<usize>;
        for (param_index, pattern) in state.patterns.iter().enumerate() {
            match &pattern.pattern {
                Pattern::Slice {
                    prefix,
                    suffix,
                    rest: Some(SliceRest::Bind(_)),
                } if prefix.len() == 1 && suffix.is_empty() => {
                    let Pattern::Char(ch) = prefix[0] else {
                        return None;
                    };
                    if !ch.is_ascii_hexdigit() {
                        if count > 0 {
                            let slice_param_index = slice_param_index?;
                            return Some((slice_param_index, count));
                        }
                        return None;
                    }
                    current_slice_index = Some(param_index);
                }
                Pattern::Name(_) | Pattern::Wildcard => {}
                _ => {
                    if count > 0 {
                        let slice_param_index = slice_param_index?;
                        return Some((slice_param_index, count));
                    }
                    return None;
                }
            }
        }
        let Some(current_slice_index) = current_slice_index else {
            if count > 0 {
                let slice_param_index = slice_param_index?;
                return Some((slice_param_index, count));
            }
            return None;
        };
        if !is_wasm_bulk_like_type(&param_types[current_slice_index]) {
            if count > 0 {
                let slice_param_index = slice_param_index?;
                return Some((slice_param_index, count));
            }
            return None;
        }
        if let Some(expected) = slice_param_index {
            if expected != current_slice_index {
                if count > 0 {
                    return Some((expected, count));
                }
                return None;
            }
        } else {
            slice_param_index = Some(current_slice_index);
        }
        let Some(args) = structural_action_self_reentry_args(&state.on_match, function_name) else {
            if count > 0 {
                let slice_param_index = slice_param_index?;
                return Some((slice_param_index, count));
            }
            return None;
        };
        if args.len() != param_types.len() {
            if count > 0 {
                let slice_param_index = slice_param_index?;
                return Some((slice_param_index, count));
            }
            return None;
        }
        count += 1;
    }
    let slice_param_index = slice_param_index?;
    (count > 0).then_some((slice_param_index, count))
}

fn structural_exec_mode_for(
    lowered: &LoweredFunction,
    checked: &CheckedFunction,
) -> StructuralExecMode {
    let LoweredKind::Structural { program, .. } = &lowered.kind else {
        return StructuralExecMode::Scalar;
    };
    if detect_structural_seq_additive_fold(lowered, checked).is_some() {
        return StructuralExecMode::StructuralBatched;
    }
    if !program.sequence_regions.is_empty() {
        return StructuralExecMode::StructuralBatched;
    }
    let (param_types, _) = checked.signature.ty.fun_parts();
    if structural_leading_ascii_whitespace_state_count(program, &param_types, &lowered.name) > 0
        || structural_entry_digit_scan_info(program, &param_types, &lowered.name).is_some()
    {
        StructuralExecMode::StructuralBatched
    } else {
        StructuralExecMode::StructuralLoop
    }
}

#[derive(Debug, Clone)]
struct StructuralSeqAdditiveFold {
    seq_param_index: usize,
    empty_state: usize,
    cons_state: usize,
    elem_prim: Prim,
    head_name: Option<String>,
    rest_name: String,
    acc_init: IrExpr,
    step_expr: IrExpr,
}

#[derive(Debug, Clone)]
struct InlineableEnumMatcherClause {
    ctor: Option<String>,
    bindings: BTreeMap<String, IrExpr>,
    body: IrExpr,
}

#[derive(Debug, Clone)]
struct InlineableEnumMatcher {
    arg_ty: Type,
    clauses: Vec<InlineableEnumMatcherClause>,
}

fn detect_structural_seq_additive_fold(
    lowered: &LoweredFunction,
    checked: &CheckedFunction,
) -> Option<StructuralSeqAdditiveFold> {
    let LoweredKind::Structural { program, .. } = &lowered.kind else {
        return None;
    };
    let (param_types, result_ty) = checked.signature.ty.fun_parts();
    if param_types.len() != 1 {
        return None;
    }
    let elem_prim = wasm_star_seq_storage_prim(&param_types[0])?;
    if !matches!(
        result_ty,
        Type::Scalar(Prim::I32 | Prim::I64 | Prim::F32 | Prim::F64)
    ) {
        return None;
    }
    let mut empty = None::<(usize, IrExpr)>;
    let mut cons = None::<(usize, Option<String>, String, IrExpr)>;
    for (index, state) in program.states.iter().enumerate() {
        if state.patterns.len() != 1 {
            return None;
        }
        match &state.patterns[0].pattern {
            Pattern::Slice {
                prefix,
                suffix,
                rest,
            } if prefix.is_empty() && suffix.is_empty() && rest.is_none() => {
                let StructuralAction::Return { expr } = &state.on_match else {
                    return None;
                };
                empty = Some((index, expr.clone()));
            }
            Pattern::Slice {
                prefix,
                suffix,
                rest: Some(SliceRest::Bind(rest_name)),
            } if prefix.len() == 1 && suffix.is_empty() => {
                let head_name = match &prefix[0] {
                    Pattern::Name(name) if name != "_" => Some(name.clone()),
                    Pattern::Wildcard => None,
                    _ => return None,
                };
                let StructuralAction::Return { expr } = &state.on_match else {
                    return None;
                };
                cons = Some((index, head_name, rest_name.clone(), expr.clone()));
            }
            _ => {}
        }
    }
    let (empty_state, acc_init) = empty?;
    let (cons_state, head_name, rest_name, cons_body) = cons?;
    let step_expr =
        rewrite_seq_additive_fold_body(&cons_body, &lowered.name, &rest_name, "__seq_fold_acc")?;
    Some(StructuralSeqAdditiveFold {
        seq_param_index: 0,
        empty_state,
        cons_state,
        elem_prim,
        head_name,
        rest_name,
        acc_init,
        step_expr,
    })
}

fn rewrite_seq_additive_fold_body(
    expr: &IrExpr,
    function_name: &str,
    rest_name: &str,
    acc_name: &str,
) -> Option<IrExpr> {
    let IrExprKind::Call {
        callee: Callee::Prim(PrimOp::Add),
        args,
    } = &expr.kind
    else {
        return None;
    };
    if args.len() != 2 {
        return None;
    }
    if is_self_seq_fold_call(&args[0], function_name, rest_name) {
        let step = args[1].clone();
        if ir_expr_contains_function_call(&step, function_name) {
            return None;
        }
        return Some(IrExpr {
            ty: expr.ty.clone(),
            kind: IrExprKind::Call {
                callee: Callee::Prim(PrimOp::Add),
                args: vec![
                    IrExpr {
                        ty: expr.ty.clone(),
                        kind: IrExprKind::Local(acc_name.to_string()),
                    },
                    step,
                ],
            },
        });
    }
    if is_self_seq_fold_call(&args[1], function_name, rest_name) {
        let step = args[0].clone();
        if ir_expr_contains_function_call(&step, function_name) {
            return None;
        }
        return Some(IrExpr {
            ty: expr.ty.clone(),
            kind: IrExprKind::Call {
                callee: Callee::Prim(PrimOp::Add),
                args: vec![
                    step,
                    IrExpr {
                        ty: expr.ty.clone(),
                        kind: IrExprKind::Local(acc_name.to_string()),
                    },
                ],
            },
        });
    }
    None
}

fn is_self_seq_fold_call(expr: &IrExpr, function_name: &str, rest_name: &str) -> bool {
    matches!(
        &expr.kind,
        IrExprKind::Call {
            callee: Callee::Function(name),
            args,
        } if name == function_name
            && args.len() == 1
            && matches!(args[0].kind, IrExprKind::Local(ref local) if local == rest_name)
    )
}

fn ir_expr_contains_function_call(expr: &IrExpr, function_name: &str) -> bool {
    match &expr.kind {
        IrExprKind::Local(_) | IrExprKind::Int(_, _) | IrExprKind::Float(_, _) => false,
        IrExprKind::Record(fields) => fields
            .values()
            .any(|field| ir_expr_contains_function_call(field, function_name)),
        IrExprKind::EnumCtor { args, .. } => args
            .iter()
            .any(|arg| ir_expr_contains_function_call(arg, function_name)),
        IrExprKind::EnumTag { value }
        | IrExprKind::EnumChildBySlot { value, .. }
        | IrExprKind::EnumNonRecField { value, .. } => {
            ir_expr_contains_function_call(value, function_name)
        }
        IrExprKind::Seq(items) => items
            .iter()
            .any(|item| ir_expr_contains_function_call(item, function_name)),
        IrExprKind::SeqSplice { prefix, tail } => {
            prefix
                .iter()
                .any(|item| ir_expr_contains_function_call(item, function_name))
                || ir_expr_contains_function_call(tail, function_name)
        }
        IrExprKind::Let { bindings, body } => {
            bindings
                .iter()
                .any(|binding| ir_expr_contains_function_call(&binding.expr, function_name))
                || ir_expr_contains_function_call(body, function_name)
        }
        IrExprKind::Call { callee, args } => match callee {
            Callee::Function(name) => {
                name == function_name
                    || args
                        .iter()
                        .any(|arg| ir_expr_contains_function_call(arg, function_name))
            }
            Callee::Prim(_) | Callee::Builtin(_) => args
                .iter()
                .any(|arg| ir_expr_contains_function_call(arg, function_name)),
        },
    }
}

fn emit_skip_ascii_whitespace_prefix_loop(
    function: &mut Function,
    ptr_local: u32,
    len_local: u32,
    scratch_local: u32,
) {
    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(len_local));
    function.instruction(&Instruction::I32Eqz);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(ptr_local));
    function.instruction(&Instruction::I32Load(memarg(0, 2)));
    function.instruction(&Instruction::LocalTee(scratch_local));
    function.instruction(&Instruction::I32Const(' ' as i32));
    function.instruction(&Instruction::I32Eq);
    function.instruction(&Instruction::LocalGet(scratch_local));
    function.instruction(&Instruction::I32Const('\n' as i32));
    function.instruction(&Instruction::I32Eq);
    function.instruction(&Instruction::I32Or);
    function.instruction(&Instruction::LocalGet(scratch_local));
    function.instruction(&Instruction::I32Const('\t' as i32));
    function.instruction(&Instruction::I32Eq);
    function.instruction(&Instruction::I32Or);
    function.instruction(&Instruction::LocalGet(scratch_local));
    function.instruction(&Instruction::I32Const('\r' as i32));
    function.instruction(&Instruction::I32Eq);
    function.instruction(&Instruction::I32Or);
    function.instruction(&Instruction::I32Eqz);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(ptr_local));
    function.instruction(&Instruction::I32Const(4));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(ptr_local));
    function.instruction(&Instruction::LocalGet(len_local));
    function.instruction(&Instruction::I32Const(1));
    function.instruction(&Instruction::I32Sub);
    function.instruction(&Instruction::LocalSet(len_local));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
}

fn emit_char_prefix_sequence_region_loop(
    function: &mut Function,
    program: &StructuralProgram,
    region: &StructuralSequenceRegion,
    state_locals: &BTreeMap<usize, BTreeMap<String, u32>>,
    temp_locals: &[(u32, u32)],
    param_local_ranges: &[(u32, u32)],
    state_local: u32,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: EnumWasmState,
    function_profile: Option<WasmFunctionInstrumentation>,
) -> Result<()> {
    if region.mode != StructuralSequenceMode::CharPrefixRun {
        return Ok(());
    }
    let (slice_ptr_local, width) = param_local_ranges[region.span_param_index];
    if width != 2 {
        return Ok(());
    }
    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    for state_id in &region.state_ids {
        let state_index = *state_id as usize;
        let state = program.states.get(state_index).ok_or_else(|| {
            SimdError::new(format!(
                "structural sequence region references missing state {}",
                state_index
            ))
        })?;
        let locals = state_locals.get(&state_index).cloned().unwrap_or_default();
        let Pattern::Slice {
            prefix,
            suffix,
            rest: Some(SliceRest::Bind(_)),
        } = &state.patterns[region.span_param_index].pattern
        else {
            return Ok(());
        };
        if prefix.len() != 1 || !suffix.is_empty() {
            return Ok(());
        }
        let Pattern::Char(ch) = prefix[0] else {
            return Ok(());
        };
        function.instruction(&Instruction::LocalGet(state_local));
        function.instruction(&Instruction::I32Const(*state_id as i32));
        function.instruction(&Instruction::I32Eq);
        function.instruction(&Instruction::If(BlockType::Empty));

        function.instruction(&Instruction::LocalGet(slice_ptr_local + 1));
        function.instruction(&Instruction::I32Eqz);
        function.instruction(&Instruction::If(BlockType::Empty));
        let miss_state = state.on_miss.ok_or_else(|| {
            SimdError::new("char-prefix sequence region state has no miss transition")
        })?;
        function.instruction(&Instruction::I32Const(miss_state as i32));
        function.instruction(&Instruction::LocalSet(state_local));
        function.instruction(&Instruction::Br(2));
        function.instruction(&Instruction::End);

        function.instruction(&Instruction::LocalGet(slice_ptr_local));
        function.instruction(&Instruction::I32Load(memarg(0, 2)));
        function.instruction(&Instruction::LocalSet(enum_state.aux0_local));
        function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
        function.instruction(&Instruction::I32Const(
            i32::try_from(u32::from(ch))
                .map_err(|_| SimdError::new("char-prefix region char does not fit in i32"))?,
        ));
        function.instruction(&Instruction::I32Eq);
        function.instruction(&Instruction::If(BlockType::Empty));
        emit_char_prefix_region_bindings(
            function,
            &state.patterns,
            &locals,
            region.span_param_index,
            enum_ctors,
            wasm_enum_layouts,
            Some(enum_state),
        )?;
        match &state.on_match {
            StructuralAction::Transition { state: next, args } => {
                emit_structural_transition_assignment(
                    function,
                    args,
                    &locals,
                    temp_locals,
                    param_local_ranges,
                    scalar_indices,
                    signatures,
                    enum_ctors,
                    wasm_enum_layouts,
                    enum_state,
                )?;
                function.instruction(&Instruction::I32Const(*next as i32));
                function.instruction(&Instruction::LocalSet(state_local));
                if region.state_ids.contains(next) {
                    function.instruction(&Instruction::Br(2));
                } else {
                    function.instruction(&Instruction::Br(3));
                }
            }
            StructuralAction::Return { expr } => {
                emit_tail_position_scalar_return(
                    function,
                    expr,
                    &locals,
                    scalar_indices,
                    signatures,
                    enum_ctors,
                    wasm_enum_layouts,
                    Some(enum_state),
                    function_profile,
                )?;
            }
        }
        function.instruction(&Instruction::End);

        function.instruction(&Instruction::I32Const(miss_state as i32));
        function.instruction(&Instruction::LocalSet(state_local));
        function.instruction(&Instruction::Br(2));
        function.instruction(&Instruction::End);
    }
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    Ok(())
}

fn emit_char_prefix_region_bindings(
    function: &mut Function,
    patterns: &[TypedPattern],
    clause_locals: &BTreeMap<String, u32>,
    span_param_index: usize,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: Option<EnumWasmState>,
) -> Result<()> {
    let mut value_local = 0u32;
    for (pattern_index, typed_pattern) in patterns.iter().enumerate() {
        if pattern_index == span_param_index {
            let Pattern::Slice {
                prefix,
                suffix,
                rest,
            } = &typed_pattern.pattern
            else {
                return Err(SimdError::new(
                    "char-prefix region binding expected slice pattern",
                ));
            };
            if prefix.len() != 1 || !suffix.is_empty() {
                return Err(SimdError::new(
                    "char-prefix region binding expected single-prefix slice",
                ));
            }
            if let Some(SliceRest::Bind(name)) = rest
                && name != "_"
                && !is_pattern_constructor_name(name, enum_ctors)
            {
                let Some(rest_ptr_local) = clause_locals.get(name).copied() else {
                    return Err(SimdError::new(format!(
                        "missing clause local '{}' for char-prefix rest binding",
                        name
                    )));
                };
                let rest_len_key = bulk_len_local_name(name);
                let Some(rest_len_local) = clause_locals.get(&rest_len_key).copied() else {
                    return Err(SimdError::new(format!(
                        "missing clause local '{}' for char-prefix rest length binding",
                        rest_len_key
                    )));
                };
                let prim = wasm_slice_pattern_prim(&typed_pattern.ty)?;
                function.instruction(&Instruction::LocalGet(value_local));
                function.instruction(&Instruction::I32Const(byte_width(prim) as i32));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalSet(rest_ptr_local));
                function.instruction(&Instruction::LocalGet(value_local + 1));
                function.instruction(&Instruction::I32Const(1));
                function.instruction(&Instruction::I32Sub);
                function.instruction(&Instruction::LocalSet(rest_len_local));
            }
        } else {
            emit_pattern_bindings(
                function,
                &typed_pattern.pattern,
                &typed_pattern.ty,
                value_local,
                clause_locals,
                enum_ctors,
                wasm_enum_layouts,
                enum_state,
            )?;
        }
        value_local += wasm_pattern_value_width(&typed_pattern.ty);
    }
    Ok(())
}

fn emit_digit_scan_prefix_loop(
    function: &mut Function,
    program: &StructuralProgram,
    function_name: &str,
    state_locals: &BTreeMap<usize, BTreeMap<String, u32>>,
    temp_locals: &[(u32, u32)],
    param_local_ranges: &[(u32, u32)],
    slice_param_index: usize,
    digit_state_count: usize,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: EnumWasmState,
) -> Result<()> {
    let (slice_ptr_local, width) = param_local_ranges[slice_param_index];
    if width != 2 {
        return Ok(());
    }
    let ascii_decimal_run = digit_state_count == 10
        && (0..10).all(|index| {
            let Pattern::Slice {
                prefix,
                suffix: _,
                rest: Some(SliceRest::Bind(_)),
            } = &program.states[index].patterns[slice_param_index].pattern
            else {
                return false;
            };
            matches!(prefix[0], Pattern::Char(ch) if ch == char::from(b'0' + index as u8))
        });
    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(slice_ptr_local + 1));
    function.instruction(&Instruction::I32Eqz);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(slice_ptr_local));
    function.instruction(&Instruction::I32Load(memarg(0, 2)));
    function.instruction(&Instruction::LocalSet(enum_state.aux0_local));
    if ascii_decimal_run {
        function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
        function.instruction(&Instruction::I32Const('0' as i32));
        function.instruction(&Instruction::I32LtU);
        function.instruction(&Instruction::BrIf(1));
        function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
        function.instruction(&Instruction::I32Const('9' as i32));
        function.instruction(&Instruction::I32GtU);
        function.instruction(&Instruction::BrIf(1));
        function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
        function.instruction(&Instruction::I32Const('0' as i32));
        function.instruction(&Instruction::I32Sub);
        function.instruction(&Instruction::LocalSet(enum_state.aux1_local));
    }
    for index in 0..digit_state_count {
        let state = &program.states[index];
        let locals = state_locals.get(&index).cloned().unwrap_or_default();
        let Pattern::Slice {
            prefix,
            suffix: _,
            rest: Some(SliceRest::Bind(_)),
        } = &state.patterns[slice_param_index].pattern
        else {
            return Ok(());
        };
        let Pattern::Char(ch) = prefix[0] else {
            return Ok(());
        };
        if ascii_decimal_run {
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Const(index as i32));
            function.instruction(&Instruction::I32Eq);
        } else {
            function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
            function.instruction(&Instruction::I32Const(
                i32::try_from(u32::from(ch))
                    .map_err(|_| SimdError::new("digit-scan char does not fit in i32"))?,
            ));
            function.instruction(&Instruction::I32Eq);
        }
        function.instruction(&Instruction::If(BlockType::Empty));
        emit_clause_bindings(
            function,
            &state.patterns,
            &locals,
            enum_ctors,
            wasm_enum_layouts,
            Some(enum_state),
        )?;
        let Some(args) = structural_action_self_reentry_args(&state.on_match, function_name) else {
            return Ok(());
        };
        emit_structural_transition_assignment(
            function,
            &args,
            &locals,
            temp_locals,
            param_local_ranges,
            scalar_indices,
            signatures,
            enum_ctors,
            wasm_enum_layouts,
            enum_state,
        )?;
        function.instruction(&Instruction::Br(1));
        function.instruction(&Instruction::End);
    }
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    Ok(())
}

fn compile_structural_function(
    lowered: &LoweredFunction,
    checked: &CheckedFunction,
    lowered_map: &BTreeMap<String, &LoweredFunction>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    function_profile: Option<WasmFunctionInstrumentation>,
) -> Result<Function> {
    let (param_types, result_ty) = checked.signature.ty.fun_parts();
    let LoweredKind::Structural { program, .. } = &lowered.kind else {
        return Err(SimdError::new(format!(
            "compile_structural_function received non-structural '{}'",
            checked.name
        )));
    };
    let result_val_type = match result_ty {
        Type::Scalar(prim) => wasm_val_type(prim),
        Type::Index(_) => ValType::I64,
        Type::Named(name, args) if is_wasm_enum_named_type(&name, &args) => ValType::I32,
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _)
            if wasm_star_seq_storage_prim(&result_ty).is_some() =>
        {
            ValType::I32
        }
        other => {
            return Err(SimdError::new(format!(
                "structural function '{}' has unsupported result {:?}",
                checked.name, other
            )));
        }
    };
    let seq_add_fold = detect_structural_seq_additive_fold(lowered, checked);
    let mut locals = Vec::<(u32, ValType)>::new();
    let mut next_local = 0u32;
    let mut param_local_ranges = Vec::<(u32, u32)>::new();
    for ty in &param_types {
        let width = if is_wasm_bulk_like_type(ty) { 2 } else { 1 };
        param_local_ranges.push((next_local, width));
        next_local += width;
    }
    let mut temp_locals = Vec::new();
    for ty in &param_types {
        let start_local = next_local;
        if is_wasm_bulk_like_type(ty) {
            next_local += 2;
            locals.push((2, ValType::I32));
            temp_locals.push((start_local, 2));
            continue;
        }
        let val_type = match ty {
            Type::Scalar(prim) => Ok(wasm_val_type(*prim)),
            Type::Index(_) => Ok(ValType::I64),
            Type::Named(name, args) if is_wasm_enum_named_type(name, args) => Ok(ValType::I32),
            Type::Record(_) => Err(SimdError::new(format!(
                "structural function '{}' has unsupported record parameters",
                checked.name
            ))),
            other => Err(SimdError::new(format!(
                "structural function '{}' has unsupported parameter {:?}",
                checked.name, other
            ))),
        }?;
        next_local += 1;
        locals.push((1, val_type));
        temp_locals.push((start_local, 1));
    }
    let state_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let fold_acc_local = seq_add_fold.as_ref().map(|_| {
        let local = next_local;
        next_local += 1;
        locals.push((1, result_val_type));
        local
    });
    let _result_local = result_val_type;
    let enum_ptr_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_base_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_save_sp_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_base_tmp_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_index_i64_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I64));
    let enum_scratch_sp_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_alloc_end_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_alloc_capacity_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_alloc_delta_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux0_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux1_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux2_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux3_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux4_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux5_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux6_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux7_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let enum_aux8_local = next_local;
    next_local += 1;
    locals.push((1, ValType::I32));
    let mut enum_child_locals = [0u32; 8];
    for local in &mut enum_child_locals {
        *local = next_local;
        next_local += 1;
        locals.push((1, ValType::I32));
    }
    let enum_state = EnumWasmState {
        ptr_local: enum_ptr_local,
        base_local: enum_base_local,
        save_sp_local: enum_save_sp_local,
        base_tmp_local: enum_base_tmp_local,
        index_i64_local: enum_index_i64_local,
        scratch_sp_local: enum_scratch_sp_local,
        alloc_end_local: enum_alloc_end_local,
        alloc_capacity_local: enum_alloc_capacity_local,
        alloc_delta_local: enum_alloc_delta_local,
        aux0_local: enum_aux0_local,
        aux1_local: enum_aux1_local,
        aux2_local: enum_aux2_local,
        aux3_local: enum_aux3_local,
        aux4_local: enum_aux4_local,
        aux5_local: enum_aux5_local,
        aux6_local: enum_aux6_local,
        aux7_local: enum_aux7_local,
        aux8_local: enum_aux8_local,
        child_locals: enum_child_locals,
    };

    let mut state_locals = BTreeMap::<usize, BTreeMap<String, u32>>::new();
    let mut shared_named_locals = BTreeMap::<String, (u32, Type)>::new();
    for (index, state) in program.states.iter().enumerate() {
        let map = build_clause_local_map(
            &state.patterns,
            &mut locals,
            &mut next_local,
            &mut shared_named_locals,
            enum_ctors,
        )?;
        state_locals.insert(index, map);
    }

    let mut function = Function::new(locals);
    emit_wasm_function_profile_enter(&mut function, function_profile);
    function.instruction(&Instruction::I32Const(ENUM_SAVE_STACK_PTR_ADDR as i32));
    function.instruction(&Instruction::I32Load(memarg(0, 2)));
    function.instruction(&Instruction::LocalTee(enum_state.scratch_sp_local));
    function.instruction(&Instruction::I32Eqz);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::I32Const(ENUM_SAVE_STACK_START));
    function.instruction(&Instruction::LocalSet(enum_state.scratch_sp_local));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::I32Const(program.entry_state as i32));
    function.instruction(&Instruction::LocalSet(state_local));
    if let (Some(fold), Some(acc_local)) = (seq_add_fold.as_ref(), fold_acc_local) {
        emit_structural_seq_additive_fold_function(
            &mut function,
            fold,
            &state_locals,
            &param_local_ranges,
            acc_local,
            lowered_map,
            scalar_indices,
            signatures,
            enum_ctors,
            wasm_enum_layouts,
            enum_state,
        )?;
        emit_wasm_function_profile_exit(&mut function, function_profile);
        function.instruction(&Instruction::End);
        return Ok(function);
    }
    let whitespace_state_count =
        structural_leading_ascii_whitespace_state_count(program, &param_types, &lowered.name);
    let digit_scan_info = structural_entry_digit_scan_info(program, &param_types, &lowered.name);
    if let Some((slice_param_index, digit_state_count)) = digit_scan_info {
        emit_digit_scan_prefix_loop(
            &mut function,
            program,
            &lowered.name,
            &state_locals,
            &temp_locals,
            &param_local_ranges,
            slice_param_index,
            digit_state_count,
            scalar_indices,
            signatures,
            enum_ctors,
            wasm_enum_layouts,
            enum_state,
        )?;
        function.instruction(&Instruction::I32Const(digit_state_count as i32));
        function.instruction(&Instruction::LocalSet(state_local));
    } else if whitespace_state_count > 0 {
        let (param_ptr_local, _) = param_local_ranges[0];
        emit_skip_ascii_whitespace_prefix_loop(
            &mut function,
            param_ptr_local,
            param_ptr_local + 1,
            enum_state.aux0_local,
        );
    } else if let Some(region) = program.sequence_regions.iter().find(|region| {
        region.entry_state == program.entry_state
            && region.mode == StructuralSequenceMode::CharPrefixRun
            && region.state_ids.len() >= 10
    }) {
        emit_char_prefix_sequence_region_loop(
            &mut function,
            program,
            region,
            &state_locals,
            &temp_locals,
            &param_local_ranges,
            state_local,
            scalar_indices,
            signatures,
            enum_ctors,
            wasm_enum_layouts,
            enum_state,
            function_profile,
        )?;
    }
    function.instruction(&Instruction::Loop(BlockType::Empty));
    for _ in (0..program.states.len()).rev() {
        function.instruction(&Instruction::Block(BlockType::Empty));
    }
    function.instruction(&Instruction::LocalGet(state_local));
    function.instruction(&Instruction::BrTable(
        (0..program.states.len() as u32).collect(),
        program.states.len() as u32,
    ));
    for (index, state) in program.states.iter().enumerate() {
        let locals = state_locals.get(&index).cloned().unwrap_or_default();
        let loop_branch_depth = (program.states.len() - index - 1) as u32;
        function.instruction(&Instruction::End);
        if clause_has_condition(&state.patterns, enum_ctors) {
            emit_matching_if(
                &mut function,
                &state.patterns,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
            )?;
            emit_clause_bindings(
                &mut function,
                &state.patterns,
                &locals,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
            )?;
            emit_structural_state_action(
                &mut function,
                &state.on_match,
                &locals,
                &temp_locals,
                &param_local_ranges,
                loop_branch_depth + 1,
                state_local,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                enum_state,
                function_profile,
            )?;
            function.instruction(&Instruction::End);
            if let Some(next_state) = state.on_miss {
                function.instruction(&Instruction::I32Const(next_state as i32));
                function.instruction(&Instruction::LocalSet(state_local));
                function.instruction(&Instruction::Br(loop_branch_depth));
            } else {
                function.instruction(&Instruction::Unreachable);
            }
        } else {
            emit_clause_bindings(
                &mut function,
                &state.patterns,
                &locals,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
            )?;
            emit_structural_state_action(
                &mut function,
                &state.on_match,
                &locals,
                &temp_locals,
                &param_local_ranges,
                loop_branch_depth,
                state_local,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                enum_state,
                function_profile,
            )?;
        }
    }
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::Unreachable);
    emit_wasm_function_profile_exit(&mut function, function_profile);
    function.instruction(&Instruction::End);
    Ok(function)
}

fn emit_structural_seq_additive_fold_function(
    function: &mut Function,
    fold: &StructuralSeqAdditiveFold,
    state_locals: &BTreeMap<usize, BTreeMap<String, u32>>,
    param_local_ranges: &[(u32, u32)],
    acc_local: u32,
    lowered_map: &BTreeMap<String, &LoweredFunction>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: EnumWasmState,
) -> Result<()> {
    let empty_locals = state_locals
        .get(&fold.empty_state)
        .cloned()
        .unwrap_or_default();
    compile_fold_step_expr_with_inline_matchers(
        function,
        &fold.acc_init,
        &empty_locals,
        lowered_map,
        scalar_indices,
        signatures,
        enum_ctors,
        wasm_enum_layouts,
        Some(enum_state),
        &BTreeMap::new(),
        &BTreeMap::new(),
        0,
    )?;
    function.instruction(&Instruction::LocalSet(acc_local));

    let (seq_ptr_local, seq_width) = param_local_ranges[fold.seq_param_index];
    if seq_width != 2 {
        return Err(SimdError::new(
            "sequence additive fold expected bulk-like sequence parameter width",
        ));
    }
    let seq_len_local = seq_ptr_local + 1;
    let cons_locals = state_locals
        .get(&fold.cons_state)
        .cloned()
        .unwrap_or_default();
    let mut step_locals = cons_locals.clone();
    step_locals.insert("__seq_fold_acc".to_string(), acc_local);
    let rest_len_key = bulk_len_local_name(&fold.rest_name);
    let rest_ptr_local = cons_locals.get(&fold.rest_name).copied().ok_or_else(|| {
        SimdError::new("sequence additive fold missing rest pointer local binding")
    })?;
    let rest_len_local = cons_locals.get(&rest_len_key).copied().ok_or_else(|| {
        SimdError::new("sequence additive fold missing rest length local binding")
    })?;

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(seq_len_local));
    function.instruction(&Instruction::I32Eqz);
    function.instruction(&Instruction::BrIf(1));
    if let Some(head_name) = &fold.head_name {
        let head_local = cons_locals
            .get(head_name)
            .copied()
            .ok_or_else(|| SimdError::new("sequence additive fold missing head local binding"))?;
        emit_bulk_element_load_at_prefix_index(function, seq_ptr_local, fold.elem_prim, 0)?;
        function.instruction(&Instruction::LocalSet(head_local));
    }
    let elem_bytes = i32::try_from(byte_width(fold.elem_prim))
        .map_err(|_| SimdError::new("sequence additive fold element width overflow"))?;
    function.instruction(&Instruction::LocalGet(seq_ptr_local));
    function.instruction(&Instruction::I32Const(elem_bytes));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(rest_ptr_local));
    function.instruction(&Instruction::LocalGet(seq_len_local));
    function.instruction(&Instruction::I32Const(1));
    function.instruction(&Instruction::I32Sub);
    function.instruction(&Instruction::LocalSet(rest_len_local));
    compile_fold_step_expr_with_inline_matchers(
        function,
        &fold.step_expr,
        &step_locals,
        lowered_map,
        scalar_indices,
        signatures,
        enum_ctors,
        wasm_enum_layouts,
        Some(enum_state),
        &BTreeMap::new(),
        &BTreeMap::new(),
        0,
    )?;
    function.instruction(&Instruction::LocalSet(acc_local));
    function.instruction(&Instruction::LocalGet(rest_ptr_local));
    function.instruction(&Instruction::LocalSet(seq_ptr_local));
    function.instruction(&Instruction::LocalGet(rest_len_local));
    function.instruction(&Instruction::LocalSet(seq_len_local));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::LocalGet(acc_local));
    Ok(())
}

fn detect_inlineable_enum_matcher(
    lowered_map: &BTreeMap<String, &LoweredFunction>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    name: &str,
) -> Option<InlineableEnumMatcher> {
    let lowered = lowered_map.get(name).copied()?;
    if lowered.tail_loop.is_some() {
        return None;
    }
    let checked = signatures.get(name).copied()?;
    let (params, _result) = checked.signature.ty.fun_parts();
    let [arg_ty] = params.as_slice() else {
        return None;
    };
    let Type::Named(enum_name, _) = arg_ty else {
        return None;
    };
    let clauses = match &lowered.kind {
        LoweredKind::Scalar { clauses } | LoweredKind::Structural { clauses, .. } => clauses,
        LoweredKind::Kernel { .. } => return None,
    };
    if clauses.is_empty() || clauses.len() > 8 {
        return None;
    }
    let mut matcher_clauses = Vec::with_capacity(clauses.len());
    let mut seen_default = false;
    let mut seen_ctors = BTreeSet::<String>::new();
    for (index, clause) in clauses.iter().enumerate() {
        let [pattern] = clause.patterns.as_slice() else {
            return None;
        };
        let (ctor, bindings) = collect_inlineable_enum_matcher_clause(
            &pattern.pattern,
            arg_ty,
            "__inline_enum_arg",
            enum_ctors,
        )?;
        if ctor.is_none() {
            if index + 1 != clauses.len() || seen_default {
                return None;
            }
            seen_default = true;
        } else if !seen_ctors.insert(ctor.clone().unwrap()) {
            return None;
        }
        matcher_clauses.push(InlineableEnumMatcherClause {
            ctor,
            bindings,
            body: clause.body.clone(),
        });
    }
    if !seen_default {
        let ctor_count = enum_ctors
            .values()
            .filter(|info| info.enum_name == *enum_name)
            .count();
        if seen_ctors.len() != ctor_count {
            return None;
        }
    }
    Some(InlineableEnumMatcher {
        arg_ty: arg_ty.clone(),
        clauses: matcher_clauses,
    })
}

fn collect_inlineable_enum_matcher_clause(
    pattern: &Pattern,
    arg_ty: &Type,
    arg_local_name: &str,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
) -> Option<(Option<String>, BTreeMap<String, IrExpr>)> {
    match pattern {
        Pattern::Wildcard => Some((None, BTreeMap::new())),
        Pattern::Name(name) => {
            if let Some(ctor) = enum_ctors.get(name) {
                let Type::Named(enum_name, _) = arg_ty else {
                    return None;
                };
                if ctor.enum_name == *enum_name && ctor.fields.is_empty() {
                    return Some((Some(name.clone()), BTreeMap::new()));
                }
            }
            Some((
                None,
                BTreeMap::from([(
                    name.clone(),
                    IrExpr {
                        ty: arg_ty.clone(),
                        kind: IrExprKind::Local(arg_local_name.to_string()),
                    },
                )]),
            ))
        }
        Pattern::Ctor(ctor_name, subpatterns) => {
            let ctor = enum_ctors.get(ctor_name)?;
            let Type::Named(enum_name, _) = arg_ty else {
                return None;
            };
            if ctor.enum_name != *enum_name || ctor.fields.len() != subpatterns.len() {
                return None;
            }
            let mut bindings = BTreeMap::new();
            let value_expr = IrExpr {
                ty: arg_ty.clone(),
                kind: IrExprKind::Local(arg_local_name.to_string()),
            };
            let mut recursive_slot = 0usize;
            for (field_index, (field_ty, subpattern)) in
                ctor.fields.iter().zip(subpatterns.iter()).enumerate()
            {
                let field_expr = if enum_field_is_recursive(field_ty, ctor) {
                    let expr = IrExpr {
                        ty: field_ty.clone(),
                        kind: IrExprKind::EnumChildBySlot {
                            value: Box::new(value_expr.clone()),
                            ctor: ctor_name.clone(),
                            slot: recursive_slot,
                        },
                    };
                    recursive_slot += 1;
                    expr
                } else {
                    IrExpr {
                        ty: field_ty.clone(),
                        kind: IrExprKind::EnumNonRecField {
                            value: Box::new(value_expr.clone()),
                            ctor: ctor_name.clone(),
                            field: field_index,
                        },
                    }
                };
                match subpattern {
                    Pattern::Wildcard => {}
                    Pattern::Name(name) => {
                        bindings.insert(name.clone(), field_expr);
                    }
                    _ => return None,
                }
            }
            Some((Some(ctor_name.clone()), bindings))
        }
        _ => None,
    }
}

fn inlineable_scalar_result_val_type(ty: &Type) -> Option<ValType> {
    match ty {
        Type::Scalar(prim) => Some(wasm_val_type(*prim)),
        Type::Index(_) => Some(ValType::I64),
        Type::Named(name, args) if is_wasm_enum_named_type(name, args) => Some(ValType::I32),
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _)
            if wasm_star_seq_storage_prim(ty).is_some() =>
        {
            Some(ValType::I32)
        }
        _ => None,
    }
}

fn wasm_scalar_call_types_match(actual: &Type, expected: &Type) -> bool {
    match (
        inlineable_scalar_result_val_type(actual),
        inlineable_scalar_result_val_type(expected),
    ) {
        (Some(actual), Some(expected)) => actual == expected,
        _ => actual == expected,
    }
}

fn compile_fold_step_expr_with_inline_matchers(
    function: &mut Function,
    expr: &IrExpr,
    locals: &BTreeMap<String, u32>,
    lowered_map: &BTreeMap<String, &LoweredFunction>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: Option<EnumWasmState>,
    hoisted_locals: &BTreeMap<HoistExprKey, u32>,
    inline_bindings: &BTreeMap<String, IrExpr>,
    inline_depth: usize,
) -> Result<()> {
    match &expr.kind {
        IrExprKind::Local(name) => {
            if let Some(inline_expr) = inline_bindings.get(name) {
                compile_fold_step_expr_with_inline_matchers(
                    function,
                    inline_expr,
                    locals,
                    lowered_map,
                    scalar_indices,
                    signatures,
                    enum_ctors,
                    wasm_enum_layouts,
                    enum_state,
                    hoisted_locals,
                    inline_bindings,
                    inline_depth,
                )
            } else {
                compile_scalar_ir_expr_with_hoists(
                    function,
                    expr,
                    locals,
                    scalar_indices,
                    signatures,
                    enum_ctors,
                    wasm_enum_layouts,
                    enum_state,
                    hoisted_locals,
                    inline_bindings,
                )
            }
        }
        IrExprKind::Call {
            callee: Callee::Prim(op),
            args,
        } => {
            compile_fold_step_expr_with_inline_matchers(
                function,
                &args[0],
                locals,
                lowered_map,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                enum_state,
                hoisted_locals,
                inline_bindings,
                inline_depth,
            )?;
            compile_fold_step_expr_with_inline_matchers(
                function,
                &args[1],
                locals,
                lowered_map,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                enum_state,
                hoisted_locals,
                inline_bindings,
                inline_depth,
            )?;
            emit_scalar_primitive(function, *op, expr.ty.prim().unwrap_or(Prim::I64), args)?;
            Ok(())
        }
        IrExprKind::Call {
            callee: Callee::Function(name),
            args,
        } if args.len() == 1 => {
            let Some(enum_state) = enum_state else {
                return compile_scalar_ir_expr_with_hoists(
                    function,
                    expr,
                    locals,
                    scalar_indices,
                    signatures,
                    enum_ctors,
                    wasm_enum_layouts,
                    enum_state,
                    hoisted_locals,
                    inline_bindings,
                );
            };
            if inline_depth >= 2 {
                return compile_scalar_ir_expr_with_hoists(
                    function,
                    expr,
                    locals,
                    scalar_indices,
                    signatures,
                    enum_ctors,
                    wasm_enum_layouts,
                    Some(enum_state),
                    hoisted_locals,
                    inline_bindings,
                );
            }
            let Some(matcher) =
                detect_inlineable_enum_matcher(lowered_map, signatures, enum_ctors, name)
            else {
                return compile_scalar_ir_expr_with_hoists(
                    function,
                    expr,
                    locals,
                    scalar_indices,
                    signatures,
                    enum_ctors,
                    wasm_enum_layouts,
                    Some(enum_state),
                    hoisted_locals,
                    inline_bindings,
                );
            };
            if matcher.arg_ty != args[0].ty || inlineable_scalar_result_val_type(&expr.ty).is_none()
            {
                return compile_scalar_ir_expr_with_hoists(
                    function,
                    expr,
                    locals,
                    scalar_indices,
                    signatures,
                    enum_ctors,
                    wasm_enum_layouts,
                    Some(enum_state),
                    hoisted_locals,
                    inline_bindings,
                );
            }
            emit_inline_enum_matcher_call(
                function,
                &matcher,
                &args[0],
                expr,
                locals,
                lowered_map,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                enum_state,
                hoisted_locals,
                inline_bindings,
                inline_depth + 1,
            )
        }
        IrExprKind::Let { bindings, body } => {
            let mut extended = inline_bindings.clone();
            for binding in bindings {
                extended.insert(binding.name.clone(), binding.expr.clone());
            }
            compile_fold_step_expr_with_inline_matchers(
                function,
                body,
                locals,
                lowered_map,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                enum_state,
                hoisted_locals,
                &extended,
                inline_depth,
            )
        }
        _ => compile_scalar_ir_expr_with_hoists(
            function,
            expr,
            locals,
            scalar_indices,
            signatures,
            enum_ctors,
            wasm_enum_layouts,
            enum_state,
            hoisted_locals,
            inline_bindings,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
fn emit_inline_enum_matcher_call(
    function: &mut Function,
    matcher: &InlineableEnumMatcher,
    arg_expr: &IrExpr,
    result_expr: &IrExpr,
    locals: &BTreeMap<String, u32>,
    lowered_map: &BTreeMap<String, &LoweredFunction>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: EnumWasmState,
    hoisted_locals: &BTreeMap<HoistExprKey, u32>,
    inline_bindings: &BTreeMap<String, IrExpr>,
    inline_depth: usize,
) -> Result<()> {
    let result_val_type = inlineable_scalar_result_val_type(&result_expr.ty)
        .ok_or_else(|| SimdError::new("inline enum matcher result type is not Wasm-scalar"))?;
    compile_scalar_ir_expr_with_hoists(
        function,
        arg_expr,
        locals,
        scalar_indices,
        signatures,
        enum_ctors,
        wasm_enum_layouts,
        Some(enum_state),
        hoisted_locals,
        inline_bindings,
    )?;
    function.instruction(&Instruction::LocalSet(enum_state.aux0_local));
    emit_enum_tag_load_from_value(function, enum_state.aux0_local, enum_state);
    function.instruction(&Instruction::LocalSet(enum_state.aux1_local));
    let mut extended_locals = locals.clone();
    extended_locals.insert("__inline_enum_arg".to_string(), enum_state.aux0_local);
    emit_inline_enum_matcher_clause_chain(
        function,
        &matcher.clauses,
        0,
        &extended_locals,
        lowered_map,
        scalar_indices,
        signatures,
        enum_ctors,
        wasm_enum_layouts,
        enum_state,
        hoisted_locals,
        inline_bindings,
        result_val_type,
        inline_depth,
    )
}

#[allow(clippy::too_many_arguments)]
fn emit_inline_enum_matcher_clause_chain(
    function: &mut Function,
    clauses: &[InlineableEnumMatcherClause],
    index: usize,
    locals: &BTreeMap<String, u32>,
    lowered_map: &BTreeMap<String, &LoweredFunction>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: EnumWasmState,
    hoisted_locals: &BTreeMap<HoistExprKey, u32>,
    inline_bindings: &BTreeMap<String, IrExpr>,
    result_val_type: ValType,
    inline_depth: usize,
) -> Result<()> {
    let clause = clauses
        .get(index)
        .ok_or_else(|| SimdError::new("inline enum matcher exhausted all clauses"))?;
    let mut clause_bindings = inline_bindings.clone();
    clause_bindings.extend(clause.bindings.clone());
    if let Some(ctor_name) = &clause.ctor {
        let ctor = enum_ctors
            .get(ctor_name)
            .ok_or_else(|| SimdError::new(format!("unknown constructor '{}'", ctor_name)))?;
        function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
        function.instruction(&Instruction::I32Const(i32::from(ctor.tag)));
        function.instruction(&Instruction::I32Eq);
        function.instruction(&Instruction::If(BlockType::Result(result_val_type)));
        compile_fold_step_expr_with_inline_matchers(
            function,
            &clause.body,
            locals,
            lowered_map,
            scalar_indices,
            signatures,
            enum_ctors,
            wasm_enum_layouts,
            Some(enum_state),
            hoisted_locals,
            &clause_bindings,
            inline_depth,
        )?;
        function.instruction(&Instruction::Else);
        emit_inline_enum_matcher_clause_chain(
            function,
            clauses,
            index + 1,
            locals,
            lowered_map,
            scalar_indices,
            signatures,
            enum_ctors,
            wasm_enum_layouts,
            enum_state,
            hoisted_locals,
            inline_bindings,
            result_val_type,
            inline_depth,
        )?;
        function.instruction(&Instruction::End);
        Ok(())
    } else {
        compile_fold_step_expr_with_inline_matchers(
            function,
            &clause.body,
            locals,
            lowered_map,
            scalar_indices,
            signatures,
            enum_ctors,
            wasm_enum_layouts,
            Some(enum_state),
            hoisted_locals,
            &clause_bindings,
            inline_depth,
        )
    }
}

fn compile_kernel_entry(
    lowered: &LoweredFunction,
    checked: &CheckedFunction,
    lowered_map: &BTreeMap<String, &LoweredFunction>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    intent: IntentClass,
    function_profile: Option<WasmFunctionInstrumentation>,
) -> Result<CompiledKernelEntry> {
    let (param_types, result_ty) = checked.signature.ty.fun_parts();
    let Type::Bulk(result_prim, _) = result_ty else {
        return Err(SimdError::new(format!(
            "kernel entry '{}' must return bulk",
            checked.name
        )));
    };
    let LoweredKind::Kernel {
        vector_width,
        clauses,
        ..
    } = &lowered.kind
    else {
        return Err(SimdError::new(format!(
            "compile_kernel_entry received scalar function '{}'",
            checked.name
        )));
    };
    let clauses = clauses
        .iter()
        .map(|clause| {
            Ok(LoweredClause {
                patterns: clause.patterns.clone(),
                body: inline_grouped_kernel_expr(&clause.body, lowered_map, &mut BTreeSet::new())?,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let mut abi_params = Vec::<KernelParam>::new();
    let mut wasm_param_index = 0u32;
    for ty in &param_types {
        match ty {
            Type::Scalar(prim) => {
                abi_params.push(KernelParam::Same {
                    prim: *prim,
                    value_local: wasm_param_index,
                });
                wasm_param_index += 1;
            }
            Type::Index(_) => {
                abi_params.push(KernelParam::Same {
                    prim: Prim::I64,
                    value_local: wasm_param_index,
                });
                wasm_param_index += 1;
            }
            Type::Bulk(prim, _) => {
                abi_params.push(KernelParam::Lane {
                    prim: *prim,
                    ptr_local: wasm_param_index,
                    len_local: wasm_param_index + 1,
                });
                wasm_param_index += 2;
            }
            Type::StarSeq(_) | Type::StarSeqWitnessed(_, _)
                if wasm_star_seq_storage_prim(ty).is_some() =>
            {
                let prim = wasm_star_seq_storage_prim(ty).unwrap();
                abi_params.push(KernelParam::SameSeq {
                    prim,
                    ptr_local: wasm_param_index,
                    len_local: wasm_param_index + 1,
                });
                wasm_param_index += 2;
            }
            Type::TypeToken(_) => {
                return Err(SimdError::new(
                    "Wasm backend does not support Type witness kernel parameters",
                ));
            }
            Type::StarSeq(_) | Type::StarSeqWitnessed(_, _) => {
                return Err(SimdError::new(
                    "Wasm backend does not yet support T[*] kernel parameters",
                ));
            }
            Type::Tuple(_) | Type::Record(_) => {
                return Err(SimdError::new(
                    "Wasm backend does not yet support tuple/record kernel parameters",
                ));
            }
            Type::Named(_, _) | Type::Var(_) | Type::Infer(_) => {
                return Err(SimdError::new(
                    "Wasm backend does not support unresolved polymorphic kernel parameters",
                ));
            }
            Type::Fun(_, _) => {
                return Err(SimdError::new(
                    "Wasm backend does not support higher-order kernel parameters",
                ));
            }
        }
    }
    let output_ptr_local = wasm_param_index;
    let output_len_local = wasm_param_index + 1;

    let mut local_decls = vec![
        (1, ValType::I32),
        (1, ValType::I32),
        (1, ValType::I32),
        (1, ValType::I32),
    ];
    let index_local = wasm_param_index + 2;
    let vector_end_local = wasm_param_index + 3;
    let vector_unrolled_end_local = wasm_param_index + 4;
    let vector_unrolled4_end_local = wasm_param_index + 5;
    let mut next_local = wasm_param_index + 6;

    let mut lane_locals = Vec::<Option<u32>>::new();
    let mut lane_ptr_locals = Vec::<Option<u32>>::new();
    for param in &abi_params {
        match param {
            KernelParam::Same { .. } | KernelParam::SameSeq { .. } => {
                lane_locals.push(None);
                lane_ptr_locals.push(None);
            }
            KernelParam::Lane { prim, .. } => {
                local_decls.push((1, wasm_val_type(*prim)));
                lane_locals.push(Some(next_local));
                next_local += 1;
                local_decls.push((1, ValType::I32));
                lane_ptr_locals.push(Some(next_local));
                next_local += 1;
            }
        }
    }
    local_decls.push((1, ValType::I32));
    let output_ptr_loop_local = next_local;
    next_local += 1;
    let enum_state = append_enum_scratch_locals(&mut local_decls, &mut next_local);

    let first_len_local = abi_params
        .iter()
        .find_map(|param| match param {
            KernelParam::Lane { len_local, .. } => Some(*len_local),
            KernelParam::Same { .. } | KernelParam::SameSeq { .. } => None,
        })
        .ok_or_else(|| SimdError::new("kernel entry requires at least one bulk parameter"))?;

    let vector_clause = vectorizable_kernel_clause(&clauses, &abi_params, result_prim);
    let op_count = clauses
        .iter()
        .map(|clause| count_primitive_ops(&clause.body))
        .sum::<usize>();
    let load_streams = abi_params
        .iter()
        .filter(|param| matches!(param, KernelParam::Lane { .. }))
        .count();
    let (vector_plan, fallback_reason) = choose_vector_plan(
        vector_clause.is_some(),
        *vector_width,
        result_prim,
        op_count,
        load_streams,
        1,
    );
    let vector_acc_local = if vector_clause.is_some() {
        let local = next_local;
        local_decls.push((1, ValType::V128));
        next_local += 1;
        Some(local)
    } else {
        None
    };
    let variant_locals = kernel_variant_locals(&clauses, &abi_params);
    let available_locals = kernel_available_locals(&clauses, &abi_params);
    let vector_hoists = vector_clause
        .as_ref()
        .map(|vector_clause| {
            if vector_clause.clauses.len() == 1 {
                collect_hoisted_exprs(
                    std::iter::once(vector_clause.clauses[0].body),
                    &available_locals,
                    &variant_locals,
                    HoistMode::Vector,
                )
            } else {
                Vec::new()
            }
        })
        .unwrap_or_default();
    for _ in &vector_hoists {
        local_decls.push((1, ValType::V128));
    }
    let cleanup_hoists = collect_hoisted_exprs(
        clauses.iter().map(|clause| &clause.body),
        &available_locals,
        &variant_locals,
        HoistMode::ScalarCleanup,
    );
    for hoisted in &cleanup_hoists {
        let prim = hoisted.expr.ty.prim().ok_or_else(|| {
            SimdError::new("cleanup hoisted expression did not have a scalar type")
        })?;
        local_decls.push((1, wasm_val_type(prim)));
    }

    let mut function = Function::new(local_decls);
    emit_wasm_function_profile_enter(&mut function, function_profile);
    let mut next_hoisted_local = next_local;
    let vector_hoisted_locals = vector_hoists
        .iter()
        .map(|hoisted| {
            let local = next_hoisted_local;
            next_hoisted_local += 1;
            (hoisted.key.clone(), local)
        })
        .collect::<BTreeMap<_, _>>();
    let cleanup_hoisted_locals = cleanup_hoists
        .iter()
        .map(|hoisted| {
            let local = next_hoisted_local;
            next_hoisted_local += 1;
            (hoisted.key.clone(), local)
        })
        .collect::<BTreeMap<_, _>>();

    for param in &abi_params {
        if let KernelParam::Lane { len_local, .. } = param {
            if *len_local == first_len_local {
                continue;
            }
            function.instruction(&Instruction::LocalGet(*len_local));
            function.instruction(&Instruction::LocalGet(first_len_local));
            function.instruction(&Instruction::I32Ne);
            function.instruction(&Instruction::If(BlockType::Empty));
            function.instruction(&Instruction::Unreachable);
            function.instruction(&Instruction::End);
        }
    }
    function.instruction(&Instruction::LocalGet(output_len_local));
    function.instruction(&Instruction::LocalGet(first_len_local));
    function.instruction(&Instruction::I32Ne);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);

    function.instruction(&Instruction::I32Const(0));
    function.instruction(&Instruction::LocalSet(index_local));

    for (param, loop_ptr_local) in abi_params.iter().zip(&lane_ptr_locals) {
        if let (KernelParam::Lane { ptr_local, .. }, Some(loop_ptr_local)) = (param, loop_ptr_local)
        {
            function.instruction(&Instruction::LocalGet(*ptr_local));
            function.instruction(&Instruction::LocalSet(*loop_ptr_local));
        }
    }
    function.instruction(&Instruction::LocalGet(output_ptr_local));
    function.instruction(&Instruction::LocalSet(output_ptr_loop_local));

    let vector_params = abi_params
        .iter()
        .zip(&lane_ptr_locals)
        .map(|(param, ptr_local)| match (param, ptr_local) {
            (KernelParam::Same { prim, value_local }, _) => KernelParam::Same {
                prim: *prim,
                value_local: *value_local,
            },
            (
                KernelParam::SameSeq {
                    prim,
                    ptr_local,
                    len_local,
                },
                _,
            ) => KernelParam::SameSeq {
                prim: *prim,
                ptr_local: *ptr_local,
                len_local: *len_local,
            },
            (
                KernelParam::Lane {
                    prim, len_local, ..
                },
                Some(loop_ptr_local),
            ) => KernelParam::Lane {
                prim: *prim,
                ptr_local: *loop_ptr_local,
                len_local: *len_local,
            },
            (KernelParam::Lane { .. }, None) => *param,
        })
        .collect::<Vec<_>>();

    if let Some(vector_clause) = vector_clause {
        emit_vector_hoists(
            &mut function,
            &vector_hoists,
            &vector_hoisted_locals,
            &vector_params,
            VectorAddressing::Pointer {
                lane_offset_elems: 0,
            },
            scalar_indices,
            &vector_clause
                .clauses
                .first()
                .ok_or_else(|| SimdError::new("vectorized kernel had no clauses"))?
                .locals,
        )?;
        function.instruction(&Instruction::LocalGet(first_len_local));
        function.instruction(&Instruction::LocalGet(first_len_local));
        function.instruction(&Instruction::I32Const(*vector_width as i32));
        function.instruction(&Instruction::I32RemU);
        function.instruction(&Instruction::I32Sub);
        function.instruction(&Instruction::LocalSet(vector_end_local));

        if matches!(vector_plan, VectorPlan::VectorUnroll4) {
            function.instruction(&Instruction::LocalGet(vector_end_local));
            function.instruction(&Instruction::LocalGet(vector_end_local));
            function.instruction(&Instruction::I32Const((*vector_width as i32) * 4));
            function.instruction(&Instruction::I32RemU);
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::LocalSet(vector_unrolled4_end_local));

            function.instruction(&Instruction::Block(BlockType::Empty));
            function.instruction(&Instruction::Loop(BlockType::Empty));
            function.instruction(&Instruction::LocalGet(index_local));
            function.instruction(&Instruction::LocalGet(vector_unrolled4_end_local));
            function.instruction(&Instruction::I32GeU);
            function.instruction(&Instruction::BrIf(1));

            for chunk in 0..4u32 {
                function.instruction(&Instruction::LocalGet(output_ptr_loop_local));
                compile_vectorized_clause_chain(
                    &mut function,
                    &vector_clause.clauses,
                    &vector_params,
                    VectorAddressing::Pointer {
                        lane_offset_elems: *vector_width as u32 * chunk,
                    },
                    scalar_indices,
                    Some(&vector_hoisted_locals),
                    vector_acc_local.ok_or_else(|| {
                        SimdError::new("vectorized kernel missing accumulator local")
                    })?,
                )?;
                function.instruction(&Instruction::V128Store(memarg(
                    u64::from(byte_width(result_prim)) * (*vector_width as u64) * chunk as u64,
                    4,
                )));
            }

            for param in &vector_params {
                if let KernelParam::Lane {
                    prim, ptr_local, ..
                } = param
                {
                    emit_pointer_bump(
                        &mut function,
                        *ptr_local,
                        byte_width(*prim) * (*vector_width as u32) * 4,
                    );
                }
            }
            emit_pointer_bump(
                &mut function,
                output_ptr_loop_local,
                byte_width(result_prim) * (*vector_width as u32) * 4,
            );

            function.instruction(&Instruction::LocalGet(index_local));
            function.instruction(&Instruction::I32Const((*vector_width as i32) * 4));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(index_local));
            function.instruction(&Instruction::Br(0));
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);
        }

        if matches!(
            vector_plan,
            VectorPlan::VectorUnroll2 | VectorPlan::VectorUnroll4
        ) {
            function.instruction(&Instruction::LocalGet(vector_end_local));
            function.instruction(&Instruction::LocalGet(vector_end_local));
            function.instruction(&Instruction::I32Const((*vector_width as i32) * 2));
            function.instruction(&Instruction::I32RemU);
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::LocalSet(vector_unrolled_end_local));

            function.instruction(&Instruction::Block(BlockType::Empty));
            function.instruction(&Instruction::Loop(BlockType::Empty));
            function.instruction(&Instruction::LocalGet(index_local));
            function.instruction(&Instruction::LocalGet(vector_unrolled_end_local));
            function.instruction(&Instruction::I32GeU);
            function.instruction(&Instruction::BrIf(1));

            for chunk in 0..2u32 {
                function.instruction(&Instruction::LocalGet(output_ptr_loop_local));
                compile_vectorized_clause_chain(
                    &mut function,
                    &vector_clause.clauses,
                    &vector_params,
                    VectorAddressing::Pointer {
                        lane_offset_elems: *vector_width as u32 * chunk,
                    },
                    scalar_indices,
                    Some(&vector_hoisted_locals),
                    vector_acc_local.ok_or_else(|| {
                        SimdError::new("vectorized kernel missing accumulator local")
                    })?,
                )?;
                function.instruction(&Instruction::V128Store(memarg(
                    u64::from(byte_width(result_prim)) * (*vector_width as u64) * chunk as u64,
                    4,
                )));
            }

            for param in &vector_params {
                if let KernelParam::Lane {
                    prim, ptr_local, ..
                } = param
                {
                    emit_pointer_bump(
                        &mut function,
                        *ptr_local,
                        byte_width(*prim) * (*vector_width as u32) * 2,
                    );
                }
            }
            emit_pointer_bump(
                &mut function,
                output_ptr_loop_local,
                byte_width(result_prim) * (*vector_width as u32) * 2,
            );

            function.instruction(&Instruction::LocalGet(index_local));
            function.instruction(&Instruction::I32Const((*vector_width as i32) * 2));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(index_local));
            function.instruction(&Instruction::Br(0));
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);
        }

        if !matches!(vector_plan, VectorPlan::ScalarOnly) {
            function.instruction(&Instruction::Block(BlockType::Empty));
            function.instruction(&Instruction::Loop(BlockType::Empty));
            function.instruction(&Instruction::LocalGet(index_local));
            function.instruction(&Instruction::LocalGet(vector_end_local));
            function.instruction(&Instruction::I32GeU);
            function.instruction(&Instruction::BrIf(1));

            function.instruction(&Instruction::LocalGet(output_ptr_loop_local));
            compile_vectorized_clause_chain(
                &mut function,
                &vector_clause.clauses,
                &vector_params,
                VectorAddressing::Pointer {
                    lane_offset_elems: 0,
                },
                scalar_indices,
                Some(&vector_hoisted_locals),
                vector_acc_local
                    .ok_or_else(|| SimdError::new("vectorized kernel missing accumulator local"))?,
            )?;
            function.instruction(&Instruction::V128Store(memarg(0, 4)));

            for param in &vector_params {
                if let KernelParam::Lane {
                    prim, ptr_local, ..
                } = param
                {
                    emit_pointer_bump(
                        &mut function,
                        *ptr_local,
                        byte_width(*prim) * (*vector_width as u32),
                    );
                }
            }
            emit_pointer_bump(
                &mut function,
                output_ptr_loop_local,
                byte_width(result_prim) * (*vector_width as u32),
            );

            function.instruction(&Instruction::LocalGet(index_local));
            function.instruction(&Instruction::I32Const(*vector_width as i32));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(index_local));
            function.instruction(&Instruction::Br(0));
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);
        }
    }

    let scalar_invariant_locals = kernel_same_locals(&clauses, &abi_params);
    emit_scalar_hoists(
        &mut function,
        &cleanup_hoists,
        &cleanup_hoisted_locals,
        &scalar_invariant_locals,
        scalar_indices,
        signatures,
    )?;
    let cleanup_inline_bindings = BTreeMap::<String, IrExpr>::new();

    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::LocalGet(first_len_local));
    function.instruction(&Instruction::I32GeU);
    function.instruction(&Instruction::BrIf(1));

    for ((param, lane_local), loop_ptr_local) in
        abi_params.iter().zip(&lane_locals).zip(&lane_ptr_locals)
    {
        if let (KernelParam::Lane { prim, .. }, Some(local), Some(loop_ptr_local)) =
            (param, lane_local, loop_ptr_local)
        {
            emit_lane_load_scalar_at_ptr(&mut function, *loop_ptr_local, *prim);
            function.instruction(&Instruction::LocalSet(*local));
        }
    }

    let scalar_locals = kernel_scalar_local_map(&clauses, &abi_params, &lane_locals);
    let empty_enum_ctors = BTreeMap::<String, EnumCtorInfo>::new();
    let empty_wasm_enum_layouts = BTreeMap::<String, WasmEnumLayout>::new();
    for clause in &clauses {
        let clause_key = clause as *const _ as usize;
        let locals = scalar_locals.get(&clause_key).cloned().unwrap_or_default();
        if clause_has_condition(&clause.patterns, &empty_enum_ctors) {
            emit_matching_if(
                &mut function,
                &clause.patterns,
                &empty_enum_ctors,
                &empty_wasm_enum_layouts,
                None,
            )?;
            function.instruction(&Instruction::LocalGet(output_ptr_loop_local));
            compile_scalar_ir_expr_with_hoists(
                &mut function,
                &clause.body,
                &locals,
                scalar_indices,
                signatures,
                &empty_enum_ctors,
                &empty_wasm_enum_layouts,
                Some(enum_state),
                &cleanup_hoisted_locals,
                &cleanup_inline_bindings,
            )?;
            emit_scalar_store(&mut function, result_prim);
            for param in &vector_params {
                if let KernelParam::Lane {
                    prim, ptr_local, ..
                } = param
                {
                    emit_pointer_bump(&mut function, *ptr_local, byte_width(*prim));
                }
            }
            emit_pointer_bump(
                &mut function,
                output_ptr_loop_local,
                byte_width(result_prim),
            );
            emit_scalar_index_bump(&mut function, index_local);
            function.instruction(&Instruction::Br(1));
            function.instruction(&Instruction::End);
        } else {
            function.instruction(&Instruction::LocalGet(output_ptr_loop_local));
            compile_scalar_ir_expr_with_hoists(
                &mut function,
                &clause.body,
                &locals,
                scalar_indices,
                signatures,
                &empty_enum_ctors,
                &empty_wasm_enum_layouts,
                Some(enum_state),
                &cleanup_hoisted_locals,
                &cleanup_inline_bindings,
            )?;
            emit_scalar_store(&mut function, result_prim);
            for param in &vector_params {
                if let KernelParam::Lane {
                    prim, ptr_local, ..
                } = param
                {
                    emit_pointer_bump(&mut function, *ptr_local, byte_width(*prim));
                }
            }
            emit_pointer_bump(
                &mut function,
                output_ptr_loop_local,
                byte_width(result_prim),
            );
            emit_scalar_index_bump(&mut function, index_local);
            function.instruction(&Instruction::Br(0));
        }
    }
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    emit_wasm_function_profile_exit(&mut function, function_profile);
    function.instruction(&Instruction::End);
    Ok(CompiledKernelEntry {
        function,
        report: WasmOptimizationReport {
            function: checked.name.clone(),
            intent,
            structural_exec: StructuralExecMode::Scalar,
            vectorizable: !matches!(vector_plan, VectorPlan::ScalarOnly),
            vector_unroll: vector_plan_unroll(vector_plan),
            fallback_reason,
            structural_scc: None,
            structural_state_count: 0,
            structural_transition_count: 0,
            structural_span_ops: 0,
            structural_enum_ops: 0,
            structural_region_count: 0,
            structural_char_prefix_regions: 0,
            structural_separated_item_regions: 0,
        },
    })
}

#[derive(Debug, Clone, Copy)]
enum KernelParam {
    Same {
        prim: Prim,
        value_local: u32,
    },
    SameSeq {
        prim: Prim,
        ptr_local: u32,
        len_local: u32,
    },
    Lane {
        prim: Prim,
        ptr_local: u32,
        len_local: u32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HoistMode {
    Vector,
    ScalarCleanup,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum HoistExprKey {
    Local(String),
    Int(i64, Prim),
    Float(u64, Prim),
    Seq(Vec<HoistExprKey>),
    Record(Vec<(String, HoistExprKey)>),
    EnumCtor {
        ctor: String,
        args: Vec<HoistExprKey>,
    },
    EnumTag {
        value: Box<HoistExprKey>,
    },
    EnumChildBySlot {
        value: Box<HoistExprKey>,
        ctor: String,
        slot: usize,
    },
    EnumNonRecField {
        value: Box<HoistExprKey>,
        ctor: String,
        field: usize,
    },
    Let {
        bindings: Vec<(String, HoistExprKey)>,
        body: Box<HoistExprKey>,
    },
    Call {
        callee: HoistCalleeKey,
        args: Vec<HoistExprKey>,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum HoistCalleeKey {
    Prim(PrimOpKey),
    Function(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum PrimOpKey {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    And,
    Or,
    Eq,
    Lt,
    Gt,
    Le,
    Ge,
}

#[derive(Debug, Clone)]
struct HoistedExpr<'a> {
    key: HoistExprKey,
    expr: &'a IrExpr,
}

fn kernel_scalar_local_map(
    clauses: &[LoweredClause],
    params: &[KernelParam],
    lane_locals: &[Option<u32>],
) -> BTreeMap<usize, BTreeMap<String, u32>> {
    let mut map = BTreeMap::new();
    for clause in clauses {
        let mut locals = BTreeMap::new();
        for ((pattern, param), lane_local) in clause.patterns.iter().zip(params).zip(lane_locals) {
            if let Pattern::Name(name) = &pattern.pattern {
                match (param, lane_local) {
                    (KernelParam::Same { value_local, .. }, _) => {
                        locals.insert(name.clone(), *value_local);
                    }
                    (
                        KernelParam::SameSeq {
                            ptr_local,
                            len_local,
                            ..
                        },
                        _,
                    ) => {
                        locals.insert(name.clone(), *ptr_local);
                        locals.insert(bulk_len_local_name(name), *len_local);
                    }
                    (KernelParam::Lane { .. }, Some(local)) => {
                        locals.insert(name.clone(), *local);
                    }
                    (KernelParam::Lane { .. }, None) => {}
                }
            }
        }
        map.insert(clause as *const _ as usize, locals);
    }
    map
}

fn kernel_variant_locals(clauses: &[LoweredClause], params: &[KernelParam]) -> BTreeSet<String> {
    let mut locals = BTreeSet::new();
    for clause in clauses {
        for (pattern, param) in clause.patterns.iter().zip(params) {
            if matches!(param, KernelParam::Lane { .. })
                && let Pattern::Name(name) = &pattern.pattern
            {
                locals.insert(name.clone());
            }
        }
    }
    locals
}

fn kernel_available_locals(clauses: &[LoweredClause], params: &[KernelParam]) -> BTreeSet<String> {
    let mut locals = kernel_variant_locals(clauses, params);
    for name in kernel_same_locals(clauses, params).into_keys() {
        locals.insert(name);
    }
    locals
}

fn kernel_same_locals(clauses: &[LoweredClause], params: &[KernelParam]) -> BTreeMap<String, u32> {
    let mut locals = BTreeMap::new();
    for clause in clauses {
        for (pattern, param) in clause.patterns.iter().zip(params) {
            if let Pattern::Name(name) = &pattern.pattern {
                match param {
                    KernelParam::Same { value_local, .. } => {
                        locals.insert(name.clone(), *value_local);
                    }
                    KernelParam::SameSeq {
                        ptr_local,
                        len_local,
                        ..
                    } => {
                        locals.insert(name.clone(), *ptr_local);
                        locals.insert(bulk_len_local_name(name), *len_local);
                    }
                    KernelParam::Lane { .. } => {}
                }
            }
        }
    }
    locals
}

fn collect_hoisted_exprs<'a, I>(
    roots: I,
    available_locals: &BTreeSet<String>,
    variant_locals: &BTreeSet<String>,
    mode: HoistMode,
) -> Vec<HoistedExpr<'a>>
where
    I: IntoIterator<Item = &'a IrExpr>,
{
    let mut seen = BTreeSet::new();
    let mut hoisted = Vec::new();
    for root in roots {
        collect_hoisted_expr(
            root,
            available_locals,
            variant_locals,
            mode,
            &mut seen,
            &mut hoisted,
        );
    }
    hoisted
}

fn collect_hoisted_expr<'a>(
    expr: &'a IrExpr,
    available_locals: &BTreeSet<String>,
    variant_locals: &BTreeSet<String>,
    mode: HoistMode,
    seen: &mut BTreeSet<HoistExprKey>,
    out: &mut Vec<HoistedExpr<'a>>,
) -> bool {
    let invariant = match &expr.kind {
        IrExprKind::Local(name) => {
            available_locals.contains(name) && !variant_locals.contains(name)
        }
        IrExprKind::Int(_, _) | IrExprKind::Float(_, _) => true,
        IrExprKind::Seq(items) => {
            let mut invariant = true;
            for item in items {
                invariant &=
                    collect_hoisted_expr(item, available_locals, variant_locals, mode, seen, out);
            }
            invariant
        }
        IrExprKind::SeqSplice { prefix, tail } => {
            let mut invariant = true;
            for item in prefix {
                invariant &=
                    collect_hoisted_expr(item, available_locals, variant_locals, mode, seen, out);
            }
            invariant &=
                collect_hoisted_expr(tail, available_locals, variant_locals, mode, seen, out);
            invariant
        }
        IrExprKind::Record(fields) => {
            let mut invariant = true;
            for field in fields.values() {
                invariant &=
                    collect_hoisted_expr(field, available_locals, variant_locals, mode, seen, out);
            }
            invariant
        }
        IrExprKind::EnumCtor { args, .. } => {
            let mut invariant = true;
            for arg in args {
                invariant &=
                    collect_hoisted_expr(arg, available_locals, variant_locals, mode, seen, out);
            }
            invariant
        }
        IrExprKind::EnumTag { value }
        | IrExprKind::EnumChildBySlot { value, .. }
        | IrExprKind::EnumNonRecField { value, .. } => {
            collect_hoisted_expr(value, available_locals, variant_locals, mode, seen, out)
        }
        IrExprKind::Let { bindings, body } => {
            let mut invariant = true;
            let mut body_available_locals = available_locals.clone();
            let mut body_variant_locals = variant_locals.clone();
            for binding in bindings {
                let binding_invariant = collect_hoisted_expr(
                    &binding.expr,
                    &body_available_locals,
                    &body_variant_locals,
                    mode,
                    seen,
                    out,
                );
                invariant &= binding_invariant;
                body_available_locals.insert(binding.name.clone());
                if binding_invariant {
                    body_variant_locals.remove(&binding.name);
                } else {
                    body_variant_locals.insert(binding.name.clone());
                }
            }
            invariant &= collect_hoisted_expr(
                body,
                &body_available_locals,
                &body_variant_locals,
                mode,
                seen,
                out,
            );
            invariant
        }
        IrExprKind::Call { args, .. } => {
            let mut invariant = true;
            for arg in args {
                invariant &=
                    collect_hoisted_expr(arg, available_locals, variant_locals, mode, seen, out);
            }
            invariant
        }
    };
    if invariant && should_hoist_expr(expr, mode) {
        let key = hoist_expr_key(expr);
        if seen.insert(key.clone()) {
            out.push(HoistedExpr { key, expr });
        }
    }
    invariant
}

fn should_hoist_expr(expr: &IrExpr, mode: HoistMode) -> bool {
    match mode {
        HoistMode::Vector => !matches!(
            expr.kind,
            IrExprKind::Local(_) | IrExprKind::Int(_, _) | IrExprKind::Float(_, _)
        ),
        HoistMode::ScalarCleanup => !matches!(
            expr.kind,
            IrExprKind::Local(_) | IrExprKind::Int(_, _) | IrExprKind::Float(_, _)
        ),
    }
}

fn hoist_expr_key(expr: &IrExpr) -> HoistExprKey {
    match &expr.kind {
        IrExprKind::Local(name) => HoistExprKey::Local(name.clone()),
        IrExprKind::Int(value, prim) => HoistExprKey::Int(*value, *prim),
        IrExprKind::Float(value, prim) => HoistExprKey::Float(value.to_bits(), *prim),
        IrExprKind::Seq(items) => HoistExprKey::Seq(items.iter().map(hoist_expr_key).collect()),
        IrExprKind::SeqSplice { prefix, tail } => HoistExprKey::Seq(
            prefix
                .iter()
                .map(hoist_expr_key)
                .chain(std::iter::once(hoist_expr_key(tail)))
                .collect(),
        ),
        IrExprKind::Record(fields) => HoistExprKey::Record(
            fields
                .iter()
                .map(|(name, field)| (name.clone(), hoist_expr_key(field)))
                .collect(),
        ),
        IrExprKind::EnumCtor { ctor, args } => HoistExprKey::EnumCtor {
            ctor: ctor.clone(),
            args: args.iter().map(hoist_expr_key).collect(),
        },
        IrExprKind::EnumTag { value } => HoistExprKey::EnumTag {
            value: Box::new(hoist_expr_key(value)),
        },
        IrExprKind::EnumChildBySlot { value, ctor, slot } => HoistExprKey::EnumChildBySlot {
            value: Box::new(hoist_expr_key(value)),
            ctor: ctor.clone(),
            slot: *slot,
        },
        IrExprKind::EnumNonRecField { value, ctor, field } => HoistExprKey::EnumNonRecField {
            value: Box::new(hoist_expr_key(value)),
            ctor: ctor.clone(),
            field: *field,
        },
        IrExprKind::Let { bindings, body } => HoistExprKey::Let {
            bindings: bindings
                .iter()
                .map(|binding| (binding.name.clone(), hoist_expr_key(&binding.expr)))
                .collect(),
            body: Box::new(hoist_expr_key(body)),
        },
        IrExprKind::Call { callee, args } => HoistExprKey::Call {
            callee: hoist_callee_key(callee),
            args: args.iter().map(hoist_expr_key).collect(),
        },
    }
}

fn hoist_callee_key(callee: &Callee) -> HoistCalleeKey {
    match callee {
        Callee::Prim(op) => HoistCalleeKey::Prim(prim_op_key(*op)),
        Callee::Function(name) => HoistCalleeKey::Function(name.clone()),
        Callee::Builtin(builtin) => HoistCalleeKey::Function(format!("__builtin::{builtin:?}")),
    }
}

fn prim_op_key(op: PrimOp) -> PrimOpKey {
    match op {
        PrimOp::Add => PrimOpKey::Add,
        PrimOp::Sub => PrimOpKey::Sub,
        PrimOp::Mul => PrimOpKey::Mul,
        PrimOp::Div => PrimOpKey::Div,
        PrimOp::Mod => PrimOpKey::Mod,
        PrimOp::And => PrimOpKey::And,
        PrimOp::Or => PrimOpKey::Or,
        PrimOp::Eq => PrimOpKey::Eq,
        PrimOp::Lt => PrimOpKey::Lt,
        PrimOp::Gt => PrimOpKey::Gt,
        PrimOp::Le => PrimOpKey::Le,
        PrimOp::Ge => PrimOpKey::Ge,
    }
}

fn emit_vector_hoists(
    function: &mut Function,
    hoisted: &[HoistedExpr<'_>],
    hoisted_locals: &BTreeMap<HoistExprKey, u32>,
    params: &[KernelParam],
    addressing: VectorAddressing,
    scalar_indices: &BTreeMap<String, u32>,
    locals: &BTreeMap<String, usize>,
) -> Result<()> {
    let mut emitted = BTreeMap::<HoistExprKey, u32>::new();
    let inline_bindings = BTreeMap::<String, IrExpr>::new();
    for hoisted_expr in hoisted {
        let local = *hoisted_locals
            .get(&hoisted_expr.key)
            .ok_or_else(|| SimdError::new("missing vector hoisted local"))?;
        compile_vector_ir_expr_with_hoists(
            function,
            hoisted_expr.expr,
            locals,
            params,
            addressing,
            scalar_indices,
            &emitted,
            &inline_bindings,
        )?;
        function.instruction(&Instruction::LocalSet(local));
        emitted.insert(hoisted_expr.key.clone(), local);
    }
    Ok(())
}

fn emit_scalar_hoists(
    function: &mut Function,
    hoisted: &[HoistedExpr<'_>],
    hoisted_locals: &BTreeMap<HoistExprKey, u32>,
    locals: &BTreeMap<String, u32>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
) -> Result<()> {
    let mut emitted = BTreeMap::<HoistExprKey, u32>::new();
    let inline_bindings = BTreeMap::<String, IrExpr>::new();
    let empty_enum_ctors = BTreeMap::<String, EnumCtorInfo>::new();
    for hoisted_expr in hoisted {
        let local = *hoisted_locals
            .get(&hoisted_expr.key)
            .ok_or_else(|| SimdError::new("missing scalar hoisted local"))?;
        compile_scalar_ir_expr_with_hoists(
            function,
            hoisted_expr.expr,
            locals,
            scalar_indices,
            signatures,
            &empty_enum_ctors,
            &BTreeMap::new(),
            None,
            &emitted,
            &inline_bindings,
        )?;
        function.instruction(&Instruction::LocalSet(local));
        emitted.insert(hoisted_expr.key.clone(), local);
    }
    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct EnumWasmState {
    ptr_local: u32,
    base_local: u32,
    save_sp_local: u32,
    base_tmp_local: u32,
    index_i64_local: u32,
    scratch_sp_local: u32,
    alloc_end_local: u32,
    alloc_capacity_local: u32,
    alloc_delta_local: u32,
    aux0_local: u32,
    aux1_local: u32,
    aux2_local: u32,
    aux3_local: u32,
    aux4_local: u32,
    aux5_local: u32,
    aux6_local: u32,
    aux7_local: u32,
    aux8_local: u32,
    child_locals: [u32; 8],
}

fn append_enum_scratch_locals(
    local_decls: &mut Vec<(u32, ValType)>,
    next_local: &mut u32,
) -> EnumWasmState {
    let ptr_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let base_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let save_sp_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let base_tmp_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let index_i64_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I64));
    let scratch_sp_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let alloc_end_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let alloc_capacity_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let alloc_delta_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let aux0_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let aux1_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let aux2_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let aux3_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let aux4_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let aux5_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let aux6_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let aux7_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let aux8_local = *next_local;
    *next_local += 1;
    local_decls.push((1, ValType::I32));
    let mut child_locals = [0u32; 8];
    for local in &mut child_locals {
        *local = *next_local;
        *next_local += 1;
        local_decls.push((1, ValType::I32));
    }
    EnumWasmState {
        ptr_local,
        base_local,
        save_sp_local,
        base_tmp_local,
        index_i64_local,
        scratch_sp_local,
        alloc_end_local,
        alloc_capacity_local,
        alloc_delta_local,
        aux0_local,
        aux1_local,
        aux2_local,
        aux3_local,
        aux4_local,
        aux5_local,
        aux6_local,
        aux7_local,
        aux8_local,
        child_locals,
    }
}

const ENUM_HEAP_PTR_ADDR: u32 = 0;
const ENUM_SAVE_STACK_PTR_ADDR: u32 = 4;
const ENUM_HEAP_START: i32 = 4096;
const ENUM_SAVE_STACK_START: i32 = 2048;
const ENUM_REF_CELL_BYTES: i32 = 8;
const ENUM_REF_HEADER_OFFSET: u64 = 0;
const ENUM_REF_ROOT_OFFSET: u64 = 4;
const ENUM_HEADER_ENUM_ID_OFFSET: u64 = 0;
const ENUM_HEADER_NODE_COUNT_OFFSET: u64 = 4;
const ENUM_HEADER_TAGS_PTR_OFFSET: u64 = 8;
const ENUM_HEADER_ENDS_PTR_OFFSET: u64 = 12;
const ENUM_HEADER_SLOTS_PTR_OFFSET: u64 = 16;
const ENUM_HEADER_CTOR_TABLE_PTR_OFFSET: u64 = 20;
const ENUM_HEADER_BYTES: i32 = 24;
const ENUM_CTOR_TABLE_ENTRY_BYTES: i32 = 8;
const ENUM_CTOR_TABLE_ROWS_PTR_OFFSET: u64 = 0;
const ENUM_CTOR_TABLE_ROW_COUNT_OFFSET: u64 = 4;
const ENUM_TAG_BYTES: i32 = 2;
const ENUM_END_BYTES: i32 = 4;
const ENUM_SLOT_BYTES: i32 = 4;

fn emit_enum_alloc(function: &mut Function, enum_state: EnumWasmState, bytes: i32) {
    function.instruction(&Instruction::I32Const(ENUM_HEAP_PTR_ADDR as i32));
    function.instruction(&Instruction::I32Load(memarg(0, 2)));
    function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
    function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
    function.instruction(&Instruction::I32Eqz);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::I32Const(ENUM_HEAP_START));
    function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
    function.instruction(&Instruction::I32Const(bytes));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(enum_state.alloc_end_local));
    function.instruction(&Instruction::MemorySize(0));
    function.instruction(&Instruction::I32Const(16));
    function.instruction(&Instruction::I32Shl);
    function.instruction(&Instruction::LocalSet(enum_state.alloc_capacity_local));
    function.instruction(&Instruction::LocalGet(enum_state.alloc_end_local));
    function.instruction(&Instruction::LocalGet(enum_state.alloc_capacity_local));
    function.instruction(&Instruction::I32GtU);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(enum_state.alloc_end_local));
    function.instruction(&Instruction::LocalGet(enum_state.alloc_capacity_local));
    function.instruction(&Instruction::I32Sub);
    function.instruction(&Instruction::I32Const(65535));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::I32Const(16));
    function.instruction(&Instruction::I32ShrU);
    function.instruction(&Instruction::LocalSet(enum_state.alloc_delta_local));
    function.instruction(&Instruction::LocalGet(enum_state.alloc_delta_local));
    function.instruction(&Instruction::MemoryGrow(0));
    function.instruction(&Instruction::I32Const(-1));
    function.instruction(&Instruction::I32Eq);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::I32Const(ENUM_HEAP_PTR_ADDR as i32));
    function.instruction(&Instruction::LocalGet(enum_state.alloc_end_local));
    function.instruction(&Instruction::I32Store(memarg(0, 2)));
}

fn emit_enum_alloc_dynamic(function: &mut Function, enum_state: EnumWasmState, bytes_local: u32) {
    function.instruction(&Instruction::I32Const(ENUM_HEAP_PTR_ADDR as i32));
    function.instruction(&Instruction::I32Load(memarg(0, 2)));
    function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
    function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
    function.instruction(&Instruction::I32Eqz);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::I32Const(ENUM_HEAP_START));
    function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
    function.instruction(&Instruction::LocalGet(bytes_local));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(enum_state.alloc_end_local));
    function.instruction(&Instruction::MemorySize(0));
    function.instruction(&Instruction::I32Const(16));
    function.instruction(&Instruction::I32Shl);
    function.instruction(&Instruction::LocalSet(enum_state.alloc_capacity_local));
    function.instruction(&Instruction::LocalGet(enum_state.alloc_end_local));
    function.instruction(&Instruction::LocalGet(enum_state.alloc_capacity_local));
    function.instruction(&Instruction::I32GtU);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(enum_state.alloc_end_local));
    function.instruction(&Instruction::LocalGet(enum_state.alloc_capacity_local));
    function.instruction(&Instruction::I32Sub);
    function.instruction(&Instruction::I32Const(65535));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::I32Const(16));
    function.instruction(&Instruction::I32ShrU);
    function.instruction(&Instruction::LocalSet(enum_state.alloc_delta_local));
    function.instruction(&Instruction::LocalGet(enum_state.alloc_delta_local));
    function.instruction(&Instruction::MemoryGrow(0));
    function.instruction(&Instruction::I32Const(-1));
    function.instruction(&Instruction::I32Eq);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::I32Const(ENUM_HEAP_PTR_ADDR as i32));
    function.instruction(&Instruction::LocalGet(enum_state.alloc_end_local));
    function.instruction(&Instruction::I32Store(memarg(0, 2)));
}

fn emit_box_seq_from_stack(function: &mut Function, enum_state: EnumWasmState) {
    function.instruction(&Instruction::LocalSet(enum_state.aux1_local));
    function.instruction(&Instruction::LocalSet(enum_state.aux0_local));
    emit_enum_alloc(function, enum_state, WASM_SEQ_HEADER_BYTES);
    function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
    function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
    function.instruction(&Instruction::I32Store(memarg(WASM_SEQ_DATA_PTR_OFFSET, 2)));
    function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
    function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
    function.instruction(&Instruction::I32Store(memarg(WASM_SEQ_LEN_OFFSET, 2)));
    function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
}

#[allow(dead_code)]
fn emit_unbox_seq_handle(function: &mut Function, handle_local: u32, _enum_state: EnumWasmState) {
    function.instruction(&Instruction::LocalGet(handle_local));
    function.instruction(&Instruction::I32Load(memarg(WASM_SEQ_DATA_PTR_OFFSET, 2)));
    function.instruction(&Instruction::LocalGet(handle_local));
    function.instruction(&Instruction::I32Load(memarg(WASM_SEQ_LEN_OFFSET, 2)));
}

fn emit_enum_load_header_root(
    function: &mut Function,
    value_local: u32,
    enum_state: EnumWasmState,
) {
    function.instruction(&Instruction::LocalGet(value_local));
    function.instruction(&Instruction::I32Load(memarg(ENUM_REF_HEADER_OFFSET, 2)));
    function.instruction(&Instruction::LocalSet(enum_state.base_local));
    function.instruction(&Instruction::LocalGet(value_local));
    function.instruction(&Instruction::I32Load(memarg(ENUM_REF_ROOT_OFFSET, 2)));
    function.instruction(&Instruction::LocalSet(enum_state.save_sp_local));
}

fn emit_enum_tag_load_from_value(
    function: &mut Function,
    value_local: u32,
    enum_state: EnumWasmState,
) {
    emit_enum_load_header_root(function, value_local, enum_state);
    function.instruction(&Instruction::LocalGet(enum_state.base_local));
    function.instruction(&Instruction::I32Load(memarg(
        ENUM_HEADER_TAGS_PTR_OFFSET,
        2,
    )));
    function.instruction(&Instruction::LocalGet(enum_state.save_sp_local));
    function.instruction(&Instruction::I32Const(ENUM_TAG_BYTES));
    function.instruction(&Instruction::I32Mul);
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::I32Load16U(memarg(0, 1)));
}

fn emit_enum_child_root(
    function: &mut Function,
    value_local: u32,
    recursive_slot: usize,
    enum_state: EnumWasmState,
) -> Result<()> {
    emit_enum_load_header_root(function, value_local, enum_state);
    function.instruction(&Instruction::LocalGet(enum_state.save_sp_local));
    function.instruction(&Instruction::I32Const(1));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(enum_state.base_tmp_local));
    for _ in 0..recursive_slot {
        function.instruction(&Instruction::LocalGet(enum_state.base_local));
        function.instruction(&Instruction::I32Load(memarg(
            ENUM_HEADER_ENDS_PTR_OFFSET,
            2,
        )));
        function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
        function.instruction(&Instruction::I32Const(ENUM_END_BYTES));
        function.instruction(&Instruction::I32Mul);
        function.instruction(&Instruction::I32Add);
        function.instruction(&Instruction::I32Load(memarg(0, 2)));
        function.instruction(&Instruction::LocalSet(enum_state.base_tmp_local));
    }
    Ok(())
}

fn emit_enum_ref_from_header_root(function: &mut Function, enum_state: EnumWasmState) {
    emit_enum_alloc(function, enum_state, ENUM_REF_CELL_BYTES);
    function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
    function.instruction(&Instruction::LocalGet(enum_state.base_local));
    function.instruction(&Instruction::I32Store(memarg(ENUM_REF_HEADER_OFFSET, 2)));
    function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
    function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
    function.instruction(&Instruction::I32Store(memarg(ENUM_REF_ROOT_OFFSET, 2)));
    function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
}

fn emit_enum_row_base_ptr_from_header(
    function: &mut Function,
    header_local: u32,
    _enum_layout: &WasmEnumLayout,
    ctor_layout: &WasmEnumCtorLayout,
    _enum_state: EnumWasmState,
) {
    function.instruction(&Instruction::LocalGet(header_local));
    function.instruction(&Instruction::I32Load(memarg(
        ENUM_HEADER_CTOR_TABLE_PTR_OFFSET,
        2,
    )));
    function.instruction(&Instruction::I32Const(
        i32::from(ctor_layout.tag) * ENUM_CTOR_TABLE_ENTRY_BYTES
            + ENUM_CTOR_TABLE_ROWS_PTR_OFFSET as i32,
    ));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::I32Load(memarg(0, 2)));
}

fn emit_store_type_at_address(
    function: &mut Function,
    ty: &Type,
    enum_state: EnumWasmState,
) -> Result<()> {
    match ty {
        Type::Scalar(Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32Store(memarg(0, 2)));
        }
        Type::Scalar(Prim::I64) => {
            function.instruction(&Instruction::I64Store(memarg(0, 3)));
        }
        Type::Scalar(Prim::F32) => {
            function.instruction(&Instruction::F32Store(memarg(0, 2)));
        }
        Type::Scalar(Prim::F64) => {
            function.instruction(&Instruction::F64Store(memarg(0, 3)));
        }
        Type::Named(name, args) if is_wasm_enum_named_type(name, args) => {
            function.instruction(&Instruction::I32Store(memarg(0, 2)));
        }
        Type::Named(name, args) if is_wasm_string_named_type(name, args) => {
            function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
            function.instruction(&Instruction::LocalSet(enum_state.base_local));
            function.instruction(&Instruction::LocalTee(enum_state.base_tmp_local));
            function.instruction(&Instruction::LocalGet(enum_state.base_local));
            function.instruction(&Instruction::I32Store(memarg(0, 2)));
            function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
            function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
            function.instruction(&Instruction::I32Store(memarg(4, 2)));
        }
        Type::Index(_) => {
            function.instruction(&Instruction::I64Store(memarg(0, 3)));
        }
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _)
            if wasm_star_seq_storage_prim(ty).is_some() =>
        {
            function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
            function.instruction(&Instruction::LocalSet(enum_state.base_local));
            function.instruction(&Instruction::LocalTee(enum_state.base_tmp_local));
            function.instruction(&Instruction::LocalGet(enum_state.base_local));
            function.instruction(&Instruction::I32Store(memarg(0, 2)));
            function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
            function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
            function.instruction(&Instruction::I32Store(memarg(4, 2)));
        }
        other => {
            return Err(SimdError::new(format!(
                "Wasm scratch store does not support value type {:?}",
                other
            )));
        }
    }
    Ok(())
}

fn emit_load_type_from_address(
    function: &mut Function,
    ty: &Type,
    enum_state: EnumWasmState,
) -> Result<()> {
    match ty {
        Type::Scalar(Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32Load(memarg(0, 2)));
        }
        Type::Scalar(Prim::I64) => {
            function.instruction(&Instruction::I64Load(memarg(0, 3)));
        }
        Type::Scalar(Prim::F32) => {
            function.instruction(&Instruction::F32Load(memarg(0, 2)));
        }
        Type::Scalar(Prim::F64) => {
            function.instruction(&Instruction::F64Load(memarg(0, 3)));
        }
        Type::Named(name, args) if is_wasm_enum_named_type(name, args) => {
            function.instruction(&Instruction::I32Load(memarg(0, 2)));
        }
        Type::Named(name, args) if is_wasm_string_named_type(name, args) => {
            function.instruction(&Instruction::LocalTee(enum_state.base_tmp_local));
            function.instruction(&Instruction::I32Load(memarg(0, 2)));
            function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
            function.instruction(&Instruction::I32Load(memarg(4, 2)));
        }
        Type::Index(_) => {
            function.instruction(&Instruction::I64Load(memarg(0, 3)));
        }
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _)
            if wasm_star_seq_storage_prim(ty).is_some() =>
        {
            function.instruction(&Instruction::LocalTee(enum_state.base_tmp_local));
            function.instruction(&Instruction::I32Load(memarg(0, 2)));
            function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
            function.instruction(&Instruction::I32Load(memarg(4, 2)));
        }
        other => {
            return Err(SimdError::new(format!(
                "Wasm scratch load does not support value type {:?}",
                other
            )));
        }
    }
    Ok(())
}

fn wasm_saved_value_bytes(ty: &Type) -> Result<i32> {
    match ty {
        Type::Scalar(Prim::I32 | Prim::Char) => Ok(4),
        Type::Scalar(Prim::I64 | Prim::F64) => Ok(8),
        Type::Scalar(Prim::F32) => Ok(4),
        Type::Named(name, args) if is_wasm_enum_named_type(name, args) => Ok(4),
        Type::Named(name, args) if is_wasm_string_named_type(name, args) => Ok(8),
        Type::Index(_) => Ok(8),
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _)
            if wasm_star_seq_storage_prim(ty).is_some() =>
        {
            Ok(8)
        }
        other => Err(SimdError::new(format!(
            "Wasm scratch storage does not support value type {:?}",
            other
        ))),
    }
}

fn emit_tape_enum_field_store(
    function: &mut Function,
    field_layout: &WasmEnumFieldLayout,
    enum_state: EnumWasmState,
) -> Result<()> {
    match &field_layout.ty {
        Type::Scalar(Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32Store(memarg(0, 2)));
        }
        Type::Scalar(Prim::I64) => {
            function.instruction(&Instruction::I64Store(memarg(0, 3)));
        }
        Type::Scalar(Prim::F32) => {
            function.instruction(&Instruction::F32Store(memarg(0, 2)));
        }
        Type::Scalar(Prim::F64) => {
            function.instruction(&Instruction::F64Store(memarg(0, 3)));
        }
        Type::Named(name, args) if is_wasm_enum_named_type(name, args) => {
            function.instruction(&Instruction::I32Store(memarg(0, 2)));
        }
        Type::Named(name, args) if is_wasm_string_named_type(name, args) => {
            function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
            function.instruction(&Instruction::LocalSet(enum_state.base_local));
            function.instruction(&Instruction::LocalTee(enum_state.base_tmp_local));
            function.instruction(&Instruction::LocalGet(enum_state.base_local));
            function.instruction(&Instruction::I32Store(memarg(0, 2)));
            function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
            function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
            function.instruction(&Instruction::I32Store(memarg(4, 2)));
        }
        Type::Index(_) => {
            function.instruction(&Instruction::I64Store(memarg(0, 3)));
        }
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _)
            if wasm_star_seq_storage_prim(&field_layout.ty).is_some() =>
        {
            function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
            function.instruction(&Instruction::LocalSet(enum_state.base_local));
            function.instruction(&Instruction::LocalTee(enum_state.base_tmp_local));
            function.instruction(&Instruction::LocalGet(enum_state.base_local));
            function.instruction(&Instruction::I32Store(memarg(0, 2)));
            function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
            function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
            function.instruction(&Instruction::I32Store(memarg(4, 2)));
        }
        other => {
            return Err(SimdError::new(format!(
                "Wasm enum field store does not support field type {:?}",
                other
            )));
        }
    }
    Ok(())
}

fn emit_enum_field_load_with_layout(
    function: &mut Function,
    value_local: u32,
    enum_layout: &WasmEnumLayout,
    ctor_layout: &WasmEnumCtorLayout,
    field_index: usize,
    field_ty: &Type,
    enum_state: EnumWasmState,
) -> Result<()> {
    if let Some(recursive_slot) = ctor_layout
        .recursive_field_indices
        .iter()
        .position(|index| *index == field_index)
    {
        emit_enum_child_root(function, value_local, recursive_slot, enum_state)?;
        emit_enum_ref_from_header_root(function, enum_state);
        return Ok(());
    }
    let field_layout = ctor_layout
        .non_recursive_fields
        .iter()
        .find(|layout| layout.source_index == field_index)
        .ok_or_else(|| {
            SimdError::new(format!(
                "constructor '{}' is missing Wasm field layout for source field {}",
                ctor_layout.ctor_name, field_index
            ))
        })?;
    emit_enum_load_header_root(function, value_local, enum_state);
    function.instruction(&Instruction::LocalGet(enum_state.base_local));
    function.instruction(&Instruction::I32Load(memarg(
        ENUM_HEADER_SLOTS_PTR_OFFSET,
        2,
    )));
    function.instruction(&Instruction::LocalGet(enum_state.save_sp_local));
    function.instruction(&Instruction::I32Const(ENUM_SLOT_BYTES));
    function.instruction(&Instruction::I32Mul);
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::I32Load(memarg(0, 2)));
    function.instruction(&Instruction::LocalSet(enum_state.base_tmp_local));
    function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
    function.instruction(&Instruction::LocalSet(enum_state.aux2_local));
    emit_enum_row_base_ptr_from_header(
        function,
        enum_state.base_local,
        enum_layout,
        ctor_layout,
        enum_state,
    );
    function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
    function.instruction(&Instruction::I32Const(ctor_layout.row_stride as i32));
    function.instruction(&Instruction::I32Mul);
    function.instruction(&Instruction::I32Add);
    if field_layout.offset != 0 {
        function.instruction(&Instruction::I32Const(field_layout.offset as i32));
        function.instruction(&Instruction::I32Add);
    }
    match field_ty {
        Type::Scalar(Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32Load(memarg(0, 2)));
        }
        Type::Scalar(Prim::I64) => {
            function.instruction(&Instruction::I64Load(memarg(0, 3)));
        }
        Type::Scalar(Prim::F32) => {
            function.instruction(&Instruction::F32Load(memarg(0, 2)));
        }
        Type::Scalar(Prim::F64) => {
            function.instruction(&Instruction::F64Load(memarg(0, 3)));
        }
        Type::Named(name, args) if is_wasm_enum_named_type(name, args) => {
            function.instruction(&Instruction::I32Load(memarg(0, 2)));
        }
        Type::Named(name, args) if is_wasm_string_named_type(name, args) => {
            function.instruction(&Instruction::I32Load(memarg(0, 2)));
            emit_enum_row_base_ptr_from_header(
                function,
                enum_state.base_local,
                enum_layout,
                ctor_layout,
                enum_state,
            );
            function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
            function.instruction(&Instruction::I32Const(ctor_layout.row_stride as i32));
            function.instruction(&Instruction::I32Mul);
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::I32Load(memarg(
                u64::from(field_layout.offset) + 4,
                2,
            )));
        }
        Type::Index(_) => {
            function.instruction(&Instruction::I64Load(memarg(0, 3)));
        }
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _)
            if wasm_star_seq_storage_prim(field_ty).is_some() =>
        {
            function.instruction(&Instruction::I32Load(memarg(0, 2)));
            emit_enum_row_base_ptr_from_header(
                function,
                enum_state.base_local,
                enum_layout,
                ctor_layout,
                enum_state,
            );
            function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
            function.instruction(&Instruction::I32Const(ctor_layout.row_stride as i32));
            function.instruction(&Instruction::I32Mul);
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::I32Load(memarg(
                u64::from(field_layout.offset) + 4,
                2,
            )));
        }
        other => {
            return Err(SimdError::new(format!(
                "Wasm enum field load does not support field type {:?}",
                other
            )));
        }
    }
    Ok(())
}

fn emit_enum_field_as_condition_value_with_layout(
    function: &mut Function,
    value_local: u32,
    enum_layout: &WasmEnumLayout,
    ctor_layout: &WasmEnumCtorLayout,
    field_index: usize,
    field_ty: &Type,
    enum_state: EnumWasmState,
) -> Result<()> {
    emit_enum_field_load_with_layout(
        function,
        value_local,
        enum_layout,
        ctor_layout,
        field_index,
        field_ty,
        enum_state,
    )
}

fn emit_enum_field_load(
    function: &mut Function,
    value_local: u32,
    enum_layout: &WasmEnumLayout,
    ctor_layout: &WasmEnumCtorLayout,
    field_index: usize,
    field_ty: &Type,
    enum_state: EnumWasmState,
) -> Result<()> {
    emit_enum_field_load_with_layout(
        function,
        value_local,
        enum_layout,
        ctor_layout,
        field_index,
        field_ty,
        enum_state,
    )
}

fn emit_enum_field_as_condition_value(
    function: &mut Function,
    value_local: u32,
    enum_layout: &WasmEnumLayout,
    ctor_layout: &WasmEnumCtorLayout,
    field_index: usize,
    field_ty: &Type,
    enum_state: EnumWasmState,
) -> Result<()> {
    emit_enum_field_as_condition_value_with_layout(
        function,
        value_local,
        enum_layout,
        ctor_layout,
        field_index,
        field_ty,
        enum_state,
    )
}

fn is_wasm_direct_self_recursive_field(field_ty: &Type, ctor: &EnumCtorInfo) -> bool {
    let Type::Named(name, args) = field_ty else {
        return false;
    };
    if name != &ctor.enum_name || args.len() != ctor.enum_params.len() {
        return false;
    }
    args.iter()
        .zip(&ctor.enum_params)
        .all(|(arg, param)| matches!(arg, Type::Var(var) if var == param))
}

fn enum_ctor_recursive_field_indices(ctor: &EnumCtorInfo) -> Vec<usize> {
    ctor.fields
        .iter()
        .enumerate()
        .filter_map(|(index, field)| {
            if is_wasm_direct_self_recursive_field(field, ctor) {
                Some(index)
            } else {
                None
            }
        })
        .collect()
}

fn enum_ctor_non_recursive_field_indices(ctor: &EnumCtorInfo) -> Vec<usize> {
    ctor.fields
        .iter()
        .enumerate()
        .filter_map(|(index, field)| {
            if is_wasm_direct_self_recursive_field(field, ctor) {
                None
            } else {
                Some(index)
            }
        })
        .collect()
}

fn is_pattern_constructor_name(name: &str, enum_ctors: &BTreeMap<String, EnumCtorInfo>) -> bool {
    is_constructor_name(name) && enum_ctors.contains_key(name)
}

fn bulk_len_local_name(name: &str) -> String {
    format!("__len${name}")
}

fn is_wasm_string_named_type(name: &str, args: &[Type]) -> bool {
    name == "string" && args.is_empty()
}

fn wasm_star_seq_scalar_prim(ty: &Type) -> Option<Prim> {
    match ty {
        Type::StarSeq(item) | Type::StarSeqWitnessed(item, _) => match item.as_ref() {
            Type::Scalar(prim) => Some(*prim),
            Type::Index(_) => Some(Prim::I64),
            _ => None,
        },
        _ => None,
    }
}

fn wasm_star_seq_storage_prim(ty: &Type) -> Option<Prim> {
    match ty {
        Type::StarSeq(item) | Type::StarSeqWitnessed(item, _) => match item.as_ref() {
            Type::Scalar(prim) => Some(*prim),
            Type::Index(_) => Some(Prim::I64),
            Type::Named(name, args) if is_wasm_enum_named_type(name, args) => Some(Prim::I32),
            _ => None,
        },
        _ => None,
    }
}

fn wasm_slice_pattern_elem_type(ty: &Type) -> Result<Type> {
    match ty {
        Type::Bulk(prim, shape) => {
            if shape.0.len() != 1 {
                return Err(SimdError::new(format!(
                    "Wasm backend currently supports slice patterns only for rank-1 bulk/string/T[*] values, found shape {:?}",
                    shape
                )));
            }
            Ok(Type::Scalar(*prim))
        }
        Type::Named(name, args) if is_wasm_string_named_type(name, args) => {
            Ok(Type::Scalar(Prim::Char))
        }
        Type::StarSeq(item) | Type::StarSeqWitnessed(item, _)
            if wasm_star_seq_storage_prim(ty).is_some() =>
        {
            Ok(item.as_ref().clone())
        }
        _ => Err(SimdError::new(format!(
            "Wasm backend currently supports slice patterns only for rank-1 bulk/string/T[*] values, found {:?}",
            ty
        ))),
    }
}

fn is_wasm_bulk_like_type(ty: &Type) -> bool {
    matches!(ty, Type::Bulk(_, _))
        || matches!(ty, Type::Named(name, args) if is_wasm_string_named_type(name, args))
        || wasm_star_seq_storage_prim(ty).is_some()
}

fn wasm_slice_pattern_prim(ty: &Type) -> Result<Prim> {
    match ty {
        Type::Bulk(prim, shape) => {
            if shape.0.len() != 1 {
                return Err(SimdError::new(format!(
                    "Wasm backend currently supports slice patterns only for rank-1 bulk/string values, found shape {:?}",
                    shape
                )));
            }
            Ok(*prim)
        }
        Type::Named(name, args) if is_wasm_string_named_type(name, args) => Ok(Prim::Char),
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _) => wasm_star_seq_storage_prim(ty).ok_or_else(|| {
            SimdError::new(format!(
                "Wasm backend currently supports T[*] slice patterns only for scalar or enum elements, found {:?}",
                ty
            ))
        }),
        _ => Err(SimdError::new(format!(
            "Wasm backend currently supports slice patterns only for rank-1 bulk/string/T[*] values, found {:?}",
            ty
        ))),
    }
}

fn wasm_local_val_type_for_pattern_binding(ty: &Type) -> Result<ValType> {
    match ty {
        Type::Scalar(prim) => Ok(wasm_val_type(*prim)),
        Type::Named(name, args) if is_wasm_enum_named_type(name, args) => Ok(ValType::I32),
        other => Err(SimdError::new(format!(
            "Wasm scalar clause local binding has unsupported type {:?}",
            other
        ))),
    }
}

fn collect_pattern_binding_types(
    pattern: &Pattern,
    ty: &Type,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    out: &mut Vec<(String, Type)>,
) -> Result<()> {
    match pattern {
        Pattern::Name(name) => {
            if !is_pattern_constructor_name(name, enum_ctors) && name != "_" {
                out.push((name.clone(), ty.clone()));
            }
            Ok(())
        }
        Pattern::Tuple(items) => {
            let Type::Tuple(item_types) = ty else {
                return Err(SimdError::new(format!(
                    "tuple pattern expects tuple type, found {:?}",
                    ty
                )));
            };
            if items.len() != item_types.len() {
                return Err(SimdError::new(format!(
                    "tuple pattern expects {} elements, found {}",
                    item_types.len(),
                    items.len()
                )));
            }
            for (item, item_ty) in items.iter().zip(item_types.iter()) {
                collect_pattern_binding_types(item, item_ty, enum_ctors, out)?;
            }
            Ok(())
        }
        Pattern::Ctor(name, subpatterns) => {
            let ctor = enum_ctors
                .get(name)
                .ok_or_else(|| SimdError::new(format!("unknown constructor '{}'", name)))?;
            let Type::Named(enum_name, enum_args) = ty else {
                return Err(SimdError::new(format!(
                    "constructor pattern '{}' requires enum type, found {:?}",
                    name, ty
                )));
            };
            if enum_name != &ctor.enum_name {
                return Err(SimdError::new(format!(
                    "constructor '{}' belongs to enum '{}', not '{}'",
                    name, ctor.enum_name, enum_name
                )));
            }
            let subst = ctor
                .enum_params
                .iter()
                .cloned()
                .zip(enum_args.iter().cloned())
                .collect::<BTreeMap<_, _>>();
            for (subpattern, field_ty) in subpatterns.iter().zip(&ctor.fields) {
                let field_ty = apply_type_subst(field_ty, &subst);
                collect_pattern_binding_types(subpattern, &field_ty, enum_ctors, out)?;
            }
            Ok(())
        }
        Pattern::Wildcard
        | Pattern::Int(_)
        | Pattern::Float(_)
        | Pattern::Char(_)
        | Pattern::Bool(_)
        | Pattern::Type(_) => Ok(()),
        Pattern::Slice {
            prefix,
            suffix,
            rest,
        } => {
            let elem_ty = wasm_slice_pattern_elem_type(ty)?;
            for subpattern in prefix {
                collect_pattern_binding_types(subpattern, &elem_ty, enum_ctors, out)?;
            }
            for subpattern in suffix {
                collect_pattern_binding_types(subpattern, &elem_ty, enum_ctors, out)?;
            }
            if let Some(SliceRest::Bind(name)) = rest
                && name != "_"
            {
                out.push((name.clone(), ty.clone()));
            }
            Ok(())
        }
    }
}

fn clause_has_condition(
    patterns: &[TypedPattern],
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
) -> bool {
    patterns.iter().any(|pattern| {
        matches!(
            pattern.pattern,
            Pattern::Int(_)
                | Pattern::Float(_)
                | Pattern::Char(_)
                | Pattern::Bool(_)
                | Pattern::Ctor(_, _)
                | Pattern::Slice { .. }
        )
            || matches!(&pattern.pattern, Pattern::Name(name) if is_pattern_constructor_name(name, enum_ctors))
    })
}

fn build_clause_local_map(
    patterns: &[TypedPattern],
    locals: &mut Vec<(u32, ValType)>,
    next_local: &mut u32,
    shared_named_locals: &mut BTreeMap<String, (u32, Type)>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
) -> Result<BTreeMap<String, u32>> {
    let mut map = BTreeMap::<String, u32>::new();
    let mut value_local = 0u32;
    for typed_pattern in patterns {
        if let Pattern::Name(name) = &typed_pattern.pattern
            && !is_pattern_constructor_name(name, enum_ctors)
            && name != "_"
        {
            map.insert(name.clone(), value_local);
            if is_wasm_bulk_like_type(&typed_pattern.ty) {
                map.insert(bulk_len_local_name(name), value_local + 1);
            }
        }
        value_local += wasm_pattern_value_width(&typed_pattern.ty);
    }
    for typed_pattern in patterns {
        let mut bindings = Vec::<(String, Type)>::new();
        collect_pattern_binding_types(
            &typed_pattern.pattern,
            &typed_pattern.ty,
            enum_ctors,
            &mut bindings,
        )?;
        for (name, ty) in bindings {
            if map.contains_key(&name) {
                continue;
            }
            if let Some((index, existing_ty)) = shared_named_locals.get(&name) {
                if existing_ty != &ty {
                    return Err(SimdError::new(format!(
                        "pattern local '{}' has inconsistent types {:?} and {:?}",
                        name, existing_ty, ty
                    )));
                }
                map.insert(name.clone(), *index);
                if is_wasm_bulk_like_type(&ty) {
                    map.insert(bulk_len_local_name(&name), *index + 1);
                }
                continue;
            }
            let index = *next_local;
            if is_wasm_bulk_like_type(&ty) {
                *next_local += 2;
                locals.push((2, ValType::I32));
                map.insert(name.clone(), index);
                map.insert(bulk_len_local_name(&name), index + 1);
            } else {
                let val_type = wasm_local_val_type_for_pattern_binding(&ty)?;
                *next_local += 1;
                locals.push((1, val_type));
                map.insert(name.clone(), index);
            }
            shared_named_locals.insert(name, (index, ty));
        }
    }
    Ok(map)
}

fn wasm_pattern_value_width(ty: &Type) -> u32 {
    match ty {
        _ if is_wasm_bulk_like_type(ty) => 2,
        Type::Record(fields) => fields
            .values()
            .map(|field_ty| wasm_pattern_value_width(field_ty))
            .sum(),
        _ => 1,
    }
}

fn emit_matching_if(
    function: &mut Function,
    patterns: &[TypedPattern],
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: Option<EnumWasmState>,
) -> Result<()> {
    emit_clause_condition(
        function,
        patterns,
        enum_ctors,
        wasm_enum_layouts,
        enum_state,
    )?;
    function.instruction(&Instruction::If(BlockType::Empty));
    Ok(())
}

fn emit_clause_condition(
    function: &mut Function,
    patterns: &[TypedPattern],
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: Option<EnumWasmState>,
) -> Result<()> {
    let mut terms = 0usize;
    let mut value_local = 0u32;
    for pattern in patterns {
        if emit_single_pattern_condition(
            function,
            &pattern.pattern,
            &pattern.ty,
            value_local,
            enum_ctors,
            wasm_enum_layouts,
            enum_state,
        )? {
            if terms > 0 {
                function.instruction(&Instruction::I32And);
            }
            terms += 1;
        }
        value_local += wasm_pattern_value_width(&pattern.ty);
    }
    if terms == 0 {
        function.instruction(&Instruction::I32Const(1));
    }
    Ok(())
}

fn emit_bulk_element_load_at_prefix_index(
    function: &mut Function,
    ptr_local: u32,
    prim: Prim,
    index: usize,
) -> Result<()> {
    let stride = byte_width(prim) as u64;
    let offset = u64::try_from(index)
        .map_err(|_| SimdError::new("slice pattern index does not fit in u64"))?
        .saturating_mul(stride);
    function.instruction(&Instruction::LocalGet(ptr_local));
    match prim {
        Prim::I32 | Prim::Char => function.instruction(&Instruction::I32Load(memarg(offset, 2))),
        Prim::I64 => function.instruction(&Instruction::I64Load(memarg(offset, 3))),
        Prim::F32 => function.instruction(&Instruction::F32Load(memarg(offset, 2))),
        Prim::F64 => function.instruction(&Instruction::F64Load(memarg(offset, 3))),
    };
    Ok(())
}

fn emit_bulk_element_load_at_suffix_index(
    function: &mut Function,
    ptr_local: u32,
    len_local: u32,
    prim: Prim,
    suffix_len: usize,
    suffix_index: usize,
) -> Result<()> {
    let suffix_len_i32 = i32::try_from(suffix_len)
        .map_err(|_| SimdError::new("slice suffix length does not fit in i32"))?;
    let suffix_index_i32 = i32::try_from(suffix_index)
        .map_err(|_| SimdError::new("slice suffix index does not fit in i32"))?;
    let stride_i32 = i32::try_from(byte_width(prim))
        .map_err(|_| SimdError::new("slice element byte-width does not fit in i32"))?;
    function.instruction(&Instruction::LocalGet(ptr_local));
    function.instruction(&Instruction::LocalGet(len_local));
    function.instruction(&Instruction::I32Const(suffix_len_i32));
    function.instruction(&Instruction::I32Sub);
    if suffix_index_i32 != 0 {
        function.instruction(&Instruction::I32Const(suffix_index_i32));
        function.instruction(&Instruction::I32Add);
    }
    function.instruction(&Instruction::I32Const(stride_i32));
    function.instruction(&Instruction::I32Mul);
    function.instruction(&Instruction::I32Add);
    match prim {
        Prim::I32 | Prim::Char => function.instruction(&Instruction::I32Load(memarg(0, 2))),
        Prim::I64 => function.instruction(&Instruction::I64Load(memarg(0, 3))),
        Prim::F32 => function.instruction(&Instruction::F32Load(memarg(0, 2))),
        Prim::F64 => function.instruction(&Instruction::F64Load(memarg(0, 3))),
    };
    Ok(())
}

fn emit_slice_element_pattern_condition(
    function: &mut Function,
    pattern: &Pattern,
    prim: Prim,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
) -> Result<bool> {
    match pattern {
        Pattern::Wildcard => Ok(false),
        Pattern::Name(name) => {
            if is_pattern_constructor_name(name, enum_ctors) {
                return Err(SimdError::new(format!(
                    "constructor '{}' cannot match a scalar bulk element",
                    name
                )));
            }
            Ok(false)
        }
        Pattern::Int(expected) => {
            if !prim.is_int() {
                return Err(SimdError::new(format!(
                    "integer slice element pattern requires integer bulk element type, found {:?}",
                    prim
                )));
            }
            emit_int_const(function, prim, *expected)?;
            emit_int_eq(function, prim);
            Ok(true)
        }
        Pattern::Float(expected) => {
            if !prim.is_float() {
                return Err(SimdError::new(format!(
                    "float slice element pattern requires float bulk element type, found {:?}",
                    prim
                )));
            }
            emit_float_bits_eq(function, prim, *expected)?;
            Ok(true)
        }
        Pattern::Char(expected) => {
            if prim != Prim::Char {
                return Err(SimdError::new(format!(
                    "char slice element pattern requires char bulk element type, found {:?}",
                    prim
                )));
            }
            function.instruction(&Instruction::I32Const(
                i32::try_from(u32::from(*expected))
                    .map_err(|_| SimdError::new("char literal codepoint does not fit in i32"))?,
            ));
            function.instruction(&Instruction::I32Eq);
            Ok(true)
        }
        Pattern::Bool(_) => Err(SimdError::new(
            "Wasm backend does not support bool slice element patterns",
        )),
        Pattern::Type(_) => Err(SimdError::new(
            "Wasm backend does not support type witness slice element patterns",
        )),
        Pattern::Ctor(_, _) => Err(SimdError::new(
            "Wasm backend does not support constructor slice element patterns",
        )),
        Pattern::Tuple(_) => Err(SimdError::new(
            "Wasm backend does not support tuple slice element patterns",
        )),
        Pattern::Slice { .. } => Err(SimdError::new(
            "Wasm backend does not support nested slice element patterns",
        )),
    }
}

fn emit_slice_element_name_binding(
    function: &mut Function,
    pattern: &Pattern,
    target_local: Option<u32>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
) -> Result<()> {
    let Some(target_local) = target_local else {
        return Ok(());
    };
    if let Pattern::Name(name) = pattern
        && name != "_"
        && !is_pattern_constructor_name(name, enum_ctors)
    {
        function.instruction(&Instruction::LocalSet(target_local));
    }
    Ok(())
}

fn emit_single_pattern_condition(
    function: &mut Function,
    pattern: &Pattern,
    ty: &Type,
    value_local: u32,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: Option<EnumWasmState>,
) -> Result<bool> {
    match pattern {
        Pattern::Wildcard => Ok(false),
        Pattern::Name(name) => {
            if !is_pattern_constructor_name(name, enum_ctors) {
                return Ok(false);
            }
            let ctor = enum_ctors
                .get(name)
                .ok_or_else(|| SimdError::new(format!("unknown constructor '{}'", name)))?;
            if !ctor.fields.is_empty() {
                return Err(SimdError::new(format!(
                    "constructor '{}' with fields must use parenthesized constructor pattern",
                    name
                )));
            }
            let Type::Named(enum_name, _) = ty else {
                return Err(SimdError::new(format!(
                    "constructor pattern '{}' requires enum type, found {:?}",
                    name, ty
                )));
            };
            if enum_name != &ctor.enum_name {
                return Err(SimdError::new(format!(
                    "constructor '{}' belongs to enum '{}', not '{}'",
                    name, ctor.enum_name, enum_name
                )));
            }
            let Some(enum_state) = enum_state else {
                return Err(SimdError::new(
                    "internal error: missing enum scratch local for constructor pattern",
                ));
            };
            emit_enum_tag_load_from_value(function, value_local, enum_state);
            function.instruction(&Instruction::I32Const(i32::from(ctor.tag)));
            function.instruction(&Instruction::I32Eq);
            Ok(true)
        }
        Pattern::Int(expected) => {
            let prim = ty.prim().ok_or_else(|| {
                SimdError::new(format!(
                    "integer pattern requires scalar primitive type, found {:?}",
                    ty
                ))
            })?;
            function.instruction(&Instruction::LocalGet(value_local));
            emit_int_const(function, prim, *expected)?;
            emit_int_eq(function, prim);
            Ok(true)
        }
        Pattern::Float(expected) => {
            let prim = ty.prim().ok_or_else(|| {
                SimdError::new(format!(
                    "float pattern requires scalar primitive type, found {:?}",
                    ty
                ))
            })?;
            function.instruction(&Instruction::LocalGet(value_local));
            emit_float_bits_eq(function, prim, *expected)?;
            Ok(true)
        }
        Pattern::Char(expected) => {
            let prim = ty.prim().ok_or_else(|| {
                SimdError::new(format!(
                    "char pattern requires scalar primitive type, found {:?}",
                    ty
                ))
            })?;
            if prim != Prim::Char {
                return Err(SimdError::new(format!(
                    "char pattern requires char type, found {:?}",
                    ty
                )));
            }
            function.instruction(&Instruction::LocalGet(value_local));
            function.instruction(&Instruction::I32Const(
                i32::try_from(u32::from(*expected))
                    .map_err(|_| SimdError::new("char literal codepoint does not fit in i32"))?,
            ));
            function.instruction(&Instruction::I32Eq);
            Ok(true)
        }
        Pattern::Bool(_) => Err(SimdError::new(
            "Wasm backend does not support bool clause patterns",
        )),
        Pattern::Type(_) => Err(SimdError::new(
            "Wasm backend does not support type witness clause patterns",
        )),
        Pattern::Tuple(_) => Err(SimdError::new(
            "Wasm backend does not support tuple clause patterns after flattening",
        )),
        Pattern::Slice {
            prefix,
            suffix,
            rest,
        } => {
            let prim = wasm_slice_pattern_prim(ty)?;

            let len_local = value_local + 1;
            let fixed = prefix.len() + suffix.len();
            let fixed_i32 = i32::try_from(fixed)
                .map_err(|_| SimdError::new("slice fixed width does not fit in i32"))?;

            function.instruction(&Instruction::LocalGet(len_local));
            function.instruction(&Instruction::I32Const(fixed_i32));
            match rest {
                None => function.instruction(&Instruction::I32Eq),
                Some(SliceRest::Ignore) | Some(SliceRest::Bind(_)) => {
                    function.instruction(&Instruction::I32GeU)
                }
            };
            function.instruction(&Instruction::If(BlockType::Result(ValType::I32)));

            let mut inner_terms = 0usize;

            for (index, subpattern) in prefix.iter().enumerate() {
                if matches!(subpattern, Pattern::Wildcard | Pattern::Name(_)) {
                    continue;
                }
                emit_bulk_element_load_at_prefix_index(function, value_local, prim, index)?;
                let emitted =
                    emit_slice_element_pattern_condition(function, subpattern, prim, enum_ctors)?;
                debug_assert!(emitted);
                if inner_terms > 0 {
                    function.instruction(&Instruction::I32And);
                }
                inner_terms += 1;
            }

            for (index, subpattern) in suffix.iter().enumerate() {
                if matches!(subpattern, Pattern::Wildcard | Pattern::Name(_)) {
                    continue;
                }
                emit_bulk_element_load_at_suffix_index(
                    function,
                    value_local,
                    len_local,
                    prim,
                    suffix.len(),
                    index,
                )?;
                let emitted =
                    emit_slice_element_pattern_condition(function, subpattern, prim, enum_ctors)?;
                debug_assert!(emitted);
                if inner_terms > 0 {
                    function.instruction(&Instruction::I32And);
                }
                inner_terms += 1;
            }

            if inner_terms == 0 {
                function.instruction(&Instruction::I32Const(1));
            }

            function.instruction(&Instruction::Else);
            function.instruction(&Instruction::I32Const(0));
            function.instruction(&Instruction::End);

            Ok(true)
        }
        Pattern::Ctor(name, subpatterns) => {
            let ctor = enum_ctors
                .get(name)
                .ok_or_else(|| SimdError::new(format!("unknown constructor '{}'", name)))?;
            let Type::Named(enum_name, enum_args) = ty else {
                return Err(SimdError::new(format!(
                    "constructor pattern '{}' requires enum type, found {:?}",
                    name, ty
                )));
            };
            if enum_name != &ctor.enum_name {
                return Err(SimdError::new(format!(
                    "constructor '{}' belongs to enum '{}', not '{}'",
                    name, ctor.enum_name, enum_name
                )));
            }
            if subpatterns.len() != ctor.fields.len() {
                return Err(SimdError::new(format!(
                    "constructor pattern '{}' expects {} fields, found {}",
                    name,
                    ctor.fields.len(),
                    subpatterns.len()
                )));
            }
            let Some(enum_state) = enum_state else {
                return Err(SimdError::new(
                    "internal error: missing enum scratch local for constructor pattern",
                ));
            };
            let wasm_layout =
                lookup_specialized_wasm_enum_layout(wasm_enum_layouts, enum_name, enum_args)?;
            let ctor_layout = lookup_specialized_wasm_enum_ctor_layout(
                wasm_enum_layouts,
                enum_name,
                enum_args,
                name,
            )?;
            emit_enum_tag_load_from_value(function, value_local, enum_state);
            function.instruction(&Instruction::I32Const(i32::from(ctor.tag)));
            function.instruction(&Instruction::I32Eq);
            let mut terms = 1usize;
            let subst = ctor
                .enum_params
                .iter()
                .cloned()
                .zip(enum_args.iter().cloned())
                .collect::<BTreeMap<_, _>>();
            for (field_index, (subpattern, field_ty_template)) in
                subpatterns.iter().zip(ctor.fields.iter()).enumerate()
            {
                let field_ty = apply_type_subst(field_ty_template, &subst);
                match subpattern {
                    Pattern::Wildcard => {}
                    Pattern::Name(local_name)
                        if !is_pattern_constructor_name(local_name, enum_ctors) =>
                    {
                        // Name bindings do not contribute to condition.
                    }
                    Pattern::Int(expected) => {
                        emit_enum_field_as_condition_value(
                            function,
                            value_local,
                            &wasm_layout,
                            &ctor_layout,
                            field_index,
                            &field_ty,
                            enum_state,
                        )?;
                        let prim = field_ty.prim().ok_or_else(|| {
                            SimdError::new(format!(
                                "integer subpattern in '{}' requires scalar field, found {:?}",
                                name, field_ty
                            ))
                        })?;
                        emit_int_const(function, prim, *expected)?;
                        emit_int_eq(function, prim);
                        function.instruction(&Instruction::I32And);
                        terms += 1;
                    }
                    Pattern::Float(expected) => {
                        emit_enum_field_as_condition_value(
                            function,
                            value_local,
                            &wasm_layout,
                            &ctor_layout,
                            field_index,
                            &field_ty,
                            enum_state,
                        )?;
                        let prim = field_ty.prim().ok_or_else(|| {
                            SimdError::new(format!(
                                "float subpattern in '{}' requires scalar field, found {:?}",
                                name, field_ty
                            ))
                        })?;
                        emit_float_bits_eq(function, prim, *expected)?;
                        function.instruction(&Instruction::I32And);
                        terms += 1;
                    }
                    Pattern::Char(expected) => {
                        emit_enum_field_as_condition_value(
                            function,
                            value_local,
                            &wasm_layout,
                            &ctor_layout,
                            field_index,
                            &field_ty,
                            enum_state,
                        )?;
                        let prim = field_ty.prim().ok_or_else(|| {
                            SimdError::new(format!(
                                "char subpattern in '{}' requires scalar field, found {:?}",
                                name, field_ty
                            ))
                        })?;
                        if prim != Prim::Char {
                            return Err(SimdError::new(format!(
                                "char subpattern in '{}' requires char field, found {:?}",
                                name, field_ty
                            )));
                        }
                        function.instruction(&Instruction::I32Const(
                            i32::try_from(u32::from(*expected)).map_err(|_| {
                                SimdError::new("char literal codepoint does not fit in i32")
                            })?,
                        ));
                        function.instruction(&Instruction::I32Eq);
                        function.instruction(&Instruction::I32And);
                        terms += 1;
                    }
                    Pattern::Name(ctor_name)
                        if is_pattern_constructor_name(ctor_name, enum_ctors) =>
                    {
                        emit_enum_field_load(
                            function,
                            value_local,
                            &wasm_layout,
                            &ctor_layout,
                            field_index,
                            &field_ty,
                            enum_state,
                        )?;
                        function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
                        if emit_single_pattern_condition(
                            function,
                            subpattern,
                            &field_ty,
                            enum_state.ptr_local,
                            enum_ctors,
                            wasm_enum_layouts,
                            enum_state.into(),
                        )? {
                            function.instruction(&Instruction::I32And);
                            terms += 1;
                        }
                    }
                    Pattern::Name(_) => {}
                    Pattern::Ctor(_, _) => {
                        emit_enum_field_load(
                            function,
                            value_local,
                            &wasm_layout,
                            &ctor_layout,
                            field_index,
                            &field_ty,
                            enum_state,
                        )?;
                        function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
                        if emit_single_pattern_condition(
                            function,
                            subpattern,
                            &field_ty,
                            enum_state.ptr_local,
                            enum_ctors,
                            wasm_enum_layouts,
                            enum_state.into(),
                        )? {
                            function.instruction(&Instruction::I32And);
                            terms += 1;
                        }
                    }
                    Pattern::Bool(_) => {
                        return Err(SimdError::new(
                            "Wasm backend does not support bool clause patterns",
                        ));
                    }
                    Pattern::Tuple(_) => {
                        return Err(SimdError::new(
                            "Wasm backend does not support tuple clause patterns after flattening",
                        ));
                    }
                    Pattern::Type(_) => {
                        return Err(SimdError::new(
                            "Wasm backend does not support type witness clause patterns",
                        ));
                    }
                    Pattern::Slice { .. } => {
                        return Err(SimdError::new(
                            "Wasm backend does not yet support slice view patterns",
                        ));
                    }
                }
            }
            Ok(terms > 0)
        }
    }
}

fn emit_clause_bindings(
    function: &mut Function,
    patterns: &[TypedPattern],
    clause_locals: &BTreeMap<String, u32>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: Option<EnumWasmState>,
) -> Result<()> {
    let mut value_local = 0u32;
    for pattern in patterns {
        emit_pattern_bindings(
            function,
            &pattern.pattern,
            &pattern.ty,
            value_local,
            clause_locals,
            enum_ctors,
            wasm_enum_layouts,
            enum_state,
        )?;
        value_local += wasm_pattern_value_width(&pattern.ty);
    }
    Ok(())
}

fn emit_pattern_bindings(
    function: &mut Function,
    pattern: &Pattern,
    ty: &Type,
    value_local: u32,
    clause_locals: &BTreeMap<String, u32>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: Option<EnumWasmState>,
) -> Result<()> {
    match pattern {
        Pattern::Name(name) => {
            if is_pattern_constructor_name(name, enum_ctors) || name == "_" {
                return Ok(());
            }
            let Some(local) = clause_locals.get(name).copied() else {
                return Ok(());
            };
            if local != value_local {
                function.instruction(&Instruction::LocalGet(value_local));
                function.instruction(&Instruction::LocalSet(local));
            }
            Ok(())
        }
        Pattern::Tuple(_) => Err(SimdError::new(
            "Wasm backend does not support tuple clause patterns after flattening",
        )),
        Pattern::Ctor(name, subpatterns) => {
            let ctor = enum_ctors
                .get(name)
                .ok_or_else(|| SimdError::new(format!("unknown constructor '{}'", name)))?;
            let Type::Named(enum_name, enum_args) = ty else {
                return Err(SimdError::new(format!(
                    "constructor pattern '{}' requires enum type, found {:?}",
                    name, ty
                )));
            };
            if enum_name != &ctor.enum_name {
                return Err(SimdError::new(format!(
                    "constructor '{}' belongs to enum '{}', not '{}'",
                    name, ctor.enum_name, enum_name
                )));
            }
            let subst = ctor
                .enum_params
                .iter()
                .cloned()
                .zip(enum_args.iter().cloned())
                .collect::<BTreeMap<_, _>>();
            let wasm_layout =
                lookup_specialized_wasm_enum_layout(wasm_enum_layouts, enum_name, enum_args)?;
            let ctor_layout = lookup_specialized_wasm_enum_ctor_layout(
                wasm_enum_layouts,
                enum_name,
                enum_args,
                name,
            )?;
            for (field_index, (subpattern, field_ty_template)) in
                subpatterns.iter().zip(ctor.fields.iter()).enumerate()
            {
                let field_ty = apply_type_subst(field_ty_template, &subst);
                match subpattern {
                    Pattern::Wildcard
                    | Pattern::Int(_)
                    | Pattern::Float(_)
                    | Pattern::Char(_)
                    | Pattern::Bool(_)
                    | Pattern::Tuple(_)
                    | Pattern::Type(_) => {}
                    Pattern::Name(local_name)
                        if !is_pattern_constructor_name(local_name, enum_ctors) =>
                    {
                        if let Some(target_local) = clause_locals.get(local_name).copied() {
                            let Some(enum_state) = enum_state else {
                                return Err(SimdError::new(
                                    "internal error: missing enum scratch local for constructor binding",
                                ));
                            };
                            emit_enum_field_load(
                                function,
                                value_local,
                                &wasm_layout,
                                &ctor_layout,
                                field_index,
                                &field_ty,
                                enum_state,
                            )?;
                            if is_wasm_bulk_like_type(&field_ty) {
                                let len_key = bulk_len_local_name(local_name);
                                let len_local =
                                    clause_locals.get(&len_key).copied().ok_or_else(|| {
                                        SimdError::new(format!(
                                            "missing clause local '{}' for enum bulk field binding",
                                            len_key
                                        ))
                                    })?;
                                function.instruction(&Instruction::LocalSet(len_local));
                                function.instruction(&Instruction::LocalSet(target_local));
                            } else {
                                function.instruction(&Instruction::LocalSet(target_local));
                            }
                        }
                    }
                    Pattern::Name(_) | Pattern::Ctor(_, _) => {
                        let Some(enum_state) = enum_state else {
                            return Err(SimdError::new(
                                "internal error: missing enum scratch local for nested constructor binding",
                            ));
                        };
                        emit_enum_field_load(
                            function,
                            value_local,
                            &wasm_layout,
                            &ctor_layout,
                            field_index,
                            &field_ty,
                            enum_state,
                        )?;
                        function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
                        emit_pattern_bindings(
                            function,
                            subpattern,
                            &field_ty,
                            enum_state.ptr_local,
                            clause_locals,
                            enum_ctors,
                            wasm_enum_layouts,
                            enum_state.into(),
                        )?;
                    }
                    Pattern::Slice { .. } => {
                        return Err(SimdError::new(
                            "Wasm backend does not yet support slice view patterns",
                        ));
                    }
                }
            }
            Ok(())
        }
        Pattern::Wildcard
        | Pattern::Int(_)
        | Pattern::Float(_)
        | Pattern::Char(_)
        | Pattern::Bool(_)
        | Pattern::Type(_) => Ok(()),
        Pattern::Slice {
            prefix,
            suffix,
            rest,
        } => {
            let prim = wasm_slice_pattern_prim(ty)?;
            let input_len_local = value_local + 1;

            for (index, subpattern) in prefix.iter().enumerate() {
                let target = match subpattern {
                    Pattern::Name(name)
                        if !is_pattern_constructor_name(name, enum_ctors) && name != "_" =>
                    {
                        clause_locals.get(name).copied()
                    }
                    Pattern::Wildcard | Pattern::Int(_) | Pattern::Float(_) | Pattern::Char(_) => {
                        None
                    }
                    Pattern::Name(_) => None,
                    Pattern::Bool(_) => {
                        return Err(SimdError::new(
                            "Wasm backend does not support bool slice element bindings",
                        ));
                    }
                    Pattern::Tuple(_) => {
                        return Err(SimdError::new(
                            "Wasm backend does not support tuple slice element bindings",
                        ));
                    }
                    Pattern::Type(_) => {
                        return Err(SimdError::new(
                            "Wasm backend does not support type witness slice element bindings",
                        ));
                    }
                    Pattern::Ctor(_, _) => {
                        return Err(SimdError::new(
                            "Wasm backend does not support constructor slice element bindings",
                        ));
                    }
                    Pattern::Slice { .. } => {
                        return Err(SimdError::new(
                            "Wasm backend does not support nested slice element bindings",
                        ));
                    }
                };
                if target.is_some() {
                    emit_bulk_element_load_at_prefix_index(function, value_local, prim, index)?;
                    emit_slice_element_name_binding(function, subpattern, target, enum_ctors)?;
                }
            }

            for (index, subpattern) in suffix.iter().enumerate() {
                let target = match subpattern {
                    Pattern::Name(name)
                        if !is_pattern_constructor_name(name, enum_ctors) && name != "_" =>
                    {
                        clause_locals.get(name).copied()
                    }
                    Pattern::Wildcard | Pattern::Int(_) | Pattern::Float(_) | Pattern::Char(_) => {
                        None
                    }
                    Pattern::Name(_) => None,
                    Pattern::Bool(_) => {
                        return Err(SimdError::new(
                            "Wasm backend does not support bool slice element bindings",
                        ));
                    }
                    Pattern::Tuple(_) => {
                        return Err(SimdError::new(
                            "Wasm backend does not support tuple slice element bindings",
                        ));
                    }
                    Pattern::Type(_) => {
                        return Err(SimdError::new(
                            "Wasm backend does not support type witness slice element bindings",
                        ));
                    }
                    Pattern::Ctor(_, _) => {
                        return Err(SimdError::new(
                            "Wasm backend does not support constructor slice element bindings",
                        ));
                    }
                    Pattern::Slice { .. } => {
                        return Err(SimdError::new(
                            "Wasm backend does not support nested slice element bindings",
                        ));
                    }
                };
                if target.is_some() {
                    emit_bulk_element_load_at_suffix_index(
                        function,
                        value_local,
                        input_len_local,
                        prim,
                        suffix.len(),
                        index,
                    )?;
                    emit_slice_element_name_binding(function, subpattern, target, enum_ctors)?;
                }
            }
            if let Some(SliceRest::Bind(name)) = rest
                && name != "_"
                && !is_pattern_constructor_name(name, enum_ctors)
            {
                let Some(rest_ptr_local) = clause_locals.get(name).copied() else {
                    return Err(SimdError::new(format!(
                        "missing clause local '{}' for slice rest binding",
                        name
                    )));
                };
                let rest_len_key = bulk_len_local_name(name);
                let Some(rest_len_local) = clause_locals.get(&rest_len_key).copied() else {
                    return Err(SimdError::new(format!(
                        "missing clause local '{}' for slice rest length binding",
                        rest_len_key
                    )));
                };
                let prefix_i32 = i32::try_from(prefix.len())
                    .map_err(|_| SimdError::new("slice prefix length does not fit in i32"))?;
                let suffix_i32 = i32::try_from(suffix.len())
                    .map_err(|_| SimdError::new("slice suffix length does not fit in i32"))?;
                let start_byte = i32::try_from(prefix.len())
                    .ok()
                    .and_then(|prefix_len| {
                        i32::try_from(byte_width(prim))
                            .ok()
                            .map(|bw| prefix_len * bw)
                    })
                    .ok_or_else(|| SimdError::new("slice rest byte offset does not fit in i32"))?;
                function.instruction(&Instruction::LocalGet(value_local));
                function.instruction(&Instruction::I32Const(start_byte));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalSet(rest_ptr_local));

                function.instruction(&Instruction::LocalGet(input_len_local));
                if prefix_i32 != 0 {
                    function.instruction(&Instruction::I32Const(prefix_i32));
                    function.instruction(&Instruction::I32Sub);
                }
                if suffix_i32 != 0 {
                    function.instruction(&Instruction::I32Const(suffix_i32));
                    function.instruction(&Instruction::I32Sub);
                }
                function.instruction(&Instruction::LocalSet(rest_len_local));
            }
            Ok(())
        }
    }
}

fn emit_int_const(function: &mut Function, prim: Prim, value: i64) -> Result<()> {
    match prim {
        Prim::I32 | Prim::Char => {
            function.instruction(&Instruction::I32Const(i32::try_from(value).map_err(
                |_| SimdError::new(format!("integer literal '{}' does not fit in i32", value)),
            )?));
        }
        Prim::I64 => {
            function.instruction(&Instruction::I64Const(value));
        }
        Prim::F32 | Prim::F64 => {
            return Err(SimdError::new(
                "integer pattern cannot be compiled against float primitive",
            ));
        }
    }
    Ok(())
}

fn emit_copy_i32_words(function: &mut Function, src_local: u32, dst_local: u32, words_local: u32) {
    function.instruction(&Instruction::Block(BlockType::Empty));
    function.instruction(&Instruction::Loop(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(words_local));
    function.instruction(&Instruction::I32Eqz);
    function.instruction(&Instruction::BrIf(1));
    function.instruction(&Instruction::LocalGet(dst_local));
    function.instruction(&Instruction::LocalGet(src_local));
    function.instruction(&Instruction::I32Load(memarg(0, 2)));
    function.instruction(&Instruction::I32Store(memarg(0, 2)));
    function.instruction(&Instruction::LocalGet(src_local));
    function.instruction(&Instruction::I32Const(4));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(src_local));
    function.instruction(&Instruction::LocalGet(dst_local));
    function.instruction(&Instruction::I32Const(4));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(dst_local));
    function.instruction(&Instruction::LocalGet(words_local));
    function.instruction(&Instruction::I32Const(1));
    function.instruction(&Instruction::I32Sub);
    function.instruction(&Instruction::LocalSet(words_local));
    function.instruction(&Instruction::Br(0));
    function.instruction(&Instruction::End);
    function.instruction(&Instruction::End);
}

fn emit_copy_i16_units(
    function: &mut Function,
    src_local: u32,
    dst_local: u32,
    units_local: u32,
    words_local: u32,
) {
    function.instruction(&Instruction::LocalGet(units_local));
    function.instruction(&Instruction::I32Const(1));
    function.instruction(&Instruction::I32ShrU);
    function.instruction(&Instruction::LocalSet(words_local));
    emit_copy_i32_words(function, src_local, dst_local, words_local);
    function.instruction(&Instruction::LocalGet(units_local));
    function.instruction(&Instruction::I32Const(1));
    function.instruction(&Instruction::I32And);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::LocalGet(dst_local));
    function.instruction(&Instruction::LocalGet(src_local));
    function.instruction(&Instruction::I32Load16U(memarg(0, 1)));
    function.instruction(&Instruction::I32Store16(memarg(0, 1)));
    function.instruction(&Instruction::End);
}

fn emit_assert_enum_root_zero(function: &mut Function, handle_local: u32) {
    function.instruction(&Instruction::LocalGet(handle_local));
    function.instruction(&Instruction::I32Load(memarg(ENUM_REF_ROOT_OFFSET, 2)));
    function.instruction(&Instruction::I32Eqz);
    function.instruction(&Instruction::If(BlockType::Empty));
    function.instruction(&Instruction::Else);
    function.instruction(&Instruction::Unreachable);
    function.instruction(&Instruction::End);
}

fn emit_int_eq(function: &mut Function, prim: Prim) {
    match prim {
        Prim::I32 | Prim::Char => {
            function.instruction(&Instruction::I32Eq);
        }
        Prim::I64 => {
            function.instruction(&Instruction::I64Eq);
        }
        Prim::F32 | Prim::F64 => unreachable!(),
    }
}

fn emit_float_bits_eq(function: &mut Function, prim: Prim, value: f64) -> Result<()> {
    match prim {
        Prim::F32 => {
            function.instruction(&Instruction::I32ReinterpretF32);
            function.instruction(&Instruction::I32Const((value as f32).to_bits() as i32));
            function.instruction(&Instruction::I32Eq);
            Ok(())
        }
        Prim::F64 => {
            function.instruction(&Instruction::I64ReinterpretF64);
            function.instruction(&Instruction::I64Const(value.to_bits() as i64));
            function.instruction(&Instruction::I64Eq);
            Ok(())
        }
        Prim::I32 | Prim::I64 | Prim::Char => Err(SimdError::new(
            "float pattern cannot be compiled against integer primitive",
        )),
    }
}

fn compile_scalar_ir_expr(
    function: &mut Function,
    expr: &IrExpr,
    locals: &BTreeMap<String, u32>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: Option<EnumWasmState>,
) -> Result<()> {
    let inline_bindings = BTreeMap::<String, IrExpr>::new();
    compile_scalar_ir_expr_with_hoists(
        function,
        expr,
        locals,
        scalar_indices,
        signatures,
        enum_ctors,
        wasm_enum_layouts,
        enum_state,
        &BTreeMap::new(),
        &inline_bindings,
    )
}

fn emit_wasm_function_profile_enter(
    function: &mut Function,
    profile: Option<WasmFunctionInstrumentation>,
) {
    if let Some(profile) = profile {
        function.instruction(&Instruction::I32Const(profile.func_id as i32));
        function.instruction(&Instruction::Call(profile.enter_import_index));
    }
}

fn emit_wasm_function_profile_exit(
    function: &mut Function,
    profile: Option<WasmFunctionInstrumentation>,
) {
    if let Some(profile) = profile {
        function.instruction(&Instruction::I32Const(profile.func_id as i32));
        function.instruction(&Instruction::Call(profile.exit_import_index));
    }
}

fn emit_tail_position_scalar_return(
    function: &mut Function,
    expr: &IrExpr,
    locals: &BTreeMap<String, u32>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: Option<EnumWasmState>,
    function_profile: Option<WasmFunctionInstrumentation>,
) -> Result<()> {
    let inline_bindings = BTreeMap::<String, IrExpr>::new();
    emit_tail_position_scalar_return_with_bindings(
        function,
        expr,
        locals,
        scalar_indices,
        signatures,
        enum_ctors,
        wasm_enum_layouts,
        enum_state,
        function_profile,
        &inline_bindings,
    )
}

fn emit_tail_position_scalar_return_with_bindings(
    function: &mut Function,
    expr: &IrExpr,
    locals: &BTreeMap<String, u32>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: Option<EnumWasmState>,
    function_profile: Option<WasmFunctionInstrumentation>,
    inline_bindings: &BTreeMap<String, IrExpr>,
) -> Result<()> {
    if let IrExprKind::Let { bindings, body } = &expr.kind {
        let mut extended = inline_bindings.clone();
        for binding in bindings {
            extended.insert(binding.name.clone(), binding.expr.clone());
        }
        return emit_tail_position_scalar_return_with_bindings(
            function,
            body,
            locals,
            scalar_indices,
            signatures,
            enum_ctors,
            wasm_enum_layouts,
            enum_state,
            function_profile,
            &extended,
        );
    }

    if let IrExprKind::Call {
        callee: Callee::Function(name),
        args,
    } = &expr.kind
    {
        let checked = signatures
            .get(name)
            .copied()
            .ok_or_else(|| SimdError::new(format!("missing checked signature for '{}'", name)))?;
        let (params, result) = checked.signature.ty.fun_parts();
        if !wasm_scalar_call_types_match(&result, &expr.ty) {
            return Err(SimdError::new(format!(
                "Wasm scalar tail call '{}' returns {:?}, expected {:?}",
                name, result, expr.ty
            )));
        }
        if params.len() != args.len() {
            return Err(SimdError::new(format!(
                "Wasm scalar tail call '{}' received {} args, expected {}",
                name,
                args.len(),
                params.len()
            )));
        }
        for (arg, ty) in args.iter().zip(params.iter()) {
            compile_scalar_ir_expr_with_hoists(
                function,
                arg,
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                enum_state,
                &BTreeMap::new(),
                inline_bindings,
            )?;
            if !wasm_scalar_call_types_match(&arg.ty, ty) {
                return Err(SimdError::new(format!(
                    "Wasm scalar tail call '{}' received {:?}, expected {:?}",
                    name, arg.ty, ty
                )));
            }
        }
        let index = scalar_indices
            .get(name)
            .copied()
            .ok_or_else(|| SimdError::new(format!("missing Wasm function index for '{}'", name)))?;
        if function_profile.is_some_and(|profile| profile.allow_return_call) {
            function.instruction(&Instruction::ReturnCall(index));
        } else {
            function.instruction(&Instruction::Call(index));
            emit_wasm_function_profile_exit(function, function_profile);
            function.instruction(&Instruction::Return);
        }
        return Ok(());
    }

    compile_scalar_ir_expr_with_hoists(
        function,
        expr,
        locals,
        scalar_indices,
        signatures,
        enum_ctors,
        wasm_enum_layouts,
        enum_state,
        &BTreeMap::new(),
        inline_bindings,
    )?;
    if wasm_star_seq_storage_prim(&expr.ty).is_some() {
        let enum_state = enum_state.ok_or_else(|| {
            SimdError::new("internal error: missing enum scratch local for T[*] return")
        })?;
        emit_box_seq_from_stack(function, enum_state);
    }
    emit_wasm_function_profile_exit(function, function_profile);
    function.instruction(&Instruction::Return);
    Ok(())
}

fn compile_scalar_ir_expr_with_hoists(
    function: &mut Function,
    expr: &IrExpr,
    locals: &BTreeMap<String, u32>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: Option<EnumWasmState>,
    hoisted_locals: &BTreeMap<HoistExprKey, u32>,
    inline_bindings: &BTreeMap<String, IrExpr>,
) -> Result<()> {
    fn emit_store_scalar_at_current_address(function: &mut Function, prim: Prim) -> Result<()> {
        match prim {
            Prim::I32 | Prim::Char => {
                function.instruction(&Instruction::I32Store(memarg(0, 2)));
            }
            Prim::I64 => {
                function.instruction(&Instruction::I64Store(memarg(0, 3)));
            }
            Prim::F32 => {
                function.instruction(&Instruction::F32Store(memarg(0, 2)));
            }
            Prim::F64 => {
                function.instruction(&Instruction::F64Store(memarg(0, 3)));
            }
        }
        Ok(())
    }

    if let Some(local) = hoisted_locals.get(&hoist_expr_key(expr)) {
        if is_wasm_bulk_like_type(&expr.ty) {
            return Err(SimdError::new(
                "Wasm scalar codegen does not support hoisted bulk locals",
            ));
        }
        function.instruction(&Instruction::LocalGet(*local));
        return Ok(());
    }
    match &expr.kind {
        IrExprKind::Seq(items) => {
            let enum_state = enum_state.ok_or_else(|| {
                SimdError::new(
                    "internal error: missing enum scratch local for sequence literal emission",
                )
            })?;
            let prim = wasm_star_seq_storage_prim(&expr.ty).ok_or_else(|| {
                SimdError::new(format!(
                    "Wasm scalar codegen only supports scalar/enum-element T[*] sequence literals, found {:?}",
                    expr.ty
                ))
            })?;
            let elem_bytes = byte_width(prim) as i32;
            let len_i32 = i32::try_from(items.len())
                .map_err(|_| SimdError::new("sequence literal length does not fit in i32"))?;
            let total_bytes = len_i32
                .checked_mul(elem_bytes)
                .ok_or_else(|| SimdError::new("sequence literal byte-size overflow"))?;
            emit_enum_alloc(function, enum_state, total_bytes);
            function.instruction(&Instruction::LocalGet(enum_state.scratch_sp_local));
            function.instruction(&Instruction::I32Const(8));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.scratch_sp_local));
            function.instruction(&Instruction::LocalGet(enum_state.scratch_sp_local));
            function.instruction(&Instruction::I32Const(4));
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
            function.instruction(&Instruction::I32Store(memarg(0, 2)));
            for (index, item) in items.iter().enumerate() {
                let offset = i32::try_from(index)
                    .map_err(|_| {
                        SimdError::new("sequence literal element index does not fit in i32")
                    })?
                    .checked_mul(elem_bytes)
                    .ok_or_else(|| SimdError::new("sequence literal element offset overflow"))?;
                function.instruction(&Instruction::LocalGet(enum_state.scratch_sp_local));
                function.instruction(&Instruction::I32Const(4));
                function.instruction(&Instruction::I32Sub);
                function.instruction(&Instruction::I32Load(memarg(0, 2)));
                if offset != 0 {
                    function.instruction(&Instruction::I32Const(offset));
                    function.instruction(&Instruction::I32Add);
                }
                compile_scalar_ir_expr_with_hoists(
                    function,
                    item,
                    locals,
                    scalar_indices,
                    signatures,
                    enum_ctors,
                    wasm_enum_layouts,
                    enum_state.into(),
                    hoisted_locals,
                    inline_bindings,
                )?;
                let item_storage_prim = match &item.ty {
                    Type::Scalar(item_prim) => *item_prim,
                    Type::Named(name, args) if is_wasm_enum_named_type(name, args) => Prim::I32,
                    other => {
                        return Err(SimdError::new(format!(
                            "sequence literal item type {:?} is not supported in Wasm",
                            other
                        )));
                    }
                };
                if item_storage_prim != prim {
                    return Err(SimdError::new(format!(
                        "sequence literal item {:?} did not match element storage primitive {:?}",
                        item.ty, prim
                    )));
                }
                emit_store_scalar_at_current_address(function, prim)?;
            }
            function.instruction(&Instruction::LocalGet(enum_state.scratch_sp_local));
            function.instruction(&Instruction::I32Const(4));
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::I32Load(memarg(0, 2)));
            function.instruction(&Instruction::I32Const(len_i32));
            function.instruction(&Instruction::LocalGet(enum_state.scratch_sp_local));
            function.instruction(&Instruction::I32Const(8));
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::LocalSet(enum_state.scratch_sp_local));
        }
        IrExprKind::SeqSplice { prefix, tail } => {
            if prefix.is_empty() {
                compile_scalar_ir_expr_with_hoists(
                    function,
                    tail,
                    locals,
                    scalar_indices,
                    signatures,
                    enum_ctors,
                    wasm_enum_layouts,
                    enum_state,
                    hoisted_locals,
                    inline_bindings,
                )?;
                return Ok(());
            }
            let enum_state = enum_state.ok_or_else(|| {
                SimdError::new(
                    "internal error: missing enum scratch local for sequence splice emission",
                )
            })?;
            let prim = wasm_star_seq_storage_prim(&expr.ty).ok_or_else(|| {
                SimdError::new(format!(
                    "Wasm scalar codegen only supports scalar/enum-element T[*] sequence splices, found {:?}",
                    expr.ty
                ))
            })?;
            let tail_prim = wasm_star_seq_storage_prim(&tail.ty).ok_or_else(|| {
                SimdError::new(format!(
                    "Wasm scalar codegen only supports T[*] tails in sequence splices, found {:?}",
                    tail.ty
                ))
            })?;
            if tail_prim != prim {
                return Err(SimdError::new(format!(
                    "sequence splice tail {:?} did not match element storage primitive {:?}",
                    tail.ty, prim
                )));
            }
            compile_scalar_ir_expr_with_hoists(
                function,
                tail,
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                enum_state.into(),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::LocalSet(enum_state.aux1_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux0_local));

            let prefix_len_i32 = i32::try_from(prefix.len())
                .map_err(|_| SimdError::new("sequence splice prefix length does not fit in i32"))?;
            let elem_bytes_i32 = i32::try_from(byte_width(prim))
                .map_err(|_| SimdError::new("sequence splice element width does not fit in i32"))?;
            let prefix_bytes_i32 = prefix_len_i32
                .checked_mul(elem_bytes_i32)
                .ok_or_else(|| SimdError::new("sequence splice prefix byte-size overflow"))?;

            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Const(prefix_len_i32));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.aux2_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
            function.instruction(&Instruction::I32Const(elem_bytes_i32));
            function.instruction(&Instruction::I32Mul);
            function.instruction(&Instruction::LocalSet(enum_state.aux3_local));

            emit_enum_alloc_dynamic(function, enum_state, enum_state.aux3_local);
            function.instruction(&Instruction::LocalGet(enum_state.scratch_sp_local));
            function.instruction(&Instruction::I32Const(8));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.scratch_sp_local));
            function.instruction(&Instruction::LocalGet(enum_state.scratch_sp_local));
            function.instruction(&Instruction::I32Const(4));
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
            function.instruction(&Instruction::I32Store(memarg(0, 2)));
            function.instruction(&Instruction::LocalGet(enum_state.scratch_sp_local));
            function.instruction(&Instruction::I32Const(4));
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::I32Load(memarg(0, 2)));
            function.instruction(&Instruction::LocalSet(enum_state.aux4_local));

            for (index, item) in prefix.iter().enumerate() {
                let offset = i32::try_from(index)
                    .map_err(|_| {
                        SimdError::new("sequence splice prefix index does not fit in i32")
                    })?
                    .checked_mul(elem_bytes_i32)
                    .ok_or_else(|| SimdError::new("sequence splice prefix offset overflow"))?;
                function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
                if offset != 0 {
                    function.instruction(&Instruction::I32Const(offset));
                    function.instruction(&Instruction::I32Add);
                }
                compile_scalar_ir_expr_with_hoists(
                    function,
                    item,
                    locals,
                    scalar_indices,
                    signatures,
                    enum_ctors,
                    wasm_enum_layouts,
                    enum_state.into(),
                    hoisted_locals,
                    inline_bindings,
                )?;
                let item_storage_prim = match &item.ty {
                    Type::Scalar(item_prim) => *item_prim,
                    Type::Named(name, args) if is_wasm_enum_named_type(name, args) => Prim::I32,
                    other => {
                        return Err(SimdError::new(format!(
                            "sequence splice prefix item type {:?} is not supported in Wasm",
                            other
                        )));
                    }
                };
                if item_storage_prim != prim {
                    return Err(SimdError::new(format!(
                        "sequence splice prefix item {:?} did not match element storage primitive {:?}",
                        item.ty, prim
                    )));
                }
                emit_store_scalar_at_current_address(function, prim)?;
            }

            compile_scalar_ir_expr_with_hoists(
                function,
                tail,
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                enum_state.into(),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::LocalSet(enum_state.aux1_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux0_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Const(prefix_len_i32));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.aux2_local));
            function.instruction(&Instruction::LocalGet(enum_state.scratch_sp_local));
            function.instruction(&Instruction::I32Const(4));
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::I32Load(memarg(0, 2)));
            function.instruction(&Instruction::LocalSet(enum_state.aux4_local));

            function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
            function.instruction(&Instruction::I32Const(prefix_bytes_i32));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Const(elem_bytes_i32));
            function.instruction(&Instruction::I32Mul);
            function.instruction(&Instruction::MemoryCopy {
                src_mem: 0,
                dst_mem: 0,
            });

            function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
            function.instruction(&Instruction::LocalGet(enum_state.scratch_sp_local));
            function.instruction(&Instruction::I32Const(8));
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::LocalSet(enum_state.scratch_sp_local));
        }
        IrExprKind::Local(name) => {
            if let Some(index) = locals.get(name).copied() {
                if is_wasm_bulk_like_type(&expr.ty) {
                    let len_key = bulk_len_local_name(name);
                    let len_index = locals.get(&len_key).copied().unwrap_or(index + 1);
                    function.instruction(&Instruction::LocalGet(index));
                    function.instruction(&Instruction::LocalGet(len_index));
                } else {
                    function.instruction(&Instruction::LocalGet(index));
                }
            } else if let Some(inline_expr) = inline_bindings.get(name) {
                compile_scalar_ir_expr_with_hoists(
                    function,
                    inline_expr,
                    locals,
                    scalar_indices,
                    signatures,
                    enum_ctors,
                    wasm_enum_layouts,
                    enum_state,
                    hoisted_locals,
                    inline_bindings,
                )?;
            } else {
                return Err(SimdError::new(format!(
                    "unknown local '{}' in Wasm codegen",
                    name
                )));
            }
        }
        IrExprKind::Int(value, prim) => emit_int_const(function, *prim, *value)?,
        IrExprKind::Float(value, prim) => match prim {
            Prim::F32 => {
                function.instruction(&Instruction::F32Const((*value as f32).into()));
            }
            Prim::F64 => {
                function.instruction(&Instruction::F64Const((*value).into()));
            }
            Prim::I32 | Prim::I64 | Prim::Char => {
                return Err(SimdError::new(format!(
                    "float literal cannot inhabit integer primitive {:?}",
                    prim
                )));
            }
        },
        IrExprKind::Record(_) => {
            return Err(SimdError::new(
                "Wasm scalar codegen does not yet support record IR expressions",
            ));
        }
        IrExprKind::EnumCtor { ctor, args } => {
            let enum_state = enum_state.ok_or_else(|| {
                SimdError::new(
                    "internal error: missing enum scratch local for constructor emission",
                )
            })?;
            let ctor_info = enum_ctors
                .get(ctor)
                .ok_or_else(|| SimdError::new(format!("unknown constructor '{}'", ctor)))?;
            let Type::Named(_, enum_args) = &expr.ty else {
                return Err(SimdError::new(format!(
                    "constructor '{}' expected enum result type, found {:?}",
                    ctor, expr.ty
                )));
            };
            let wasm_layout = lookup_specialized_wasm_enum_layout(
                wasm_enum_layouts,
                &ctor_info.enum_name,
                enum_args,
            )?;
            let ctor_layout = lookup_specialized_wasm_enum_ctor_layout(
                wasm_enum_layouts,
                &ctor_info.enum_name,
                enum_args,
                ctor,
            )?
            .clone();
            let mut enum_ctor_layouts = wasm_layout.ctors.values().cloned().collect::<Vec<_>>();
            enum_ctor_layouts.sort_by_key(|layout| layout.tag);
            let row_bearing_ctor_layouts = enum_ctor_layouts
                .iter()
                .filter(|layout| layout.row_stride != 0)
                .cloned()
                .collect::<Vec<_>>();
            if ctor_info.fields.len() != args.len() {
                return Err(SimdError::new(format!(
                    "constructor '{}' expects {} args, found {}",
                    ctor,
                    ctor_info.fields.len(),
                    args.len()
                )));
            }
            let tag_count = usize::from(wasm_layout.max_tag) + 1;
            let tag_count_i32 = i32::try_from(tag_count)
                .map_err(|_| SimdError::new("enum tag-count does not fit in i32"))?;
            if ctor_layout.recursive_field_indices.len() > enum_state.child_locals.len() {
                return Err(SimdError::new(format!(
                    "constructor '{}' has too many recursive fields for Wasm codegen",
                    ctor
                )));
            }
            let mut arg_offsets = Vec::with_capacity(args.len());
            let mut arg_scratch_bytes = 0i32;
            for (field_index, arg) in args.iter().enumerate() {
                if ctor_layout.recursive_field_indices.contains(&field_index) {
                    arg_offsets.push(None);
                } else {
                    arg_offsets.push(Some(arg_scratch_bytes));
                    arg_scratch_bytes = arg_scratch_bytes
                        .checked_add(wasm_saved_value_bytes(&arg.ty)?)
                        .ok_or_else(|| SimdError::new("enum arg scratch size overflow"))?;
                }
            }
            let counts_base_offset = arg_scratch_bytes;
            let child_offsets_base_offset = counts_base_offset
                .checked_add(
                    tag_count_i32
                        .checked_mul(4)
                        .ok_or_else(|| SimdError::new("enum tag-count scratch size overflow"))?,
                )
                .ok_or_else(|| SimdError::new("enum scratch offset overflow"))?;
            let child_handles_base_offset = child_offsets_base_offset
                .checked_add(
                    tag_count_i32
                        .checked_mul(4)
                        .ok_or_else(|| SimdError::new("enum child-offset scratch size overflow"))?,
                )
                .ok_or_else(|| SimdError::new("enum scratch offset overflow"))?;
            let child_handle_count_i32 = i32::try_from(ctor_layout.recursive_field_indices.len())
                .map_err(|_| {
                SimdError::new("enum recursive child count does not fit in i32")
            })?;
            let total_scratch_bytes = child_handles_base_offset
                .checked_add(
                    child_handle_count_i32
                        .checked_mul(4)
                        .ok_or_else(|| SimdError::new("enum child-handle scratch size overflow"))?,
                )
                .ok_or_else(|| SimdError::new("enum scratch size overflow"))?;

            function.instruction(&Instruction::LocalGet(enum_state.scratch_sp_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux0_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
            function.instruction(&Instruction::I32Const(total_scratch_bytes));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.scratch_sp_local));
            function.instruction(&Instruction::I32Const(ENUM_SAVE_STACK_PTR_ADDR as i32));
            function.instruction(&Instruction::LocalGet(enum_state.scratch_sp_local));
            function.instruction(&Instruction::I32Store(memarg(0, 2)));

            let mut recursive_local_index = 0usize;
            for (field_index, arg) in args.iter().enumerate() {
                if ctor_layout.recursive_field_indices.contains(&field_index) {
                    compile_scalar_ir_expr_with_hoists(
                        function,
                        arg,
                        locals,
                        scalar_indices,
                        signatures,
                        enum_ctors,
                        wasm_enum_layouts,
                        enum_state.into(),
                        hoisted_locals,
                        inline_bindings,
                    )?;
                    function.instruction(&Instruction::LocalGet(enum_state.scratch_sp_local));
                    function.instruction(&Instruction::I32Const(total_scratch_bytes));
                    function.instruction(&Instruction::I32Sub);
                    function.instruction(&Instruction::LocalSet(enum_state.aux0_local));
                    function.instruction(&Instruction::LocalSet(enum_state.base_tmp_local));
                    function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                    function.instruction(&Instruction::I32Const(
                        child_handles_base_offset
                            + i32::try_from(recursive_local_index).map_err(|_| {
                                SimdError::new("enum recursive child index does not fit in i32")
                            })? * 4,
                    ));
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
                    function.instruction(&Instruction::I32Store(memarg(0, 2)));
                    recursive_local_index += 1;
                } else {
                    let arg_offset = arg_offsets[field_index]
                        .ok_or_else(|| SimdError::new("missing enum arg scratch offset"))?;
                    function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                    if arg_offset != 0 {
                        function.instruction(&Instruction::I32Const(arg_offset));
                        function.instruction(&Instruction::I32Add);
                    }
                    compile_scalar_ir_expr_with_hoists(
                        function,
                        arg,
                        locals,
                        scalar_indices,
                        signatures,
                        enum_ctors,
                        wasm_enum_layouts,
                        enum_state.into(),
                        hoisted_locals,
                        inline_bindings,
                    )?;
                    emit_store_type_at_address(function, &arg.ty, enum_state)?;
                    function.instruction(&Instruction::LocalGet(enum_state.scratch_sp_local));
                    function.instruction(&Instruction::I32Const(total_scratch_bytes));
                    function.instruction(&Instruction::I32Sub);
                    function.instruction(&Instruction::LocalSet(enum_state.aux0_local));
                }
            }

            if ctor_layout.recursive_field_indices.is_empty() {
                let leaf_total_bytes = ENUM_REF_CELL_BYTES
                    .checked_add(ENUM_HEADER_BYTES)
                    .and_then(|v| v.checked_add(ENUM_TAG_BYTES))
                    .and_then(|v| v.checked_add(ENUM_END_BYTES))
                    .and_then(|v| v.checked_add(ENUM_SLOT_BYTES))
                    .and_then(|v| {
                        v.checked_add(
                            tag_count_i32
                                .checked_mul(ENUM_CTOR_TABLE_ENTRY_BYTES)
                                .unwrap_or(i32::MAX),
                        )
                    })
                    .and_then(|v| v.checked_add(ctor_layout.row_stride as i32))
                    .ok_or_else(|| SimdError::new("enum leaf constructor size overflow"))?;
                emit_enum_alloc(function, enum_state, leaf_total_bytes);

                function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
                function.instruction(&Instruction::I32Const(ENUM_REF_CELL_BYTES));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalSet(enum_state.aux4_local));

                function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
                function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
                function.instruction(&Instruction::I32Store(memarg(ENUM_REF_HEADER_OFFSET, 2)));
                function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
                function.instruction(&Instruction::I32Const(0));
                function.instruction(&Instruction::I32Store(memarg(ENUM_REF_ROOT_OFFSET, 2)));

                function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
                function.instruction(&Instruction::I32Const(wasm_layout.enum_id as i32));
                function.instruction(&Instruction::I32Store(memarg(
                    ENUM_HEADER_ENUM_ID_OFFSET,
                    2,
                )));
                function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
                function.instruction(&Instruction::I32Const(1));
                function.instruction(&Instruction::I32Store(memarg(
                    ENUM_HEADER_NODE_COUNT_OFFSET,
                    2,
                )));

                function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
                function.instruction(&Instruction::I32Const(ENUM_HEADER_BYTES));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalSet(enum_state.aux5_local));
                function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
                function.instruction(&Instruction::LocalGet(enum_state.aux5_local));
                function.instruction(&Instruction::I32Store(memarg(
                    ENUM_HEADER_TAGS_PTR_OFFSET,
                    2,
                )));

                function.instruction(&Instruction::LocalGet(enum_state.aux5_local));
                function.instruction(&Instruction::I32Const(ENUM_TAG_BYTES));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalSet(enum_state.aux6_local));
                function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
                function.instruction(&Instruction::LocalGet(enum_state.aux6_local));
                function.instruction(&Instruction::I32Store(memarg(
                    ENUM_HEADER_ENDS_PTR_OFFSET,
                    2,
                )));

                function.instruction(&Instruction::LocalGet(enum_state.aux6_local));
                function.instruction(&Instruction::I32Const(ENUM_END_BYTES));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalSet(enum_state.aux7_local));
                function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
                function.instruction(&Instruction::LocalGet(enum_state.aux7_local));
                function.instruction(&Instruction::I32Store(memarg(
                    ENUM_HEADER_SLOTS_PTR_OFFSET,
                    2,
                )));

                function.instruction(&Instruction::LocalGet(enum_state.aux7_local));
                function.instruction(&Instruction::I32Const(ENUM_SLOT_BYTES));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalSet(enum_state.aux2_local));
                function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
                function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
                function.instruction(&Instruction::I32Store(memarg(
                    ENUM_HEADER_CTOR_TABLE_PTR_OFFSET,
                    2,
                )));

                for tag in 0..tag_count {
                    function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
                    function.instruction(&Instruction::I32Const(
                        i32::try_from(tag)
                            .map_err(|_| SimdError::new("enum tag index does not fit in i32"))?
                            * ENUM_CTOR_TABLE_ENTRY_BYTES,
                    ));
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::LocalTee(enum_state.aux3_local));
                    function.instruction(&Instruction::I32Const(0));
                    function.instruction(&Instruction::I32Store(memarg(
                        ENUM_CTOR_TABLE_ROWS_PTR_OFFSET,
                        2,
                    )));
                    function.instruction(&Instruction::LocalGet(enum_state.aux3_local));
                    function.instruction(&Instruction::I32Const(0));
                    function.instruction(&Instruction::I32Store(memarg(
                        ENUM_CTOR_TABLE_ROW_COUNT_OFFSET,
                        2,
                    )));
                }

                function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
                function.instruction(&Instruction::I32Const(
                    tag_count_i32 * ENUM_CTOR_TABLE_ENTRY_BYTES,
                ));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalSet(enum_state.aux3_local));

                function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
                function.instruction(&Instruction::I32Const(
                    i32::from(ctor_layout.tag) * ENUM_CTOR_TABLE_ENTRY_BYTES,
                ));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalTee(enum_state.base_tmp_local));
                if ctor_layout.row_stride == 0 {
                    function.instruction(&Instruction::I32Const(0));
                } else {
                    function.instruction(&Instruction::LocalGet(enum_state.aux3_local));
                }
                function.instruction(&Instruction::I32Store(memarg(
                    ENUM_CTOR_TABLE_ROWS_PTR_OFFSET,
                    2,
                )));
                function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
                function.instruction(&Instruction::I32Const(1));
                function.instruction(&Instruction::I32Store(memarg(
                    ENUM_CTOR_TABLE_ROW_COUNT_OFFSET,
                    2,
                )));

                function.instruction(&Instruction::LocalGet(enum_state.aux5_local));
                function.instruction(&Instruction::I32Const(i32::from(ctor_layout.tag)));
                function.instruction(&Instruction::I32Store16(memarg(0, 1)));
                function.instruction(&Instruction::LocalGet(enum_state.aux6_local));
                function.instruction(&Instruction::I32Const(1));
                function.instruction(&Instruction::I32Store(memarg(0, 2)));
                function.instruction(&Instruction::LocalGet(enum_state.aux7_local));
                function.instruction(&Instruction::I32Const(0));
                function.instruction(&Instruction::I32Store(memarg(0, 2)));

                if ctor_layout.row_stride != 0 {
                    function.instruction(&Instruction::LocalGet(enum_state.aux3_local));
                    function.instruction(&Instruction::LocalSet(enum_state.base_local));
                    for field_layout in &ctor_layout.non_recursive_fields {
                        let arg_offset =
                            arg_offsets[field_layout.source_index].ok_or_else(|| {
                                SimdError::new("missing enum row field scratch offset")
                            })?;
                        function.instruction(&Instruction::LocalGet(enum_state.base_local));
                        if field_layout.offset != 0 {
                            function
                                .instruction(&Instruction::I32Const(field_layout.offset as i32));
                            function.instruction(&Instruction::I32Add);
                        }
                        function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                        if arg_offset != 0 {
                            function.instruction(&Instruction::I32Const(arg_offset));
                            function.instruction(&Instruction::I32Add);
                        }
                        emit_load_type_from_address(function, &field_layout.ty, enum_state)?;
                        emit_tape_enum_field_store(function, field_layout, enum_state)?;
                    }
                }

                function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                function.instruction(&Instruction::LocalSet(enum_state.scratch_sp_local));
                function.instruction(&Instruction::I32Const(ENUM_SAVE_STACK_PTR_ADDR as i32));
                function.instruction(&Instruction::LocalGet(enum_state.scratch_sp_local));
                function.instruction(&Instruction::I32Store(memarg(0, 2)));
                function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
                function.instruction(&Instruction::I32Const(ENUM_REF_CELL_BYTES));
                function.instruction(&Instruction::I32Sub);
                return Ok(());
            }

            for tag in 0..tag_count {
                function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                function.instruction(&Instruction::I32Const(
                    counts_base_offset
                        + i32::try_from(tag).map_err(|_| {
                            SimdError::new("enum scratch tag index does not fit in i32")
                        })? * 4,
                ));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::I32Const(0));
                function.instruction(&Instruction::I32Store(memarg(0, 2)));
            }

            function.instruction(&Instruction::I32Const(1));
            function.instruction(&Instruction::LocalSet(enum_state.aux1_local));

            let mut recursive_local_index = 0usize;
            for (field_index, field_ty) in ctor_info.fields.iter().enumerate() {
                if !is_wasm_direct_self_recursive_field(field_ty, ctor_info) {
                    continue;
                }
                function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                function.instruction(&Instruction::I32Const(
                    child_handles_base_offset
                        + i32::try_from(recursive_local_index).map_err(|_| {
                            SimdError::new("enum recursive child index does not fit in i32")
                        })? * 4,
                ));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::I32Load(memarg(0, 2)));
                function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
                recursive_local_index += 1;
                emit_assert_enum_root_zero(function, enum_state.ptr_local);
                emit_enum_load_header_root(function, enum_state.ptr_local, enum_state);
                function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
                function.instruction(&Instruction::LocalGet(enum_state.base_local));
                function.instruction(&Instruction::I32Load(memarg(
                    ENUM_HEADER_NODE_COUNT_OFFSET,
                    2,
                )));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalSet(enum_state.aux1_local));
                for child_ctor_layout in &row_bearing_ctor_layouts {
                    function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                    function.instruction(&Instruction::I32Const(
                        counts_base_offset + i32::from(child_ctor_layout.tag) * 4,
                    ));
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::LocalTee(enum_state.aux2_local));
                    function.instruction(&Instruction::I32Load(memarg(0, 2)));
                    function.instruction(&Instruction::LocalGet(enum_state.base_local));
                    function.instruction(&Instruction::I32Load(memarg(
                        ENUM_HEADER_CTOR_TABLE_PTR_OFFSET,
                        2,
                    )));
                    function.instruction(&Instruction::I32Const(
                        i32::from(child_ctor_layout.tag) * ENUM_CTOR_TABLE_ENTRY_BYTES,
                    ));
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::I32Load(memarg(
                        ENUM_CTOR_TABLE_ROW_COUNT_OFFSET,
                        2,
                    )));
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::LocalSet(enum_state.aux3_local));
                    function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
                    function.instruction(&Instruction::LocalGet(enum_state.aux3_local));
                    function.instruction(&Instruction::I32Store(memarg(0, 2)));
                }
                let _ = field_index;
            }

            function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
            function.instruction(&Instruction::I32Const(
                counts_base_offset + i32::from(ctor_layout.tag) * 4,
            ));
            function.instruction(&Instruction::I32Add);
            if ctor_layout.row_stride != 0 {
                function.instruction(&Instruction::LocalTee(enum_state.aux2_local));
                function.instruction(&Instruction::I32Load(memarg(0, 2)));
                function.instruction(&Instruction::I32Const(1));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalSet(enum_state.aux3_local));
                function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
                function.instruction(&Instruction::LocalGet(enum_state.aux3_local));
                function.instruction(&Instruction::I32Store(memarg(0, 2)));
            } else {
                function.instruction(&Instruction::Drop);
            }

            let structural_bytes = ENUM_REF_CELL_BYTES
                .checked_add(ENUM_HEADER_BYTES)
                .and_then(|v| {
                    v.checked_add(
                        tag_count_i32
                            .checked_mul(ENUM_CTOR_TABLE_ENTRY_BYTES)
                            .unwrap_or(i32::MAX),
                    )
                })
                .ok_or_else(|| SimdError::new("enum structural size overflow"))?;
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Const(ENUM_TAG_BYTES));
            function.instruction(&Instruction::I32Mul);
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Const(ENUM_END_BYTES));
            function.instruction(&Instruction::I32Mul);
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Const(ENUM_SLOT_BYTES));
            function.instruction(&Instruction::I32Mul);
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::I32Const(structural_bytes));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.aux2_local));
            for child_ctor_layout in &row_bearing_ctor_layouts {
                function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
                function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                function.instruction(&Instruction::I32Const(
                    counts_base_offset + i32::from(child_ctor_layout.tag) * 4,
                ));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::I32Load(memarg(0, 2)));
                function.instruction(&Instruction::I32Const(child_ctor_layout.row_stride as i32));
                function.instruction(&Instruction::I32Mul);
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalSet(enum_state.aux2_local));
            }
            emit_enum_alloc_dynamic(function, enum_state, enum_state.aux2_local);

            function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
            function.instruction(&Instruction::I32Const(ENUM_REF_CELL_BYTES));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.aux4_local));

            function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
            function.instruction(&Instruction::I32Store(memarg(ENUM_REF_HEADER_OFFSET, 2)));
            function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
            function.instruction(&Instruction::I32Const(0));
            function.instruction(&Instruction::I32Store(memarg(ENUM_REF_ROOT_OFFSET, 2)));

            function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
            function.instruction(&Instruction::I32Const(wasm_layout.enum_id as i32));
            function.instruction(&Instruction::I32Store(memarg(
                ENUM_HEADER_ENUM_ID_OFFSET,
                2,
            )));
            function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Store(memarg(
                ENUM_HEADER_NODE_COUNT_OFFSET,
                2,
            )));

            function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
            function.instruction(&Instruction::I32Const(ENUM_HEADER_BYTES));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.aux5_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux5_local));
            function.instruction(&Instruction::I32Store(memarg(
                ENUM_HEADER_TAGS_PTR_OFFSET,
                2,
            )));

            function.instruction(&Instruction::LocalGet(enum_state.aux5_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Const(ENUM_TAG_BYTES));
            function.instruction(&Instruction::I32Mul);
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.aux6_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux6_local));
            function.instruction(&Instruction::I32Store(memarg(
                ENUM_HEADER_ENDS_PTR_OFFSET,
                2,
            )));

            function.instruction(&Instruction::LocalGet(enum_state.aux6_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Const(ENUM_END_BYTES));
            function.instruction(&Instruction::I32Mul);
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.aux7_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux7_local));
            function.instruction(&Instruction::I32Store(memarg(
                ENUM_HEADER_SLOTS_PTR_OFFSET,
                2,
            )));

            function.instruction(&Instruction::LocalGet(enum_state.aux7_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Const(ENUM_SLOT_BYTES));
            function.instruction(&Instruction::I32Mul);
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.aux2_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
            function.instruction(&Instruction::I32Store(memarg(
                ENUM_HEADER_CTOR_TABLE_PTR_OFFSET,
                2,
            )));

            for tag in 0..tag_count {
                function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
                function.instruction(&Instruction::I32Const(
                    i32::try_from(tag)
                        .map_err(|_| SimdError::new("enum tag index does not fit in i32"))?
                        * ENUM_CTOR_TABLE_ENTRY_BYTES,
                ));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalTee(enum_state.aux3_local));
                function.instruction(&Instruction::I32Const(0));
                function.instruction(&Instruction::I32Store(memarg(
                    ENUM_CTOR_TABLE_ROWS_PTR_OFFSET,
                    2,
                )));
                function.instruction(&Instruction::LocalGet(enum_state.aux3_local));
                function.instruction(&Instruction::I32Const(0));
                function.instruction(&Instruction::I32Store(memarg(
                    ENUM_CTOR_TABLE_ROW_COUNT_OFFSET,
                    2,
                )));
            }

            function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
            function.instruction(&Instruction::I32Const(
                tag_count_i32 * ENUM_CTOR_TABLE_ENTRY_BYTES,
            ));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.aux3_local));
            for child_ctor_layout in &row_bearing_ctor_layouts {
                function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                function.instruction(&Instruction::I32Const(
                    counts_base_offset + i32::from(child_ctor_layout.tag) * 4,
                ));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::I32Load(memarg(0, 2)));
                function.instruction(&Instruction::LocalSet(enum_state.base_local));
                function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
                function.instruction(&Instruction::I32Const(
                    i32::from(child_ctor_layout.tag) * ENUM_CTOR_TABLE_ENTRY_BYTES,
                ));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalTee(enum_state.base_tmp_local));
                if child_ctor_layout.row_stride == 0 {
                    function.instruction(&Instruction::I32Const(0));
                } else {
                    function.instruction(&Instruction::LocalGet(enum_state.aux3_local));
                }
                function.instruction(&Instruction::I32Store(memarg(
                    ENUM_CTOR_TABLE_ROWS_PTR_OFFSET,
                    2,
                )));
                function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
                function.instruction(&Instruction::LocalGet(enum_state.base_local));
                function.instruction(&Instruction::I32Store(memarg(
                    ENUM_CTOR_TABLE_ROW_COUNT_OFFSET,
                    2,
                )));
                if child_ctor_layout.row_stride != 0 {
                    function.instruction(&Instruction::LocalGet(enum_state.aux3_local));
                    function.instruction(&Instruction::LocalGet(enum_state.base_local));
                    function
                        .instruction(&Instruction::I32Const(child_ctor_layout.row_stride as i32));
                    function.instruction(&Instruction::I32Mul);
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::LocalSet(enum_state.aux3_local));
                }
            }

            for tag in 0..tag_count {
                for scratch_offset in [counts_base_offset, child_offsets_base_offset] {
                    function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                    function.instruction(&Instruction::I32Const(
                        scratch_offset
                            + i32::try_from(tag).map_err(|_| {
                                SimdError::new("enum scratch tag index does not fit in i32")
                            })? * 4,
                    ));
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::I32Const(0));
                    function.instruction(&Instruction::I32Store(memarg(0, 2)));
                }
            }

            function.instruction(&Instruction::LocalGet(enum_state.aux5_local));
            function.instruction(&Instruction::I32Const(i32::from(ctor_layout.tag)));
            function.instruction(&Instruction::I32Store16(memarg(0, 1)));
            function.instruction(&Instruction::LocalGet(enum_state.aux6_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Store(memarg(0, 2)));
            function.instruction(&Instruction::LocalGet(enum_state.aux7_local));
            function.instruction(&Instruction::I32Const(0));
            function.instruction(&Instruction::I32Store(memarg(0, 2)));
            if ctor_layout.row_stride != 0 {
                function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                function.instruction(&Instruction::I32Const(
                    counts_base_offset + i32::from(ctor_layout.tag) * 4,
                ));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::I32Const(1));
                function.instruction(&Instruction::I32Store(memarg(0, 2)));
            }

            if ctor_layout.row_stride != 0 {
                emit_enum_row_base_ptr_from_header(
                    function,
                    enum_state.aux4_local,
                    &wasm_layout,
                    &ctor_layout,
                    enum_state,
                );
                function.instruction(&Instruction::LocalSet(enum_state.base_local));
                for field_layout in &ctor_layout.non_recursive_fields {
                    let arg_offset = arg_offsets[field_layout.source_index]
                        .ok_or_else(|| SimdError::new("missing enum row field scratch offset"))?;
                    function.instruction(&Instruction::LocalGet(enum_state.base_local));
                    if field_layout.offset != 0 {
                        function.instruction(&Instruction::I32Const(field_layout.offset as i32));
                        function.instruction(&Instruction::I32Add);
                    }
                    function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                    if arg_offset != 0 {
                        function.instruction(&Instruction::I32Const(arg_offset));
                        function.instruction(&Instruction::I32Add);
                    }
                    emit_load_type_from_address(function, &field_layout.ty, enum_state)?;
                    emit_tape_enum_field_store(function, field_layout, enum_state)?;
                }
            }

            function.instruction(&Instruction::I32Const(1));
            function.instruction(&Instruction::LocalSet(enum_state.aux1_local));
            let mut recursive_local_index = 0usize;
            for field_ty in &ctor_info.fields {
                if !is_wasm_direct_self_recursive_field(field_ty, ctor_info) {
                    continue;
                }
                let child_position = recursive_local_index;
                function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                function.instruction(&Instruction::I32Const(
                    child_handles_base_offset
                        + i32::try_from(recursive_local_index).map_err(|_| {
                            SimdError::new("enum recursive child index does not fit in i32")
                        })? * 4,
                ));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::I32Load(memarg(0, 2)));
                function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
                recursive_local_index += 1;
                emit_assert_enum_root_zero(function, enum_state.ptr_local);
                emit_enum_load_header_root(function, enum_state.ptr_local, enum_state);

                for child_ctor_layout in &row_bearing_ctor_layouts {
                    function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                    function.instruction(&Instruction::I32Const(
                        counts_base_offset + i32::from(child_ctor_layout.tag) * 4,
                    ));
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::I32Load(memarg(0, 2)));
                    function.instruction(&Instruction::LocalSet(enum_state.save_sp_local));
                    function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                    function.instruction(&Instruction::I32Const(
                        child_offsets_base_offset + i32::from(child_ctor_layout.tag) * 4,
                    ));
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::LocalGet(enum_state.save_sp_local));
                    function.instruction(&Instruction::I32Store(memarg(0, 2)));

                    function.instruction(&Instruction::LocalGet(enum_state.base_local));
                    function.instruction(&Instruction::I32Load(memarg(
                        ENUM_HEADER_CTOR_TABLE_PTR_OFFSET,
                        2,
                    )));
                    function.instruction(&Instruction::I32Const(
                        i32::from(child_ctor_layout.tag) * ENUM_CTOR_TABLE_ENTRY_BYTES,
                    ));
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::LocalTee(enum_state.aux3_local));
                    function.instruction(&Instruction::I32Load(memarg(
                        ENUM_CTOR_TABLE_ROW_COUNT_OFFSET,
                        2,
                    )));
                    function.instruction(&Instruction::LocalSet(enum_state.aux8_local));

                    function.instruction(&Instruction::LocalGet(enum_state.aux8_local));
                    function.instruction(&Instruction::I32Eqz);
                    function.instruction(&Instruction::If(BlockType::Empty));
                    function.instruction(&Instruction::Else);
                    if child_ctor_layout.row_stride != 0 {
                        emit_enum_row_base_ptr_from_header(
                            function,
                            enum_state.base_local,
                            &wasm_layout,
                            child_ctor_layout,
                            enum_state,
                        );
                        function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
                        emit_enum_row_base_ptr_from_header(
                            function,
                            enum_state.aux4_local,
                            &wasm_layout,
                            child_ctor_layout,
                            enum_state,
                        );
                        function.instruction(&Instruction::LocalGet(enum_state.save_sp_local));
                        function.instruction(&Instruction::I32Const(
                            child_ctor_layout.row_stride as i32,
                        ));
                        function.instruction(&Instruction::I32Mul);
                        function.instruction(&Instruction::I32Add);
                        function.instruction(&Instruction::LocalSet(enum_state.aux3_local));
                        function.instruction(&Instruction::LocalGet(enum_state.aux8_local));
                        function.instruction(&Instruction::I32Const(
                            i32::try_from(child_ctor_layout.row_stride / 4).map_err(|_| {
                                SimdError::new("enum row stride words do not fit in i32")
                            })?,
                        ));
                        function.instruction(&Instruction::I32Mul);
                        function.instruction(&Instruction::LocalSet(enum_state.scratch_sp_local));
                        emit_copy_i32_words(
                            function,
                            enum_state.ptr_local,
                            enum_state.aux3_local,
                            enum_state.scratch_sp_local,
                        );
                    }
                    function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                    function.instruction(&Instruction::I32Const(
                        counts_base_offset + i32::from(child_ctor_layout.tag) * 4,
                    ));
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::LocalGet(enum_state.save_sp_local));
                    function.instruction(&Instruction::LocalGet(enum_state.aux8_local));
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::I32Store(memarg(0, 2)));
                    function.instruction(&Instruction::End);
                }

                function.instruction(&Instruction::LocalGet(enum_state.base_local));
                function.instruction(&Instruction::I32Load(memarg(
                    ENUM_HEADER_TAGS_PTR_OFFSET,
                    2,
                )));
                function.instruction(&Instruction::LocalSet(enum_state.save_sp_local));
                function.instruction(&Instruction::LocalGet(enum_state.aux5_local));
                function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
                function.instruction(&Instruction::I32Const(ENUM_TAG_BYTES));
                function.instruction(&Instruction::I32Mul);
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalSet(enum_state.aux3_local));
                function.instruction(&Instruction::LocalGet(enum_state.base_local));
                function.instruction(&Instruction::I32Load(memarg(
                    ENUM_HEADER_NODE_COUNT_OFFSET,
                    2,
                )));
                function.instruction(&Instruction::LocalSet(enum_state.aux8_local));
                emit_copy_i16_units(
                    function,
                    enum_state.save_sp_local,
                    enum_state.aux3_local,
                    enum_state.aux8_local,
                    enum_state.aux2_local,
                );

                let skip_slot_rebase = row_bearing_ctor_layouts.is_empty()
                    || (ctor_layout.row_stride == 0 && child_position == 0);
                if skip_slot_rebase {
                    function.instruction(&Instruction::LocalGet(enum_state.base_local));
                    function.instruction(&Instruction::I32Load(memarg(
                        ENUM_HEADER_SLOTS_PTR_OFFSET,
                        2,
                    )));
                    function.instruction(&Instruction::LocalSet(enum_state.save_sp_local));
                    function.instruction(&Instruction::LocalGet(enum_state.aux7_local));
                    function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
                    function.instruction(&Instruction::I32Const(ENUM_SLOT_BYTES));
                    function.instruction(&Instruction::I32Mul);
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::LocalSet(enum_state.aux3_local));
                    function.instruction(&Instruction::LocalGet(enum_state.base_local));
                    function.instruction(&Instruction::I32Load(memarg(
                        ENUM_HEADER_NODE_COUNT_OFFSET,
                        2,
                    )));
                    function.instruction(&Instruction::LocalSet(enum_state.aux8_local));
                    emit_copy_i32_words(
                        function,
                        enum_state.save_sp_local,
                        enum_state.aux3_local,
                        enum_state.aux8_local,
                    );
                }

                function.instruction(&Instruction::I32Const(0));
                function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
                function.instruction(&Instruction::Block(BlockType::Empty));
                function.instruction(&Instruction::Loop(BlockType::Empty));
                function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
                function.instruction(&Instruction::LocalGet(enum_state.base_local));
                function.instruction(&Instruction::I32Load(memarg(
                    ENUM_HEADER_NODE_COUNT_OFFSET,
                    2,
                )));
                function.instruction(&Instruction::I32GeU);
                function.instruction(&Instruction::BrIf(1));

                function.instruction(&Instruction::LocalGet(enum_state.aux6_local));
                function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
                function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::I32Const(ENUM_END_BYTES));
                function.instruction(&Instruction::I32Mul);
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalGet(enum_state.base_local));
                function.instruction(&Instruction::I32Load(memarg(
                    ENUM_HEADER_ENDS_PTR_OFFSET,
                    2,
                )));
                function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
                function.instruction(&Instruction::I32Const(ENUM_END_BYTES));
                function.instruction(&Instruction::I32Mul);
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::I32Load(memarg(0, 2)));
                function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::I32Store(memarg(0, 2)));

                if !skip_slot_rebase {
                    function.instruction(&Instruction::LocalGet(enum_state.aux7_local));
                    function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
                    function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::I32Const(ENUM_SLOT_BYTES));
                    function.instruction(&Instruction::I32Mul);
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::LocalGet(enum_state.base_local));
                    function.instruction(&Instruction::I32Load(memarg(
                        ENUM_HEADER_SLOTS_PTR_OFFSET,
                        2,
                    )));
                    function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
                    function.instruction(&Instruction::I32Const(ENUM_SLOT_BYTES));
                    function.instruction(&Instruction::I32Mul);
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::I32Load(memarg(0, 2)));
                    function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                    function.instruction(&Instruction::I32Const(child_offsets_base_offset));
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::LocalGet(enum_state.base_local));
                    function.instruction(&Instruction::I32Load(memarg(
                        ENUM_HEADER_TAGS_PTR_OFFSET,
                        2,
                    )));
                    function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
                    function.instruction(&Instruction::I32Const(ENUM_TAG_BYTES));
                    function.instruction(&Instruction::I32Mul);
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::I32Load16U(memarg(0, 1)));
                    function.instruction(&Instruction::I32Const(4));
                    function.instruction(&Instruction::I32Mul);
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::I32Load(memarg(0, 2)));
                    function.instruction(&Instruction::I32Add);
                    function.instruction(&Instruction::I32Store(memarg(0, 2)));
                }

                function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
                function.instruction(&Instruction::I32Const(1));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
                function.instruction(&Instruction::Br(0));
                function.instruction(&Instruction::End);
                function.instruction(&Instruction::End);

                function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
                function.instruction(&Instruction::LocalGet(enum_state.base_local));
                function.instruction(&Instruction::I32Load(memarg(
                    ENUM_HEADER_NODE_COUNT_OFFSET,
                    2,
                )));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalSet(enum_state.aux1_local));
            }

            function.instruction(&Instruction::LocalGet(enum_state.aux6_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Store(memarg(0, 2)));
            for child_ctor_layout in &row_bearing_ctor_layouts {
                function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
                function.instruction(&Instruction::I32Const(
                    i32::from(child_ctor_layout.tag) * ENUM_CTOR_TABLE_ENTRY_BYTES,
                ));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
                function.instruction(&Instruction::I32Const(
                    counts_base_offset + i32::from(child_ctor_layout.tag) * 4,
                ));
                function.instruction(&Instruction::I32Add);
                function.instruction(&Instruction::I32Load(memarg(0, 2)));
                function.instruction(&Instruction::I32Store(memarg(
                    ENUM_CTOR_TABLE_ROW_COUNT_OFFSET,
                    2,
                )));
            }

            function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
            function.instruction(&Instruction::I32Const(ENUM_REF_CELL_BYTES));
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
            function.instruction(&Instruction::LocalSet(enum_state.scratch_sp_local));
            function.instruction(&Instruction::I32Const(ENUM_SAVE_STACK_PTR_ADDR as i32));
            function.instruction(&Instruction::LocalGet(enum_state.scratch_sp_local));
            function.instruction(&Instruction::I32Store(memarg(0, 2)));
        }
        IrExprKind::EnumTag { value } => {
            let enum_state = enum_state.ok_or_else(|| {
                SimdError::new("internal error: missing enum scratch local for enum tag load")
            })?;
            compile_scalar_ir_expr_with_hoists(
                function,
                value,
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                enum_state.into(),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
            emit_enum_tag_load_from_value(function, enum_state.ptr_local, enum_state);
            if matches!(expr.ty, Type::Scalar(Prim::I64)) {
                function.instruction(&Instruction::I64ExtendI32U);
            }
        }
        IrExprKind::EnumChildBySlot { value, ctor, slot } => {
            let enum_state = enum_state.ok_or_else(|| {
                SimdError::new(
                    "internal error: missing enum scratch local for enum child projection",
                )
            })?;
            let ctor_info = enum_ctors
                .get(ctor)
                .ok_or_else(|| SimdError::new(format!("unknown constructor '{}'", ctor)))?;
            let Type::Named(_, enum_args) = &value.ty else {
                return Err(SimdError::new(format!(
                    "enum child projection '{}' requires enum value type, found {:?}",
                    ctor, value.ty
                )));
            };
            let wasm_layout = lookup_specialized_wasm_enum_layout(
                wasm_enum_layouts,
                &ctor_info.enum_name,
                enum_args,
            )?;
            let ctor_layout = lookup_specialized_wasm_enum_ctor_layout(
                wasm_enum_layouts,
                &ctor_info.enum_name,
                enum_args,
                ctor,
            )?;
            let recursive_fields = enum_ctor_recursive_field_indices(ctor_info);
            let source_field = *recursive_fields.get(*slot).ok_or_else(|| {
                SimdError::new(format!(
                    "enum child slot {} is out of bounds for constructor '{}'",
                    slot, ctor
                ))
            })?;
            let field_ty = ctor_info.fields.get(source_field).ok_or_else(|| {
                SimdError::new(format!(
                    "constructor '{}' is missing recursive field {}",
                    ctor, source_field
                ))
            })?;
            compile_scalar_ir_expr_with_hoists(
                function,
                value,
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                enum_state.into(),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
            emit_enum_field_load(
                function,
                enum_state.ptr_local,
                &wasm_layout,
                &ctor_layout,
                source_field,
                field_ty,
                enum_state,
            )?;
        }
        IrExprKind::EnumNonRecField { value, ctor, field } => {
            let enum_state = enum_state.ok_or_else(|| {
                SimdError::new(
                    "internal error: missing enum scratch local for enum non-recursive projection",
                )
            })?;
            let ctor_info = enum_ctors
                .get(ctor)
                .ok_or_else(|| SimdError::new(format!("unknown constructor '{}'", ctor)))?;
            let Type::Named(_, enum_args) = &value.ty else {
                return Err(SimdError::new(format!(
                    "enum field projection '{}' requires enum value type, found {:?}",
                    ctor, value.ty
                )));
            };
            let wasm_layout = lookup_specialized_wasm_enum_layout(
                wasm_enum_layouts,
                &ctor_info.enum_name,
                enum_args,
            )?;
            let ctor_layout = lookup_specialized_wasm_enum_ctor_layout(
                wasm_enum_layouts,
                &ctor_info.enum_name,
                enum_args,
                ctor,
            )?;
            let non_recursive_fields = enum_ctor_non_recursive_field_indices(ctor_info);
            let source_field = *non_recursive_fields.get(*field).ok_or_else(|| {
                SimdError::new(format!(
                    "enum non-rec field {} is out of bounds for constructor '{}'",
                    field, ctor
                ))
            })?;
            compile_scalar_ir_expr_with_hoists(
                function,
                value,
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                enum_state.into(),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::LocalSet(enum_state.ptr_local));
            emit_enum_field_load(
                function,
                enum_state.ptr_local,
                &wasm_layout,
                &ctor_layout,
                source_field,
                &expr.ty,
                enum_state,
            )?;
        }
        IrExprKind::Let { bindings, body } => {
            let mut extended = inline_bindings.clone();
            for binding in bindings {
                extended.insert(binding.name.clone(), binding.expr.clone());
            }
            compile_scalar_ir_expr_with_hoists(
                function,
                body,
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                enum_state,
                hoisted_locals,
                &extended,
            )?;
        }
        IrExprKind::Call { callee, args } => match callee {
            Callee::Prim(op) => {
                compile_scalar_ir_expr_with_hoists(
                    function,
                    &args[0],
                    locals,
                    scalar_indices,
                    signatures,
                    enum_ctors,
                    wasm_enum_layouts,
                    enum_state,
                    hoisted_locals,
                    inline_bindings,
                )?;
                compile_scalar_ir_expr_with_hoists(
                    function,
                    &args[1],
                    locals,
                    scalar_indices,
                    signatures,
                    enum_ctors,
                    wasm_enum_layouts,
                    enum_state,
                    hoisted_locals,
                    inline_bindings,
                )?;
                emit_scalar_primitive(function, *op, expr.ty.prim().unwrap_or(Prim::I64), args)?;
            }
            Callee::Function(name) => {
                let checked = signatures.get(name).copied().ok_or_else(|| {
                    SimdError::new(format!("missing checked signature for '{}'", name))
                })?;
                let (params, _result) = checked.signature.ty.fun_parts();
                for (arg, ty) in args.iter().zip(params) {
                    compile_scalar_ir_expr_with_hoists(
                        function,
                        arg,
                        locals,
                        scalar_indices,
                        signatures,
                        enum_ctors,
                        wasm_enum_layouts,
                        enum_state,
                        hoisted_locals,
                        inline_bindings,
                    )?;
                    if !wasm_scalar_call_types_match(&arg.ty, &ty) {
                        return Err(SimdError::new(format!(
                            "Wasm scalar call '{}' received {:?}, expected {:?}",
                            name, arg.ty, ty
                        )));
                    }
                }
                let index = scalar_indices.get(name).copied().ok_or_else(|| {
                    SimdError::new(format!("missing Wasm function index for '{}'", name))
                })?;
                function.instruction(&Instruction::Call(index));
                if wasm_star_seq_storage_prim(&expr.ty).is_some() {
                    let enum_state = enum_state.ok_or_else(|| {
                        SimdError::new(
                            "internal error: missing enum scratch local for T[*] call result",
                        )
                    })?;
                    function.instruction(&Instruction::LocalSet(enum_state.base_local));
                    emit_unbox_seq_handle(function, enum_state.base_local, enum_state);
                }
            }
            Callee::Builtin(builtin) => {
                emit_builtin_scalar_call(
                    function,
                    builtin,
                    args,
                    expr,
                    locals,
                    scalar_indices,
                    signatures,
                    enum_ctors,
                    wasm_enum_layouts,
                    enum_state,
                    hoisted_locals,
                    inline_bindings,
                )?;
            }
        },
    }
    Ok(())
}

fn emit_builtin_scalar_call(
    function: &mut Function,
    builtin: &BuiltinFamilyCallee,
    args: &[IrExpr],
    expr: &IrExpr,
    locals: &BTreeMap<String, u32>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    enum_state: Option<EnumWasmState>,
    hoisted_locals: &BTreeMap<HoistExprKey, u32>,
    inline_bindings: &BTreeMap<String, IrExpr>,
) -> Result<()> {
    match builtin {
        BuiltinFamilyCallee::ReverseSeq(_) => {
            if args.len() != 1 {
                return Err(SimdError::new(format!(
                    "Wasm builtin reverse expected 1 arg, found {}",
                    args.len()
                )));
            }
            let enum_state = enum_state.ok_or_else(|| {
                SimdError::new("internal error: missing scratch locals for builtin reverse")
            })?;
            let prim = wasm_star_seq_storage_prim(&expr.ty).ok_or_else(|| {
                SimdError::new(format!(
                    "Wasm builtin reverse expects a Wasm-storable T[*] result, found {:?}",
                    expr.ty
                ))
            })?;
            compile_scalar_ir_expr_with_hoists(
                function,
                &args[0],
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::LocalSet(enum_state.aux1_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux0_local));

            let elem_bytes = i32::try_from(byte_width(prim))
                .map_err(|_| SimdError::new("builtin reverse element width overflow"))?;
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Const(elem_bytes));
            function.instruction(&Instruction::I32Mul);
            function.instruction(&Instruction::LocalSet(enum_state.aux2_local));
            emit_enum_alloc_dynamic(function, enum_state, enum_state.aux2_local);
            function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux3_local));

            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::If(BlockType::Empty));
            function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Const(elem_bytes));
            function.instruction(&Instruction::I32Mul);
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::I32Const(elem_bytes));
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::LocalSet(enum_state.aux4_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux3_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux5_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux6_local));

            function.instruction(&Instruction::Block(BlockType::Empty));
            function.instruction(&Instruction::Loop(BlockType::Empty));
            function.instruction(&Instruction::LocalGet(enum_state.aux5_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
            match prim {
                Prim::I32 | Prim::Char => {
                    function.instruction(&Instruction::I32Load(memarg(0, 2)));
                    function.instruction(&Instruction::I32Store(memarg(0, 2)));
                }
                Prim::I64 => {
                    function.instruction(&Instruction::I64Load(memarg(0, 3)));
                    function.instruction(&Instruction::I64Store(memarg(0, 3)));
                }
                Prim::F32 => {
                    function.instruction(&Instruction::F32Load(memarg(0, 2)));
                    function.instruction(&Instruction::F32Store(memarg(0, 2)));
                }
                Prim::F64 => {
                    function.instruction(&Instruction::F64Load(memarg(0, 3)));
                    function.instruction(&Instruction::F64Store(memarg(0, 3)));
                }
            }
            function.instruction(&Instruction::LocalGet(enum_state.aux5_local));
            function.instruction(&Instruction::I32Const(elem_bytes));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.aux5_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
            function.instruction(&Instruction::I32Const(elem_bytes));
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::LocalSet(enum_state.aux4_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux6_local));
            function.instruction(&Instruction::I32Const(1));
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::LocalTee(enum_state.aux6_local));
            function.instruction(&Instruction::BrIf(0));
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);

            function.instruction(&Instruction::LocalGet(enum_state.aux3_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
        }
        BuiltinFamilyCallee::GatherSeq(prim) => {
            if args.len() != 2 {
                return Err(SimdError::new(format!(
                    "Wasm builtin gather expected 2 args, found {}",
                    args.len()
                )));
            }
            let enum_state = enum_state.ok_or_else(|| {
                SimdError::new("internal error: missing scratch locals for builtin gather")
            })?;
            compile_scalar_ir_expr_with_hoists(
                function,
                &args[0],
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::LocalSet(enum_state.aux1_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux0_local));
            compile_scalar_ir_expr_with_hoists(
                function,
                &args[1],
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            if args[1].ty != Type::Scalar(Prim::I64) {
                return Err(SimdError::new(format!(
                    "Wasm builtin gather expected i64 index, found {:?}",
                    args[1].ty
                )));
            }
            function.instruction(&Instruction::LocalSet(enum_state.index_i64_local));

            function.instruction(&Instruction::LocalGet(enum_state.index_i64_local));
            function.instruction(&Instruction::I64Const(0));
            function.instruction(&Instruction::I64LtS);
            function.instruction(&Instruction::If(BlockType::Empty));
            function.instruction(&Instruction::Unreachable);
            function.instruction(&Instruction::End);

            function.instruction(&Instruction::LocalGet(enum_state.index_i64_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I64ExtendI32U);
            function.instruction(&Instruction::I64GeU);
            function.instruction(&Instruction::If(BlockType::Empty));
            function.instruction(&Instruction::Unreachable);
            function.instruction(&Instruction::End);

            function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
            function.instruction(&Instruction::LocalGet(enum_state.index_i64_local));
            function.instruction(&Instruction::I32WrapI64);
            let elem_bytes = i32::try_from(byte_width(*prim))
                .map_err(|_| SimdError::new("builtin gather element width overflow"))?;
            if elem_bytes != 1 {
                function.instruction(&Instruction::I32Const(elem_bytes));
                function.instruction(&Instruction::I32Mul);
            }
            function.instruction(&Instruction::I32Add);
            match prim {
                Prim::I32 | Prim::Char => {
                    function.instruction(&Instruction::I32Load(memarg(0, 2)));
                }
                Prim::I64 => {
                    function.instruction(&Instruction::I64Load(memarg(0, 3)));
                }
                Prim::F32 => {
                    function.instruction(&Instruction::F32Load(memarg(0, 2)));
                }
                Prim::F64 => {
                    function.instruction(&Instruction::F64Load(memarg(0, 3)));
                }
            }
            if expr.ty != Type::Scalar(*prim) {
                return Err(SimdError::new(format!(
                    "Wasm builtin gather result type {:?} did not match {:?}",
                    expr.ty, prim
                )));
            }
        }
        BuiltinFamilyCallee::ScatterSeq(prim) | BuiltinFamilyCallee::ScatterAddSeq(prim) => {
            if args.len() != 3 {
                return Err(SimdError::new(format!(
                    "Wasm builtin {} expected 3 args, found {}",
                    if matches!(builtin, BuiltinFamilyCallee::ScatterAddSeq(_)) {
                        "scatter_add"
                    } else {
                        "scatter"
                    },
                    args.len()
                )));
            }
            let enum_state = enum_state.ok_or_else(|| {
                SimdError::new("internal error: missing scratch locals for builtin scatter")
            })?;
            compile_scalar_ir_expr_with_hoists(
                function,
                &args[0],
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::LocalSet(enum_state.aux1_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux0_local));
            compile_scalar_ir_expr_with_hoists(
                function,
                &args[1],
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::LocalSet(enum_state.aux3_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux2_local));
            compile_scalar_ir_expr_with_hoists(
                function,
                &args[2],
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::LocalSet(enum_state.aux5_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux4_local));

            if !matches!(&args[1].ty, Type::StarSeq(item) | Type::StarSeqWitnessed(item, _) if matches!(item.as_ref(), Type::Scalar(Prim::I64)))
            {
                return Err(SimdError::new(format!(
                    "Wasm builtin scatter expected i64[*] indices, found {:?}",
                    args[1].ty
                )));
            }
            if !matches!(&args[2].ty, Type::StarSeq(item) | Type::StarSeqWitnessed(item, _) if matches!(item.as_ref(), Type::Scalar(arg_prim) if arg_prim == prim))
            {
                return Err(SimdError::new(format!(
                    "Wasm builtin scatter expected {:?}[*] values, found {:?}",
                    prim, args[2].ty
                )));
            }
            if wasm_star_seq_storage_prim(&expr.ty) != Some(*prim) {
                return Err(SimdError::new(format!(
                    "Wasm builtin scatter result type {:?} did not match {:?}",
                    expr.ty, prim
                )));
            }

            function.instruction(&Instruction::LocalGet(enum_state.aux3_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux5_local));
            function.instruction(&Instruction::I32Ne);
            function.instruction(&Instruction::If(BlockType::Empty));
            function.instruction(&Instruction::Unreachable);
            function.instruction(&Instruction::End);

            let elem_bytes = i32::try_from(byte_width(*prim))
                .map_err(|_| SimdError::new("builtin scatter element width overflow"))?;
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Const(elem_bytes));
            function.instruction(&Instruction::I32Mul);
            function.instruction(&Instruction::LocalSet(enum_state.aux6_local));
            emit_enum_alloc_dynamic(function, enum_state, enum_state.aux6_local);
            function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux7_local));

            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::If(BlockType::Empty));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux6_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux7_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux8_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux0_local));
            function.instruction(&Instruction::LocalSet(enum_state.base_tmp_local));
            function.instruction(&Instruction::Block(BlockType::Empty));
            function.instruction(&Instruction::Loop(BlockType::Empty));
            function.instruction(&Instruction::LocalGet(enum_state.aux8_local));
            function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
            match prim {
                Prim::I32 | Prim::Char => {
                    function.instruction(&Instruction::I32Load(memarg(0, 2)));
                    function.instruction(&Instruction::I32Store(memarg(0, 2)));
                }
                Prim::I64 => {
                    function.instruction(&Instruction::I64Load(memarg(0, 3)));
                    function.instruction(&Instruction::I64Store(memarg(0, 3)));
                }
                Prim::F32 => {
                    function.instruction(&Instruction::F32Load(memarg(0, 2)));
                    function.instruction(&Instruction::F32Store(memarg(0, 2)));
                }
                Prim::F64 => {
                    function.instruction(&Instruction::F64Load(memarg(0, 3)));
                    function.instruction(&Instruction::F64Store(memarg(0, 3)));
                }
            }
            function.instruction(&Instruction::LocalGet(enum_state.aux8_local));
            function.instruction(&Instruction::I32Const(elem_bytes));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.aux8_local));
            function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
            function.instruction(&Instruction::I32Const(elem_bytes));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.base_tmp_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux6_local));
            function.instruction(&Instruction::I32Const(1));
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::LocalTee(enum_state.aux6_local));
            function.instruction(&Instruction::BrIf(0));
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);

            function.instruction(&Instruction::LocalGet(enum_state.aux3_local));
            function.instruction(&Instruction::If(BlockType::Empty));
            function.instruction(&Instruction::LocalGet(enum_state.aux3_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux6_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux2_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux8_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
            function.instruction(&Instruction::LocalSet(enum_state.base_tmp_local));
            function.instruction(&Instruction::Block(BlockType::Empty));
            function.instruction(&Instruction::Loop(BlockType::Empty));

            function.instruction(&Instruction::LocalGet(enum_state.aux8_local));
            function.instruction(&Instruction::I64Load(memarg(0, 3)));
            function.instruction(&Instruction::LocalSet(enum_state.index_i64_local));

            function.instruction(&Instruction::LocalGet(enum_state.index_i64_local));
            function.instruction(&Instruction::I64Const(0));
            function.instruction(&Instruction::I64LtS);
            function.instruction(&Instruction::If(BlockType::Empty));
            function.instruction(&Instruction::Unreachable);
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::LocalGet(enum_state.index_i64_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I64ExtendI32U);
            function.instruction(&Instruction::I64GeU);
            function.instruction(&Instruction::If(BlockType::Empty));
            function.instruction(&Instruction::Unreachable);
            function.instruction(&Instruction::End);

            function.instruction(&Instruction::LocalGet(enum_state.aux7_local));
            function.instruction(&Instruction::LocalGet(enum_state.index_i64_local));
            function.instruction(&Instruction::I32WrapI64);
            if elem_bytes != 1 {
                function.instruction(&Instruction::I32Const(elem_bytes));
                function.instruction(&Instruction::I32Mul);
            }
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.base_local));

            function.instruction(&Instruction::LocalGet(enum_state.base_local));
            if matches!(builtin, BuiltinFamilyCallee::ScatterAddSeq(_)) {
                function.instruction(&Instruction::LocalGet(enum_state.base_local));
                match prim {
                    Prim::I32 | Prim::Char => {
                        function.instruction(&Instruction::I32Load(memarg(0, 2)));
                    }
                    Prim::I64 => {
                        function.instruction(&Instruction::I64Load(memarg(0, 3)));
                    }
                    Prim::F32 => {
                        function.instruction(&Instruction::F32Load(memarg(0, 2)));
                    }
                    Prim::F64 => {
                        function.instruction(&Instruction::F64Load(memarg(0, 3)));
                    }
                };
                function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
                match prim {
                    Prim::I32 | Prim::Char => {
                        function.instruction(&Instruction::I32Load(memarg(0, 2)));
                        function.instruction(&Instruction::I32Add);
                        function.instruction(&Instruction::I32Store(memarg(0, 2)));
                    }
                    Prim::I64 => {
                        function.instruction(&Instruction::I64Load(memarg(0, 3)));
                        function.instruction(&Instruction::I64Add);
                        function.instruction(&Instruction::I64Store(memarg(0, 3)));
                    }
                    Prim::F32 => {
                        function.instruction(&Instruction::F32Load(memarg(0, 2)));
                        function.instruction(&Instruction::F32Add);
                        function.instruction(&Instruction::F32Store(memarg(0, 2)));
                    }
                    Prim::F64 => {
                        function.instruction(&Instruction::F64Load(memarg(0, 3)));
                        function.instruction(&Instruction::F64Add);
                        function.instruction(&Instruction::F64Store(memarg(0, 3)));
                    }
                }
            } else {
                function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
                match prim {
                    Prim::I32 | Prim::Char => {
                        function.instruction(&Instruction::I32Load(memarg(0, 2)));
                        function.instruction(&Instruction::I32Store(memarg(0, 2)));
                    }
                    Prim::I64 => {
                        function.instruction(&Instruction::I64Load(memarg(0, 3)));
                        function.instruction(&Instruction::I64Store(memarg(0, 3)));
                    }
                    Prim::F32 => {
                        function.instruction(&Instruction::F32Load(memarg(0, 2)));
                        function.instruction(&Instruction::F32Store(memarg(0, 2)));
                    }
                    Prim::F64 => {
                        function.instruction(&Instruction::F64Load(memarg(0, 3)));
                        function.instruction(&Instruction::F64Store(memarg(0, 3)));
                    }
                }
            }

            function.instruction(&Instruction::LocalGet(enum_state.aux8_local));
            function.instruction(&Instruction::I32Const(8));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.aux8_local));
            function.instruction(&Instruction::LocalGet(enum_state.base_tmp_local));
            function.instruction(&Instruction::I32Const(elem_bytes));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.base_tmp_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux6_local));
            function.instruction(&Instruction::I32Const(1));
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::LocalTee(enum_state.aux6_local));
            function.instruction(&Instruction::BrIf(0));
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);

            function.instruction(&Instruction::LocalGet(enum_state.aux7_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
        }
        BuiltinFamilyCallee::CheckedGatherSeq(prim) => {
            if args.len() != 2 {
                return Err(SimdError::new(format!(
                    "Wasm builtin try_get expected 2 args, found {}",
                    args.len()
                )));
            }
            let enum_state = enum_state.ok_or_else(|| {
                SimdError::new("internal error: missing scratch locals for builtin try_get")
            })?;
            compile_scalar_ir_expr_with_hoists(
                function,
                &args[0],
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::LocalSet(enum_state.aux1_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux0_local));
            compile_scalar_ir_expr_with_hoists(
                function,
                &args[1],
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            if args[1].ty != Type::Scalar(Prim::I64) {
                return Err(SimdError::new(format!(
                    "Wasm builtin try_get expected i64 index, found {:?}",
                    args[1].ty
                )));
            }
            function.instruction(&Instruction::LocalSet(enum_state.index_i64_local));
            let result_val_type = inlineable_scalar_result_val_type(&expr.ty).ok_or_else(|| {
                SimdError::new(format!(
                    "Wasm builtin try_get expected inlineable result type, found {:?}",
                    expr.ty
                ))
            })?;
            let none_expr = IrExpr {
                ty: expr.ty.clone(),
                kind: IrExprKind::EnumCtor {
                    ctor: "None".to_string(),
                    args: Vec::new(),
                },
            };
            let some_expr = IrExpr {
                ty: expr.ty.clone(),
                kind: IrExprKind::EnumCtor {
                    ctor: "Some".to_string(),
                    args: vec![IrExpr {
                        ty: Type::Scalar(*prim),
                        kind: IrExprKind::Call {
                            callee: Callee::Builtin(BuiltinFamilyCallee::GatherSeq(*prim)),
                            args: vec![args[0].clone(), args[1].clone()],
                        },
                    }],
                },
            };

            function.instruction(&Instruction::LocalGet(enum_state.index_i64_local));
            function.instruction(&Instruction::I64Const(0));
            function.instruction(&Instruction::I64LtS);
            function.instruction(&Instruction::If(BlockType::Result(result_val_type)));
            compile_scalar_ir_expr_with_hoists(
                function,
                &none_expr,
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::Else);
            function.instruction(&Instruction::LocalGet(enum_state.index_i64_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I64ExtendI32U);
            function.instruction(&Instruction::I64GeU);
            function.instruction(&Instruction::If(BlockType::Result(result_val_type)));
            compile_scalar_ir_expr_with_hoists(
                function,
                &none_expr,
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::Else);
            compile_scalar_ir_expr_with_hoists(
                function,
                &some_expr,
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);
        }
        BuiltinFamilyCallee::CheckIndexSeq(witness) => {
            if args.len() != 2 {
                return Err(SimdError::new(format!(
                    "Wasm builtin check_index expected 2 args, found {}",
                    args.len()
                )));
            }
            let enum_state = enum_state.ok_or_else(|| {
                SimdError::new("internal error: missing scratch locals for builtin check_index")
            })?;
            compile_scalar_ir_expr_with_hoists(
                function,
                &args[0],
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::LocalSet(enum_state.aux1_local));
            function.instruction(&Instruction::Drop);
            compile_scalar_ir_expr_with_hoists(
                function,
                &args[1],
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            if args[1].ty != Type::Scalar(Prim::I64) {
                return Err(SimdError::new(format!(
                    "Wasm builtin check_index expected i64 index, found {:?}",
                    args[1].ty
                )));
            }
            function.instruction(&Instruction::LocalSet(enum_state.index_i64_local));
            let result_val_type = inlineable_scalar_result_val_type(&expr.ty).ok_or_else(|| {
                SimdError::new(format!(
                    "Wasm builtin check_index expected inlineable result type, found {:?}",
                    expr.ty
                ))
            })?;
            let none_expr = IrExpr {
                ty: expr.ty.clone(),
                kind: IrExprKind::EnumCtor {
                    ctor: "None".to_string(),
                    args: Vec::new(),
                },
            };
            let some_expr = IrExpr {
                ty: expr.ty.clone(),
                kind: IrExprKind::EnumCtor {
                    ctor: "Some".to_string(),
                    args: vec![IrExpr {
                        ty: Type::Index(witness.clone()),
                        kind: args[1].kind.clone(),
                    }],
                },
            };

            function.instruction(&Instruction::LocalGet(enum_state.index_i64_local));
            function.instruction(&Instruction::I64Const(0));
            function.instruction(&Instruction::I64LtS);
            function.instruction(&Instruction::If(BlockType::Result(result_val_type)));
            compile_scalar_ir_expr_with_hoists(
                function,
                &none_expr,
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::Else);
            function.instruction(&Instruction::LocalGet(enum_state.index_i64_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I64ExtendI32U);
            function.instruction(&Instruction::I64GeU);
            function.instruction(&Instruction::If(BlockType::Result(result_val_type)));
            compile_scalar_ir_expr_with_hoists(
                function,
                &none_expr,
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::Else);
            compile_scalar_ir_expr_with_hoists(
                function,
                &some_expr,
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);
        }
        BuiltinFamilyCallee::IndicesSeq(_) => {
            if args.len() != 1 {
                return Err(SimdError::new(format!(
                    "Wasm builtin indices expected 1 arg, found {}",
                    args.len()
                )));
            }
            let enum_state = enum_state.ok_or_else(|| {
                SimdError::new("internal error: missing scratch locals for builtin indices")
            })?;
            let prim = wasm_star_seq_storage_prim(&expr.ty).ok_or_else(|| {
                SimdError::new(format!(
                    "Wasm builtin indices expects a Wasm-storable T[*] result, found {:?}",
                    expr.ty
                ))
            })?;
            compile_scalar_ir_expr_with_hoists(
                function,
                &args[0],
                locals,
                scalar_indices,
                signatures,
                enum_ctors,
                wasm_enum_layouts,
                Some(enum_state),
                hoisted_locals,
                inline_bindings,
            )?;
            function.instruction(&Instruction::LocalSet(enum_state.aux1_local));
            function.instruction(&Instruction::Drop);

            let elem_bytes = i32::try_from(byte_width(prim))
                .map_err(|_| SimdError::new("builtin indices element width overflow"))?;
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::I32Const(elem_bytes));
            function.instruction(&Instruction::I32Mul);
            function.instruction(&Instruction::LocalSet(enum_state.aux2_local));
            emit_enum_alloc_dynamic(function, enum_state, enum_state.aux2_local);
            function.instruction(&Instruction::LocalGet(enum_state.ptr_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux3_local));

            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::If(BlockType::Empty));
            function.instruction(&Instruction::LocalGet(enum_state.aux3_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux4_local));
            function.instruction(&Instruction::I64Const(0));
            function.instruction(&Instruction::LocalSet(enum_state.index_i64_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
            function.instruction(&Instruction::LocalSet(enum_state.aux6_local));

            function.instruction(&Instruction::Block(BlockType::Empty));
            function.instruction(&Instruction::Loop(BlockType::Empty));
            function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
            function.instruction(&Instruction::LocalGet(enum_state.index_i64_local));
            function.instruction(&Instruction::I64Store(memarg(0, 3)));
            function.instruction(&Instruction::LocalGet(enum_state.aux4_local));
            function.instruction(&Instruction::I32Const(elem_bytes));
            function.instruction(&Instruction::I32Add);
            function.instruction(&Instruction::LocalSet(enum_state.aux4_local));
            function.instruction(&Instruction::LocalGet(enum_state.index_i64_local));
            function.instruction(&Instruction::I64Const(1));
            function.instruction(&Instruction::I64Add);
            function.instruction(&Instruction::LocalSet(enum_state.index_i64_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux6_local));
            function.instruction(&Instruction::I32Const(1));
            function.instruction(&Instruction::I32Sub);
            function.instruction(&Instruction::LocalTee(enum_state.aux6_local));
            function.instruction(&Instruction::BrIf(0));
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);
            function.instruction(&Instruction::End);

            function.instruction(&Instruction::LocalGet(enum_state.aux3_local));
            function.instruction(&Instruction::LocalGet(enum_state.aux1_local));
        }
        _ => {
            return Err(SimdError::new(
                "Wasm scalar codegen does not support this builtin call",
            ));
        }
    }
    Ok(())
}

fn emit_scalar_primitive(
    function: &mut Function,
    op: PrimOp,
    result_prim: Prim,
    args: &[IrExpr],
) -> Result<()> {
    let operand_prim = args[0]
        .ty
        .prim()
        .ok_or_else(|| SimdError::new("primitive operand did not have a scalar primitive type"))?;
    match (op, operand_prim) {
        (PrimOp::Add, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32Add);
        }
        (PrimOp::Sub, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32Sub);
        }
        (PrimOp::Mul, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32Mul);
        }
        (PrimOp::Div, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32DivS);
        }
        (PrimOp::Mod, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32RemS);
        }
        (PrimOp::Add, Prim::I64) => {
            function.instruction(&Instruction::I64Add);
        }
        (PrimOp::Sub, Prim::I64) => {
            function.instruction(&Instruction::I64Sub);
        }
        (PrimOp::Mul, Prim::I64) => {
            function.instruction(&Instruction::I64Mul);
        }
        (PrimOp::Div, Prim::I64) => {
            function.instruction(&Instruction::I64DivS);
        }
        (PrimOp::Mod, Prim::I64) => {
            function.instruction(&Instruction::I64RemS);
        }
        (PrimOp::Add, Prim::F32) => {
            function.instruction(&Instruction::F32Add);
        }
        (PrimOp::Sub, Prim::F32) => {
            function.instruction(&Instruction::F32Sub);
        }
        (PrimOp::Mul, Prim::F32) => {
            function.instruction(&Instruction::F32Mul);
        }
        (PrimOp::Div, Prim::F32) => {
            function.instruction(&Instruction::F32Div);
        }
        (PrimOp::Add, Prim::F64) => {
            function.instruction(&Instruction::F64Add);
        }
        (PrimOp::Sub, Prim::F64) => {
            function.instruction(&Instruction::F64Sub);
        }
        (PrimOp::Mul, Prim::F64) => {
            function.instruction(&Instruction::F64Mul);
        }
        (PrimOp::Div, Prim::F64) => {
            function.instruction(&Instruction::F64Div);
        }
        (PrimOp::Eq, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32Eq);
            if result_prim == Prim::I64 {
                function.instruction(&Instruction::I64ExtendI32U);
            }
        }
        (PrimOp::Lt, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32LtS);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Gt, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32GtS);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Le, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32LeS);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Ge, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32GeS);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Eq, Prim::I64) => {
            function.instruction(&Instruction::I64Eq);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Lt, Prim::I64) => {
            function.instruction(&Instruction::I64LtS);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Gt, Prim::I64) => {
            function.instruction(&Instruction::I64GtS);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Le, Prim::I64) => {
            function.instruction(&Instruction::I64LeS);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Ge, Prim::I64) => {
            function.instruction(&Instruction::I64GeS);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Eq, Prim::F32) => {
            function.instruction(&Instruction::F32Eq);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Lt, Prim::F32) => {
            function.instruction(&Instruction::F32Lt);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Gt, Prim::F32) => {
            function.instruction(&Instruction::F32Gt);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Le, Prim::F32) => {
            function.instruction(&Instruction::F32Le);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Ge, Prim::F32) => {
            function.instruction(&Instruction::F32Ge);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Eq, Prim::F64) => {
            function.instruction(&Instruction::F64Eq);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Lt, Prim::F64) => {
            function.instruction(&Instruction::F64Lt);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Gt, Prim::F64) => {
            function.instruction(&Instruction::F64Gt);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Le, Prim::F64) => {
            function.instruction(&Instruction::F64Le);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Ge, Prim::F64) => {
            function.instruction(&Instruction::F64Ge);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        _ => {
            return Err(SimdError::new(format!(
                "Wasm backend does not support scalar primitive {:?} on {:?}",
                op, operand_prim
            )));
        }
    }
    Ok(())
}

struct VectorizableClause<'a> {
    patterns: &'a [TypedPattern],
    body: &'a IrExpr,
    locals: BTreeMap<String, usize>,
}

struct VectorizableKernel<'a> {
    clauses: Vec<VectorizableClause<'a>>,
}

fn vectorizable_kernel_clause<'a>(
    clauses: &'a [LoweredClause],
    params: &'a [KernelParam],
    result_prim: Prim,
) -> Option<VectorizableKernel<'a>> {
    let empty_enum_ctors = BTreeMap::<String, EnumCtorInfo>::new();
    if clauses.is_empty() {
        return None;
    }
    if clause_has_condition(&clauses.last()?.patterns, &empty_enum_ctors) {
        return None;
    }

    let mut vectorizable = Vec::with_capacity(clauses.len());
    for clause in clauses {
        let mut locals = BTreeMap::new();
        for (index, (pattern, param)) in clause.patterns.iter().zip(params).enumerate() {
            if let Pattern::Name(name) = &pattern.pattern
                && matches!(param, KernelParam::Same { .. } | KernelParam::Lane { .. })
            {
                locals.insert(name.clone(), index);
            }
        }
        if !patterns_are_vectorizable(&clause.patterns, params, result_prim) {
            return None;
        }
        if !is_vectorizable_expr(&clause.body, &locals, result_prim) {
            return None;
        }
        vectorizable.push(VectorizableClause {
            patterns: &clause.patterns,
            body: &clause.body,
            locals,
        });
    }

    Some(VectorizableKernel {
        clauses: vectorizable,
    })
}

fn patterns_are_vectorizable(
    patterns: &[TypedPattern],
    params: &[KernelParam],
    result_prim: Prim,
) -> bool {
    for (index, pattern) in patterns.iter().enumerate() {
        let Some(param) = params.get(index) else {
            return false;
        };
        let prim = match param {
            KernelParam::Same { prim, .. } | KernelParam::Lane { prim, .. } => *prim,
            KernelParam::SameSeq { .. } => return false,
        };
        if matches!(param, KernelParam::Lane { .. })
            && prim.lane_width() != result_prim.lane_width()
        {
            return false;
        }
        match pattern.pattern {
            Pattern::Int(_) => {
                if !prim.is_int() {
                    return false;
                }
            }
            Pattern::Float(_) => {
                if !prim.is_float() {
                    return false;
                }
            }
            Pattern::Char(_) => {
                if prim != Prim::Char {
                    return false;
                }
            }
            Pattern::Bool(_) => return false,
            Pattern::Type(_) => return false,
            Pattern::Tuple(_) => return false,
            Pattern::Ctor(_, _) => return false,
            Pattern::Slice { .. } => return false,
            Pattern::Name(_) | Pattern::Wildcard => {}
        }
    }
    true
}

fn is_vectorizable_expr(
    expr: &IrExpr,
    locals: &BTreeMap<String, usize>,
    result_prim: Prim,
) -> bool {
    match &expr.kind {
        IrExprKind::Local(name) => locals.contains_key(name),
        IrExprKind::Int(_, prim) => prim.is_int(),
        IrExprKind::Float(_, prim) => prim.is_float(),
        IrExprKind::Record(_) => false,
        IrExprKind::EnumCtor { .. }
        | IrExprKind::EnumTag { .. }
        | IrExprKind::EnumChildBySlot { .. }
        | IrExprKind::EnumNonRecField { .. } => false,
        IrExprKind::Seq(_) | IrExprKind::SeqSplice { .. } => false,
        IrExprKind::Let { bindings, body } => {
            bindings
                .iter()
                .all(|binding| is_vectorizable_expr(&binding.expr, locals, result_prim))
                && is_vectorizable_expr(body, locals, result_prim)
        }
        IrExprKind::Call { callee, args } => match callee {
            Callee::Function(_) => false,
            Callee::Builtin(_) => false,
            Callee::Prim(op) => match op {
                PrimOp::Add | PrimOp::Sub | PrimOp::Mul => args
                    .iter()
                    .all(|arg| is_vectorizable_expr(arg, locals, result_prim)),
                PrimOp::Div => {
                    args.iter()
                        .all(|arg| is_vectorizable_expr(arg, locals, result_prim))
                        && matches!(result_prim, Prim::F32 | Prim::F64)
                }
                PrimOp::Eq | PrimOp::Lt | PrimOp::Gt | PrimOp::Le | PrimOp::Ge => {
                    if !args
                        .iter()
                        .all(|arg| is_vectorizable_expr(arg, locals, result_prim))
                    {
                        return false;
                    }
                    let Some(operand_prim) = args[0].ty.prim() else {
                        return false;
                    };
                    result_prim == Prim::I64
                        && matches!(operand_prim, Prim::I32 | Prim::I64 | Prim::Char | Prim::F64)
                }
                _ => false,
            },
        },
    }
}

fn emit_vector_clause_mask(
    function: &mut Function,
    patterns: &[TypedPattern],
    params: &[KernelParam],
    addressing: VectorAddressing,
) -> Result<()> {
    let mut terms = 0usize;
    for (index, typed_pattern) in patterns.iter().enumerate() {
        match &typed_pattern.pattern {
            Pattern::Wildcard | Pattern::Name(_) => {}
            Pattern::Int(expected) => {
                let prim = emit_vector_pattern_value(function, params, index, addressing)?;
                emit_vector_splat_int(function, prim, *expected)?;
                emit_vector_pattern_eq(function, prim, false)?;
                if terms > 0 {
                    function.instruction(&Instruction::V128And);
                }
                terms += 1;
            }
            Pattern::Float(expected) => {
                let prim = emit_vector_pattern_value(function, params, index, addressing)?;
                emit_vector_splat_float(function, prim, *expected)?;
                emit_vector_pattern_eq(function, prim, true)?;
                if terms > 0 {
                    function.instruction(&Instruction::V128And);
                }
                terms += 1;
            }
            Pattern::Char(_) => {
                return Err(SimdError::new(
                    "Wasm backend does not support char vector clause patterns",
                ));
            }
            Pattern::Bool(_) => {
                return Err(SimdError::new(
                    "Wasm backend does not support bool clause patterns",
                ));
            }
            Pattern::Tuple(_) => {
                return Err(SimdError::new(
                    "Wasm backend does not support tuple vector clause patterns",
                ));
            }
            Pattern::Ctor(_, _) => {
                return Err(SimdError::new(
                    "Wasm backend does not support enum constructor clause patterns yet",
                ));
            }
            Pattern::Type(_) => {
                return Err(SimdError::new(
                    "Wasm backend does not support type witness clause patterns",
                ));
            }
            Pattern::Slice { .. } => {
                return Err(SimdError::new(
                    "Wasm backend does not yet support slice view patterns",
                ));
            }
        }
    }
    if terms == 0 {
        emit_vector_all_ones_mask(function);
    }
    Ok(())
}

fn emit_vector_pattern_value(
    function: &mut Function,
    params: &[KernelParam],
    index: usize,
    addressing: VectorAddressing,
) -> Result<Prim> {
    let param = params
        .get(index)
        .ok_or_else(|| SimdError::new("pattern arity mismatch in vector codegen"))?;
    match param {
        KernelParam::Same { prim, value_local } => {
            emit_vector_splat_local(function, *prim, *value_local);
            Ok(*prim)
        }
        KernelParam::SameSeq { .. } => Err(SimdError::new(
            "Wasm backend cannot vectorize T[*] same parameters",
        )),
        KernelParam::Lane {
            prim, ptr_local, ..
        } => {
            emit_vector_load(function, *prim, *ptr_local, addressing);
            Ok(*prim)
        }
    }
}

fn emit_vector_pattern_eq(function: &mut Function, prim: Prim, float_bits: bool) -> Result<()> {
    match (prim, float_bits) {
        (Prim::I32 | Prim::Char, false) => {
            function.instruction(&Instruction::I32x4Eq);
        }
        (Prim::I64, false) => {
            function.instruction(&Instruction::I64x2Eq);
        }
        // Float patterns use bit-equality semantics to match scalar evaluator behavior.
        (Prim::F32, true) => {
            function.instruction(&Instruction::I32x4Eq);
        }
        (Prim::F64, true) => {
            function.instruction(&Instruction::I64x2Eq);
        }
        _ => {
            return Err(SimdError::new(format!(
                "vector pattern equality mismatch for {:?}",
                prim
            )));
        }
    }
    Ok(())
}

fn emit_vector_all_ones_mask(function: &mut Function) {
    function.instruction(&Instruction::I32Const(-1));
    function.instruction(&Instruction::I32x4Splat);
}

fn compile_vectorized_clause_chain(
    function: &mut Function,
    clauses: &[VectorizableClause<'_>],
    params: &[KernelParam],
    addressing: VectorAddressing,
    scalar_indices: &BTreeMap<String, u32>,
    hoisted_locals: Option<&BTreeMap<HoistExprKey, u32>>,
    acc_local: u32,
) -> Result<()> {
    let empty_enum_ctors = BTreeMap::<String, EnumCtorInfo>::new();
    let last = clauses
        .last()
        .ok_or_else(|| SimdError::new("vectorized kernel missing clauses"))?;
    compile_vector_ir_expr(
        function,
        last.body,
        &last.locals,
        params,
        addressing,
        scalar_indices,
        hoisted_locals,
    )?;
    function.instruction(&Instruction::LocalSet(acc_local));

    for clause in clauses.iter().rev().skip(1) {
        if !clause_has_condition(clause.patterns, &empty_enum_ctors) {
            compile_vector_ir_expr(
                function,
                clause.body,
                &clause.locals,
                params,
                addressing,
                scalar_indices,
                hoisted_locals,
            )?;
            function.instruction(&Instruction::LocalSet(acc_local));
            continue;
        }
        compile_vector_ir_expr(
            function,
            clause.body,
            &clause.locals,
            params,
            addressing,
            scalar_indices,
            hoisted_locals,
        )?;
        function.instruction(&Instruction::LocalGet(acc_local));
        emit_vector_clause_mask(function, clause.patterns, params, addressing)?;
        function.instruction(&Instruction::V128Bitselect);
        function.instruction(&Instruction::LocalSet(acc_local));
    }

    function.instruction(&Instruction::LocalGet(acc_local));
    Ok(())
}

fn compile_grouped_vectorized_clause_chain(
    function: &mut Function,
    clauses: &[VectorizableGroupedClause<'_>],
    output_index: usize,
    params: &[KernelParam],
    addressing: VectorAddressing,
    scalar_indices: &BTreeMap<String, u32>,
    hoisted_locals: Option<&BTreeMap<HoistExprKey, u32>>,
    acc_local: u32,
) -> Result<()> {
    let empty_enum_ctors = BTreeMap::<String, EnumCtorInfo>::new();
    let last = clauses
        .last()
        .ok_or_else(|| SimdError::new("vectorized grouped kernel missing clauses"))?;
    let body = last
        .bodies
        .get(output_index)
        .ok_or_else(|| SimdError::new("vectorized grouped kernel missing output body"))?;
    compile_vector_ir_expr(
        function,
        body,
        &last.locals,
        params,
        addressing,
        scalar_indices,
        hoisted_locals,
    )?;
    function.instruction(&Instruction::LocalSet(acc_local));

    for clause in clauses.iter().rev().skip(1) {
        if !clause_has_condition(clause.patterns, &empty_enum_ctors) {
            let body = clause
                .bodies
                .get(output_index)
                .ok_or_else(|| SimdError::new("vectorized grouped kernel missing output body"))?;
            compile_vector_ir_expr(
                function,
                body,
                &clause.locals,
                params,
                addressing,
                scalar_indices,
                hoisted_locals,
            )?;
            function.instruction(&Instruction::LocalSet(acc_local));
            continue;
        }
        let body = clause
            .bodies
            .get(output_index)
            .ok_or_else(|| SimdError::new("vectorized grouped kernel missing output body"))?;
        compile_vector_ir_expr(
            function,
            body,
            &clause.locals,
            params,
            addressing,
            scalar_indices,
            hoisted_locals,
        )?;
        function.instruction(&Instruction::LocalGet(acc_local));
        emit_vector_clause_mask(function, clause.patterns, params, addressing)?;
        function.instruction(&Instruction::V128Bitselect);
        function.instruction(&Instruction::LocalSet(acc_local));
    }

    function.instruction(&Instruction::LocalGet(acc_local));
    Ok(())
}

fn compile_vector_ir_expr(
    function: &mut Function,
    expr: &IrExpr,
    locals: &BTreeMap<String, usize>,
    params: &[KernelParam],
    addressing: VectorAddressing,
    scalar_indices: &BTreeMap<String, u32>,
    hoisted_locals: Option<&BTreeMap<HoistExprKey, u32>>,
) -> Result<()> {
    let empty: BTreeMap<HoistExprKey, u32> = BTreeMap::new();
    let inline_bindings = BTreeMap::<String, IrExpr>::new();
    compile_vector_ir_expr_with_hoists(
        function,
        expr,
        locals,
        params,
        addressing,
        scalar_indices,
        hoisted_locals.unwrap_or(&empty),
        &inline_bindings,
    )
}

fn compile_vector_ir_expr_with_hoists(
    function: &mut Function,
    expr: &IrExpr,
    locals: &BTreeMap<String, usize>,
    params: &[KernelParam],
    addressing: VectorAddressing,
    scalar_indices: &BTreeMap<String, u32>,
    hoisted_locals: &BTreeMap<HoistExprKey, u32>,
    inline_bindings: &BTreeMap<String, IrExpr>,
) -> Result<()> {
    let _ = scalar_indices;
    if let Some(local) = hoisted_locals.get(&hoist_expr_key(expr)) {
        function.instruction(&Instruction::LocalGet(*local));
        return Ok(());
    }
    match &expr.kind {
        IrExprKind::Seq(_) | IrExprKind::SeqSplice { .. } => {
            return Err(SimdError::new(
                "Wasm vector codegen does not yet support sequence IR",
            ));
        }
        IrExprKind::Local(name) => {
            if let Some(slot) = locals.get(name) {
                match params[*slot] {
                    KernelParam::Same { prim, value_local } => {
                        emit_vector_splat_local(function, prim, value_local)
                    }
                    KernelParam::SameSeq { .. } => {
                        return Err(SimdError::new(
                            "Wasm vector codegen does not support T[*] same locals",
                        ));
                    }
                    KernelParam::Lane {
                        prim, ptr_local, ..
                    } => emit_vector_load(function, prim, ptr_local, addressing),
                }
            } else if let Some(inline_expr) = inline_bindings.get(name) {
                compile_vector_ir_expr_with_hoists(
                    function,
                    inline_expr,
                    locals,
                    params,
                    addressing,
                    scalar_indices,
                    hoisted_locals,
                    inline_bindings,
                )?;
            } else {
                return Err(SimdError::new(format!("unknown vector local '{}'", name)));
            }
        }
        IrExprKind::Int(value, prim) => emit_vector_splat_int(function, *prim, *value)?,
        IrExprKind::Float(value, prim) => emit_vector_splat_float(function, *prim, *value)?,
        IrExprKind::Record(_) => {
            return Err(SimdError::new(
                "Wasm vector codegen does not yet support record IR expressions",
            ));
        }
        IrExprKind::EnumCtor { .. }
        | IrExprKind::EnumTag { .. }
        | IrExprKind::EnumChildBySlot { .. }
        | IrExprKind::EnumNonRecField { .. } => {
            return Err(SimdError::new(
                "Wasm vector codegen does not yet support enum IR expressions",
            ));
        }
        IrExprKind::Let { bindings, body } => {
            let mut extended = inline_bindings.clone();
            for binding in bindings {
                extended.insert(binding.name.clone(), binding.expr.clone());
            }
            compile_vector_ir_expr_with_hoists(
                function,
                body,
                locals,
                params,
                addressing,
                scalar_indices,
                hoisted_locals,
                &extended,
            )?;
        }
        IrExprKind::Call { callee, args } => match callee {
            Callee::Function(name) => {
                return Err(SimdError::new(format!(
                    "vectorized kernel cannot call scalar helper '{}'",
                    name
                )));
            }
            Callee::Prim(op) => {
                compile_vector_ir_expr_with_hoists(
                    function,
                    &args[0],
                    locals,
                    params,
                    addressing,
                    scalar_indices,
                    hoisted_locals,
                    inline_bindings,
                )?;
                compile_vector_ir_expr_with_hoists(
                    function,
                    &args[1],
                    locals,
                    params,
                    addressing,
                    scalar_indices,
                    hoisted_locals,
                    inline_bindings,
                )?;
                emit_vector_primitive(
                    function,
                    *op,
                    args[0].ty.prim().unwrap(),
                    expr.ty.prim().unwrap_or(args[0].ty.prim().unwrap()),
                )?;
            }
            Callee::Builtin(_) => {
                return Err(SimdError::new(
                    "vectorized kernel cannot call builtin helper",
                ));
            }
        },
    }
    Ok(())
}

fn emit_vector_splat_local(function: &mut Function, prim: Prim, local: u32) {
    function.instruction(&Instruction::LocalGet(local));
    match prim {
        Prim::I32 | Prim::Char => {
            function.instruction(&Instruction::I32x4Splat);
        }
        Prim::I64 => {
            function.instruction(&Instruction::I64x2Splat);
        }
        Prim::F32 => {
            function.instruction(&Instruction::F32x4Splat);
        }
        Prim::F64 => {
            function.instruction(&Instruction::F64x2Splat);
        }
    }
}

fn emit_vector_splat_int(function: &mut Function, prim: Prim, value: i64) -> Result<()> {
    emit_int_const(function, prim, value)?;
    match prim {
        Prim::I32 | Prim::Char => {
            function.instruction(&Instruction::I32x4Splat);
        }
        Prim::I64 => {
            function.instruction(&Instruction::I64x2Splat);
        }
        Prim::F32 | Prim::F64 => {
            return Err(SimdError::new(
                "integer vector splat requested for float primitive",
            ));
        }
    }
    Ok(())
}

fn emit_vector_splat_float(function: &mut Function, prim: Prim, value: f64) -> Result<()> {
    match prim {
        Prim::F32 => {
            function.instruction(&Instruction::F32Const((value as f32).into()));
            function.instruction(&Instruction::F32x4Splat);
        }
        Prim::F64 => {
            function.instruction(&Instruction::F64Const(value.into()));
            function.instruction(&Instruction::F64x2Splat);
        }
        Prim::I32 | Prim::I64 | Prim::Char => {
            return Err(SimdError::new(
                "float vector splat requested for integer primitive",
            ));
        }
    }
    Ok(())
}

fn emit_vector_load(
    function: &mut Function,
    prim: Prim,
    ptr_local: u32,
    addressing: VectorAddressing,
) {
    match addressing {
        VectorAddressing::Pointer { lane_offset_elems } => {
            function.instruction(&Instruction::LocalGet(ptr_local));
            let offset = (lane_offset_elems as u64) * u64::from(byte_width(prim));
            function.instruction(&Instruction::V128Load(memarg(offset, 4)));
        }
    }
}

fn emit_vector_primitive(
    function: &mut Function,
    op: PrimOp,
    operand_prim: Prim,
    result_prim: Prim,
) -> Result<()> {
    match (op, operand_prim) {
        (PrimOp::Add, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32x4Add);
        }
        (PrimOp::Sub, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32x4Sub);
        }
        (PrimOp::Mul, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32x4Mul);
        }
        (PrimOp::Add, Prim::I64) => {
            function.instruction(&Instruction::I64x2Add);
        }
        (PrimOp::Sub, Prim::I64) => {
            function.instruction(&Instruction::I64x2Sub);
        }
        (PrimOp::Mul, Prim::I64) => {
            function.instruction(&Instruction::I64x2Mul);
        }
        (PrimOp::Add, Prim::F32) => {
            function.instruction(&Instruction::F32x4Add);
        }
        (PrimOp::Sub, Prim::F32) => {
            function.instruction(&Instruction::F32x4Sub);
        }
        (PrimOp::Mul, Prim::F32) => {
            function.instruction(&Instruction::F32x4Mul);
        }
        (PrimOp::Div, Prim::F32) => {
            function.instruction(&Instruction::F32x4Div);
        }
        (PrimOp::Add, Prim::F64) => {
            function.instruction(&Instruction::F64x2Add);
        }
        (PrimOp::Sub, Prim::F64) => {
            function.instruction(&Instruction::F64x2Sub);
        }
        (PrimOp::Mul, Prim::F64) => {
            function.instruction(&Instruction::F64x2Mul);
        }
        (PrimOp::Div, Prim::F64) => {
            function.instruction(&Instruction::F64x2Div);
        }
        (PrimOp::Eq, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32x4Eq);
            if result_prim == Prim::I64 {
                emit_vector_bool_to_i64(function)?;
            }
        }
        (PrimOp::Lt, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32x4LtS);
            if result_prim == Prim::I64 {
                emit_vector_bool_to_i64(function)?;
            }
        }
        (PrimOp::Gt, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32x4GtS);
            if result_prim == Prim::I64 {
                emit_vector_bool_to_i64(function)?;
            }
        }
        (PrimOp::Le, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32x4LeS);
            if result_prim == Prim::I64 {
                emit_vector_bool_to_i64(function)?;
            }
        }
        (PrimOp::Ge, Prim::I32 | Prim::Char) => {
            function.instruction(&Instruction::I32x4GeS);
            if result_prim == Prim::I64 {
                emit_vector_bool_to_i64(function)?;
            }
        }
        (PrimOp::Eq, Prim::I64) => {
            function.instruction(&Instruction::I64x2Eq);
            if result_prim == Prim::I64 {
                emit_vector_bool_to_i64(function)?;
            }
        }
        (PrimOp::Lt, Prim::I64) => {
            function.instruction(&Instruction::I64x2LtS);
            if result_prim == Prim::I64 {
                emit_vector_bool_to_i64(function)?;
            }
        }
        (PrimOp::Gt, Prim::I64) => {
            function.instruction(&Instruction::I64x2GtS);
            if result_prim == Prim::I64 {
                emit_vector_bool_to_i64(function)?;
            }
        }
        (PrimOp::Le, Prim::I64) => {
            function.instruction(&Instruction::I64x2LeS);
            if result_prim == Prim::I64 {
                emit_vector_bool_to_i64(function)?;
            }
        }
        (PrimOp::Ge, Prim::I64) => {
            function.instruction(&Instruction::I64x2GeS);
            if result_prim == Prim::I64 {
                emit_vector_bool_to_i64(function)?;
            }
        }
        (PrimOp::Eq, Prim::F64) => {
            function.instruction(&Instruction::F64x2Eq);
            if result_prim == Prim::I64 {
                emit_vector_bool_to_i64(function)?;
            }
        }
        (PrimOp::Lt, Prim::F64) => {
            function.instruction(&Instruction::F64x2Lt);
            if result_prim == Prim::I64 {
                emit_vector_bool_to_i64(function)?;
            }
        }
        (PrimOp::Gt, Prim::F64) => {
            function.instruction(&Instruction::F64x2Gt);
            if result_prim == Prim::I64 {
                emit_vector_bool_to_i64(function)?;
            }
        }
        (PrimOp::Le, Prim::F64) => {
            function.instruction(&Instruction::F64x2Le);
            if result_prim == Prim::I64 {
                emit_vector_bool_to_i64(function)?;
            }
        }
        (PrimOp::Ge, Prim::F64) => {
            function.instruction(&Instruction::F64x2Ge);
            if result_prim == Prim::I64 {
                emit_vector_bool_to_i64(function)?;
            }
        }
        _ => {
            return Err(SimdError::new(format!(
                "Wasm SIMD backend does not support vector primitive {:?} on {:?}",
                op, operand_prim
            )));
        }
    }
    Ok(())
}

fn emit_vector_bool_to_i64(function: &mut Function) -> Result<()> {
    emit_int_const(function, Prim::I64, 1)?;
    function.instruction(&Instruction::I64x2Splat);
    function.instruction(&Instruction::V128And);
    Ok(())
}

fn emit_lane_load_scalar_at_ptr(function: &mut Function, ptr_local: u32, prim: Prim) {
    function.instruction(&Instruction::LocalGet(ptr_local));
    match prim {
        Prim::I32 | Prim::Char => {
            function.instruction(&Instruction::I32Load(memarg(0, 2)));
        }
        Prim::I64 => {
            function.instruction(&Instruction::I64Load(memarg(0, 3)));
        }
        Prim::F32 => {
            function.instruction(&Instruction::F32Load(memarg(0, 2)));
        }
        Prim::F64 => {
            function.instruction(&Instruction::F64Load(memarg(0, 3)));
        }
    }
}

fn emit_scalar_store(function: &mut Function, prim: Prim) {
    match prim {
        Prim::I32 | Prim::Char => {
            function.instruction(&Instruction::I32Store(memarg(0, 2)));
        }
        Prim::I64 => {
            function.instruction(&Instruction::I64Store(memarg(0, 3)));
        }
        Prim::F32 => {
            function.instruction(&Instruction::F32Store(memarg(0, 2)));
        }
        Prim::F64 => {
            function.instruction(&Instruction::F64Store(memarg(0, 3)));
        }
    }
}

fn emit_pointer_bump(function: &mut Function, ptr_local: u32, bytes: u32) {
    function.instruction(&Instruction::LocalGet(ptr_local));
    function.instruction(&Instruction::I32Const(bytes as i32));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(ptr_local));
}

fn emit_scalar_index_bump(function: &mut Function, index_local: u32) {
    function.instruction(&Instruction::LocalGet(index_local));
    function.instruction(&Instruction::I32Const(1));
    function.instruction(&Instruction::I32Add);
    function.instruction(&Instruction::LocalSet(index_local));
}

fn memarg(offset: u64, align: u32) -> MemArg {
    MemArg {
        offset,
        align,
        memory_index: 0,
    }
}

fn byte_width(prim: Prim) -> u32 {
    match prim {
        Prim::I32 | Prim::Char | Prim::F32 => 4,
        Prim::I64 | Prim::F64 => 8,
    }
}

fn wasm_enum_field_storage_bytes(ty: &Type) -> Result<u32> {
    match ty {
        Type::Scalar(prim) => Ok(byte_width(*prim)),
        Type::Named(name, args) if is_wasm_enum_named_type(name, args) => Ok(4),
        Type::Named(name, args) if is_wasm_string_named_type(name, args) => Ok(8),
        Type::Index(_) => Ok(8),
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _)
            if wasm_star_seq_storage_prim(ty).is_some() =>
        {
            Ok(8)
        }
        other => Err(SimdError::new(format!(
            "Wasm enum tape rows do not support non-recursive field type {:?}",
            other
        ))),
    }
}

fn build_specialized_wasm_enum_ctor_layout(
    ctor_name: &str,
    ctor: &EnumCtorInfo,
    enum_args: &[Type],
) -> Result<WasmEnumCtorLayout> {
    if ctor.enum_params.len() != enum_args.len() {
        return Err(SimdError::new(format!(
            "constructor '{}.{}' expected {} enum args, found {}",
            ctor.enum_name,
            ctor_name,
            ctor.enum_params.len(),
            enum_args.len()
        )));
    }
    let subst = ctor
        .enum_params
        .iter()
        .cloned()
        .zip(enum_args.iter().cloned())
        .collect::<BTreeMap<_, _>>();
    let mut row_stride = 0u32;
    let mut recursive_field_indices = Vec::new();
    let mut non_recursive_fields = Vec::new();
    for (field_index, field_ty) in ctor.fields.iter().enumerate() {
        let specialized_ty = apply_type_subst(field_ty, &subst);
        if is_wasm_direct_self_recursive_field(field_ty, ctor) {
            recursive_field_indices.push(field_index);
            continue;
        }
        let offset = row_stride;
        row_stride = row_stride
            .checked_add(wasm_enum_field_storage_bytes(&specialized_ty)?)
            .ok_or_else(|| SimdError::new("enum row stride overflow"))?;
        non_recursive_fields.push(WasmEnumFieldLayout {
            source_index: field_index,
            row_index: non_recursive_fields.len(),
            offset,
            ty: specialized_ty,
        });
    }
    Ok(WasmEnumCtorLayout {
        ctor_name: ctor_name.to_string(),
        tag: ctor.tag,
        row_stride,
        recursive_field_indices,
        non_recursive_fields,
    })
}

fn build_specialized_wasm_enum_layout(
    enum_name: &str,
    enum_args: &[Type],
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
) -> Result<WasmEnumLayout> {
    let enum_names = enum_ctors
        .values()
        .map(|ctor| ctor.enum_name.clone())
        .collect::<BTreeSet<_>>();
    let enum_id = enum_names
        .iter()
        .position(|name| name == enum_name)
        .ok_or_else(|| {
            SimdError::new(format!("unknown enum '{}' in Wasm layout build", enum_name))
        })?;
    let mut max_tag = 0u16;
    let mut ctors = BTreeMap::new();
    for (ctor_name, ctor) in enum_ctors
        .iter()
        .filter(|(_, ctor)| ctor.enum_name == enum_name)
    {
        max_tag = max_tag.max(ctor.tag);
        ctors.insert(
            ctor_name.clone(),
            build_specialized_wasm_enum_ctor_layout(ctor_name, ctor, enum_args)?,
        );
    }
    Ok(WasmEnumLayout {
        enum_name: enum_name.to_string(),
        enum_id: u32::try_from(enum_id)
            .map_err(|_| SimdError::new("enum id does not fit in u32"))?,
        max_tag,
        ctors,
    })
}

fn specialized_wasm_enum_layout_key(enum_name: &str, enum_args: &[Type]) -> String {
    format!("{enum_name}::{enum_args:?}")
}

fn enqueue_precomputed_wasm_enum_layout_types(
    ty: &Type,
    seen: &mut BTreeSet<String>,
    queue: &mut VecDeque<(String, Vec<Type>)>,
) {
    match ty {
        Type::Scalar(_) | Type::Bulk(_, _) | Type::Var(_) | Type::Infer(_) | Type::Index(_) => {}
        Type::StarSeq(item) | Type::StarSeqWitnessed(item, _) => {
            enqueue_precomputed_wasm_enum_layout_types(item, seen, queue)
        }
        Type::TypeToken(inner) => {
            enqueue_precomputed_wasm_enum_layout_types(inner, seen, queue);
        }
        Type::Tuple(items) => {
            for item_ty in items {
                enqueue_precomputed_wasm_enum_layout_types(item_ty, seen, queue);
            }
        }
        Type::Record(fields) => {
            for field_ty in fields.values() {
                enqueue_precomputed_wasm_enum_layout_types(field_ty, seen, queue);
            }
        }
        Type::Fun(args, ret) => {
            for arg in args {
                enqueue_precomputed_wasm_enum_layout_types(arg, seen, queue);
            }
            enqueue_precomputed_wasm_enum_layout_types(ret, seen, queue);
        }
        Type::Named(name, args) => {
            for arg in args {
                enqueue_precomputed_wasm_enum_layout_types(arg, seen, queue);
            }
            if is_wasm_enum_named_type(name, args) {
                let key = specialized_wasm_enum_layout_key(name, args);
                if seen.insert(key) {
                    queue.push_back((name.clone(), args.clone()));
                }
            }
        }
    }
}

fn enqueue_precomputed_wasm_enum_layout_expr_types(
    expr: &TypedExpr,
    seen: &mut BTreeSet<String>,
    queue: &mut VecDeque<(String, Vec<Type>)>,
) {
    enqueue_precomputed_wasm_enum_layout_types(&expr.ty, seen, queue);
    match &expr.kind {
        TypedExprKind::Local(_)
        | TypedExprKind::FunctionRef { .. }
        | TypedExprKind::ConstructorRef { .. }
        | TypedExprKind::Int(_, _)
        | TypedExprKind::Float(_, _)
        | TypedExprKind::Bool(_)
        | TypedExprKind::Char(_)
        | TypedExprKind::String(_)
        | TypedExprKind::TypeToken(_) => {}
        TypedExprKind::Seq(items) => {
            for item in items {
                enqueue_precomputed_wasm_enum_layout_expr_types(item, seen, queue);
            }
        }
        TypedExprKind::SeqSplice { prefix, tail } => {
            for item in prefix {
                enqueue_precomputed_wasm_enum_layout_expr_types(item, seen, queue);
            }
            enqueue_precomputed_wasm_enum_layout_expr_types(tail, seen, queue);
        }
        TypedExprKind::Lambda { body, .. } => {
            enqueue_precomputed_wasm_enum_layout_expr_types(body, seen, queue);
        }
        TypedExprKind::Let { bindings, body } => {
            for binding in bindings {
                enqueue_precomputed_wasm_enum_layout_expr_types(&binding.expr, seen, queue);
            }
            enqueue_precomputed_wasm_enum_layout_expr_types(body, seen, queue);
        }
        TypedExprKind::Tuple(items) => {
            for item in items {
                enqueue_precomputed_wasm_enum_layout_expr_types(item, seen, queue);
            }
        }
        TypedExprKind::Record(fields) => {
            for field in fields.values() {
                enqueue_precomputed_wasm_enum_layout_expr_types(field, seen, queue);
            }
        }
        TypedExprKind::Project { base, .. } | TypedExprKind::TupleProject { base, .. } => {
            enqueue_precomputed_wasm_enum_layout_expr_types(base, seen, queue);
        }
        TypedExprKind::Index { base, index, .. } => {
            enqueue_precomputed_wasm_enum_layout_expr_types(base, seen, queue);
            enqueue_precomputed_wasm_enum_layout_expr_types(index, seen, queue);
        }
        TypedExprKind::RecordUpdate { base, fields } => {
            enqueue_precomputed_wasm_enum_layout_expr_types(base, seen, queue);
            for field in fields.values() {
                enqueue_precomputed_wasm_enum_layout_expr_types(field, seen, queue);
            }
        }
        TypedExprKind::Call { args, .. } => {
            for arg in args {
                enqueue_precomputed_wasm_enum_layout_expr_types(&arg.expr, seen, queue);
            }
        }
        TypedExprKind::Apply { callee, arg } => {
            enqueue_precomputed_wasm_enum_layout_expr_types(callee, seen, queue);
            enqueue_precomputed_wasm_enum_layout_expr_types(arg, seen, queue);
        }
    }
}

fn build_precomputed_wasm_enum_layouts(
    checked_program: &CheckedProgram,
) -> Result<BTreeMap<String, WasmEnumLayout>> {
    let mut seen = BTreeSet::<String>::new();
    let mut queue = VecDeque::<(String, Vec<Type>)>::new();
    for function in &checked_program.functions {
        enqueue_precomputed_wasm_enum_layout_types(&function.signature.ty, &mut seen, &mut queue);
        for clause in &function.clauses {
            for pattern in &clause.patterns {
                enqueue_precomputed_wasm_enum_layout_types(&pattern.ty, &mut seen, &mut queue);
            }
            enqueue_precomputed_wasm_enum_layout_expr_types(&clause.body, &mut seen, &mut queue);
        }
    }

    let mut layouts = BTreeMap::<String, WasmEnumLayout>::new();
    while let Some((enum_name, enum_args)) = queue.pop_front() {
        let key = specialized_wasm_enum_layout_key(&enum_name, &enum_args);
        if layouts.contains_key(&key) {
            continue;
        }
        let layout = build_specialized_wasm_enum_layout(
            &enum_name,
            &enum_args,
            &checked_program.enum_ctors,
        )?;
        for ctor_layout in layout.ctors.values() {
            for field_layout in &ctor_layout.non_recursive_fields {
                enqueue_precomputed_wasm_enum_layout_types(&field_layout.ty, &mut seen, &mut queue);
            }
        }
        layouts.insert(key, layout);
    }
    Ok(layouts)
}

fn lookup_specialized_wasm_enum_layout<'a>(
    wasm_enum_layouts: &'a BTreeMap<String, WasmEnumLayout>,
    enum_name: &str,
    enum_args: &[Type],
) -> Result<&'a WasmEnumLayout> {
    let key = specialized_wasm_enum_layout_key(enum_name, enum_args);
    wasm_enum_layouts.get(&key).ok_or_else(|| {
        SimdError::new(format!(
            "missing precomputed Wasm enum layout for '{}<{:?}>'",
            enum_name, enum_args
        ))
    })
}

fn lookup_specialized_wasm_enum_ctor_layout<'a>(
    wasm_enum_layouts: &'a BTreeMap<String, WasmEnumLayout>,
    enum_name: &str,
    enum_args: &[Type],
    ctor_name: &str,
) -> Result<&'a WasmEnumCtorLayout> {
    lookup_specialized_wasm_enum_layout(wasm_enum_layouts, enum_name, enum_args)?
        .ctors
        .get(ctor_name)
        .ok_or_else(|| {
            SimdError::new(format!(
                "missing precomputed Wasm ctor layout for '{}.{}<{:?}>'",
                enum_name, ctor_name, enum_args
            ))
        })
}

#[derive(Debug, Clone)]
struct WasmPlan {
    checked: CheckedProgram,
    params: Vec<WasmParamAbi>,
    result: WasmResultAbi,
    leaf_exports: Vec<WasmLeafExport>,
}

fn build_wasm_plan(checked_program: &CheckedProgram, main: &str) -> Result<WasmPlan> {
    let checked_map: BTreeMap<_, _> = checked_program
        .functions
        .iter()
        .map(|function| (function.name.clone(), function))
        .collect();
    let main_function = checked_map
        .get(main)
        .copied()
        .ok_or_else(|| SimdError::new(format!("unknown entry function '{}'", main)))?;

    let mut result_leaf_names = BTreeMap::<String, BTreeMap<LeafPath, String>>::new();
    let mut result_leaf_types = BTreeMap::<String, Vec<TypeLeaf>>::new();
    for function in &checked_program.functions {
        let (_, result) = function.signature.ty.fun_parts();
        let leaves = flatten_type_leaves(&result);
        result_leaf_names.insert(
            function.name.clone(),
            leaves
                .iter()
                .map(|leaf| {
                    (
                        leaf.path.clone(),
                        leaf_export_name(&function.name, &leaf.path),
                    )
                })
                .collect(),
        );
        result_leaf_types.insert(function.name.clone(), leaves);
    }

    let mut leaf_functions = Vec::new();
    for function in &checked_program.functions {
        let leaves = result_leaf_types
            .get(&function.name)
            .cloned()
            .ok_or_else(|| {
                SimdError::new(format!("missing result leaves for '{}'", function.name))
            })?;
        for leaf in leaves {
            leaf_functions.push(build_leaf_function(
                function,
                &leaf.path,
                &checked_map,
                &result_leaf_names,
            )?);
        }
    }

    let leaf_exports = result_leaf_types
        .get(main)
        .ok_or_else(|| SimdError::new(format!("missing result leaves for '{}'", main)))?
        .iter()
        .map(|leaf| {
            Ok(WasmLeafExport {
                leaf_path: leaf.path.clone(),
                export_name: leaf_export_name(main, &leaf.path),
                result: wasm_leaf_result_abi_from_type(&leaf.ty)?,
                used_param_leaves: Vec::new(),
                reusable_param_leaf: None,
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let main_param_types = main_function.signature.ty.fun_parts().0;
    let params = wasm_param_abi_from_type(&main_param_types)?;
    let result =
        wasm_result_abi_from_type(&main_function.signature.ty.fun_parts().1, &main_param_types)?;

    Ok(WasmPlan {
        checked: CheckedProgram {
            enum_names: checked_program.enum_names.clone(),
            enum_ctors: checked_program.enum_ctors.clone(),
            enum_layouts: checked_program.enum_layouts.clone(),
            functions: leaf_functions,
        },
        params,
        result,
        leaf_exports,
    })
}

fn build_leaf_function(
    function: &CheckedFunction,
    leaf_path: &LeafPath,
    checked_map: &BTreeMap<String, &CheckedFunction>,
    result_leaf_names: &BTreeMap<String, BTreeMap<LeafPath, String>>,
) -> Result<CheckedFunction> {
    let (param_types, result_ty) = function.signature.ty.fun_parts();
    let leaf_result_ty = type_at_leaf_path(&result_ty, leaf_path)?;
    let leaf_param_types = param_types
        .iter()
        .flat_map(flatten_type_leaves)
        .map(|leaf| leaf.ty)
        .collect::<Vec<_>>();

    let clauses = function
        .clauses
        .iter()
        .map(|clause| {
            let (local_types, local_leaf_names) = build_clause_env(clause, &param_types);
            let patterns = flatten_clause_patterns(&clause.patterns, &param_types)?;
            let body = normalize_expr_for_leaf(
                &clause.body,
                leaf_path,
                &local_types,
                &local_leaf_names,
                checked_map,
                result_leaf_names,
            )?;
            Ok(TypedClause { patterns, body })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(CheckedFunction {
        name: leaf_export_name(&function.name, leaf_path),
        signature: Signature {
            name: leaf_export_name(&function.name, leaf_path),
            ty: Type::Fun(leaf_param_types, Box::new(leaf_result_ty)),
            operator_instance: None,
            family_instance: None,
        },
        clauses,
        pointwise: function.pointwise,
        tailrec: function.tailrec.clone(),
    })
}

fn build_clause_env(
    clause: &TypedClause,
    param_types: &[Type],
) -> (
    BTreeMap<String, Type>,
    BTreeMap<String, BTreeMap<LeafPath, String>>,
) {
    let mut local_types = BTreeMap::new();
    let mut local_leaf_names = BTreeMap::new();
    for (pattern, ty) in clause.patterns.iter().zip(param_types) {
        let mut bindings = Vec::new();
        collect_non_ctor_pattern_bindings(&pattern.pattern, ty, &mut bindings);
        for (name, binding_ty) in bindings {
            local_types.insert(name.clone(), binding_ty.clone());
            let mut leaf_names = BTreeMap::new();
            for leaf in flatten_type_leaves(&binding_ty) {
                leaf_names.insert(leaf.path.clone(), leaf_symbol_name(&name, &leaf.path));
            }
            local_leaf_names.insert(name, leaf_names);
        }
    }
    (local_types, local_leaf_names)
}

fn collect_non_ctor_pattern_bindings(pattern: &Pattern, ty: &Type, out: &mut Vec<(String, Type)>) {
    match pattern {
        Pattern::Name(name) => {
            if name != "_" && !is_constructor_name(name) {
                out.push((name.clone(), ty.clone()));
            }
        }
        Pattern::Tuple(items) => {
            if let Type::Tuple(item_types) = ty {
                for (item, item_ty) in items.iter().zip(item_types.iter()) {
                    collect_non_ctor_pattern_bindings(item, item_ty, out);
                }
            }
        }
        Pattern::Ctor(_, _)
        | Pattern::Wildcard
        | Pattern::Int(_)
        | Pattern::Float(_)
        | Pattern::Char(_)
        | Pattern::Bool(_)
        | Pattern::Type(_)
        | Pattern::Slice { .. } => {}
    }
}

fn tuple_leaf_name(index: usize) -> String {
    index.to_string()
}

fn parse_tuple_leaf_name(name: &str) -> Option<usize> {
    name.parse::<usize>().ok()
}

fn pattern_at_leaf(pattern: &Pattern, ty: &Type, path: &LeafPath) -> Result<Pattern> {
    if path.is_root() {
        return Ok(pattern.clone());
    }
    let Some((head, tail)) = path.split_first() else {
        unreachable!("non-root tuple/record leaf path must have a head");
    };
    match (pattern, ty) {
        (Pattern::Name(name), _) => Ok(Pattern::Name(leaf_symbol_name(name, path))),
        (Pattern::Wildcard, _) => Ok(Pattern::Wildcard),
        (Pattern::Tuple(items), Type::Tuple(item_types)) => {
            let Some(index) = parse_tuple_leaf_name(head) else {
                return Err(SimdError::new(format!(
                    "tuple leaf '{}' is not a valid tuple index",
                    head
                )));
            };
            let item = items.get(index).ok_or_else(|| {
                SimdError::new(format!("tuple pattern is missing element {}", index))
            })?;
            let item_ty = item_types.get(index).ok_or_else(|| {
                SimdError::new(format!("tuple type is missing element {}", index))
            })?;
            pattern_at_leaf(item, item_ty, &tail)
        }
        (
            Pattern::Ctor(_, _)
            | Pattern::Slice { .. }
            | Pattern::Int(_)
            | Pattern::Float(_)
            | Pattern::Char(_)
            | Pattern::Bool(_)
            | Pattern::Type(_),
            Type::Tuple(_) | Type::Record(_),
        ) => Err(SimdError::new(
            "tuple/record parameters cannot use literal patterns",
        )),
        (pattern, Type::Record(_)) => Err(SimdError::new(format!(
            "cannot select leaf {:?} from record pattern {:?}",
            path, pattern
        ))),
        (pattern, other) => Err(SimdError::new(format!(
            "cannot select leaf {:?} from pattern {:?} over {:?}",
            path, pattern, other
        ))),
    }
}

fn flatten_clause_patterns(
    patterns: &[TypedPattern],
    param_types: &[Type],
) -> Result<Vec<TypedPattern>> {
    let mut flattened = Vec::new();
    for (pattern, ty) in patterns.iter().zip(param_types) {
        match ty {
            Type::Scalar(_) | Type::Bulk(_, _) | Type::Index(_) => flattened.push(TypedPattern {
                pattern: pattern.pattern.clone(),
                ty: ty.clone(),
            }),
            Type::StarSeq(_) | Type::StarSeqWitnessed(_, _)
                if wasm_star_seq_storage_prim(ty).is_some() =>
            {
                flattened.push(TypedPattern {
                    pattern: pattern.pattern.clone(),
                    ty: ty.clone(),
                });
            }
            Type::StarSeq(_) | Type::StarSeqWitnessed(_, _) => {
                return Err(SimdError::new(
                    "Wasm backend only supports scalar/enum-element T[*] parameters in flattened clauses",
                ));
            }
            Type::Named(name, args) if is_wasm_string_named_type(name, args) => {
                flattened.push(TypedPattern {
                    pattern: pattern.pattern.clone(),
                    ty: ty.clone(),
                });
            }
            Type::Named(name, args) if is_wasm_enum_named_type(name, args) => {
                flattened.push(TypedPattern {
                    pattern: pattern.pattern.clone(),
                    ty: ty.clone(),
                });
            }
            Type::TypeToken(_) => {
                return Err(SimdError::new(
                    "Wasm backend does not support type witness parameters in flattened clauses",
                ));
            }
            Type::Record(_) | Type::Tuple(_) => {
                for leaf in flatten_type_leaves(ty) {
                    let leaf_pattern = pattern_at_leaf(&pattern.pattern, ty, &leaf.path)?;
                    flattened.push(TypedPattern {
                        pattern: leaf_pattern,
                        ty: leaf.ty,
                    });
                }
            }
            Type::Fun(_, _) => {
                return Err(SimdError::new(
                    "function parameters are not supported in Wasm record lowering",
                ));
            }
            Type::Named(_, _) | Type::Var(_) | Type::Infer(_) => {
                return Err(SimdError::new(
                    "unresolved polymorphic parameters are not supported in Wasm record lowering",
                ));
            }
        }
    }
    Ok(flattened)
}

fn normalize_expr_for_leaf(
    expr: &TypedExpr,
    leaf_path: &LeafPath,
    local_types: &BTreeMap<String, Type>,
    local_leaf_names: &BTreeMap<String, BTreeMap<LeafPath, String>>,
    checked_map: &BTreeMap<String, &CheckedFunction>,
    result_leaf_names: &BTreeMap<String, BTreeMap<LeafPath, String>>,
) -> Result<TypedExpr> {
    let leaf_ty = match (&expr.ty, leaf_path.is_root()) {
        (Type::Fun(_, _), true) => expr.ty.clone(),
        _ => type_at_leaf_path(&expr.ty, leaf_path)?,
    };
    match &expr.kind {
        TypedExprKind::Local(name) => {
            let Some(local_ty) = local_types.get(name) else {
                let synthesized = leaf_symbol_name(name, leaf_path);
                if local_types.contains_key(&synthesized) {
                    return Ok(TypedExpr {
                        ty: leaf_ty,
                        kind: TypedExprKind::Local(synthesized),
                    });
                }
                if !leaf_path.is_root() {
                    return Err(SimdError::new(format!(
                        "unknown local '{}' in Wasm normalization for non-root leaf {:?}",
                        name, leaf_path
                    )));
                }
                return Ok(TypedExpr {
                    ty: leaf_ty,
                    kind: TypedExprKind::Local(name.clone()),
                });
            };
            let local_name = local_leaf_names
                .get(name)
                .and_then(|names| names.get(leaf_path))
                .cloned()
                .unwrap_or_else(|| name.clone());
            if !leaf_path.is_root() {
                let projected = project_local_path(name, local_ty, leaf_path)?;
                return Ok(projected);
            }
            if local_leaf_names.get(name).is_some()
                && local_leaf_names
                    .get(name)
                    .and_then(|names| names.get(leaf_path))
                    .is_none()
                && matches!(local_ty, Type::Record(_))
            {
                let projected = project_local_path(name, local_ty, leaf_path)?;
                return Ok(projected);
            }
            Ok(TypedExpr {
                ty: leaf_ty,
                kind: TypedExprKind::Local(local_name),
            })
        }
        TypedExprKind::Int(value, prim) => {
            if !leaf_path.is_root() {
                return Err(SimdError::new(
                    "integer literal cannot be projected into a record leaf",
                ));
            }
            Ok(TypedExpr {
                ty: leaf_ty,
                kind: TypedExprKind::Int(*value, *prim),
            })
        }
        TypedExprKind::Float(value, prim) => {
            if !leaf_path.is_root() {
                return Err(SimdError::new(
                    "float literal cannot be projected into a record leaf",
                ));
            }
            Ok(TypedExpr {
                ty: leaf_ty,
                kind: TypedExprKind::Float(*value, *prim),
            })
        }
        TypedExprKind::Bool(_) => Err(SimdError::new(
            "Wasm record normalization does not support bool expressions",
        )),
        TypedExprKind::Char(value) => {
            if !leaf_path.is_root() {
                return Err(SimdError::new(
                    "char literal cannot be projected into a record leaf",
                ));
            }
            Ok(TypedExpr {
                ty: leaf_ty,
                kind: TypedExprKind::Char(*value),
            })
        }
        TypedExprKind::String(value) => {
            if !leaf_path.is_root() {
                return Err(SimdError::new(
                    "string literal cannot be projected into a record leaf",
                ));
            }
            Ok(TypedExpr {
                ty: leaf_ty,
                kind: TypedExprKind::String(value.clone()),
            })
        }
        TypedExprKind::Seq(items) => {
            if !leaf_path.is_root() {
                return Err(SimdError::new(
                    "sequence literal cannot be projected into a record leaf",
                ));
            }
            Ok(TypedExpr {
                ty: leaf_ty,
                kind: TypedExprKind::Seq(
                    items
                        .iter()
                        .map(|item| {
                            normalize_expr_for_leaf(
                                item,
                                &LeafPath::root(),
                                local_types,
                                local_leaf_names,
                                checked_map,
                                result_leaf_names,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?,
                ),
            })
        }
        TypedExprKind::SeqSplice { prefix, tail } => {
            if !leaf_path.is_root() {
                return Err(SimdError::new(
                    "sequence splice cannot be projected into a record leaf",
                ));
            }
            Ok(TypedExpr {
                ty: leaf_ty,
                kind: TypedExprKind::SeqSplice {
                    prefix: prefix
                        .iter()
                        .map(|item| {
                            normalize_expr_for_leaf(
                                item,
                                &LeafPath::root(),
                                local_types,
                                local_leaf_names,
                                checked_map,
                                result_leaf_names,
                            )
                        })
                        .collect::<Result<Vec<_>>>()?,
                    tail: Box::new(normalize_expr_for_leaf(
                        tail,
                        &LeafPath::root(),
                        local_types,
                        local_leaf_names,
                        checked_map,
                        result_leaf_names,
                    )?),
                },
            })
        }
        TypedExprKind::TypeToken(_) => Err(SimdError::new(
            "Wasm record normalization does not support type witness expressions",
        )),
        TypedExprKind::ConstructorRef { name } => {
            if !leaf_path.is_root() {
                return Err(SimdError::new(
                    "enum constructor reference cannot be projected into a record leaf",
                ));
            }
            Ok(TypedExpr {
                ty: leaf_ty,
                kind: TypedExprKind::ConstructorRef { name: name.clone() },
            })
        }
        TypedExprKind::FunctionRef { name } => {
            let function = checked_map.get(name).ok_or_else(|| {
                SimdError::new(format!(
                    "unknown function '{}' in Wasm record normalization",
                    name
                ))
            })?;
            let expected_arity = function.signature.ty.arity();
            if expected_arity != 0 {
                return Err(SimdError::new(
                    "Wasm record normalization does not support higher-order expressions",
                ));
            }
            let return_ty = match &function.signature.ty {
                Type::Fun(_, ret) => ret.as_ref().clone(),
                other => other.clone(),
            };
            match &return_ty {
                Type::Scalar(_) | Type::Bulk(_, _) | Type::Record(_) => {}
                other => {
                    return Err(SimdError::new(format!(
                        "unsupported zero-arity function '{}' type in Wasm record normalization: {:?}",
                        name, other
                    )));
                }
            }
            if function.clauses.len() != 1 || !function.clauses[0].patterns.is_empty() {
                return Err(SimdError::new(format!(
                    "zero-arity function '{}' must be a single-clause constant",
                    name
                )));
            }
            normalize_expr_for_leaf(
                &function.clauses[0].body,
                leaf_path,
                local_types,
                local_leaf_names,
                checked_map,
                result_leaf_names,
            )
        }
        TypedExprKind::Lambda { .. } => Err(SimdError::new(
            "Wasm record normalization does not support higher-order expressions",
        )),
        TypedExprKind::Apply { callee, arg } => {
            let (head, _) = typed_expr_apply_chain_root(expr);
            if !matches!(head.kind, TypedExprKind::ConstructorRef { .. }) {
                return Err(SimdError::new(
                    "Wasm record normalization does not support higher-order expressions",
                ));
            }
            if !leaf_path.is_root() {
                return Err(SimdError::new(
                    "constructor application cannot be projected into a record leaf",
                ));
            }
            let normalized_callee = normalize_expr_for_leaf(
                callee,
                &LeafPath::root(),
                local_types,
                local_leaf_names,
                checked_map,
                result_leaf_names,
            )?;
            let normalized_arg = normalize_expr_for_leaf(
                arg,
                &LeafPath::root(),
                local_types,
                local_leaf_names,
                checked_map,
                result_leaf_names,
            )?;
            Ok(TypedExpr {
                ty: leaf_ty,
                kind: TypedExprKind::Apply {
                    callee: Box::new(normalized_callee),
                    arg: Box::new(normalized_arg),
                },
            })
        }
        TypedExprKind::Tuple(items) => {
            let (field, tail) = leaf_path
                .split_first()
                .ok_or_else(|| SimdError::new("tuple expression requires a non-empty leaf path"))?;
            let Some(index) = parse_tuple_leaf_name(field) else {
                return Err(SimdError::new(format!(
                    "tuple leaf '{}' is not a valid tuple index",
                    field
                )));
            };
            let item_expr = items.get(index).ok_or_else(|| {
                SimdError::new(format!("tuple expression is missing element {}", index))
            })?;
            normalize_expr_for_leaf(
                item_expr,
                &tail,
                local_types,
                local_leaf_names,
                checked_map,
                result_leaf_names,
            )
        }
        TypedExprKind::Record(fields) => {
            let (field, tail) = leaf_path.split_first().ok_or_else(|| {
                SimdError::new("record expression requires a non-empty leaf path")
            })?;
            let field_expr = fields.get(field).ok_or_else(|| {
                SimdError::new(format!("record expression is missing field '{}'", field))
            })?;
            normalize_expr_for_leaf(
                field_expr,
                &tail,
                local_types,
                local_leaf_names,
                checked_map,
                result_leaf_names,
            )
        }
        TypedExprKind::Project { base, field } => normalize_expr_for_leaf(
            base,
            &advance_field_for_projection(base, leaf_path, field),
            local_types,
            local_leaf_names,
            checked_map,
            result_leaf_names,
        ),
        TypedExprKind::TupleProject { base, index } => normalize_expr_for_leaf(
            base,
            &advance_field_for_projection(base, leaf_path, &tuple_leaf_name(*index)),
            local_types,
            local_leaf_names,
            checked_map,
            result_leaf_names,
        ),
        TypedExprKind::Index {
            base,
            index,
            checked,
        } => {
            if !leaf_path.is_root() {
                return Err(SimdError::new(
                    "index expression cannot be normalized into a non-root leaf",
                ));
            }
            Ok(TypedExpr {
                ty: leaf_ty,
                kind: TypedExprKind::Index {
                    base: Box::new(normalize_expr_for_leaf(
                        base,
                        &LeafPath::root(),
                        local_types,
                        local_leaf_names,
                        checked_map,
                        result_leaf_names,
                    )?),
                    index: Box::new(normalize_expr_for_leaf(
                        index,
                        &LeafPath::root(),
                        local_types,
                        local_leaf_names,
                        checked_map,
                        result_leaf_names,
                    )?),
                    checked: *checked,
                },
            })
        }
        TypedExprKind::RecordUpdate { base, fields } => {
            if let Some((head, tail)) = leaf_path.split_first() {
                if let Some(field_expr) = fields.get(head) {
                    normalize_expr_for_leaf(
                        field_expr,
                        &tail,
                        local_types,
                        local_leaf_names,
                        checked_map,
                        result_leaf_names,
                    )
                } else {
                    normalize_expr_for_leaf(
                        base,
                        leaf_path,
                        local_types,
                        local_leaf_names,
                        checked_map,
                        result_leaf_names,
                    )
                }
            } else {
                normalize_expr_for_leaf(
                    base,
                    leaf_path,
                    local_types,
                    local_leaf_names,
                    checked_map,
                    result_leaf_names,
                )
            }
        }
        TypedExprKind::Let { bindings, body } => {
            let mut normalized_bindings = Vec::new();
            let mut local_types = local_types.clone();
            let mut local_leaf_names = local_leaf_names.clone();
            for binding in bindings {
                if binding.name == "_" {
                    continue;
                }
                let binding_leaves = flatten_type_leaves(&binding.expr.ty);
                let mut leaf_names = BTreeMap::new();
                for leaf in binding_leaves {
                    let local_name = leaf_symbol_name(&binding.name, &leaf.path);
                    let expr = normalize_expr_for_leaf(
                        &binding.expr,
                        &leaf.path,
                        &local_types,
                        &local_leaf_names,
                        checked_map,
                        result_leaf_names,
                    )?;
                    normalized_bindings.push(TypedLetBinding {
                        name: local_name.clone(),
                        expr,
                    });
                    leaf_names.insert(leaf.path.clone(), local_name);
                }
                local_types.insert(binding.name.clone(), binding.expr.ty.clone());
                local_leaf_names.insert(binding.name.clone(), leaf_names);
            }
            let body = normalize_expr_for_leaf(
                body,
                leaf_path,
                &local_types,
                &local_leaf_names,
                checked_map,
                result_leaf_names,
            )?;
            if normalized_bindings.is_empty() {
                Ok(body)
            } else {
                Ok(TypedExpr {
                    ty: leaf_ty,
                    kind: TypedExprKind::Let {
                        bindings: normalized_bindings,
                        body: Box::new(body),
                    },
                })
            }
        }
        TypedExprKind::Call {
            callee,
            args,
            lifted_shape,
        } => match callee {
            Callee::Prim(op) => {
                let normalized_args = args
                    .iter()
                    .map(|arg| {
                        Ok(TypedArg {
                            mode: arg.mode,
                            expr: Box::new(normalize_expr_for_leaf(
                                &arg.expr,
                                &LeafPath::root(),
                                local_types,
                                local_leaf_names,
                                checked_map,
                                result_leaf_names,
                            )?),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(TypedExpr {
                    ty: leaf_ty,
                    kind: TypedExprKind::Call {
                        callee: Callee::Prim(*op),
                        args: normalized_args,
                        lifted_shape: lifted_shape.clone(),
                    },
                })
            }
            Callee::Builtin(builtin) => {
                let arg_leaf_paths = match builtin {
                    BuiltinFamilyCallee::GatherSeq(_) => vec![LeafPath::root(), leaf_path.clone()],
                    BuiltinFamilyCallee::ScatterSeq(_) | BuiltinFamilyCallee::ScatterAddSeq(_) => {
                        vec![LeafPath::root(), LeafPath::root(), LeafPath::root()]
                    }
                    BuiltinFamilyCallee::CheckedGatherSeq(_)
                    | BuiltinFamilyCallee::CheckIndexSeq(_) => {
                        vec![LeafPath::root(), LeafPath::root()]
                    }
                    BuiltinFamilyCallee::IndicesSeq(_) | BuiltinFamilyCallee::ReverseSeq(_) => {
                        vec![LeafPath::root()]
                    }
                    _ => {
                        return Err(SimdError::new(
                            "Wasm normalization does not support this builtin call",
                        ));
                    }
                };
                if arg_leaf_paths.len() != args.len() {
                    return Err(SimdError::new(
                        "builtin normalization arity did not match call arguments",
                    ));
                }
                let normalized_args = args
                    .iter()
                    .zip(arg_leaf_paths.iter())
                    .map(|(arg, arg_leaf_path)| {
                        Ok(TypedArg {
                            mode: arg.mode,
                            expr: Box::new(normalize_expr_for_leaf(
                                &arg.expr,
                                arg_leaf_path,
                                local_types,
                                local_leaf_names,
                                checked_map,
                                result_leaf_names,
                            )?),
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(TypedExpr {
                    ty: leaf_ty,
                    kind: TypedExprKind::Call {
                        callee: Callee::Builtin(builtin.clone()),
                        args: normalized_args,
                        lifted_shape: lifted_shape.clone(),
                    },
                })
            }
            Callee::Function(name) => {
                let function = checked_map.get(name).copied().ok_or_else(|| {
                    SimdError::new(format!("unknown callee '{}' in Wasm normalization", name))
                })?;
                let (_, result_ty) = function.signature.ty.fun_parts();
                let callee_leaf_name = result_leaf_names
                    .get(name)
                    .and_then(|names| names.get(leaf_path))
                    .cloned()
                    .ok_or_else(|| {
                        SimdError::new(format!(
                            "missing leaf export for '{}' at path {:?}",
                            name, leaf_path
                        ))
                    })?;
                let normalized_args = args
                    .iter()
                    .zip(function.signature.ty.fun_parts().0.iter())
                    .flat_map(|(arg, param_ty)| {
                        flatten_type_leaves(param_ty).into_iter().map(move |leaf| {
                            Ok(TypedArg {
                                mode: arg.mode,
                                expr: Box::new(normalize_expr_for_leaf(
                                    &arg.expr,
                                    &leaf.path,
                                    local_types,
                                    local_leaf_names,
                                    checked_map,
                                    result_leaf_names,
                                )?),
                            })
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(TypedExpr {
                    ty: type_at_leaf_path(&result_ty, leaf_path)?,
                    kind: TypedExprKind::Call {
                        callee: Callee::Function(callee_leaf_name),
                        args: normalized_args,
                        lifted_shape: lifted_shape.clone(),
                    },
                })
            }
        },
    }
}

fn project_local_path(name: &str, local_ty: &Type, path: &LeafPath) -> Result<TypedExpr> {
    if path.is_root() {
        return Ok(TypedExpr {
            ty: local_ty.clone(),
            kind: TypedExprKind::Local(name.to_string()),
        });
    }
    let mut expr = TypedExpr {
        ty: local_ty.clone(),
        kind: TypedExprKind::Local(name.to_string()),
    };
    for index in 0..path.0.len() {
        let field = &path.0[index];
        let is_tuple_base = matches!(expr.ty, Type::Tuple(_));
        let field_ty = match &expr.ty {
            Type::Record(fields) => fields.get(field).cloned().ok_or_else(|| {
                SimdError::new(format!(
                    "cannot project field '{}' from type {:?} for local '{}'",
                    field, local_ty, name
                ))
            })?,
            Type::Tuple(items) => {
                let Some(tuple_index) = parse_tuple_leaf_name(field) else {
                    return Err(SimdError::new(format!(
                        "cannot project tuple field '{}' from type {:?} for local '{}'",
                        field, local_ty, name
                    )));
                };
                items.get(tuple_index).cloned().ok_or_else(|| {
                    SimdError::new(format!(
                        "cannot project tuple field '{}' from type {:?} for local '{}'",
                        field, local_ty, name
                    ))
                })?
            }
            _ => {
                return Err(SimdError::new(format!(
                    "cannot project field '{}' from type {:?} for local '{}'",
                    field, local_ty, name
                )));
            }
        };
        expr = TypedExpr {
            ty: field_ty,
            kind: if is_tuple_base {
                TypedExprKind::TupleProject {
                    base: Box::new(expr),
                    index: parse_tuple_leaf_name(field).unwrap_or(0),
                }
            } else {
                TypedExprKind::Project {
                    base: Box::new(expr),
                    field: field.clone(),
                }
            },
        };
        if index + 1 == path.0.len() {
            break;
        }
    }
    Ok(expr)
}

fn advance_field_for_projection(base: &TypedExpr, leaf_path: &LeafPath, field: &str) -> LeafPath {
    if leaf_path.is_root() {
        if matches!(base.ty, Type::Record(_) | Type::Tuple(_)) {
            return LeafPath(vec![field.to_string()]);
        }
        return leaf_path.clone();
    }
    let mut path = Vec::with_capacity(leaf_path.0.len() + 1);
    path.push(field.to_string());
    path.extend(leaf_path.0.iter().cloned());
    LeafPath(path)
}

fn wasm_param_abi_from_type(ty: &[Type]) -> Result<Vec<WasmParamAbi>> {
    ty.iter().map(wasm_param_abi_from_single_type).collect()
}

fn wasm_param_abi_from_single_type(ty: &Type) -> Result<WasmParamAbi> {
    match ty {
        Type::Scalar(prim) => Ok(WasmParamAbi::Scalar { prim: *prim }),
        Type::Index(_) => Ok(WasmParamAbi::Scalar { prim: Prim::I64 }),
        Type::Bulk(prim, _) => Ok(WasmParamAbi::Bulk { prim: *prim }),
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _) => wasm_star_seq_scalar_prim(ty)
            .map(|prim| WasmParamAbi::Bulk { prim })
            .ok_or_else(|| {
                SimdError::new("Wasm backend only supports scalar-element T[*] entry parameters")
            }),
        Type::Named(name, args) if is_wasm_string_named_type(name, args) => {
            Ok(WasmParamAbi::Bulk { prim: Prim::Char })
        }
        Type::TypeToken(_) => Err(SimdError::new(
            "Wasm backend does not support Type witness entry parameters",
        )),
        Type::Tuple(items) => Ok(WasmParamAbi::Record {
            fields: items
                .iter()
                .enumerate()
                .map(|(index, item_ty)| {
                    Ok((
                        tuple_leaf_name(index),
                        wasm_param_abi_from_single_type(item_ty)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
        }),
        Type::Record(fields) => Ok(WasmParamAbi::Record {
            fields: fields
                .iter()
                .map(|(name, field_ty)| {
                    Ok((name.clone(), wasm_param_abi_from_single_type(field_ty)?))
                })
                .collect::<Result<Vec<_>>>()?,
        }),
        Type::Fun(_, _) => Err(SimdError::new(
            "Wasm backend does not support higher-order entry parameters",
        )),
        Type::Named(name, args) if is_wasm_enum_named_type(name, args) => {
            Ok(WasmParamAbi::Scalar { prim: Prim::I32 })
        }
        Type::Named(_, _) | Type::Var(_) | Type::Infer(_) => Err(SimdError::new(
            "Wasm backend does not support unresolved polymorphic entry parameters",
        )),
    }
}

fn wasm_result_abi_from_type(ty: &Type, param_types: &[Type]) -> Result<WasmResultAbi> {
    match ty {
        Type::Scalar(prim) => Ok(WasmResultAbi::Scalar { prim: *prim }),
        Type::Index(_) => Ok(WasmResultAbi::Scalar { prim: Prim::I64 }),
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _) => wasm_star_seq_scalar_prim(ty)
            .map(|prim| WasmResultAbi::StarSeq { prim })
            .ok_or_else(|| {
                SimdError::new("Wasm backend only supports scalar-element T[*] entry results")
            }),
        Type::Bulk(prim, _) => Ok(WasmResultAbi::Bulk {
            prim: *prim,
            shape_param: param_types
                .iter()
                .position(type_contains_bulk_leaf)
                .ok_or_else(|| {
                    SimdError::new(
                        "Wasm bulk entry result requires a top-level bulk input shape source",
                    )
                })?,
        }),
        Type::Named(name, args) if is_wasm_string_named_type(name, args) => Err(SimdError::new(
            "Wasm backend does not yet support string entry results",
        )),
        Type::TypeToken(_) => Err(SimdError::new(
            "Wasm backend does not support Type witness entry results",
        )),
        Type::Tuple(items) => Ok(WasmResultAbi::Record {
            fields: items
                .iter()
                .enumerate()
                .map(|(index, item_ty)| {
                    Ok((
                        tuple_leaf_name(index),
                        wasm_result_abi_from_type(item_ty, param_types)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
        }),
        Type::Record(fields) => Ok(WasmResultAbi::Record {
            fields: fields
                .iter()
                .map(|(name, field_ty)| {
                    Ok((
                        name.clone(),
                        wasm_result_abi_from_type(field_ty, param_types)?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?,
        }),
        Type::Fun(_, _) => Err(SimdError::new(
            "Wasm backend does not support higher-order entry results",
        )),
        Type::Named(name, args) if is_wasm_enum_named_type(name, args) => {
            Ok(WasmResultAbi::Scalar { prim: Prim::I32 })
        }
        Type::Named(_, _) | Type::Var(_) | Type::Infer(_) => Err(SimdError::new(
            "Wasm backend does not support unresolved polymorphic entry results",
        )),
    }
}

fn type_contains_bulk_leaf(ty: &Type) -> bool {
    match ty {
        Type::Bulk(_, _) => true,
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _) => false,
        Type::Index(_) => false,
        Type::Named(name, args) if is_wasm_string_named_type(name, args) => true,
        Type::Tuple(items) => items.iter().any(type_contains_bulk_leaf),
        Type::Record(fields) => fields
            .iter()
            .any(|(_, field_ty)| type_contains_bulk_leaf(field_ty)),
        Type::TypeToken(_) => false,
        Type::Scalar(_) | Type::Named(_, _) | Type::Var(_) | Type::Infer(_) | Type::Fun(_, _) => {
            false
        }
    }
}

fn wasm_leaf_result_abi_from_type(ty: &Type) -> Result<WasmLeafResultAbi> {
    match ty {
        Type::Scalar(prim) => Ok(WasmLeafResultAbi::Scalar { prim: *prim }),
        Type::Index(_) => Ok(WasmLeafResultAbi::Scalar { prim: Prim::I64 }),
        Type::StarSeq(_) | Type::StarSeqWitnessed(_, _) => wasm_star_seq_scalar_prim(ty)
            .map(|prim| WasmLeafResultAbi::StarSeq { prim })
            .ok_or_else(|| {
                SimdError::new("leaf result ABI only supports scalar-element T[*] values")
            }),
        Type::Bulk(prim, _) => Ok(WasmLeafResultAbi::Bulk { prim: *prim }),
        Type::Named(name, args) if is_wasm_string_named_type(name, args) => {
            Ok(WasmLeafResultAbi::Bulk { prim: Prim::Char })
        }
        Type::TypeToken(_) => Err(SimdError::new(
            "leaf result ABI cannot contain Type witness values",
        )),
        Type::Tuple(_) | Type::Record(_) => Err(SimdError::new(
            "leaf result ABI cannot contain tuple/record products",
        )),
        Type::Fun(_, _) => Err(SimdError::new("leaf result ABI cannot contain functions")),
        Type::Named(name, args) if is_wasm_enum_named_type(name, args) => {
            Ok(WasmLeafResultAbi::Scalar { prim: Prim::I32 })
        }
        Type::Named(_, _) | Type::Var(_) | Type::Infer(_) => Err(SimdError::new(
            "leaf result ABI cannot contain unresolved polymorphic types",
        )),
    }
}

fn leaf_export_name(source_name: &str, leaf_path: &LeafPath) -> String {
    if leaf_path.is_root() {
        source_name.to_string()
    } else {
        let mut name = source_name.to_string();
        name.push('$');
        name.push_str(&leaf_path.suffix());
        name
    }
}

fn leaf_symbol_name(base: &str, leaf_path: &LeafPath) -> String {
    if leaf_path.is_root() {
        base.to_string()
    } else {
        let mut name = base.to_string();
        name.push('$');
        name.push_str(&leaf_path.suffix());
        name
    }
}

fn type_at_leaf_path(ty: &Type, leaf_path: &LeafPath) -> Result<Type> {
    if leaf_path.is_root() {
        return match ty {
            Type::Scalar(_) | Type::Bulk(_, _) => Ok(ty.clone()),
            Type::Index(_) => Ok(ty.clone()),
            Type::StarSeq(_) | Type::StarSeqWitnessed(_, _) => Ok(ty.clone()),
            Type::Named(name, args) if is_wasm_string_named_type(name, args) => Ok(ty.clone()),
            Type::TypeToken(_) => Err(SimdError::new(
                "Type witness values cannot be used as leaf values",
            )),
            Type::Tuple(_) | Type::Record(_) => Err(SimdError::new(
                "tuple/record type requires a non-empty leaf path",
            )),
            Type::Fun(_, _) => Err(SimdError::new(
                "function types cannot be used as leaf values",
            )),
            Type::Named(name, args) if is_wasm_enum_named_type(name, args) => Ok(ty.clone()),
            Type::Named(_, _) | Type::Var(_) | Type::Infer(_) => Err(SimdError::new(
                "unresolved polymorphic types cannot be used as leaf values",
            )),
        };
    }
    let Some((head, tail)) = leaf_path.split_first() else {
        unreachable!("non-root leaf path must have a head");
    };
    match ty {
        Type::Tuple(items) => {
            let Some(index) = parse_tuple_leaf_name(head) else {
                return Err(SimdError::new(format!(
                    "tuple leaf '{}' is not a valid tuple index",
                    head
                )));
            };
            let item = items
                .get(index)
                .ok_or_else(|| SimdError::new(format!("tuple type is missing field '{}'", head)))?;
            type_at_leaf_path(item, &tail)
        }
        Type::Record(fields) => {
            let field = fields.get(head).ok_or_else(|| {
                SimdError::new(format!("record type is missing field '{}'", head))
            })?;
            type_at_leaf_path(field, &tail)
        }
        _ => Err(SimdError::new(format!(
            "cannot select field '{}' from non-product type {:?}",
            head, ty
        ))),
    }
}

fn is_wasm_enum_named_type(name: &str, args: &[Type]) -> bool {
    !(args.is_empty() && (name == "string" || name == "bool"))
}

fn ensure_wasmtime_cache_config() -> Result<String> {
    let cache_root = if let Some(root) = std::env::var_os("SIMD_WASMTIME_CACHE_ROOT") {
        std::path::PathBuf::from(root)
    } else {
        let cwd = std::env::current_dir().map_err(|error| {
            SimdError::new(format!("failed to read current directory: {}", error))
        })?;
        cwd.join(".tmp/wasmtime-cache")
    };
    let cache_dir = cache_root.join("artifacts");
    fs::create_dir_all(&cache_dir).map_err(|error| {
        SimdError::new(format!(
            "failed to create Wasmtime cache directory '{}': {}",
            cache_dir.display(),
            error
        ))
    })?;
    let config_path = cache_root.join("config.toml");
    let config_text = format!("[cache]\ndirectory = '{}'\n", cache_dir.display());
    let should_write = match fs::read_to_string(&config_path) {
        Ok(existing) => existing != config_text,
        Err(_) => true,
    };
    if should_write {
        fs::write(&config_path, config_text).map_err(|error| {
            SimdError::new(format!(
                "failed to write Wasmtime cache config '{}': {}",
                config_path.display(),
                error
            ))
        })?;
    }
    Ok(config_path.display().to_string())
}

fn build_engine() -> Result<Engine> {
    let mut config = Config::new();
    config.wasm_simd(true);
    config.wasm_tail_call(true);
    config.cranelift_opt_level(OptLevel::Speed);
    let cache_config_path = ensure_wasmtime_cache_config()?;
    config.cache(Some(
        Cache::from_file(Some(std::path::Path::new(&cache_config_path))).map_err(|error| {
            SimdError::new(format!("failed to load Wasmtime cache config: {error}"))
        })?,
    ));
    Engine::new(&config).map_err(|error| SimdError::new(format!("wasmtime engine error: {error}")))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum UsageCount {
    Zero,
    One,
    Many,
}

fn summarize_param_uses(program: &LoweredProgram) -> BTreeMap<String, Vec<UsageCount>> {
    let functions = program
        .functions
        .iter()
        .map(|function| (function.name.clone(), function))
        .collect::<BTreeMap<_, _>>();
    let mut memo = BTreeMap::<String, Vec<UsageCount>>::new();
    let mut visiting = Vec::<String>::new();
    for name in functions.keys() {
        let _ = summarize_function_param_uses(name, &functions, &mut memo, &mut visiting);
    }
    memo
}

fn summarize_function_param_uses(
    name: &str,
    functions: &BTreeMap<String, &LoweredFunction>,
    memo: &mut BTreeMap<String, Vec<UsageCount>>,
    visiting: &mut Vec<String>,
) -> Vec<UsageCount> {
    if let Some(summary) = memo.get(name) {
        return summary.clone();
    }
    let Some(function) = functions.get(name).copied() else {
        return Vec::new();
    };
    let arity = function.param_access.len();
    if visiting.iter().any(|item| item == name) {
        return vec![UsageCount::Many; arity];
    }
    visiting.push(name.to_string());

    let mut summary = vec![UsageCount::Zero; arity];
    let clauses = match &function.kind {
        LoweredKind::Scalar { clauses }
        | LoweredKind::Structural { clauses, .. }
        | LoweredKind::Kernel { clauses, .. } => clauses,
    };
    for clause in clauses {
        let clause_summary = summarize_clause_param_uses(clause, arity, functions, memo, visiting);
        join_usage_vec(&mut summary, &clause_summary);
    }

    visiting.pop();
    memo.insert(name.to_string(), summary.clone());
    summary
}

fn summarize_clause_param_uses(
    clause: &LoweredClause,
    arity: usize,
    functions: &BTreeMap<String, &LoweredFunction>,
    memo: &mut BTreeMap<String, Vec<UsageCount>>,
    visiting: &mut Vec<String>,
) -> Vec<UsageCount> {
    let mut locals = BTreeMap::<String, usize>::new();
    for (index, pattern) in clause.patterns.iter().enumerate() {
        if let Pattern::Name(name) = &pattern.pattern {
            locals.insert(name.clone(), index);
        }
    }
    summarize_ir_param_uses(&clause.body, arity, &locals, functions, memo, visiting)
}

fn summarize_ir_param_uses(
    expr: &IrExpr,
    arity: usize,
    locals: &BTreeMap<String, usize>,
    functions: &BTreeMap<String, &LoweredFunction>,
    memo: &mut BTreeMap<String, Vec<UsageCount>>,
    visiting: &mut Vec<String>,
) -> Vec<UsageCount> {
    summarize_ir_param_uses_with_lets(
        expr,
        arity,
        locals,
        &BTreeMap::new(),
        functions,
        memo,
        visiting,
    )
}

fn summarize_ir_param_uses_with_lets(
    expr: &IrExpr,
    arity: usize,
    locals: &BTreeMap<String, usize>,
    let_bindings: &BTreeMap<String, Vec<UsageCount>>,
    functions: &BTreeMap<String, &LoweredFunction>,
    memo: &mut BTreeMap<String, Vec<UsageCount>>,
    visiting: &mut Vec<String>,
) -> Vec<UsageCount> {
    match &expr.kind {
        IrExprKind::Local(name) => {
            if let Some(summary) = let_bindings.get(name) {
                summary.clone()
            } else {
                let mut summary = vec![UsageCount::Zero; arity];
                if let Some(index) = locals.get(name) {
                    summary[*index] = UsageCount::One;
                }
                summary
            }
        }
        IrExprKind::Int(_, _) | IrExprKind::Float(_, _) => vec![UsageCount::Zero; arity],
        IrExprKind::Record(fields) => {
            let mut summary = vec![UsageCount::Zero; arity];
            for field in fields.values() {
                let field_summary = summarize_ir_param_uses_with_lets(
                    field,
                    arity,
                    locals,
                    let_bindings,
                    functions,
                    memo,
                    visiting,
                );
                add_usage_vec(&mut summary, &field_summary);
            }
            summary
        }
        IrExprKind::EnumCtor { args, .. } => {
            let mut summary = vec![UsageCount::Zero; arity];
            for arg in args {
                let arg_summary = summarize_ir_param_uses_with_lets(
                    arg,
                    arity,
                    locals,
                    let_bindings,
                    functions,
                    memo,
                    visiting,
                );
                add_usage_vec(&mut summary, &arg_summary);
            }
            summary
        }
        IrExprKind::Seq(items) => {
            let mut summary = vec![UsageCount::Zero; arity];
            for item in items {
                let item_summary = summarize_ir_param_uses_with_lets(
                    item,
                    arity,
                    locals,
                    let_bindings,
                    functions,
                    memo,
                    visiting,
                );
                add_usage_vec(&mut summary, &item_summary);
            }
            summary
        }
        IrExprKind::SeqSplice { prefix, tail } => {
            let mut summary = vec![UsageCount::Zero; arity];
            for item in prefix {
                let item_summary = summarize_ir_param_uses_with_lets(
                    item,
                    arity,
                    locals,
                    let_bindings,
                    functions,
                    memo,
                    visiting,
                );
                add_usage_vec(&mut summary, &item_summary);
            }
            let tail_summary = summarize_ir_param_uses_with_lets(
                tail,
                arity,
                locals,
                let_bindings,
                functions,
                memo,
                visiting,
            );
            add_usage_vec(&mut summary, &tail_summary);
            summary
        }
        IrExprKind::EnumTag { value }
        | IrExprKind::EnumChildBySlot { value, .. }
        | IrExprKind::EnumNonRecField { value, .. } => summarize_ir_param_uses_with_lets(
            value,
            arity,
            locals,
            let_bindings,
            functions,
            memo,
            visiting,
        ),
        IrExprKind::Let { bindings, body } => {
            let mut extended = let_bindings.clone();
            for binding in bindings {
                let binding_summary = summarize_ir_param_uses_with_lets(
                    &binding.expr,
                    arity,
                    locals,
                    &extended,
                    functions,
                    memo,
                    visiting,
                );
                extended.insert(binding.name.clone(), binding_summary);
            }
            summarize_ir_param_uses_with_lets(
                body, arity, locals, &extended, functions, memo, visiting,
            )
        }
        IrExprKind::Call { callee, args } => match callee {
            Callee::Prim(_) => {
                let mut summary = vec![UsageCount::Zero; arity];
                for arg in args {
                    let arg_summary = summarize_ir_param_uses_with_lets(
                        arg,
                        arity,
                        locals,
                        let_bindings,
                        functions,
                        memo,
                        visiting,
                    );
                    add_usage_vec(&mut summary, &arg_summary);
                }
                summary
            }
            Callee::Function(name) => {
                let callee_summary = summarize_function_param_uses(name, functions, memo, visiting);
                let mut summary = vec![UsageCount::Zero; arity];
                for (index, arg) in args.iter().enumerate() {
                    let arg_summary = summarize_ir_param_uses_with_lets(
                        arg,
                        arity,
                        locals,
                        let_bindings,
                        functions,
                        memo,
                        visiting,
                    );
                    let usage = callee_summary
                        .get(index)
                        .copied()
                        .unwrap_or(UsageCount::Many);
                    let scaled = scale_usage_vec(&arg_summary, usage);
                    add_usage_vec(&mut summary, &scaled);
                }
                summary
            }
            Callee::Builtin(_) => {
                vec![UsageCount::Many; arity]
            }
        },
    }
}

fn analyze_leaf_buffer_reuse(
    flat_param_abis: &[WasmParamAbi],
    result: &WasmLeafResultAbi,
    param_summary: &[UsageCount],
) -> Result<(Vec<usize>, Option<usize>)> {
    let WasmLeafResultAbi::Bulk { prim: result_prim } = result else {
        return Ok((Vec::new(), None));
    };

    let used_param_leaves = param_summary
        .iter()
        .enumerate()
        .filter_map(|(index, usage)| (*usage != UsageCount::Zero).then_some(index))
        .collect::<Vec<_>>();
    let reusable_param_leaf =
        flat_param_abis
            .iter()
            .enumerate()
            .find_map(|(index, abi)| match abi {
                WasmParamAbi::Bulk { prim }
                    if *prim == *result_prim
                        && matches!(param_summary.get(index), Some(UsageCount::One)) =>
                {
                    Some(index)
                }
                _ => None,
            });
    Ok((used_param_leaves, reusable_param_leaf))
}

fn add_usage_vec(into: &mut [UsageCount], other: &[UsageCount]) {
    for (slot, value) in into.iter_mut().zip(other.iter().copied()) {
        *slot = add_usage(*slot, value);
    }
}

fn scale_usage_vec(values: &[UsageCount], scale: UsageCount) -> Vec<UsageCount> {
    values
        .iter()
        .copied()
        .map(|value| scale_usage(value, scale))
        .collect()
}

fn join_usage_vec(into: &mut [UsageCount], other: &[UsageCount]) {
    for (slot, value) in into.iter_mut().zip(other.iter().copied()) {
        *slot = join_usage(*slot, value);
    }
}

fn add_usage(left: UsageCount, right: UsageCount) -> UsageCount {
    match (left, right) {
        (UsageCount::Zero, other) | (other, UsageCount::Zero) => other,
        (UsageCount::One, UsageCount::One) => UsageCount::Many,
        _ => UsageCount::Many,
    }
}

fn scale_usage(value: UsageCount, scale: UsageCount) -> UsageCount {
    match scale {
        UsageCount::Zero => UsageCount::Zero,
        UsageCount::One => value,
        UsageCount::Many => match value {
            UsageCount::Zero => UsageCount::Zero,
            UsageCount::One | UsageCount::Many => UsageCount::Many,
        },
    }
}

fn join_usage(left: UsageCount, right: UsageCount) -> UsageCount {
    std::cmp::max(left, right)
}

#[derive(Debug, Clone)]
struct LeafInputLayout {
    wasm_args: Vec<Val>,
    bulk_ptrs: Vec<Option<usize>>,
}

#[derive(Debug, Clone)]
struct BulkShapeInfo {
    shape: Vec<usize>,
    len: usize,
}

fn execute_wasm_artifact_grouped_in_runtime(
    runtime: &mut WasmRuntime,
    artifact: &WasmArtifact,
    args: &[Value],
) -> Result<Value> {
    if args.len() != artifact.params.len() {
        return Err(SimdError::new(format!(
            "Wasm entry expects {} arguments, found {}",
            artifact.params.len(),
            args.len()
        )));
    }

    let flat_param_abis = flatten_wasm_param_abis(&artifact.params)?;
    let (flat_args, bulk_info) = flatten_wasm_args(&artifact.params, args)?;
    runtime.begin_run();

    let memory = runtime.memory.clone();
    let leaf_funcs = artifact
        .leaf_exports
        .iter()
        .map(|leaf| runtime_func(runtime, &leaf.export_name))
        .collect::<Result<Vec<_>>>()?;
    let input_layout = build_leaf_input_args(runtime, &flat_param_abis, &flat_args)?;
    let mut remaining_uses = vec![0usize; flat_args.len()];
    for leaf in &artifact.leaf_exports {
        for index in &leaf.used_param_leaves {
            remaining_uses[*index] += 1;
        }
    }
    let mut leaf_values = BTreeMap::<LeafPath, Value>::new();

    for (leaf, func) in artifact.leaf_exports.iter().zip(leaf_funcs.iter()) {
        let mut wasm_args = input_layout.wasm_args.clone();
        let mut results = match leaf.result {
            WasmLeafResultAbi::Scalar { .. } | WasmLeafResultAbi::StarSeq { .. } => {
                vec![Val::I64(0)]
            }
            WasmLeafResultAbi::Bulk { .. } => Vec::new(),
        };
        let mut output_ptr = None::<usize>;
        let mut output_len = None::<usize>;
        if matches!(leaf.result, WasmLeafResultAbi::Bulk { .. }) {
            let bulk = bulk_info.as_ref().ok_or_else(|| {
                SimdError::new("Wasm bulk leaf result requires at least one bulk input")
            })?;
            let output_base = if leaf
                .reusable_param_leaf
                .is_some_and(|index| remaining_uses[index] == 1)
            {
                let index = leaf.reusable_param_leaf.unwrap();
                input_layout
                    .bulk_ptrs
                    .get(index)
                    .and_then(|ptr| *ptr)
                    .ok_or_else(|| {
                        SimdError::new(format!(
                            "reusable bulk input {} did not have an allocated pointer",
                            index
                        ))
                    })?
            } else {
                ensure_output_bulk_buffer(
                    runtime,
                    &leaf.leaf_path,
                    leaf_result_prim(&leaf.result),
                    bulk,
                )?
            };
            wasm_args.push(Val::I32(i32::try_from(output_base).map_err(|_| {
                SimdError::new("bulk output pointer does not fit in i32")
            })?));
            wasm_args.push(Val::I32(i32::try_from(bulk.len).map_err(|_| {
                SimdError::new("bulk output length does not fit in i32")
            })?));
            output_ptr = Some(output_base);
            output_len = Some(bulk.len);
        }
        runtime.sync_enum_heap_ptr()?;
        func.call(&mut runtime.store, &wasm_args, &mut results)
            .map_err(|error| SimdError::new(format!("Wasm execution failed: {error}")))?;
        let value = match leaf.result {
            WasmLeafResultAbi::Scalar { prim } => {
                let value = results
                    .into_iter()
                    .next()
                    .ok_or_else(|| SimdError::new("Wasm scalar leaf did not produce a result"))?;
                Value::Scalar(wasmtime_to_scalar(value, prim)?)
            }
            WasmLeafResultAbi::StarSeq { prim } => {
                let value = results
                    .into_iter()
                    .next()
                    .ok_or_else(|| SimdError::new("Wasm T[*] leaf did not produce a handle"))?;
                let Val::I32(handle) = value else {
                    return Err(SimdError::new(
                        "Wasm T[*] leaf did not produce an i32 handle",
                    ));
                };
                read_wasm_star_seq_from_handle(&memory, &runtime.store, handle, prim)?
            }
            WasmLeafResultAbi::Bulk { prim } => {
                let bulk = bulk_info.as_ref().ok_or_else(|| {
                    SimdError::new("Wasm bulk leaf result requires a bulk input shape")
                })?;
                let ptr =
                    output_ptr.ok_or_else(|| SimdError::new("missing bulk output pointer"))?;
                let len = output_len.ok_or_else(|| SimdError::new("missing bulk output length"))?;
                Value::Bulk(read_bulk_from_memory(
                    &memory,
                    &runtime.store,
                    ptr,
                    len,
                    prim,
                    bulk.shape.clone(),
                )?)
            }
        };
        leaf_values.insert(leaf.leaf_path.clone(), value);
        for index in &leaf.used_param_leaves {
            remaining_uses[*index] -= 1;
        }
    }

    rebuild_wasm_value_from_abi(&artifact.result, &leaf_values, &LeafPath::root())
}

fn execute_wasm_artifact_grouped_export_in_runtime(
    runtime: &mut WasmRuntime,
    artifact: &WasmArtifact,
    args: &[Value],
) -> Result<Value> {
    let grouped = artifact
        .grouped_export
        .as_ref()
        .ok_or_else(|| SimdError::new("missing grouped export metadata"))?;

    if args.len() != artifact.params.len() {
        return Err(SimdError::new(format!(
            "Wasm entry expects {} arguments, found {}",
            artifact.params.len(),
            args.len()
        )));
    }

    let flat_param_abis = flatten_wasm_param_abis(&artifact.params)?;
    let (flat_args, bulk_info) = flatten_wasm_args(&artifact.params, args)?;
    runtime.begin_run();

    let memory = runtime.memory.clone();
    let func = runtime_func(runtime, &grouped.export_name)?;
    let input_layout = build_leaf_input_args(runtime, &flat_param_abis, &flat_args)?;
    let mut remaining_uses = vec![0usize; flat_args.len()];
    for leaf in &grouped.leaf_exports {
        for index in &leaf.used_param_leaves {
            remaining_uses[*index] += 1;
        }
    }

    let mut wasm_args = input_layout.wasm_args.clone();
    let mut output_layouts = Vec::<(LeafPath, WasmLeafResultAbi, usize, usize)>::new();

    for leaf in &grouped.leaf_exports {
        match leaf.result {
            WasmLeafResultAbi::Scalar { prim } => {
                let output_base = ensure_output_scalar_buffer(runtime, &leaf.leaf_path, prim)?;
                wasm_args.push(Val::I32(i32::try_from(output_base).map_err(|_| {
                    SimdError::new("scalar output pointer does not fit in i32")
                })?));
                output_layouts.push((leaf.leaf_path.clone(), leaf.result.clone(), output_base, 1));
            }
            WasmLeafResultAbi::Bulk { prim } => {
                let bulk = bulk_info.as_ref().ok_or_else(|| {
                    SimdError::new("Wasm grouped bulk result requires at least one bulk input")
                })?;
                let output_base = if leaf
                    .reusable_param_leaf
                    .is_some_and(|index| remaining_uses[index] == 1)
                {
                    let index = leaf.reusable_param_leaf.unwrap();
                    input_layout
                        .bulk_ptrs
                        .get(index)
                        .and_then(|ptr| *ptr)
                        .ok_or_else(|| {
                            SimdError::new(format!(
                                "reusable bulk input {} did not have an allocated pointer",
                                index
                            ))
                        })?
                } else {
                    ensure_output_bulk_buffer(runtime, &leaf.leaf_path, prim, bulk)?
                };
                wasm_args.push(Val::I32(i32::try_from(output_base).map_err(|_| {
                    SimdError::new("bulk output pointer does not fit in i32")
                })?));
                wasm_args.push(Val::I32(i32::try_from(bulk.len).map_err(|_| {
                    SimdError::new("bulk output length does not fit in i32")
                })?));
                output_layouts.push((
                    leaf.leaf_path.clone(),
                    leaf.result.clone(),
                    output_base,
                    bulk.len,
                ));
            }
            WasmLeafResultAbi::StarSeq { .. } => {
                return Err(SimdError::new(
                    "grouped Wasm export does not yet support T[*] leaf outputs",
                ));
            }
        }
    }

    runtime.sync_enum_heap_ptr()?;
    func.call(&mut runtime.store, &wasm_args, &mut [])
        .map_err(|error| SimdError::new(format!("Wasm grouped execution failed: {error}")))?;

    let mut leaf_values = BTreeMap::<LeafPath, Value>::new();
    for (leaf_path, result, output_base, output_len) in output_layouts {
        let value = match result {
            WasmLeafResultAbi::Scalar { prim } => Value::Scalar(read_scalar_from_memory(
                &memory,
                &runtime.store,
                output_base,
                prim,
            )?),
            WasmLeafResultAbi::StarSeq { .. } => {
                return Err(SimdError::new(
                    "grouped Wasm export does not yet support T[*] leaf outputs",
                ));
            }
            WasmLeafResultAbi::Bulk { prim } => {
                let bulk = bulk_info.as_ref().ok_or_else(|| {
                    SimdError::new("Wasm grouped bulk result requires a bulk input shape")
                })?;
                Value::Bulk(read_bulk_from_memory(
                    &memory,
                    &runtime.store,
                    output_base,
                    output_len,
                    prim,
                    bulk.shape.clone(),
                )?)
            }
        };
        leaf_values.insert(leaf_path, value);
    }

    rebuild_wasm_value_from_abi(&artifact.result, &leaf_values, &LeafPath::root())
}

fn build_leaf_input_args(
    runtime: &mut WasmRuntime,
    flat_param_abis: &[WasmParamAbi],
    flat_args: &[Value],
) -> Result<LeafInputLayout> {
    let mut wasm_args = Vec::new();
    let mut bulk_ptrs = Vec::with_capacity(flat_args.len());
    for (index, (abi, value)) in flat_param_abis.iter().zip(flat_args).enumerate() {
        match (abi, value) {
            (WasmParamAbi::Scalar { prim }, Value::Scalar(value)) => {
                wasm_args.push(scalar_to_wasmtime(value, *prim)?);
                bulk_ptrs.push(None);
            }
            (WasmParamAbi::Bulk { prim }, Value::Bulk(bulk)) => {
                if bulk.prim != *prim {
                    return Err(SimdError::new(format!(
                        "Wasm bulk argument expected {:?}, found {:?}",
                        prim, bulk.prim
                    )));
                }
                let ptr = ensure_input_bulk_buffer(runtime, index, bulk)?;
                write_bulk_to_memory(&runtime.memory, &mut runtime.store, ptr, bulk)?;
                wasm_args.push(Val::I32(i32::try_from(ptr).map_err(|_| {
                    SimdError::new("bulk input pointer does not fit in i32")
                })?));
                wasm_args.push(Val::I32(i32::try_from(bulk.elements.len()).map_err(
                    |_| SimdError::new("bulk input length does not fit in i32"),
                )?));
                bulk_ptrs.push(Some(ptr));
            }
            (WasmParamAbi::Bulk { prim }, Value::StarSeq(seq)) => {
                let elements = seq
                    .items
                    .iter()
                    .map(|item| match item {
                        Value::Scalar(value) if value.prim() == *prim => Ok(value.clone()),
                        Value::Scalar(value) => Err(SimdError::new(format!(
                            "Wasm T[*] argument expected {:?} elements, found {:?}",
                            prim,
                            value.prim()
                        ))),
                        other => Err(SimdError::new(format!(
                            "Wasm backend only supports scalar-element T[*] runtime arguments, found {:?}",
                            other
                        ))),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let bulk = BulkValue {
                    prim: *prim,
                    shape: vec![elements.len()],
                    elements,
                };
                let ptr = ensure_input_bulk_buffer(runtime, index, &bulk)?;
                write_bulk_to_memory(&runtime.memory, &mut runtime.store, ptr, &bulk)?;
                wasm_args.push(Val::I32(i32::try_from(ptr).map_err(|_| {
                    SimdError::new("bulk input pointer does not fit in i32")
                })?));
                wasm_args.push(Val::I32(i32::try_from(bulk.elements.len()).map_err(
                    |_| SimdError::new("bulk input length does not fit in i32"),
                )?));
                bulk_ptrs.push(Some(ptr));
            }
            (WasmParamAbi::Scalar { prim }, other) => {
                return Err(SimdError::new(format!(
                    "Wasm scalar argument expected {:?}, found {:?}",
                    prim, other
                )));
            }
            (WasmParamAbi::Bulk { prim }, other) => {
                return Err(SimdError::new(format!(
                    "Wasm bulk argument expected {:?}, found {:?}",
                    prim, other
                )));
            }
            (WasmParamAbi::Record { .. }, _) => {
                return Err(SimdError::new(
                    "flattened Wasm arguments should not contain record ABIs",
                ));
            }
        }
    }
    Ok(LeafInputLayout {
        wasm_args,
        bulk_ptrs,
    })
}

fn ensure_input_bulk_buffer(
    runtime: &mut WasmRuntime,
    param_leaf: usize,
    bulk: &BulkValue,
) -> Result<usize> {
    let key = InputBufferKey {
        param_leaf,
        prim: bulk.prim,
        shape: bulk.shape.clone(),
    };
    if let Some(ptr) = runtime.input_buffers.get(&key) {
        return Ok(*ptr);
    }
    let byte_len = bulk.elements.len() * byte_width(bulk.prim) as usize;
    let ptr = runtime.alloc_pinned(byte_len)?;
    runtime.input_buffers.insert(key, ptr);
    Ok(ptr)
}

fn ensure_output_scalar_buffer(
    runtime: &mut WasmRuntime,
    leaf_path: &LeafPath,
    prim: Prim,
) -> Result<usize> {
    let key = OutputBufferKey {
        leaf_path: leaf_path.clone(),
        prim,
        kind: OutputBufferKind::Scalar,
        shape: Vec::new(),
    };
    if let Some(ptr) = runtime.output_buffers.get(&key) {
        return Ok(*ptr);
    }
    let ptr = runtime.alloc_pinned(byte_width(prim) as usize)?;
    runtime.output_buffers.insert(key, ptr);
    Ok(ptr)
}

fn ensure_output_bulk_buffer(
    runtime: &mut WasmRuntime,
    leaf_path: &LeafPath,
    prim: Prim,
    bulk: &BulkShapeInfo,
) -> Result<usize> {
    let key = OutputBufferKey {
        leaf_path: leaf_path.clone(),
        prim,
        kind: OutputBufferKind::Bulk,
        shape: bulk.shape.clone(),
    };
    if let Some(ptr) = runtime.output_buffers.get(&key) {
        return Ok(*ptr);
    }
    let byte_len = bulk.len * byte_width(prim) as usize;
    let ptr = runtime.alloc_pinned(byte_len)?;
    runtime.output_buffers.insert(key, ptr);
    Ok(ptr)
}

fn flatten_wasm_param_abis(params: &[WasmParamAbi]) -> Result<Vec<WasmParamAbi>> {
    let mut flat = Vec::new();
    for param in params {
        flatten_wasm_param_abi(param, &mut flat)?;
    }
    Ok(flat)
}

fn flatten_wasm_param_abi(param: &WasmParamAbi, out: &mut Vec<WasmParamAbi>) -> Result<()> {
    match param {
        WasmParamAbi::Scalar { prim } => out.push(WasmParamAbi::Scalar { prim: *prim }),
        WasmParamAbi::Bulk { prim } => out.push(WasmParamAbi::Bulk { prim: *prim }),
        WasmParamAbi::Record { fields } => {
            for (_, field) in fields {
                flatten_wasm_param_abi(field, out)?;
            }
        }
    }
    Ok(())
}

fn flatten_wasm_args(
    params: &[WasmParamAbi],
    args: &[Value],
) -> Result<(Vec<Value>, Option<BulkShapeInfo>)> {
    let mut flat = Vec::new();
    let mut bulk_info = None::<BulkShapeInfo>;
    for (value, abi) in args.iter().zip(params) {
        flatten_wasm_value(value, abi, &mut flat, &mut bulk_info)?;
    }
    Ok((flat, bulk_info))
}

fn flatten_wasm_value(
    value: &Value,
    abi: &WasmParamAbi,
    out: &mut Vec<Value>,
    bulk_info: &mut Option<BulkShapeInfo>,
) -> Result<()> {
    let register_bulk_shape =
        |bulk: &BulkValue, bulk_info: &mut Option<BulkShapeInfo>| -> Result<()> {
            match bulk_info {
                None => {
                    *bulk_info = Some(BulkShapeInfo {
                        shape: bulk.shape.clone(),
                        len: bulk.elements.len(),
                    });
                }
                Some(existing)
                    if existing.shape == bulk.shape && existing.len == bulk.elements.len() => {}
                Some(existing) => {
                    return Err(SimdError::new(format!(
                        "Wasm bulk arguments must share a shape, found {:?} and {:?}",
                        &existing.shape, &bulk.shape
                    )));
                }
            }
            Ok(())
        };
    match (value, abi) {
        (Value::Scalar(_), WasmParamAbi::Scalar { .. }) => out.push(value.clone()),
        (Value::String(text), WasmParamAbi::Bulk { prim }) if *prim == Prim::Char => {
            let elements = text.chars().map(ScalarValue::Char).collect::<Vec<_>>();
            let bulk = BulkValue {
                prim: Prim::Char,
                shape: vec![elements.len()],
                elements,
            };
            register_bulk_shape(&bulk, bulk_info)?;
            out.push(Value::Bulk(bulk));
        }
        (Value::Bulk(bulk), WasmParamAbi::Bulk { prim }) => {
            if bulk.prim != *prim {
                return Err(SimdError::new(format!(
                    "Wasm bulk argument expected {:?}, found {:?}",
                    prim, bulk.prim
                )));
            }
            register_bulk_shape(bulk, bulk_info)?;
            out.push(value.clone());
        }
        (Value::StarSeq(seq), WasmParamAbi::Bulk { prim }) => {
            let elements = seq
                .items
                .iter()
                .map(|item| match item {
                    Value::Scalar(value) if value.prim() == *prim => Ok(value.clone()),
                    Value::Scalar(value) => Err(SimdError::new(format!(
                        "Wasm T[*] argument expected {:?} elements, found {:?}",
                        prim,
                        value.prim()
                    ))),
                    other => Err(SimdError::new(format!(
                        "Wasm backend only supports scalar-element T[*] runtime arguments, found {:?}",
                        other
                    ))),
                })
                .collect::<Result<Vec<_>>>()?;
            let bulk = BulkValue {
                prim: *prim,
                shape: vec![elements.len()],
                elements,
            };
            register_bulk_shape(&bulk, bulk_info)?;
            out.push(Value::Bulk(bulk));
        }
        (Value::Record(fields), WasmParamAbi::Record { fields: abi_fields }) => {
            for (name, field_abi) in abi_fields {
                let field_value = fields.get(name).ok_or_else(|| {
                    SimdError::new(format!(
                        "record runtime argument is missing field '{}'",
                        name
                    ))
                })?;
                flatten_wasm_value(field_value, field_abi, out, bulk_info)?;
            }
        }
        (Value::Tuple(items), WasmParamAbi::Record { fields: abi_fields }) => {
            for (name, field_abi) in abi_fields {
                let Some(index) = parse_tuple_leaf_name(name) else {
                    return Err(SimdError::new(format!(
                        "tuple Wasm ABI field '{}' is not a numeric tuple index",
                        name
                    )));
                };
                let field_value = items.get(index).ok_or_else(|| {
                    SimdError::new(format!(
                        "tuple runtime argument is missing element {}",
                        index
                    ))
                })?;
                flatten_wasm_value(field_value, field_abi, out, bulk_info)?;
            }
        }
        (Value::Scalar(_), WasmParamAbi::Bulk { prim }) => {
            return Err(SimdError::new(format!(
                "Wasm bulk argument expected {:?}, found scalar",
                prim
            )));
        }
        (Value::Bulk(bulk), WasmParamAbi::Scalar { prim }) => {
            return Err(SimdError::new(format!(
                "Wasm scalar argument expected {:?}, found {:?}",
                prim, bulk.prim
            )));
        }
        (Value::Record(_), WasmParamAbi::Scalar { prim }) => {
            return Err(SimdError::new(format!(
                "Wasm scalar argument expected {:?}, found record",
                prim
            )));
        }
        (Value::Record(_), WasmParamAbi::Bulk { prim }) => {
            return Err(SimdError::new(format!(
                "Wasm bulk argument expected {:?}, found record",
                prim
            )));
        }
        (Value::Tuple(_), WasmParamAbi::Scalar { prim }) => {
            return Err(SimdError::new(format!(
                "Wasm scalar argument expected {:?}, found tuple",
                prim
            )));
        }
        (Value::Tuple(_), WasmParamAbi::Bulk { prim }) => {
            return Err(SimdError::new(format!(
                "Wasm bulk argument expected {:?}, found tuple",
                prim
            )));
        }
        (Value::StarSeq(_), WasmParamAbi::Scalar { .. })
        | (Value::StarSeq(_), WasmParamAbi::Record { .. }) => {
            return Err(SimdError::new(
                "Wasm backend only supports scalar-element T[*] runtime arguments",
            ));
        }
        (Value::String(_), WasmParamAbi::Scalar { prim }) => {
            return Err(SimdError::new(format!(
                "Wasm scalar argument expected {:?}, found string",
                prim
            )));
        }
        (Value::String(_), WasmParamAbi::Bulk { prim }) => {
            return Err(SimdError::new(format!(
                "Wasm bulk argument expected {:?}, found string",
                prim
            )));
        }
        (Value::String(_), WasmParamAbi::Record { .. }) => {
            return Err(SimdError::new(
                "Wasm record argument expected a record value, found string",
            ));
        }
        (Value::Bool(_), WasmParamAbi::Scalar { prim }) => {
            return Err(SimdError::new(format!(
                "Wasm scalar argument expected {:?}, found bool",
                prim
            )));
        }
        (Value::Bool(_), WasmParamAbi::Bulk { prim }) => {
            return Err(SimdError::new(format!(
                "Wasm bulk argument expected {:?}, found bool",
                prim
            )));
        }
        (Value::Bool(_), WasmParamAbi::Record { .. }) => {
            return Err(SimdError::new(
                "Wasm record argument expected a record value, found bool",
            ));
        }
        (Value::TypeToken(_), WasmParamAbi::Scalar { .. })
        | (Value::TypeToken(_), WasmParamAbi::Bulk { .. }) => {
            return Err(SimdError::new(
                "Wasm backend does not support runtime type witness arguments",
            ));
        }
        (Value::Enum(_), WasmParamAbi::Scalar { prim }) => {
            return Err(SimdError::new(format!(
                "Wasm scalar argument expected {:?}, found enum",
                prim
            )));
        }
        (Value::Enum(_), WasmParamAbi::Bulk { prim }) => {
            return Err(SimdError::new(format!(
                "Wasm bulk argument expected {:?}, found enum",
                prim
            )));
        }
        (value, WasmParamAbi::Record { .. }) => {
            return Err(SimdError::new(format!(
                "record/tuple Wasm ABI expected a product value, found {:?}",
                value
            )));
        }
    }
    Ok(())
}

fn rebuild_wasm_value_from_abi(
    abi: &WasmResultAbi,
    leaves: &BTreeMap<LeafPath, Value>,
    prefix: &LeafPath,
) -> Result<Value> {
    match abi {
        WasmResultAbi::Scalar { .. }
        | WasmResultAbi::StarSeq { .. }
        | WasmResultAbi::Bulk { .. } => leaves
            .get(prefix)
            .cloned()
            .ok_or_else(|| SimdError::new(format!("missing leaf result for path {:?}", prefix))),
        WasmResultAbi::Record { fields } => {
            let tuple_like = fields
                .iter()
                .enumerate()
                .all(|(index, (name, _))| parse_tuple_leaf_name(name) == Some(index));
            if tuple_like {
                let mut items = Vec::with_capacity(fields.len());
                for (name, field_abi) in fields {
                    items.push(rebuild_wasm_value_from_abi(
                        field_abi,
                        leaves,
                        &prefix.child(name),
                    )?);
                }
                Ok(Value::Tuple(items))
            } else {
                let mut record = BTreeMap::new();
                for (name, field_abi) in fields {
                    record.insert(
                        name.clone(),
                        rebuild_wasm_value_from_abi(field_abi, leaves, &prefix.child(name))?,
                    );
                }
                Ok(Value::Record(record))
            }
        }
    }
}

fn leaf_result_prim(result: &WasmLeafResultAbi) -> Prim {
    match result {
        WasmLeafResultAbi::Scalar { prim }
        | WasmLeafResultAbi::StarSeq { prim }
        | WasmLeafResultAbi::Bulk { prim } => *prim,
    }
}

const WASM_SEQ_DATA_PTR_OFFSET: u64 = 0;
const WASM_SEQ_LEN_OFFSET: u64 = 4;
const WASM_SEQ_HEADER_BYTES: i32 = 8;

fn read_wasm_star_seq_from_handle(
    memory: &Memory,
    store: &Store<()>,
    handle: i32,
    prim: Prim,
) -> Result<Value> {
    let handle = usize::try_from(handle)
        .map_err(|_| SimdError::new("Wasm T[*] handle must be non-negative"))?;
    let data = memory.data(store);
    let header = data
        .get(handle..handle + WASM_SEQ_HEADER_BYTES as usize)
        .ok_or_else(|| SimdError::new("Wasm T[*] handle is out of bounds"))?;
    let data_ptr =
        u32::from_le_bytes(header[0..4].try_into().map_err(|_| {
            SimdError::new("Wasm T[*] header is truncated while reading data pointer")
        })?) as usize;
    let len = u32::from_le_bytes(
        header[4..8]
            .try_into()
            .map_err(|_| SimdError::new("Wasm T[*] header is truncated while reading length"))?,
    ) as usize;
    let bulk = read_bulk_from_memory(memory, store, data_ptr, len, prim, vec![len])?;
    Ok(Value::StarSeq(StarSeqValue {
        elem_ty: Type::Scalar(prim),
        items: bulk.elements.into_iter().map(Value::Scalar).collect(),
    }))
}

fn build_runtime(
    engine: &Engine,
    module: &WasmtimeModule,
    artifact: &WasmArtifact,
) -> Result<WasmRuntime> {
    let mut store = Store::new(engine, ());
    let function_profiler = if artifact.function_profile_names.is_empty() {
        None
    } else {
        Some(Arc::new(Mutex::new(WasmFunctionProfilerState::new(
            artifact.function_profile_names.clone(),
        ))))
    };
    let instance = if let Some(profiler) = function_profiler.clone() {
        let enter_profiler = profiler.clone();
        let enter = Func::wrap(&mut store, move |func_id: i32| {
            if let Ok(index) = usize::try_from(func_id) {
                if let Ok(mut profiler) = enter_profiler.lock() {
                    profiler.enter(index);
                }
            }
        });
        let exit_profiler = profiler.clone();
        let exit = Func::wrap(&mut store, move |func_id: i32| {
            if let Ok(index) = usize::try_from(func_id) {
                if let Ok(mut profiler) = exit_profiler.lock() {
                    profiler.exit(index);
                }
            }
        });
        Instance::new(&mut store, module, &[enter.into(), exit.into()]).map_err(|error| {
            SimdError::new(format!("failed to instantiate Wasm module: {error}"))
        })?
    } else {
        Instance::new(&mut store, module, &[]).map_err(|error| {
            SimdError::new(format!("failed to instantiate Wasm module: {error}"))
        })?
    };
    let memory = instance
        .get_memory(&mut store, "memory")
        .ok_or_else(|| SimdError::new("compiled Wasm module did not export memory"))?;

    let mut export_names = BTreeSet::<String>::new();
    export_names.insert(artifact.export_name.clone());
    if let Some(grouped) = &artifact.grouped_export {
        export_names.insert(grouped.export_name.clone());
    }
    for leaf in &artifact.leaf_exports {
        export_names.insert(leaf.export_name.clone());
    }

    let mut funcs = BTreeMap::new();
    for name in export_names {
        let func = instance.get_func(&mut store, &name).ok_or_else(|| {
            SimdError::new(format!(
                "compiled Wasm module did not export function '{}'",
                name
            ))
        })?;
        funcs.insert(name, func);
    }

    Ok(WasmRuntime {
        store,
        memory,
        funcs,
        input_buffers: BTreeMap::new(),
        output_buffers: BTreeMap::new(),
        arena_pinned_end: 0,
        arena_cursor: 0,
        function_profiler,
    })
}

fn runtime_func(runtime: &WasmRuntime, name: &str) -> Result<Func> {
    runtime
        .funcs
        .get(name)
        .cloned()
        .ok_or_else(|| SimdError::new(format!("compiled Wasm module did not export '{}'", name)))
}

fn execute_wasm_artifact_in_runtime(
    runtime: &mut WasmRuntime,
    artifact: &WasmArtifact,
    args: &[Value],
) -> Result<Value> {
    if can_use_direct_wasm_path(artifact) {
        execute_wasm_artifact_direct_in_runtime(runtime, artifact, args)
    } else if artifact.grouped_export.is_some() {
        execute_wasm_artifact_grouped_export_in_runtime(runtime, artifact, args)
    } else {
        execute_wasm_artifact_grouped_in_runtime(runtime, artifact, args)
    }
}

fn can_use_direct_wasm_path(artifact: &WasmArtifact) -> bool {
    let shape_contract_ok = match &artifact.result {
        WasmResultAbi::Scalar { .. } => true,
        WasmResultAbi::StarSeq { .. } => true,
        WasmResultAbi::Bulk { shape_param, .. } => artifact
            .params
            .get(*shape_param)
            .is_some_and(|abi| matches!(abi, WasmParamAbi::Bulk { .. })),
        WasmResultAbi::Record { .. } => false,
    };
    artifact.leaf_exports.len() == 1
        && artifact.leaf_exports[0].leaf_path.is_root()
        && artifact.leaf_exports[0].export_name == artifact.export_name
        && artifact
            .params
            .iter()
            .all(|abi| !matches!(abi, WasmParamAbi::Record { .. }))
        && !matches!(artifact.result, WasmResultAbi::Record { .. })
        && shape_contract_ok
}

fn execute_wasm_artifact_direct_in_runtime(
    runtime: &mut WasmRuntime,
    artifact: &WasmArtifact,
    args: &[Value],
) -> Result<Value> {
    if args.len() != artifact.params.len() {
        return Err(SimdError::new(format!(
            "Wasm entry expects {} arguments, found {}",
            artifact.params.len(),
            args.len()
        )));
    }

    let memory = runtime.memory.clone();
    let func = runtime_func(runtime, &artifact.export_name)?;
    runtime.begin_run();

    let mut wasm_args = Vec::<Val>::new();
    let mut output_shape = None::<Vec<usize>>;
    let mut output_ptr = None::<usize>;
    let mut output_len = None::<usize>;

    let string_as_char_bulk = |text: &str| -> BulkValue {
        let elements = text.chars().map(ScalarValue::Char).collect::<Vec<_>>();
        BulkValue {
            prim: Prim::Char,
            shape: vec![elements.len()],
            elements,
        }
    };

    for (index, (abi, value)) in artifact.params.iter().zip(args).enumerate() {
        match (abi, value) {
            (WasmParamAbi::Scalar { prim }, Value::Scalar(value)) => {
                wasm_args.push(scalar_to_wasmtime(value, *prim)?);
            }
            (WasmParamAbi::Bulk { prim }, Value::String(text)) if *prim == Prim::Char => {
                let bulk = string_as_char_bulk(text);
                let ptr = ensure_input_bulk_buffer(runtime, index, &bulk)?;
                write_bulk_to_memory(&memory, &mut runtime.store, ptr, &bulk)?;
                wasm_args.push(Val::I32(i32::try_from(ptr).map_err(|_| {
                    SimdError::new("bulk input pointer does not fit in i32")
                })?));
                wasm_args.push(Val::I32(i32::try_from(bulk.elements.len()).map_err(
                    |_| SimdError::new("bulk input length does not fit in i32"),
                )?));
            }
            (WasmParamAbi::Bulk { prim }, Value::Bulk(bulk)) => {
                if bulk.prim != *prim {
                    return Err(SimdError::new(format!(
                        "Wasm bulk argument expected {:?}, found {:?}",
                        prim, bulk.prim
                    )));
                }
                let ptr = ensure_input_bulk_buffer(runtime, index, bulk)?;
                write_bulk_to_memory(&memory, &mut runtime.store, ptr, bulk)?;
                wasm_args.push(Val::I32(i32::try_from(ptr).map_err(|_| {
                    SimdError::new("bulk input pointer does not fit in i32")
                })?));
                wasm_args.push(Val::I32(i32::try_from(bulk.elements.len()).map_err(
                    |_| SimdError::new("bulk input length does not fit in i32"),
                )?));
            }
            (WasmParamAbi::Bulk { prim }, Value::StarSeq(seq)) => {
                let elements = seq
                    .items
                    .iter()
                    .map(|item| match item {
                        Value::Scalar(value) if value.prim() == *prim => Ok(value.clone()),
                        Value::Scalar(value) => Err(SimdError::new(format!(
                            "Wasm T[*] argument expected {:?} elements, found {:?}",
                            prim,
                            value.prim()
                        ))),
                        other => Err(SimdError::new(format!(
                            "Wasm backend only supports scalar-element T[*] runtime arguments, found {:?}",
                            other
                        ))),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let bulk = BulkValue {
                    prim: *prim,
                    shape: vec![elements.len()],
                    elements,
                };
                let ptr = ensure_input_bulk_buffer(runtime, index, &bulk)?;
                write_bulk_to_memory(&memory, &mut runtime.store, ptr, &bulk)?;
                wasm_args.push(Val::I32(i32::try_from(ptr).map_err(|_| {
                    SimdError::new("bulk input pointer does not fit in i32")
                })?));
                wasm_args.push(Val::I32(i32::try_from(bulk.elements.len()).map_err(
                    |_| SimdError::new("bulk input length does not fit in i32"),
                )?));
            }
            (WasmParamAbi::Scalar { prim }, other) => {
                return Err(SimdError::new(format!(
                    "Wasm scalar argument expected {:?}, found {:?}",
                    prim, other
                )));
            }
            (WasmParamAbi::Bulk { prim }, other) => {
                return Err(SimdError::new(format!(
                    "Wasm bulk argument expected {:?}, found {:?}",
                    prim, other
                )));
            }
            (WasmParamAbi::Record { .. }, _) => {
                return Err(SimdError::new(
                    "direct Wasm execution does not support record ABIs",
                ));
            }
        }
    }

    if let WasmResultAbi::Bulk { prim, shape_param } = &artifact.result {
        let shape_source = args
            .get(*shape_param)
            .ok_or_else(|| SimdError::new("missing bulk argument for Wasm result shape"))?;
        let (shape, len) = match shape_source {
            Value::Bulk(shape_value) => (shape_value.shape.clone(), shape_value.elements.len()),
            Value::String(text) => (vec![text.chars().count()], text.chars().count()),
            _ => {
                return Err(SimdError::new(
                    "Wasm bulk result shape must come from a bulk/string input argument",
                ));
            }
        };
        let bulk = BulkShapeInfo {
            shape: shape.clone(),
            len,
        };
        let out_ptr = ensure_output_bulk_buffer(runtime, &LeafPath::root(), *prim, &bulk)?;
        wasm_args.push(Val::I32(i32::try_from(out_ptr).map_err(|_| {
            SimdError::new("bulk output pointer does not fit in i32")
        })?));
        wasm_args.push(Val::I32(i32::try_from(len).map_err(|_| {
            SimdError::new("bulk output length does not fit in i32")
        })?));
        output_shape = Some(shape);
        output_ptr = Some(out_ptr);
        output_len = Some(len);
    }

    let mut results = if matches!(
        artifact.result,
        WasmResultAbi::Scalar { .. } | WasmResultAbi::StarSeq { .. }
    ) {
        vec![Val::I64(0)]
    } else {
        Vec::new()
    };
    runtime.sync_enum_heap_ptr()?;
    func.call(&mut runtime.store, &wasm_args, &mut results)
        .map_err(|error| SimdError::new(format!("Wasm execution failed: {error}")))?;

    match &artifact.result {
        WasmResultAbi::Scalar { prim } => {
            let value = results
                .into_iter()
                .next()
                .ok_or_else(|| SimdError::new("Wasm scalar entry did not produce a result"))?;
            if let Type::Named(enum_name, args) = &artifact.result_type {
                if is_wasm_enum_named_type(enum_name, args) {
                    let ptr = match value {
                        Val::I32(ptr) => ptr,
                        other => {
                            return Err(SimdError::new(format!(
                                "Wasm enum entry '{}' returned non-i32 handle: {:?}",
                                artifact.export_name, other
                            )));
                        }
                    };
                    let wasm_layout = lookup_specialized_wasm_enum_layout(
                        &artifact.wasm_enum_layouts,
                        enum_name,
                        args,
                    )?;
                    return decode_wasm_enum_value_from_ptr(
                        &memory,
                        &runtime.store,
                        ptr,
                        enum_name,
                        &artifact.enum_ctors,
                        &artifact.wasm_enum_layouts,
                        wasm_layout,
                    );
                }
            }
            Ok(Value::Scalar(wasmtime_to_scalar(value, *prim)?))
        }
        WasmResultAbi::StarSeq { prim } => {
            let value = results
                .into_iter()
                .next()
                .ok_or_else(|| SimdError::new("Wasm T[*] entry did not produce a result"))?;
            let Val::I32(handle) = value else {
                return Err(SimdError::new(
                    "Wasm T[*] entry did not produce an i32 handle",
                ));
            };
            read_wasm_star_seq_from_handle(&memory, &runtime.store, handle, *prim)
        }
        WasmResultAbi::Bulk { prim, .. } => {
            let ptr = output_ptr.ok_or_else(|| SimdError::new("missing bulk output pointer"))?;
            let len = output_len.ok_or_else(|| SimdError::new("missing bulk output length"))?;
            let shape = output_shape.ok_or_else(|| SimdError::new("missing bulk output shape"))?;
            Ok(Value::Bulk(read_bulk_from_memory(
                &memory,
                &runtime.store,
                ptr,
                len,
                *prim,
                shape,
            )?))
        }
        WasmResultAbi::Record { .. } => Err(SimdError::new(
            "direct Wasm execution does not support record results",
        )),
    }
}

fn decode_wasm_enum_value_from_ptr(
    memory: &Memory,
    store: &Store<()>,
    ptr: i32,
    enum_name: &str,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
    wasm_layout: &WasmEnumLayout,
) -> Result<Value> {
    let ref_ptr = usize::try_from(ptr)
        .map_err(|_| SimdError::new(format!("invalid negative enum handle '{}'", ptr)))?;
    if ref_ptr == 0 {
        return Err(SimdError::new("enum handle 0 is invalid"));
    }
    let header_ptr = usize::try_from(read_i32_from_memory(
        memory,
        store,
        ref_ptr + ENUM_REF_HEADER_OFFSET as usize,
    )?)
    .map_err(|_| SimdError::new("enum ref header pointer conversion failed"))?;
    let root = u32::try_from(read_i32_from_memory(
        memory,
        store,
        ref_ptr + ENUM_REF_ROOT_OFFSET as usize,
    )?)
    .map_err(|_| SimdError::new("enum ref root conversion failed"))?;
    let node_count = usize::try_from(read_i32_from_memory(
        memory,
        store,
        header_ptr + ENUM_HEADER_NODE_COUNT_OFFSET as usize,
    )?)
    .map_err(|_| SimdError::new("enum header node_count conversion failed"))?;
    if root as usize >= node_count {
        return Err(SimdError::new(format!(
            "enum root {} is out of bounds for node_count {}",
            root, node_count
        )));
    }
    let tags_ptr = usize::try_from(read_i32_from_memory(
        memory,
        store,
        header_ptr + ENUM_HEADER_TAGS_PTR_OFFSET as usize,
    )?)
    .map_err(|_| SimdError::new("enum tags pointer conversion failed"))?;
    let _ends_ptr = usize::try_from(read_i32_from_memory(
        memory,
        store,
        header_ptr + ENUM_HEADER_ENDS_PTR_OFFSET as usize,
    )?)
    .map_err(|_| SimdError::new("enum ends pointer conversion failed"))?;
    let _slots_ptr = usize::try_from(read_i32_from_memory(
        memory,
        store,
        header_ptr + ENUM_HEADER_SLOTS_PTR_OFFSET as usize,
    )?)
    .map_err(|_| SimdError::new("enum slots pointer conversion failed"))?;
    let ctor_table_ptr = usize::try_from(read_i32_from_memory(
        memory,
        store,
        header_ptr + ENUM_HEADER_CTOR_TABLE_PTR_OFFSET as usize,
    )?)
    .map_err(|_| SimdError::new("enum ctor-table pointer conversion failed"))?;

    let mut tags = Vec::with_capacity(node_count);
    for index in 0..node_count {
        tags.push(read_u16_from_memory(
            memory,
            store,
            tags_ptr + index * ENUM_TAG_BYTES as usize,
        )?);
    }
    let tag_names = enum_ctors
        .iter()
        .filter(|(_, info)| info.enum_name == enum_name)
        .map(|(name, info)| (info.tag, name.clone()))
        .collect::<BTreeMap<_, _>>();
    let ctor_recursive_mask = enum_ctors
        .iter()
        .filter(|(_, info)| info.enum_name == enum_name)
        .map(|(name, info)| {
            (
                name.clone(),
                info.fields
                    .iter()
                    .map(|field| enum_field_is_recursive(field, info))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();

    let mut slots = Vec::with_capacity(node_count);
    let mut next_slot_by_tag = BTreeMap::<u16, u32>::new();
    for tag in &tags {
        let slot = next_slot_by_tag.entry(*tag).or_insert(0);
        slots.push(*slot);
        *slot += 1;
    }

    let mut ends = vec![0u32; node_count];
    for index in (0..node_count).rev() {
        let tag = tags[index];
        let ctor_name = tag_names.get(&tag).ok_or_else(|| {
            SimdError::new(format!("missing constructor name for enum tag {}", tag))
        })?;
        let recursive_count = ctor_recursive_mask
            .get(ctor_name)
            .map(|mask| mask.iter().filter(|is_recursive| **is_recursive).count())
            .ok_or_else(|| {
                SimdError::new(format!(
                    "missing recursive-field mask for constructor '{}'",
                    ctor_name
                ))
            })?;
        if recursive_count == 0 {
            ends[index] = (index + 1) as u32;
            continue;
        }
        let mut child = index + 1;
        for _ in 0..recursive_count {
            let next_child = *ends.get(child).ok_or_else(|| {
                SimdError::new(format!(
                    "enum child root {} is out of bounds while rebuilding tape ends",
                    child
                ))
            })? as usize;
            child = next_child;
        }
        ends[index] = child as u32;
    }

    let mut ctor_rows = BTreeMap::<String, Vec<EnumCtorRow>>::new();
    let mut ctor_layouts = wasm_layout.ctors.values().cloned().collect::<Vec<_>>();
    ctor_layouts.sort_by_key(|layout| layout.tag);
    let mut row_cursor = ctor_table_ptr
        + (usize::from(wasm_layout.max_tag) + 1) * ENUM_CTOR_TABLE_ENTRY_BYTES as usize;
    for ctor_layout in &ctor_layouts {
        let row_count =
            usize::try_from(next_slot_by_tag.get(&ctor_layout.tag).copied().unwrap_or(0))
                .map_err(|_| SimdError::new("enum reconstructed row-count conversion failed"))?;
        let rows_ptr = row_cursor;
        let mut rows = Vec::with_capacity(row_count);
        for row_index in 0..row_count {
            let row_base = rows_ptr + row_index * ctor_layout.row_stride as usize;
            let mut fields = Vec::with_capacity(ctor_layout.non_recursive_fields.len());
            for field_layout in &ctor_layout.non_recursive_fields {
                let field_ptr = row_base + field_layout.offset as usize;
                fields.push(decode_wasm_enum_row_field_value(
                    memory,
                    store,
                    field_ptr,
                    &field_layout.ty,
                    enum_ctors,
                    wasm_enum_layouts,
                )?);
            }
            rows.push(EnumCtorRow { fields });
        }
        ctor_rows.insert(ctor_layout.ctor_name.clone(), rows);
        row_cursor += row_count * ctor_layout.row_stride as usize;
    }

    Ok(Value::Enum(EnumValue {
        enum_name: enum_name.to_string(),
        tape: Rc::new(EnumTape {
            enum_name: enum_name.to_string(),
            tags,
            ends,
            slots,
            tag_names,
            ctor_recursive_mask,
            ctor_rows,
        }),
        root,
    }))
}

fn decode_wasm_enum_row_field_value(
    memory: &Memory,
    store: &Store<()>,
    field_ptr: usize,
    field_ty: &Type,
    enum_ctors: &BTreeMap<String, EnumCtorInfo>,
    wasm_enum_layouts: &BTreeMap<String, WasmEnumLayout>,
) -> Result<Value> {
    match field_ty {
        Type::Scalar(prim) => Ok(Value::Scalar(read_scalar_from_memory(
            memory, store, field_ptr, *prim,
        )?)),
        Type::Index(_) => {
            let ScalarValue::I64(value) =
                read_scalar_from_memory(memory, store, field_ptr, Prim::I64)?
            else {
                unreachable!();
            };
            Ok(Value::Scalar(ScalarValue::I64(value)))
        }
        Type::Named(name, args) if is_wasm_enum_named_type(name, args) => {
            let value_ptr = read_i32_from_memory(memory, store, field_ptr)?;
            if value_ptr < 0 {
                return Err(SimdError::new(format!(
                    "invalid enum field pointer {} in enum tape row",
                    value_ptr
                )));
            }
            let wasm_layout = lookup_specialized_wasm_enum_layout(wasm_enum_layouts, name, args)?;
            decode_wasm_enum_value_from_ptr(
                memory,
                store,
                value_ptr,
                name,
                enum_ctors,
                wasm_enum_layouts,
                wasm_layout,
            )
        }
        Type::Named(name, args) if is_wasm_string_named_type(name, args) => {
            let string_ptr = read_i32_from_memory(memory, store, field_ptr)?;
            let string_len = read_i32_from_memory(memory, store, field_ptr + 4)?;
            if string_ptr < 0 || string_len < 0 {
                return Err(SimdError::new(format!(
                    "invalid string field pointer/len ({}, {}) in enum tape row",
                    string_ptr, string_len
                )));
            }
            let bulk = read_bulk_from_memory(
                memory,
                store,
                usize::try_from(string_ptr)
                    .map_err(|_| SimdError::new("string field pointer conversion failed"))?,
                usize::try_from(string_len)
                    .map_err(|_| SimdError::new("string field length conversion failed"))?,
                Prim::Char,
                vec![
                    usize::try_from(string_len)
                        .map_err(|_| SimdError::new("string field shape conversion failed"))?,
                ],
            )?;
            let mut text = String::new();
            for scalar in bulk.elements {
                let ScalarValue::Char(ch) = scalar else {
                    return Err(SimdError::new(
                        "string enum field contained non-char scalar in Wasm memory",
                    ));
                };
                text.push(ch);
            }
            Ok(Value::String(text))
        }
        Type::StarSeq(item) | Type::StarSeqWitnessed(item, _) => {
            let seq_ptr = read_i32_from_memory(memory, store, field_ptr)?;
            let seq_len = read_i32_from_memory(memory, store, field_ptr + 4)?;
            if seq_ptr < 0 || seq_len < 0 {
                return Err(SimdError::new(format!(
                    "invalid T[*] field pointer/len ({}, {}) in enum tape row",
                    seq_ptr, seq_len
                )));
            }
            let seq_ptr = usize::try_from(seq_ptr)
                .map_err(|_| SimdError::new("T[*] field pointer conversion failed"))?;
            let len = usize::try_from(seq_len)
                .map_err(|_| SimdError::new("T[*] field length conversion failed"))?;
            match item.as_ref() {
                Type::Scalar(prim) => {
                    let bulk =
                        read_bulk_from_memory(memory, store, seq_ptr, len, *prim, vec![len])?;
                    Ok(Value::StarSeq(StarSeqValue {
                        elem_ty: Type::Scalar(*prim),
                        items: bulk.elements.into_iter().map(Value::Scalar).collect(),
                    }))
                }
                Type::Index(witness) => {
                    let bulk =
                        read_bulk_from_memory(memory, store, seq_ptr, len, Prim::I64, vec![len])?;
                    Ok(Value::StarSeq(StarSeqValue {
                        elem_ty: Type::Index(witness.clone()),
                        items: bulk.elements.into_iter().map(Value::Scalar).collect(),
                    }))
                }
                Type::Named(name, args) if is_wasm_enum_named_type(name, args) => {
                    let wasm_layout =
                        lookup_specialized_wasm_enum_layout(wasm_enum_layouts, name, args)?;
                    let mut items = Vec::with_capacity(len);
                    for index in 0..len {
                        let handle = read_i32_from_memory(memory, store, seq_ptr + index * 4)?;
                        if handle < 0 {
                            return Err(SimdError::new(format!(
                                "invalid enum handle {} in T[*] row field",
                                handle
                            )));
                        }
                        items.push(decode_wasm_enum_value_from_ptr(
                            memory,
                            store,
                            handle,
                            name,
                            enum_ctors,
                            wasm_enum_layouts,
                            wasm_layout,
                        )?);
                    }
                    Ok(Value::StarSeq(StarSeqValue {
                        elem_ty: item.as_ref().clone(),
                        items,
                    }))
                }
                other => Err(SimdError::new(format!(
                    "Wasm enum tape decode does not support T[*] field element type {:?}",
                    other
                ))),
            }
        }
        other => Err(SimdError::new(format!(
            "Wasm enum tape decode does not support field type {:?}",
            other
        ))),
    }
}

fn read_u16_from_memory(memory: &Memory, store: &Store<()>, ptr: usize) -> Result<u16> {
    let data = memory.data(store);
    let bytes = data
        .get(ptr..ptr + 2)
        .ok_or_else(|| SimdError::new(format!("u16 memory read out of bounds at {}", ptr)))?;
    Ok(u16::from_le_bytes([bytes[0], bytes[1]]))
}

fn read_i32_from_memory(memory: &Memory, store: &Store<()>, ptr: usize) -> Result<i32> {
    let scalar = read_scalar_from_memory(memory, store, ptr, Prim::I32)?;
    let ScalarValue::I32(value) = scalar else {
        unreachable!("i32 memory read must produce i32 scalar");
    };
    Ok(value)
}

fn ensure_memory_size(memory: &Memory, store: &mut Store<()>, bytes: usize) -> Result<()> {
    let current = memory.data_size(&*store);
    if current >= bytes {
        return Ok(());
    }
    let needed = bytes - current;
    let pages = (needed + 65535) / 65536;
    memory
        .grow(store, pages as u64)
        .map_err(|error| SimdError::new(format!("failed to grow Wasm memory: {error}")))?;
    Ok(())
}

fn write_bulk_to_memory(
    memory: &Memory,
    store: &mut Store<()>,
    ptr: usize,
    bulk: &BulkValue,
) -> Result<()> {
    let data = memory.data_mut(store);
    let mut cursor = ptr;
    for value in &bulk.elements {
        write_scalar_bytes(data, &mut cursor, value)?;
    }
    Ok(())
}

fn read_bulk_from_memory(
    memory: &Memory,
    store: &Store<()>,
    ptr: usize,
    len: usize,
    prim: Prim,
    shape: Vec<usize>,
) -> Result<BulkValue> {
    let data = memory.data(&*store);
    let mut cursor = ptr;
    let mut elements = Vec::with_capacity(len);
    for _ in 0..len {
        elements.push(read_scalar_bytes(data, &mut cursor, prim)?);
    }
    Ok(BulkValue {
        prim,
        shape,
        elements,
    })
}

fn read_scalar_from_memory(
    memory: &Memory,
    store: &Store<()>,
    ptr: usize,
    prim: Prim,
) -> Result<ScalarValue> {
    let data = memory.data(&*store);
    let mut cursor = ptr;
    read_scalar_bytes(data, &mut cursor, prim)
}

fn write_scalar_bytes(data: &mut [u8], cursor: &mut usize, value: &ScalarValue) -> Result<()> {
    match value {
        ScalarValue::I32(value) => write_bytes(data, cursor, &value.to_le_bytes()),
        ScalarValue::I64(value) => write_bytes(data, cursor, &value.to_le_bytes()),
        ScalarValue::F32(value) => write_bytes(data, cursor, &value.to_bits().to_le_bytes()),
        ScalarValue::F64(value) => write_bytes(data, cursor, &value.to_bits().to_le_bytes()),
        ScalarValue::Char(value) => write_bytes(data, cursor, &u32::from(*value).to_le_bytes()),
    }
}

fn read_scalar_bytes(data: &[u8], cursor: &mut usize, prim: Prim) -> Result<ScalarValue> {
    match prim {
        Prim::I32 => Ok(ScalarValue::I32(i32::from_le_bytes(read_bytes::<4>(
            data, cursor,
        )?))),
        Prim::Char => {
            let codepoint = u32::from_le_bytes(read_bytes::<4>(data, cursor)?);
            let value = char::from_u32(codepoint).ok_or_else(|| {
                SimdError::new(format!(
                    "invalid char codepoint '{}' in Wasm memory",
                    codepoint
                ))
            })?;
            Ok(ScalarValue::Char(value))
        }
        Prim::I64 => Ok(ScalarValue::I64(i64::from_le_bytes(read_bytes::<8>(
            data, cursor,
        )?))),
        Prim::F32 => Ok(ScalarValue::F32(f32::from_bits(u32::from_le_bytes(
            read_bytes::<4>(data, cursor)?,
        )))),
        Prim::F64 => Ok(ScalarValue::F64(f64::from_bits(u64::from_le_bytes(
            read_bytes::<8>(data, cursor)?,
        )))),
    }
}

fn write_bytes(data: &mut [u8], cursor: &mut usize, bytes: &[u8]) -> Result<()> {
    let end = *cursor + bytes.len();
    let slice = data
        .get_mut(*cursor..end)
        .ok_or_else(|| SimdError::new("Wasm memory write was out of bounds"))?;
    slice.copy_from_slice(bytes);
    *cursor = end;
    Ok(())
}

fn read_bytes<const N: usize>(data: &[u8], cursor: &mut usize) -> Result<[u8; N]> {
    let end = *cursor + N;
    let slice = data
        .get(*cursor..end)
        .ok_or_else(|| SimdError::new("Wasm memory read was out of bounds"))?;
    let mut bytes = [0u8; N];
    bytes.copy_from_slice(slice);
    *cursor = end;
    Ok(bytes)
}

fn scalar_to_wasmtime(value: &ScalarValue, prim: Prim) -> Result<Val> {
    match (value, prim) {
        (ScalarValue::I32(value), Prim::I32) => Ok(Val::I32(*value)),
        (ScalarValue::I32(value), Prim::Char) => Ok(Val::I32(*value)),
        (ScalarValue::Char(value), Prim::Char) => Ok(Val::I32(
            i32::try_from(u32::from(*value))
                .map_err(|_| SimdError::new("char scalar does not fit in i32 for Wasm input"))?,
        )),
        (ScalarValue::I64(value), Prim::I64) => Ok(Val::I64(*value)),
        (ScalarValue::F32(value), Prim::F32) => Ok(Val::F32(value.to_bits())),
        (ScalarValue::F64(value), Prim::F64) => Ok(Val::F64(value.to_bits())),
        _ => Err(SimdError::new(format!(
            "Wasm scalar argument expected {:?}, found {:?}",
            prim, value
        ))),
    }
}

fn wasmtime_to_scalar(value: Val, prim: Prim) -> Result<ScalarValue> {
    match (value, prim) {
        (Val::I32(value), Prim::I32) => Ok(ScalarValue::I32(value)),
        (Val::I32(value), Prim::Char) => {
            let codepoint = u32::try_from(value).map_err(|_| {
                SimdError::new(format!("invalid negative char codepoint '{}'", value))
            })?;
            let ch = char::from_u32(codepoint).ok_or_else(|| {
                SimdError::new(format!(
                    "invalid char codepoint '{}' returned from Wasm",
                    codepoint
                ))
            })?;
            Ok(ScalarValue::Char(ch))
        }
        (Val::I64(value), Prim::I64) => Ok(ScalarValue::I64(value)),
        (Val::F32(value), Prim::F32) => Ok(ScalarValue::F32(f32::from_bits(value))),
        (Val::F64(value), Prim::F64) => Ok(ScalarValue::F64(f64::from_bits(value))),
        (other, _) => Err(SimdError::new(format!(
            "unexpected Wasm result {:?} for primitive {:?}",
            other, prim
        ))),
    }
}

fn align_to(value: usize, align: usize) -> usize {
    if align == 0 {
        value
    } else {
        ((value + align - 1) / align) * align
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wasm_run(src: &str, main: &str, args_json: &str) -> String {
        run_wasm_main(src, main, args_json)
            .expect("Wasm program should run")
            .to_json_string()
    }

    fn typed_expr_has_lambda_or_apply(expr: &TypedExpr) -> bool {
        typed_expr_contains_lambda_or_apply(expr)
    }

    #[test]
    fn wasm_matches_i64_inc_for_full_vector() {
        let src = "inc : i64 -> i64\ninc x = x + 1\nmain : i64[n] -> i64[n]\nmain xs = inc xs\n";
        assert_eq!(wasm_run(src, "main", "[[1,2]]"), "[2,3]");
    }

    #[test]
    fn wasm_matches_i64_inc_for_tail_cleanup() {
        let src = "inc : i64 -> i64\ninc x = x + 1\nmain : i64[n] -> i64[n]\nmain xs = inc xs\n";
        assert_eq!(wasm_run(src, "main", "[[1,2,3]]"), "[2,3,4]");
    }

    #[test]
    fn wasm_matches_i64_inc_for_small_input() {
        let src = "inc : i64 -> i64\ninc x = x + 1\nmain : i64[n] -> i64[n]\nmain xs = inc xs\n";
        assert_eq!(wasm_run(src, "main", "[[41]]"), "[42]");
    }

    #[test]
    fn wasm_star_seq_gather_runs() {
        let src = include_str!("../examples/star_seq_gather_i64.simd");
        assert_eq!(
            wasm_run(src, "main", "[[10,20,30,40],[3,1,0,2]]"),
            "[40,20,10,30]"
        );
    }

    #[test]
    fn wasm_witnessed_safe_index_runs() {
        let src = "main : i64[*n] -> Index n -> i64\nmain xs i = xs[i]\n";
        assert_eq!(wasm_run(src, "main", "[[10,20,30],1]"), "20");
    }

    #[test]
    fn wasm_checked_index_returns_some_and_none() {
        let src = "main : i64[*n] -> i64 -> Maybe i64\nmain xs i = xs[i]?\n";
        assert_eq!(
            wasm_run(src, "main", "[[10,20,30],1]"),
            "{\"$enum\":\"Some\",\"fields\":[20]}"
        );
        assert_eq!(
            wasm_run(src, "main", "[[10,20,30],5]"),
            "{\"$enum\":\"None\",\"fields\":[]}"
        );
    }

    #[test]
    fn wasm_witnessed_helper_builtins_run() {
        let get_src = "main : i64[*n] -> Index n -> i64\nmain xs i = get xs i\n";
        assert_eq!(wasm_run(get_src, "main", "[[10,20,30],1]"), "20");

        let try_get_src = "main : i64[*n] -> i64 -> Maybe i64\nmain xs i = try_get xs i\n";
        assert_eq!(
            wasm_run(try_get_src, "main", "[[10,20,30],5]"),
            "{\"$enum\":\"None\",\"fields\":[]}"
        );

        let check_index_src =
            "main : i64[*n] -> i64 -> Maybe (Index n)\nmain xs i = check_index xs i\n";
        assert_eq!(
            wasm_run(check_index_src, "main", "[[10,20,30],2]"),
            "{\"$enum\":\"Some\",\"fields\":[2]}"
        );

        let indices_src = "main : i64[*n] -> Index n[*n]\nmain xs = indices xs\n";
        assert_eq!(wasm_run(indices_src, "main", "[[10,20,30]]"), "[0,1,2]");
    }

    #[test]
    fn wasm_scatter_builtins_run() {
        let scatter_src = "main : f32[*] -> i64[*] -> f32[*] -> f32[*]\nmain dest idx values = scatter dest idx values\n";
        assert_eq!(
            wasm_run(scatter_src, "main", "[[0,0,0,0],[1,1,3],[0.5,0.25,1.0]]"),
            "[0.0,0.25,0.0,1.0]"
        );

        let scatter_add_src = "main : f32[*] -> i64[*] -> f32[*] -> f32[*]\nmain dest idx values = scatter_add dest idx values\n";
        assert_eq!(
            wasm_run(
                scatter_add_src,
                "main",
                "[[0,0,0,0],[1,1,3],[0.5,0.25,1.0]]"
            ),
            "[0.0,0.75,0.0,1.0]"
        );
    }

    #[test]
    fn wasm_star_seq_reverse_runs() {
        let src = "main : i64[*] -> i64[*]\nmain xs = reverse xs\n";
        assert_eq!(wasm_run(src, "main", "[[10,20,30,40]]"), "[40,30,20,10]");
    }

    #[test]
    fn wat_includes_optimizer_report_comments() {
        let src = "axpy : i64 -> i64 -> i64 -> i64\naxpy a x y = a * x + y\nmain : i64 -> i64[n] -> i64[n] -> i64[n]\nmain a xs ys = axpy a xs ys\n";
        let wat = wat_main(src, "main").expect("WAT should be printable");
        assert!(wat.contains(";; optimizer reports:"));
        assert!(wat.contains("fn=main"));
        assert!(
            wat.contains("plan=vec1") || wat.contains("plan=vec2") || wat.contains("plan=vec4")
        );
    }

    #[test]
    fn structural_optimizer_reports_use_structural_loop_exec() {
        let src = "enum List a =\n  | Nil\n  | Cons a (List a)\n\nlen : List i64 -> i64\nlen Nil = 0\nlen (Cons _ xs) = 1 + len xs\n\nmain : i64\nmain = len (Cons 1 (Cons 2 Nil))\n";
        let artifact = compile_wasm_main(src, "main").expect("artifact should compile");
        let report = artifact
            .optimizer_reports
            .iter()
            .find(|report| report.function == "len")
            .expect("len optimizer report should exist");
        assert_eq!(report.intent, IntentClass::Structural);
        assert_eq!(report.structural_exec, StructuralExecMode::StructuralLoop);
        let wat = wat_main(src, "main").expect("WAT should be printable");
        assert!(wat.contains("fn=len intent=Structural exec=structural-loop"));
    }

    #[test]
    fn json_parser_item_helpers_report_structural_batched_exec() {
        let src = include_str!("../examples/json_parser_adt.simd");
        let artifact = compile_wasm_main(src, "main").expect("artifact should compile");
        for function in ["parse_array_items", "parse_object_items"] {
            let report = artifact
                .optimizer_reports
                .iter()
                .find(|report| report.function == function)
                .unwrap_or_else(|| panic!("missing optimizer report for {}", function));
            assert_eq!(report.intent, IntentClass::Structural);
            assert_eq!(
                report.structural_exec,
                StructuralExecMode::StructuralBatched
            );
            assert!(report.structural_region_count > 0);
        }
    }

    #[test]
    fn artifact_includes_vector_plan_fallback_reason() {
        let src = "pow2 : i64 -> i64 -> i64\npow2 0 x = x\npow2 n x = pow2 (n - 1) (x * 2)\nmain : i64[n] -> i64[n]\nmain xs = pow2 3 xs\n";
        let artifact = compile_wasm_main(src, "main").expect("artifact should compile");
        let main_report = artifact
            .optimizer_reports
            .iter()
            .find(|report| report.function == "main")
            .expect("main optimizer report should be present");
        assert_eq!(main_report.vector_unroll, 0);
        assert_eq!(
            main_report.fallback_reason.as_deref(),
            Some("non-vectorizable clause")
        );
    }

    #[test]
    fn wat_uses_pointer_induction_without_index_stride_multiply() {
        let src = "axpy : i64 -> i64 -> i64 -> i64\naxpy a x y = a * x + y\nmain : i64 -> i64[n] -> i64[n] -> i64[n]\nmain a xs ys = axpy a xs ys\n";
        let wat = wat_main(src, "main").expect("WAT should be printable");
        assert!(!wat.contains("i32.mul"));
    }

    #[test]
    fn wat_uses_return_call_for_scalar_tail_position_call() {
        let src = "id : i64 -> i64\nid x = x\nforward : i64 -> i64\nforward x = id x\nmain : i64 -> i64\nmain x = forward x\n";
        let wat = wat_main(src, "main").expect("WAT should be printable");
        assert!(wat.contains("return_call"));
        assert_eq!(wasm_run(src, "main", "[41]"), "41");
    }

    #[test]
    fn wat_uses_return_call_for_tail_let_wrapped_call() {
        let src = "id : i64 -> i64\nid x = x\nforward : i64 -> i64\nforward x = let y = x in id y\nmain : i64 -> i64\nmain x = forward x\n";
        let wat = wat_main(src, "main").expect("WAT should be printable");
        assert!(wat.contains("return_call"));
        assert_eq!(wasm_run(src, "main", "[7]"), "7");
    }

    #[test]
    fn string_sdf_demo_compiles_for_wasm() {
        let src = include_str!("../examples/string_sdf_f32.simd");
        compile_wasm_main(src, "main").expect("string_sdf_f32 demo should compile to Wasm");
    }

    #[test]
    fn image_stipple_field_demo_compiles_for_wasm() {
        let src = include_str!("../examples/image_stipple_field_f32.simd");
        compile_wasm_main(src, "main")
            .expect("image_stipple_field_f32 demo should compile to Wasm");
    }

    #[test]
    fn image_stipple_force_demo_compiles_for_wasm() {
        let src = include_str!("../examples/image_stipple_force_f32.simd");
        compile_wasm_main(src, "main")
            .expect("image_stipple_force_f32 demo should compile to Wasm");
    }

    #[test]
    fn dense_kernel_can_select_vec4_unroll() {
        let src = "mix : i64 -> i64 -> i64 -> i64 -> i64 -> i64 -> i64\nmix a b c d x y = ((a * x + y) * (b * x + y)) + ((c * x + y) * (d * x + y))\nmain : i64 -> i64 -> i64 -> i64 -> i64[n] -> i64[n] -> i64[n]\nmain a b c d xs ys = mix a b c d xs ys\n";
        let artifact = compile_wasm_main(src, "main").expect("artifact should compile");
        let report = artifact
            .optimizer_reports
            .iter()
            .find(|report| report.function == "main")
            .expect("main optimizer report should exist");
        assert_eq!(report.vector_unroll, 4);
    }

    #[test]
    fn multiclausal_kernel_vectorizes_with_bitselect() {
        let src = "main : i64 -> i64[n] -> i64[n]\nmain 0 xs = xs * 0\nmain a xs = a * xs\n";
        let artifact = compile_wasm_main(src, "main").expect("artifact should compile");
        let report = artifact
            .optimizer_reports
            .iter()
            .find(|report| report.function == "main")
            .expect("main optimizer report should exist");
        assert!(report.vectorizable);
        let wat = wat_main(src, "main").expect("WAT should be printable");
        assert!(wat.contains("v128.bitselect"));
        assert_eq!(wasm_run(src, "main", "[0,[1,2,3,4]]"), "[0,0,0,0]");
        assert_eq!(wasm_run(src, "main", "[3,[1,2,3,4]]"), "[3,6,9,12]");
    }

    #[test]
    fn i64_comparison_kernel_vectorizes() {
        let src = "lt : i64 -> i64 -> i64\nlt x y = x < y\nmain : i64[n] -> i64[n] -> i64[n]\nmain xs ys = lt xs ys\n";
        let artifact = compile_wasm_main(src, "main").expect("artifact should compile");
        let report = artifact
            .optimizer_reports
            .iter()
            .find(|report| report.function == "main")
            .expect("main optimizer report should exist");
        assert!(report.vectorizable);
        assert_eq!(wasm_run(src, "main", "[[1,9],[2,3]]"), "[1,0]");
    }

    #[test]
    fn wasm_direct_path_eligibility_is_structural() {
        let artifact = WasmArtifact {
            bytes: Vec::new(),
            export_name: "alias".to_string(),
            params: vec![WasmParamAbi::Scalar { prim: Prim::I64 }],
            result: WasmResultAbi::Scalar { prim: Prim::I64 },
            result_type: Type::Scalar(Prim::I64),
            enum_ctors: BTreeMap::new(),
            wasm_enum_layouts: BTreeMap::new(),
            grouped_export: None,
            leaf_exports: vec![WasmLeafExport {
                leaf_path: LeafPath::root(),
                export_name: "alias".to_string(),
                result: WasmLeafResultAbi::Scalar { prim: Prim::I64 },
                used_param_leaves: Vec::new(),
                reusable_param_leaf: None,
            }],
            optimizer_reports: Vec::new(),
            higher_order_reports: Vec::new(),
            function_profile_names: Vec::new(),
        };
        assert!(can_use_direct_wasm_path(&artifact));

        let mut grouped = artifact.clone();
        grouped.leaf_exports[0].leaf_path = LeafPath(vec!["x".to_string()]);
        assert!(!can_use_direct_wasm_path(&grouped));
    }

    #[test]
    fn wasm_bulk_result_shape_param_uses_top_level_bulk_index() {
        let src = "axpy : i64 -> i64 -> i64 -> i64\naxpy a x y = a * x + y\nmain : i64 -> i64[n] -> i64[n] -> i64[n]\nmain a xs ys = axpy a xs ys\n";
        let artifact = compile_wasm_main(src, "main").expect("axpy artifact should compile");
        assert_eq!(
            artifact.result,
            WasmResultAbi::Bulk {
                prim: Prim::I64,
                shape_param: 1,
            }
        );
        assert!(can_use_direct_wasm_path(&artifact));
    }

    #[test]
    fn wasm_runtime_reuses_direct_input_and_output_buffers_for_same_shape() {
        let src = "inc : i64 -> i64\ninc x = x + 1\nmain : i64[n] -> i64[n]\nmain xs = inc xs\n";
        let artifact = compile_wasm_main(src, "main").expect("artifact should compile");
        let executable = prepare_wasm_artifact(artifact).expect("artifact should prepare");
        let args_a = vec![Value::Bulk(BulkValue {
            prim: Prim::I64,
            shape: vec![4],
            elements: vec![
                ScalarValue::I64(1),
                ScalarValue::I64(2),
                ScalarValue::I64(3),
                ScalarValue::I64(4),
            ],
        })];
        let args_b = vec![Value::Bulk(BulkValue {
            prim: Prim::I64,
            shape: vec![4],
            elements: vec![
                ScalarValue::I64(10),
                ScalarValue::I64(20),
                ScalarValue::I64(30),
                ScalarValue::I64(40),
            ],
        })];
        let out_a = executable
            .run_hot(&args_a)
            .expect("first run should succeed");
        assert_eq!(out_a.to_json_string(), "[2,3,4,5]");
        let (input_ptr_a, output_ptr_a) = {
            let runtime = executable.runtime.borrow();
            assert_eq!(runtime.input_buffers.len(), 1);
            assert_eq!(runtime.output_buffers.len(), 1);
            (
                *runtime
                    .input_buffers
                    .values()
                    .next()
                    .expect("expected one input buffer"),
                *runtime
                    .output_buffers
                    .values()
                    .next()
                    .expect("expected one output buffer"),
            )
        };

        let out_b = executable
            .run_hot(&args_b)
            .expect("second run should succeed");
        assert_eq!(out_b.to_json_string(), "[11,21,31,41]");
        let (input_ptr_b, output_ptr_b) = {
            let runtime = executable.runtime.borrow();
            (
                *runtime
                    .input_buffers
                    .values()
                    .next()
                    .expect("expected one input buffer"),
                *runtime
                    .output_buffers
                    .values()
                    .next()
                    .expect("expected one output buffer"),
            )
        };

        assert_eq!(input_ptr_a, input_ptr_b);
        assert_eq!(output_ptr_a, output_ptr_b);
    }

    #[test]
    fn wasm_runtime_allocates_new_buffers_when_shape_changes() {
        let src = "inc : i64 -> i64\ninc x = x + 1\nmain : i64[n] -> i64[n]\nmain xs = inc xs\n";
        let artifact = compile_wasm_main(src, "main").expect("artifact should compile");
        let executable = prepare_wasm_artifact(artifact).expect("artifact should prepare");

        let args_short = vec![Value::Bulk(BulkValue {
            prim: Prim::I64,
            shape: vec![2],
            elements: vec![ScalarValue::I64(1), ScalarValue::I64(2)],
        })];
        let args_long = vec![Value::Bulk(BulkValue {
            prim: Prim::I64,
            shape: vec![5],
            elements: vec![
                ScalarValue::I64(1),
                ScalarValue::I64(2),
                ScalarValue::I64(3),
                ScalarValue::I64(4),
                ScalarValue::I64(5),
            ],
        })];

        executable
            .run_hot(&args_short)
            .expect("short shape run should succeed");
        executable
            .run_hot(&args_long)
            .expect("long shape run should succeed");

        let runtime = executable.runtime.borrow();
        assert_eq!(runtime.input_buffers.len(), 2);
        assert_eq!(runtime.output_buffers.len(), 2);
    }

    fn find_slot_index(
        slots: &[PreparedSlotMetadata],
        role: PreparedSlotRole,
        kind: PreparedSlotKind,
    ) -> usize {
        slots
            .iter()
            .find(|slot| slot.role == role && slot.kind == kind)
            .map(|slot| slot.index)
            .expect("expected prepared slot")
    }

    #[test]
    fn prepared_run_supports_typed_bulk_io() {
        let src = "inc : i64 -> i64\ninc x = x + 1\nmain : i64[n] -> i64[n]\nmain xs = inc xs\n";
        let prepared = prepare_wasm_main(src, "main").expect("prepared main should compile");
        let layout = PreparedLayout {
            bulk_input_shapes: vec![vec![3]],
        };
        let mut bound = prepared.bind(layout).expect("bind should succeed");
        let slots = bound.slots();
        let input_slot = find_slot_index(&slots, PreparedSlotRole::Input, PreparedSlotKind::Bulk);
        let output_slot = find_slot_index(&slots, PreparedSlotRole::Output, PreparedSlotKind::Bulk);

        bound
            .write_bulk_i64(input_slot, &[1, 2, 3])
            .expect("typed input write should succeed");
        bound.run().expect("prepared run should succeed");
        let mut out = vec![0i64; 3];
        bound
            .read_bulk_i64(output_slot, &mut out)
            .expect("typed output read should succeed");
        assert_eq!(out, vec![2, 3, 4]);
    }

    #[test]
    fn prepared_bind_rejects_wrong_bulk_shape_count() {
        let src = "inc : i64 -> i64\ninc x = x + 1\nmain : i64[n] -> i64[n]\nmain xs = inc xs\n";
        let prepared = prepare_wasm_main(src, "main").expect("prepared main should compile");
        let error = prepared
            .bind(PreparedLayout {
                bulk_input_shapes: vec![],
            })
            .expect_err("bind should fail for missing bulk shape");
        assert!(error.to_string().contains("expected 1 bulk input shapes"));
    }

    #[test]
    fn prepared_run_matches_record_wasm_output() {
        let src = "axpy2 : i64 -> {x:i64,y:i64} -> {x:i64,y:i64} -> {x:i64,y:i64}\naxpy2 a u v = { x = a * u.x + v.x, y = a * u.y + v.y }\nmain : i64 -> {x:i64,y:i64}[n] -> {x:i64,y:i64}[n] -> {x:i64,y:i64}[n]\nmain a us vs = axpy2 a us vs\n";
        let args = vec![
            Value::Scalar(ScalarValue::I64(2)),
            Value::Record(BTreeMap::from([
                (
                    "x".to_string(),
                    Value::Bulk(BulkValue {
                        prim: Prim::I64,
                        shape: vec![2],
                        elements: vec![ScalarValue::I64(1), ScalarValue::I64(2)],
                    }),
                ),
                (
                    "y".to_string(),
                    Value::Bulk(BulkValue {
                        prim: Prim::I64,
                        shape: vec![2],
                        elements: vec![ScalarValue::I64(3), ScalarValue::I64(4)],
                    }),
                ),
            ])),
            Value::Record(BTreeMap::from([
                (
                    "x".to_string(),
                    Value::Bulk(BulkValue {
                        prim: Prim::I64,
                        shape: vec![2],
                        elements: vec![ScalarValue::I64(10), ScalarValue::I64(20)],
                    }),
                ),
                (
                    "y".to_string(),
                    Value::Bulk(BulkValue {
                        prim: Prim::I64,
                        shape: vec![2],
                        elements: vec![ScalarValue::I64(30), ScalarValue::I64(40)],
                    }),
                ),
            ])),
        ];
        let prepared = prepare_wasm_main(src, "main").expect("prepared main should compile");
        let layout = prepared
            .layout_from_args(&args)
            .expect("layout should infer");
        let mut bound = prepared.bind(layout).expect("bind should succeed");
        bound.load_args(&args).expect("loading args should succeed");
        bound.run().expect("prepared run should succeed");
        let prepared_value = bound.read_result().expect("prepared read should succeed");

        let artifact = compile_wasm_main(src, "main").unwrap();
        let executable = prepare_wasm_artifact(artifact).unwrap();
        let wasm_value = executable.run(&args).unwrap();
        assert_eq!(prepared_value, wasm_value);
    }

    #[test]
    fn wasm_matches_scalar_record_update() {
        let src = concat!(
            "bump : { x: i64, y: i64 } -> { x: i64, y: i64 }\n",
            "bump p = p { x = p.x + 1 }\n",
            "main : { x: i64, y: i64 } -> { x: i64, y: i64 }\n",
            "main p = bump p\n",
        );
        let artifact = compile_wasm_main(src, "main").expect("record Wasm artifact should compile");
        assert_eq!(artifact.leaf_exports.len(), 2);
        assert!(matches!(artifact.params[0], WasmParamAbi::Record { .. }));
        assert!(!can_use_direct_wasm_path(&artifact));
        assert_eq!(
            wasm_run(src, "main", "[{\"x\":1,\"y\":2}]"),
            "{\"x\":2,\"y\":2}"
        );
        let eval = run_main(src, "main", "[{\"x\":1,\"y\":2}]")
            .unwrap()
            .to_json_string();
        let wasm = run_wasm_main(src, "main", "[{\"x\":1,\"y\":2}]")
            .unwrap()
            .to_json_string();
        assert_eq!(eval, wasm);
    }

    #[test]
    fn wasm_matches_bulk_record_aos() {
        let src = concat!(
            "bump : { x: i64, y: i64 } -> { x: i64, y: i64 }\n",
            "bump p = p { x = p.x + 1, y = p.y + 2 }\n",
            "main : { x: i64, y: i64 }[n] -> { x: i64, y: i64 }[n]\n",
            "main ps = bump ps\n",
        );
        let artifact =
            compile_wasm_main(src, "main").expect("bulk record Wasm artifact should compile");
        assert_eq!(artifact.leaf_exports.len(), 2);
        assert!(
            artifact
                .leaf_exports
                .iter()
                .any(|leaf| leaf.export_name.ends_with("$x"))
        );
        assert!(
            artifact
                .leaf_exports
                .iter()
                .any(|leaf| leaf.export_name.ends_with("$y"))
        );
        assert_eq!(
            wasm_run(src, "main", "[[{\"x\":1,\"y\":2},{\"x\":3,\"y\":4}]]"),
            "[{\"x\":2,\"y\":4},{\"x\":4,\"y\":6}]"
        );
        let eval = run_main(src, "main", "[[{\"x\":1,\"y\":2},{\"x\":3,\"y\":4}]]")
            .unwrap()
            .to_json_string();
        let wasm = run_wasm_main(src, "main", "[[{\"x\":1,\"y\":2},{\"x\":3,\"y\":4}]]")
            .unwrap()
            .to_json_string();
        assert_eq!(eval, wasm);
    }

    #[test]
    fn debug_custom_record_types() {
        let src = include_str!("../examples/custom_record_types.simd");
        let (_surface, _module, checked) = compile_frontend(src).unwrap();
        let specialized = specialize_checked_program_for_main(&checked, "main").unwrap();
        let lowered_ready = eliminate_non_escaping_lambdas_program(&specialized.checked).unwrap();
        let canonical = canonicalize_backend_higher_order_program(&lowered_ready).unwrap();
        let normalized = normalize_records(&canonical).unwrap();
        let lowered = lower_program(&normalized).unwrap();

        eprintln!("normalized functions:");
        for function in &normalized.functions {
            eprintln!("  {} : {:?}", function.name, function.signature.ty);
            for (idx, clause) in function.clauses.iter().enumerate() {
                eprintln!("    clause {} patterns={:?}", idx, clause.patterns);
                eprintln!("    body={:?}", clause.body);
            }
        }

        eprintln!("lowered functions:");
        for function in &lowered.functions {
            eprintln!("  {} : {:?}", function.name, function.kind);
            eprintln!("    result={:?}", function.result);
            match &function.kind {
                LoweredKind::Scalar { clauses }
                | LoweredKind::Structural { clauses, .. }
                | LoweredKind::Kernel { clauses, .. } => {
                    for (idx, clause) in clauses.iter().enumerate() {
                        eprintln!("    clause {} patterns={:?}", idx, clause.patterns);
                        eprintln!("    body={:?}", clause.body);
                    }
                }
            }
            if let Some(loop_form) = &function.tail_loop {
                eprintln!("    tail_loop clauses={:?}", loop_form.clauses);
            }
        }
    }

    #[test]
    fn debug_axpy2_record_wasm_pipeline() {
        let src = "axpy2 : i64 -> {x:i64,y:i64} -> {x:i64,y:i64} -> {x:i64,y:i64}\naxpy2 a u v = { x = a * u.x + v.x, y = a * u.y + v.y }\nmain : i64 -> {x:i64[n],y:i64[n]} -> {x:i64[n],y:i64[n]} -> {x:i64[n],y:i64[n]}\nmain a us vs = axpy2 a us vs";
        let (_surface, _module, checked) = compile_frontend(src).unwrap();
        let specialized = specialize_checked_program_for_main(&checked, "main").unwrap();
        let lowered_ready = eliminate_non_escaping_lambdas_program(&specialized.checked).unwrap();
        let canonical = canonicalize_backend_higher_order_program(&lowered_ready).unwrap();
        eprintln!("canonical checked:\n{:#?}", canonical);
        let normalized = normalize_records(&canonical).unwrap();
        eprintln!("normalized function signatures:");
        for function in &normalized.functions {
            eprintln!("  {} : {:?}", function.name, function.signature.ty);
            for clause in &function.clauses {
                eprintln!("    body {:?}", clause.body);
            }
        }
    }

    #[test]
    fn debug_scalar_record_update_wasm_pipeline() {
        let src = concat!(
            "bump : { x: i64, y: i64 } -> { x: i64, y: i64 }\n",
            "bump p = p { x = p.x + 1 }\n",
            "main : { x: i64, y: i64 } -> { x: i64, y: i64 }\n",
            "main p = bump p\n",
        );
        let (_surface, _module, checked) = compile_frontend(src).unwrap();
        let specialized = specialize_checked_program_for_main(&checked, "main").unwrap();
        let lowered_ready = eliminate_non_escaping_lambdas_program(&specialized.checked).unwrap();
        let canonical = canonicalize_backend_higher_order_program(&lowered_ready).unwrap();
        eprintln!("canonical checked:\n{:#?}", canonical);
        let normalized = normalize_records(&canonical);
        eprintln!(
            "normalize_records result: {:?}",
            normalized.as_ref().map(|_| "ok")
        );
        let normalized = normalized.unwrap();
        eprintln!("normalized functions:");
        for function in &normalized.functions {
            eprintln!("  {} : {:?}", function.name, function.signature.ty);
            for clause in &function.clauses {
                eprintln!("    patterns={:?}", clause.patterns);
                eprintln!("    body={:?}", clause.body);
            }
        }

        let plan = build_wasm_plan(&canonical, "main").expect("plan should build");
        eprintln!("plan.checked:");
        for function in &plan.checked.functions {
            eprintln!("  {} : {:?}", function.name, function.signature.ty);
            for clause in &function.clauses {
                eprintln!("    patterns={:?}", clause.patterns);
                eprintln!("    body={:?}", clause.body);
            }
        }
        let normalized_from_plan = normalize_records(&plan.checked);
        eprintln!(
            "normalize plan.checked result: {:?}",
            normalized_from_plan.as_ref().map(|_| "ok")
        );
        let _ = normalized_from_plan.expect("plan checked should normalize");
    }

    #[test]
    fn wasm_matches_f32_square() {
        let src =
            "square : f32 -> f32\nsquare x = x * x\nmain : f32[n] -> f32[n]\nmain xs = square xs\n";
        assert_eq!(
            wasm_run(src, "main", "[[1.5,2.0,3.0,4.0]]"),
            "[2.25,4.0,9.0,16.0]"
        );
    }

    #[test]
    fn wasm_matches_axpy() {
        let src = "axpy : i64 -> i64 -> i64 -> i64\naxpy a x y = a * x + y\nmain : i64 -> i64[n] -> i64[n] -> i64[n]\nmain a xs ys = axpy a xs ys\n";
        assert_eq!(
            wasm_run(src, "main", "[2,[1,2,3],[10,20,30]]"),
            "[12,24,36]"
        );
    }

    #[test]
    fn wasm_monomorphizes_generic_direct_call() {
        let src =
            "square : t -> t\nsquare x = x * x\nmain : i64[n] -> i64[n]\nmain xs = square xs\n";
        assert_eq!(wasm_run(src, "main", "[[2,3,4]]"), "[4,9,16]");
    }

    #[test]
    fn wasm_monomorphizes_transitive_generic_calls() {
        let src = "id : t -> t\nid x = x\nsquare : t -> t\nsquare x = let y = id x in y * y\nmain : i64 -> i64\nmain x = square x\n";
        assert_eq!(wasm_run(src, "main", "[7]"), "49");
    }

    #[test]
    fn wasm_monomorphizes_generic_literal_for_float_call() {
        let src = "inc : t -> t\ninc x = x + 1\nmain : f32 -> f32\nmain x = inc x\n";
        assert_eq!(wasm_run(src, "main", "[1.5]"), "2.5");
    }

    #[test]
    fn wasm_monomorphizes_type_witness_dispatch_when_main_is_concrete() {
        let src = "my_func : Type t -> t -> t\nmy_func i64 x = x + 1\nmy_func _ x = x\nmain : i64 -> i64\nmain x = my_func i64 x\n";
        let eval = run_main(src, "main", "[41]")
            .expect("evaluator should run")
            .to_json_string();
        let wasm = run_wasm_main(src, "main", "[41]")
            .expect("wasm should run")
            .to_json_string();
        assert_eq!(eval, wasm);
        assert_eq!(wasm, "42");
    }

    #[test]
    fn wasm_rejects_runtime_type_witness_entry_params() {
        let src = "main : Type i64 -> i64 -> i64\nmain i64 x = x + 1\nmain _ x = x\n";
        let error = compile_wasm_main(src, "main").expect_err("Wasm compile should reject");
        assert!(
            error
                .to_string()
                .contains("cannot expose Type witness parameters")
        );
    }

    #[test]
    fn wasm_rejects_nonliteral_type_witness_call_args() {
        let src = "my_func : Type t -> t -> t\nmy_func i64 x = x + 1\nmy_func _ x = x\nmain : i64 -> i64\nmain x = let ty = i64 in my_func ty x\n";
        let error = compile_wasm_main(src, "main").expect_err("Wasm compile should reject");
        assert!(error.to_string().contains(
            "type witness argument 0 in call to 'my_func' to be a literal primitive token"
        ));
    }

    #[test]
    fn wasm_specializes_higher_order_helper_with_known_function_arg() {
        let src = "apply : (i64 -> i64) -> i64 -> i64\napply f x = f x\ninc : i64 -> i64\ninc x = x + 1\nmain : i64[n] -> i64[n]\nmain xs = apply inc xs\n";
        let artifact = compile_wasm_main(src, "main").expect("artifact should compile");
        assert_eq!(wasm_run(src, "main", "[[1,2,3]]"), "[2,3,4]");
        assert!(
            artifact
                .higher_order_reports
                .iter()
                .any(|report| report.function.contains("$mono$"))
        );
        assert!(
            artifact
                .higher_order_reports
                .iter()
                .all(|report| report.rejection_reason.is_none())
        );
    }

    #[test]
    fn wasm_specializes_transitive_higher_order_calls() {
        let src = "twice : (i64 -> i64) -> i64 -> i64\ntwice f x = f (f x)\nwrap : (i64 -> i64) -> i64 -> i64\nwrap f x = twice f x\ninc : i64 -> i64\ninc x = x + 1\nmain : i64[n] -> i64[n]\nmain xs = wrap inc xs\n";
        assert_eq!(wasm_run(src, "main", "[[1,2,3]]"), "[3,4,5]");
    }

    #[test]
    fn wasm_specialization_is_deterministic() {
        let src = "apply : (i64 -> i64) -> i64 -> i64\napply f x = f x\ninc : i64 -> i64\ninc x = x + 1\nmain : i64[n] -> i64[n]\nmain xs = apply inc xs\n";
        let first = compile_wasm_main(src, "main").expect("first compile should succeed");
        let second = compile_wasm_main(src, "main").expect("second compile should succeed");
        assert_eq!(first.bytes, second.bytes);
        assert_eq!(first.higher_order_reports, second.higher_order_reports);
    }

    #[test]
    fn backend_canonicalizes_known_partial_application_saturation() {
        let src = "add : i64 -> i64 -> i64\nadd a b = a + b\nmain : i64 -> i64\nmain x = let plus2 = add 2 in plus2 x\n";
        let (_surface, _module, checked) = compile_frontend(src).expect("frontend should compile");
        let specialized =
            specialize_checked_program_for_main(&checked, "main").expect("specialization works");
        let lowered_ready = eliminate_non_escaping_lambdas_program(&specialized.checked)
            .expect("lambda elimination should work");
        let canonical = canonicalize_backend_higher_order_program(&lowered_ready)
            .expect("backend canonicalization should succeed");
        for function in &canonical.functions {
            for clause in &function.clauses {
                assert!(
                    !typed_expr_has_lambda_or_apply(&clause.body),
                    "function '{}' still contains lambda/apply after backend canonicalization",
                    function.name
                );
            }
        }
        assert_eq!(wasm_run(src, "main", "[7]"), "9");
    }

    #[test]
    fn wasm_compiles_non_capturing_escaping_lambda_value_in_higher_order_call() {
        let src = "apply : (i64 -> i64) -> i64 -> i64\napply f x = f x\nmain : i64 -> i64\nmain x = apply (\\y -> y + 1) x\n";
        assert_eq!(wasm_run(src, "main", "[9]"), "10");
    }

    #[test]
    fn wasm_compiles_capturing_scalar_lambda_value_in_higher_order_call() {
        let src = "apply : (i64 -> i64) -> i64 -> i64\napply f x = f x\nmain : i64 -> i64 -> i64\nmain a x = apply (\\y -> y + a) x\n";
        assert_eq!(wasm_run(src, "main", "[5,9]"), "14");
    }

    #[test]
    fn wasm_rejects_lambda_with_function_capture_for_now() {
        let src = "callk : ((i64 -> i64) -> i64) -> i64\ncallk k = k (\\x -> x + 1)\nmain : i64\nmain = callk (\\f -> f 7)\n";
        let _error = compile_wasm_main(src, "main").expect_err("Wasm compile should reject");
    }

    #[test]
    fn wasm_compiles_higher_order_function_results_after_conversion() {
        let src = "mk : i64 -> i64 -> i64\nmk a x = x + a\nmain : i64 -> i64\nmain x = (mk 1) x\n";
        assert_eq!(wasm_run(src, "main", "[9]"), "10");
    }

    #[test]
    fn wasm_compiles_nested_lambda_returning_lambda_chain() {
        let src = "mk2 : i64 -> i64 -> i64 -> i64\nmk2 a b x = x + a + b\nmain : i64 -> i64\nmain x = ((mk2 1) 2) x\n";
        assert_eq!(wasm_run(src, "main", "[9]"), "12");
    }

    #[test]
    fn wasm_compiles_bulk_capture_lambda_value_in_higher_order_call() {
        let src = "applyv : (i64[n] -> i64[n]) -> i64[n] -> i64[n]\napplyv f xs = f xs\nmain : i64[n] -> i64[n] -> i64[n]\nmain ys xs = applyv (\\x -> x + ys) xs\n";
        assert_eq!(wasm_run(src, "main", "[[10,20,30],[1,2,3]]"), "[11,22,33]");
    }

    #[test]
    fn wasm_matches_pow2_via_scalar_helper() {
        let src = "pow2 : i64 -> i64 -> i64\npow2 0 x = x\npow2 n x = pow2 (n - 1) (x * 2)\nmain : i64[n] -> i64[n]\nmain xs = pow2 3 xs\n";
        assert_eq!(wasm_run(src, "main", "[[1,2,3]]"), "[8,16,24]");
    }

    #[test]
    fn wasm_matches_invariant_hoisting_kernel() {
        let src = "affine : i64 -> i64 -> i64\naffine a x = (a + 1) * x + (a + 1)\nmain : i64 -> i64[n] -> i64[n]\nmain a xs = affine a xs\n";
        let eval = run_main(src, "main", "[2,[1,2,3]]")
            .unwrap()
            .to_json_string();
        let wasm = run_wasm_main(src, "main", "[2,[1,2,3]]")
            .unwrap()
            .to_json_string();
        assert_eq!(eval, wasm);
        assert_eq!(wasm, "[6,9,12]");
    }

    #[test]
    fn wasm_matches_let_bound_kernel() {
        let src = "affine : i64 -> i64 -> i64\naffine a x = let b = a + 1; c = b * x in c + b\nmain : i64 -> i64[n] -> i64[n]\nmain a xs = affine a xs\n";
        let eval = run_main(src, "main", "[2,[1,2,3]]")
            .unwrap()
            .to_json_string();
        let wasm = run_wasm_main(src, "main", "[2,[1,2,3]]")
            .unwrap()
            .to_json_string();
        assert_eq!(eval, wasm);
        assert_eq!(wasm, "[6,9,12]");
    }

    #[test]
    fn wasm_matches_rank2_square() {
        let src = "square : i64 -> i64\nsquare x = x * x\nmain : i64[n,m] -> i64[n,m]\nmain xs = square xs\n";
        assert_eq!(
            wasm_run(src, "main", "[[[1,2,3],[4,5,6]]]"),
            "[[1,4,9],[16,25,36]]"
        );
    }

    #[test]
    fn wasm_matches_scalar_entry() {
        let src = "pow2 : i64 -> i64 -> i64\npow2 0 x = x\npow2 n x = pow2 (n - 1) (x * 2)\nmain : i64 -> i64\nmain x = pow2 3 x\n";
        assert_eq!(wasm_run(src, "main", "[7]"), "56");
    }

    #[test]
    fn wasm_matches_prefix_slice_pattern_for_rank1_bulk() {
        let src = "main : i64[n] -> i64\nmain [1, 2, ...] = 7\nmain _ = 0\n";
        assert_eq!(wasm_run(src, "main", "[[1,2,3,4]]"), "7");
        assert_eq!(wasm_run(src, "main", "[[1,3,4]]"), "0");
        assert_eq!(wasm_run(src, "main", "[[1]]"), "0");
    }

    #[test]
    fn wasm_matches_suffix_slice_pattern_for_rank1_bulk() {
        let src = "main : i64[n] -> i64\nmain [..., 4, 5] = 9\nmain _ = 0\n";
        assert_eq!(wasm_run(src, "main", "[[1,4,5]]"), "9");
        assert_eq!(wasm_run(src, "main", "[[4,6]]"), "0");
    }

    #[test]
    fn wasm_matches_slice_rest_binding_for_rank1_bulk() {
        let src = "head : i64[n] -> i64\nhead [x, ...] = x\nhead _ = 0\nmain : i64[n] -> i64\nmain [_, ...rest] = head rest\nmain _ = 0\n";
        assert_eq!(wasm_run(src, "main", "[[5,6,7]]"), "6");
        assert_eq!(wasm_run(src, "main", "[[5]]"), "0");
    }

    #[test]
    fn wasm_lowers_string_param_to_char_bulk_abi() {
        let src = "main : string -> i64\nmain _ = 0\n";
        let artifact = compile_wasm_main(src, "main").expect("Wasm should compile");
        assert_eq!(
            artifact.params,
            vec![WasmParamAbi::Bulk { prim: Prim::Char }]
        );
        assert_eq!(artifact.result, WasmResultAbi::Scalar { prim: Prim::I64 });
    }

    #[test]
    fn wasm_matches_char_entry_roundtrip() {
        let src = "main : char -> char\nmain c = c\n";
        assert_eq!(wasm_run(src, "main", "[\"a\"]"), "\"a\"");
    }

    #[test]
    fn wasm_matches_prefix_slice_pattern_for_string() {
        let src = "starts : string -> i64\nstarts ['c', 'a', 'r', ...] = 1\nstarts _ = 0\nmain : string -> i64\nmain s = starts s\n";
        assert_eq!(wasm_run(src, "main", "[\"carpet\"]"), "1");
        assert_eq!(wasm_run(src, "main", "[\"cat\"]"), "0");
    }

    #[test]
    fn wasm_matches_slice_rest_binding_for_string() {
        let src = "head : string -> char\nhead [x, ...] = x\nhead _ = 'z'\nmain : string -> char\nmain [_, ...rest] = head rest\nmain _ = 'z'\n";
        assert_eq!(wasm_run(src, "main", "[\"car\"]"), "\"a\"");
        assert_eq!(wasm_run(src, "main", "[\"c\"]"), "\"z\"");
    }

    #[test]
    fn wasm_json_parser_adt_example_runs() {
        let src = include_str!("../examples/json_parser_adt.simd");
        let eval_ok = run_main(src, "main", "[\"null\"]")
            .expect("json parser adt example should run in evaluator")
            .to_json_string();
        let wasm_ok = run_wasm_main(src, "main", "[\"null\"]")
            .expect("json parser adt example should run in wasm")
            .to_json_string();
        assert_eq!(eval_ok, wasm_ok);
        assert_eq!(wasm_ok, "10");

        let eval_nested = run_main(src, "main", "[\"[1,[2],null]\"]")
            .expect("json parser adt example should run in evaluator")
            .to_json_string();
        let wasm_nested = run_wasm_main(src, "main", "[\"[1,[2],null]\"]")
            .expect("json parser adt example should run in wasm")
            .to_json_string();
        assert_eq!(eval_nested, wasm_nested);
        assert_eq!(wasm_nested, "813");

        let eval_long = run_main(src, "main", "[\"[0,1,2,3,4,5,6,7,8,9,10]\"]")
            .expect("json parser adt example should handle longer arrays in evaluator")
            .to_json_string();
        let wasm_long = run_wasm_main(src, "main", "[\"[0,1,2,3,4,5,6,7,8,9,10]\"]")
            .expect("json parser adt example should handle longer arrays in wasm")
            .to_json_string();
        assert_eq!(eval_long, wasm_long);
        assert_eq!(wasm_long, "1455");

        let eval_long_spaced = run_main(src, "main", "[\"[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\"]")
            .expect("json parser adt example should handle spaced longer arrays in evaluator")
            .to_json_string();
        let wasm_long_spaced =
            run_wasm_main(src, "main", "[\"[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]\"]")
                .expect("json parser adt example should handle spaced longer arrays in wasm")
                .to_json_string();
        assert_eq!(eval_long_spaced, wasm_long_spaced);
        assert_eq!(wasm_long_spaced, "1400");

        let eval_err = run_main(src, "main", "[\"oops\"]")
            .expect("json parser adt example should run in evaluator")
            .to_json_string();
        let wasm_err = run_wasm_main(src, "main", "[\"oops\"]")
            .expect("json parser adt example should run in wasm")
            .to_json_string();
        assert_eq!(eval_err, wasm_err);
        assert_eq!(wasm_err, "-100");

        let compiled = compile_source(src).expect("json parser adt example should compile");
        let eval_object = run_compiled_main(
            &compiled,
            "main",
            &[Value::String("{\"a\":1,\"b\":[2]}".to_string())],
        )
        .expect("json parser adt example should run in evaluator with direct string args")
        .to_json_string();
        let artifact =
            compile_wasm_main(src, "main").expect("json parser adt example should compile to wasm");
        let wasm_object = run_wasm_artifact(
            &artifact,
            &[Value::String("{\"a\":1,\"b\":[2]}".to_string())],
        )
        .expect("json parser adt example should run in wasm with direct string args")
        .to_json_string();
        assert_eq!(eval_object, wasm_object);
        assert_eq!(wasm_object, "905");

        let escaped_input = Value::String("\"\\\\\"".to_string());
        let eval_escaped = run_compiled_main(&compiled, "main", &[escaped_input.clone()])
            .expect("json parser adt example should handle escaped chars in evaluator")
            .to_json_string();
        let wasm_escaped = run_wasm_artifact(&artifact, &[escaped_input])
            .expect("json parser adt example should handle escaped chars in wasm")
            .to_json_string();
        assert_eq!(eval_escaped, wasm_escaped);
        assert_eq!(wasm_escaped, "201");

        let eval_true = run_compiled_main(&compiled, "main", &[Value::String("true".to_string())])
            .expect("json parser adt example should parse true in evaluator")
            .to_json_string();
        let wasm_true = run_wasm_artifact(&artifact, &[Value::String("true".to_string())])
            .expect("json parser adt example should parse true in wasm")
            .to_json_string();
        assert_eq!(eval_true, wasm_true);
        assert_eq!(wasm_true, "21");

        let eval_neg = run_compiled_main(&compiled, "main", &[Value::String("-12".to_string())])
            .expect("json parser adt example should parse negative integer in evaluator")
            .to_json_string();
        let wasm_neg = run_wasm_artifact(&artifact, &[Value::String("-12".to_string())])
            .expect("json parser adt example should parse negative integer in wasm")
            .to_json_string();
        assert_eq!(eval_neg, wasm_neg);
        assert_eq!(wasm_neg, "88");

        let eval_unicode = run_compiled_main(
            &compiled,
            "main",
            &[Value::String("\"\\u1234\"".to_string())],
        )
        .expect("json parser adt example should parse unicode escape shape in evaluator")
        .to_json_string();
        let wasm_unicode =
            run_wasm_artifact(&artifact, &[Value::String("\"\\u1234\"".to_string())])
                .expect("json parser adt example should parse unicode escape shape in wasm")
                .to_json_string();
        assert_eq!(eval_unicode, wasm_unicode);
        assert_eq!(wasm_unicode, "201");

        let eval_unicode_12af = run_compiled_main(
            &compiled,
            "main",
            &[Value::String("\"\\u12af\"".to_string())],
        )
        .expect("json parser adt example should parse lowercase unicode escape in evaluator")
        .to_json_string();
        let wasm_unicode_12af =
            run_wasm_artifact(&artifact, &[Value::String("\"\\u12af\"".to_string())])
                .expect("json parser adt example should parse lowercase unicode escape in wasm")
                .to_json_string();
        assert_eq!(eval_unicode_12af, wasm_unicode_12af);
        assert_eq!(wasm_unicode_12af, "201");

        let eval_invalid_escape =
            run_compiled_main(&compiled, "main", &[Value::String("\"\\x\"".to_string())])
                .expect("json parser adt example should report invalid escape in evaluator")
                .to_json_string();
        let wasm_invalid_escape =
            run_wasm_artifact(&artifact, &[Value::String("\"\\x\"".to_string())])
                .expect("json parser adt example should report invalid escape in wasm")
                .to_json_string();
        assert_eq!(eval_invalid_escape, wasm_invalid_escape);
        assert_eq!(wasm_invalid_escape, "-202");

        let eval_invalid_unicode = run_compiled_main(
            &compiled,
            "main",
            &[Value::String("\"\\u12x4\"".to_string())],
        )
        .expect("json parser adt example should reject non-hex unicode escape in evaluator")
        .to_json_string();
        let wasm_invalid_unicode =
            run_wasm_artifact(&artifact, &[Value::String("\"\\u12x4\"".to_string())])
                .expect("json parser adt example should reject non-hex unicode escape in wasm")
                .to_json_string();
        assert_eq!(eval_invalid_unicode, wasm_invalid_unicode);
        assert_eq!(wasm_invalid_unicode, "-202");

        let surrogate_pair_input = Value::String("\"\\ud834\\udd1e\"".to_string());
        let eval_surrogate_pair =
            run_compiled_main(&compiled, "main", &[surrogate_pair_input.clone()])
                .expect("json parser adt example should reject surrogate pairs in evaluator")
                .to_json_string();
        let wasm_surrogate_pair = run_wasm_artifact(&artifact, &[surrogate_pair_input])
            .expect("json parser adt example should reject surrogate pairs in wasm")
            .to_json_string();
        assert_eq!(eval_surrogate_pair, wasm_surrogate_pair);
        assert_eq!(wasm_surrogate_pair, "-202");

        let eval_unpaired_high = run_compiled_main(
            &compiled,
            "main",
            &[Value::String("\"\\ud834\"".to_string())],
        )
        .expect("json parser adt example should reject unpaired high surrogates in evaluator")
        .to_json_string();
        let wasm_unpaired_high =
            run_wasm_artifact(&artifact, &[Value::String("\"\\ud834\"".to_string())])
                .expect("json parser adt example should reject unpaired high surrogates in wasm")
                .to_json_string();
        assert_eq!(eval_unpaired_high, wasm_unpaired_high);
        assert_eq!(wasm_unpaired_high, "-202");

        let eval_unpaired_low = run_compiled_main(
            &compiled,
            "main",
            &[Value::String("\"\\udd1e\"".to_string())],
        )
        .expect("json parser adt example should reject unpaired low surrogates in evaluator")
        .to_json_string();
        let wasm_unpaired_low =
            run_wasm_artifact(&artifact, &[Value::String("\"\\udd1e\"".to_string())])
                .expect("json parser adt example should reject unpaired low surrogates in wasm")
                .to_json_string();
        assert_eq!(eval_unpaired_low, wasm_unpaired_low);
        assert_eq!(wasm_unpaired_low, "-202");

        let eval_trailing =
            run_compiled_main(&compiled, "main", &[Value::String("nullx".to_string())])
                .expect("json parser adt example should report trailing garbage in evaluator")
                .to_json_string();
        let wasm_trailing = run_wasm_artifact(&artifact, &[Value::String("nullx".to_string())])
            .expect("json parser adt example should report trailing garbage in wasm")
            .to_json_string();
        assert_eq!(eval_trailing, wasm_trailing);
        assert_eq!(wasm_trailing, "-999");

        let eval_decimal =
            run_compiled_main(&compiled, "main", &[Value::String("1.25".to_string())])
                .expect("json parser adt example should parse decimal number in evaluator")
                .to_json_string();
        let wasm_decimal = run_wasm_artifact(&artifact, &[Value::String("1.25".to_string())])
            .expect("json parser adt example should parse decimal number in wasm")
            .to_json_string();
        assert_eq!(eval_decimal, wasm_decimal);
        assert_eq!(wasm_decimal, "101");

        let eval_exp = run_compiled_main(&compiled, "main", &[Value::String("2e9".to_string())])
            .expect("json parser adt example should parse exponent number in evaluator")
            .to_json_string();
        let wasm_exp = run_wasm_artifact(&artifact, &[Value::String("2e9".to_string())])
            .expect("json parser adt example should parse exponent number in wasm")
            .to_json_string();
        assert_eq!(eval_exp, wasm_exp);
        assert_eq!(wasm_exp, "102");

        let eval_exp_negative =
            run_compiled_main(&compiled, "main", &[Value::String("-3.4e5".to_string())])
                .expect("json parser adt example should parse signed exponent number in evaluator")
                .to_json_string();
        let wasm_exp_negative =
            run_wasm_artifact(&artifact, &[Value::String("-3.4e5".to_string())])
                .expect("json parser adt example should parse signed exponent number in wasm")
                .to_json_string();
        assert_eq!(eval_exp_negative, wasm_exp_negative);
        assert_eq!(wasm_exp_negative, "97");

        let eval_decimal_dot =
            run_compiled_main(&compiled, "main", &[Value::String("1.".to_string())])
                .expect("json parser adt example should reject dangling decimal point in evaluator")
                .to_json_string();
        let wasm_decimal_dot = run_wasm_artifact(&artifact, &[Value::String("1.".to_string())])
            .expect("json parser adt example should reject dangling decimal point in wasm")
            .to_json_string();
        assert_eq!(eval_decimal_dot, wasm_decimal_dot);
        assert_eq!(wasm_decimal_dot, "-302");

        let eval_exponent =
            run_compiled_main(&compiled, "main", &[Value::String("1e".to_string())])
                .expect("json parser adt example should reject dangling exponent in evaluator")
                .to_json_string();
        let wasm_exponent = run_wasm_artifact(&artifact, &[Value::String("1e".to_string())])
            .expect("json parser adt example should reject dangling exponent in wasm")
            .to_json_string();
        assert_eq!(eval_exponent, wasm_exponent);
        assert_eq!(wasm_exponent, "-303");

        let eval_empty_array =
            run_compiled_main(&compiled, "main", &[Value::String("[]".to_string())])
                .expect("json parser adt example should parse empty array in evaluator")
                .to_json_string();
        let wasm_empty_array = run_wasm_artifact(&artifact, &[Value::String("[]".to_string())])
            .expect("json parser adt example should parse empty array in wasm")
            .to_json_string();
        assert_eq!(eval_empty_array, wasm_empty_array);
        assert_eq!(wasm_empty_array, "300");

        let eval_empty_object =
            run_compiled_main(&compiled, "main", &[Value::String("{}".to_string())])
                .expect("json parser adt example should parse empty object in evaluator")
                .to_json_string();
        let wasm_empty_object = run_wasm_artifact(&artifact, &[Value::String("{}".to_string())])
            .expect("json parser adt example should parse empty object in wasm")
            .to_json_string();
        assert_eq!(eval_empty_object, wasm_empty_object);
        assert_eq!(wasm_empty_object, "400");

        let whitespace_case = Value::String("[ 1 , [ 2 , 3 ] ] ".to_string());
        let eval_whitespace = run_compiled_main(&compiled, "main", &[whitespace_case.clone()])
            .expect("json parser adt example should parse whitespace nested values in evaluator")
            .to_json_string();
        let wasm_whitespace = run_wasm_artifact(&artifact, &[whitespace_case])
            .expect("json parser adt example should parse whitespace nested values in wasm")
            .to_json_string();
        assert_eq!(eval_whitespace, wasm_whitespace);
        assert_eq!(wasm_whitespace, "906");

        let valid_json_corpus = [
            "null",
            "true",
            "false",
            "0",
            "-0.0",
            "3.14",
            "\"\"",
            "\"line1\\nline2\"",
            "\"tab\\tchar\"",
            "\"quote:\\\"hi\\\"\"",
            "\"backspace:\\b and formfeed:\\f\"",
            "\"slash\\\\path\\\\to\\\\file\"",
            "\"snowman:\\u2603\"",
            "[ 1 , 2 , 3 ] ",
            "{ \"a\" : [ 1 , { \"b\" : true } , 3 ] }",
            "{\n\t\"a\":1,\n\t\"b\":[\n\t\tnull,\n\t\tfalse\n\t]\n}",
            "{}",
            "[]",
            "{\"a\":[],\"b\":{}}",
            "[1,\"two\",false,null,{\"x\":3}]",
            "{\"num\":1,\"str\":\"x\",\"bool\":true,\"nil\":null,\"arr\":[1,2,\"3\"]}",
            "[{\"a\":1},2,\"3\",true,[false,null],{\"x\":{}}]",
            "{\"a\":{\"b\":{\"c\":{\"d\":{\"e\":{\"f\":1}}}}}}",
            "[[[1,[2,[3,[4,[5]]]]]]]",
            "{\"root\":{\"level1\":[{\"level2\":{\"level3\":[{\"level4\":null}]}}]}}",
            "[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]",
            "[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]",
            "[{\"id\":1},{\"id\":2},{\"id\":3},{\"id\":4},{\"id\":5},{\"id\":6},{\"id\":7},{\"id\":8},{\"id\":9},{\"id\":10}]",
        ];
        for input in valid_json_corpus {
            let eval_value =
                run_compiled_main(&compiled, "main", &[Value::String(input.to_string())])
                    .expect("json parser adt valid corpus should run in evaluator")
                    .to_json_string();
            let wasm_value = run_wasm_artifact(&artifact, &[Value::String(input.to_string())])
                .expect("json parser adt valid corpus should run in wasm")
                .to_json_string();
            assert_eq!(
                eval_value, wasm_value,
                "json corpus parity mismatch for input: {input}"
            );
            let score = eval_value
                .parse::<i64>()
                .expect("json parser adt corpus output should be numeric i64");
            assert!(
                score > 0,
                "json parser adt should accept valid corpus input, got score {score} for input: {input}"
            );
        }

        let eval_malformed_minus =
            run_compiled_main(&compiled, "main", &[Value::String("-abc".to_string())])
                .expect("json parser adt example should report malformed minus in evaluator")
                .to_json_string();
        let wasm_malformed_minus =
            run_wasm_artifact(&artifact, &[Value::String("-abc".to_string())])
                .expect("json parser adt example should report malformed minus in wasm")
                .to_json_string();
        assert_eq!(eval_malformed_minus, wasm_malformed_minus);
        assert_eq!(wasm_malformed_minus, "-301");

        let object_case = Value::String("{\"ab\":1,\"c\":2}".to_string());
        let eval_key_lengths = run_compiled_main(&compiled, "main", &[object_case.clone()])
            .expect("json parser adt example should account for key lengths in evaluator")
            .to_json_string();
        let wasm_key_lengths = run_wasm_artifact(&artifact, &[object_case])
            .expect("json parser adt example should account for key lengths in wasm")
            .to_json_string();
        assert_eq!(eval_key_lengths, wasm_key_lengths);
        assert_eq!(wasm_key_lengths, "606");

        let eval_debug = run_main(src, "main_json", "[\"[1,null]\"]")
            .expect("json parser adt debug entry should run in evaluator")
            .to_json_string();
        let wasm_debug = run_wasm_main(src, "main_json", "[\"[1,null]\"]")
            .expect("json parser adt debug entry should run in wasm")
            .to_json_string();
        assert_eq!(eval_debug, wasm_debug);
        assert!(wasm_debug.contains("\"$enum\""));
        assert!(wasm_debug.contains("JArray"));

        let eval_debug_err = run_main(src, "main_json", "[\"1.\"]")
            .expect("json parser adt debug error should run in evaluator")
            .to_json_string();
        let wasm_debug_err = run_wasm_main(src, "main_json", "[\"1.\"]")
            .expect("json parser adt debug error should run in wasm")
            .to_json_string();
        assert_eq!(eval_debug_err, wasm_debug_err);
        assert!(wasm_debug_err.contains("JNum"));
        assert!(wasm_debug_err.contains("-302"));
    }

    #[test]
    fn wasm_matches_mixed_recursive_and_row_enum_constructor() {
        let src = concat!(
            "enum Json =\n",
            "  | JNull\n",
            "  | JNum i64\n",
            "  | JField i64 Json\n",
            "\n",
            "weight : Json -> i64\n",
            "weight JNull = 10\n",
            "weight (JNum n) = 100 + n\n",
            "weight (JField key_len value) = key_len + weight value\n",
            "\n",
            "main : i64\n",
            "main = weight (JField 2 (JNum 1))\n",
        );
        let eval = run_main(src, "main", "[]")
            .expect("mixed recursive constructor should run in evaluator")
            .to_json_string();
        let wasm = run_wasm_main(src, "main", "[]")
            .expect("mixed recursive constructor should run in wasm")
            .to_json_string();
        assert_eq!(eval, wasm);
        assert_eq!(wasm, "103");
    }

    #[test]
    fn wasm_matches_enum_star_seq_reverse_over_mixed_recursive_entries() {
        let src = concat!(
            "enum Json =\n",
            "  | JNull\n",
            "  | JNum i64\n",
            "  | JObject Json[*]\n",
            "  | JField i64 Json\n",
            "\n",
            "reverse_json_seq : Json[*] -> Json[*]\n",
            "reverse_json_seq xs = reverse xs\n",
            "\n",
            "json_weight : Json -> i64\n",
            "json_weight JNull = 10\n",
            "json_weight (JNum n) = 100 + n\n",
            "json_weight (JObject fields) = 400 + object_seq_weight fields\n",
            "json_weight (JField key_len value) = key_len + json_weight value\n",
            "json_weight _ = 0\n",
            "\n",
            "object_seq_weight : Json[*] -> i64\n",
            "object_seq_weight [] = 0\n",
            "object_seq_weight [entry, ...rest] = json_weight entry + object_seq_weight rest\n",
            "\n",
            "main : i64\n",
            "main = json_weight (JObject (reverse_json_seq [JField 1 (JNum 2), JField 2 (JNum 1)]))\n",
        );
        let eval = run_main(src, "main", "[]")
            .expect("enum T[*] reverse should run in evaluator")
            .to_json_string();
        let wasm = run_wasm_main(src, "main", "[]")
            .expect("enum T[*] reverse should run in wasm")
            .to_json_string();
        assert_eq!(eval, wasm);
        assert_eq!(wasm, "606");
    }

    #[test]
    fn wasm_matches_object_seed_with_flat_field_seq() {
        let src = concat!(
            "enum Json =\n",
            "  | JNull\n",
            "  | JNum i64\n",
            "  | JObject Json[*]\n",
            "  | JField i64 Json\n",
            "\n",
            "enum Either a b =\n",
            "  | Left a\n",
            "  | Right b\n",
            "\n",
            "enum Parsed a =\n",
            "  | Parsed a string\n",
            "\n",
            "enum ParsedMember a b =\n",
            "  | ParsedMember a b string\n",
            "\n",
            "parse_object_seed : Json[*] -> Either i64 (ParsedMember i64 Json) -> Either i64 (Parsed Json)\n",
            "parse_object_seed _ (Left c) = Left c\n",
            "parse_object_seed acc (Right (ParsedMember key_len value rest)) = parse_object_items [JField key_len value, ...acc] rest\n",
            "\n",
            "parse_object_items : Json[*] -> string -> Either i64 (Parsed Json)\n",
            "parse_object_items acc [' ', ...rest] = parse_object_items acc rest\n",
            "parse_object_items acc ['}', ...rest] = Right (Parsed (JObject acc) rest)\n",
            "parse_object_items _ _ = Left 501\n",
            "\n",
            "json_weight : Json -> i64\n",
            "json_weight JNull = 10\n",
            "json_weight (JNum n) = 100 + n\n",
            "json_weight (JObject fields) = 400 + object_seq_weight fields\n",
            "json_weight (JField key_len value) = key_len + json_weight value\n",
            "json_weight _ = 0\n",
            "\n",
            "object_seq_weight : Json[*] -> i64\n",
            "object_seq_weight [] = 0\n",
            "object_seq_weight [entry, ...rest] = json_weight entry + object_seq_weight rest\n",
            "\n",
            "parse_status : Either i64 (Parsed Json) -> i64\n",
            "parse_status (Left c) = 0 - c\n",
            "parse_status (Right (Parsed value rest)) = parse_status_rest value rest\n",
            "\n",
            "parse_status_rest : Json -> string -> i64\n",
            "parse_status_rest value [' ', ...rest] = parse_status_rest value rest\n",
            "parse_status_rest value [] = json_weight value\n",
            "parse_status_rest _ _ = 0 - 999\n",
            "\n",
            "main : string -> i64\n",
            "main s = parse_status (parse_object_seed [] (Right (ParsedMember 2 (JNum 1) s)))\n",
        );
        let eval = run_main(src, "main", "[\"}\"]")
            .expect("flat object seed should run in evaluator")
            .to_json_string();
        let wasm = run_wasm_main(src, "main", "[\"}\"]")
            .expect("flat object seed should run in wasm")
            .to_json_string();
        assert_eq!(eval, wasm);
        assert_eq!(wasm, "503");
    }

    #[test]
    fn wasm_matches_enum_star_seq_splice_with_constructor_prefix() {
        let src = concat!(
            "enum Json =\n",
            "  | JNull\n",
            "  | JNum i64\n",
            "  | JObject Json[*]\n",
            "  | JField i64 Json\n",
            "\n",
            "append_one : Json[*] -> Json[*]\n",
            "append_one acc = [JField 2 (JNum 1), ...acc]\n",
            "\n",
            "json_weight : Json -> i64\n",
            "json_weight JNull = 10\n",
            "json_weight (JNum n) = 100 + n\n",
            "json_weight (JObject fields) = 400 + object_seq_weight fields\n",
            "json_weight (JField key_len value) = key_len + json_weight value\n",
            "json_weight _ = 0\n",
            "\n",
            "object_seq_weight : Json[*] -> i64\n",
            "object_seq_weight [] = 0\n",
            "object_seq_weight [entry, ...rest] = json_weight entry + object_seq_weight rest\n",
            "\n",
            "main : i64\n",
            "main = json_weight (JObject (append_one []))\n",
        );
        let eval = run_main(src, "main", "[]")
            .expect("enum T[*] splice with constructor prefix should run in evaluator")
            .to_json_string();
        let wasm = run_wasm_main(src, "main", "[]")
            .expect("enum T[*] splice with constructor prefix should run in wasm")
            .to_json_string();
        assert_eq!(eval, wasm);
        assert_eq!(wasm, "503");
    }

    #[test]
    fn wasm_json_parser_adt_string_leaf_chars_harness_runs() {
        let src = include_str!("../examples/json_parser_adt.simd");
        let compiled = compile_source(src).expect("json parser adt example should compile");
        let artifact = compile_wasm_main(src, "main_string_leaf_chars")
            .expect("json parser adt string leaf harness should compile to wasm");

        let cases = [
            ("\"car\"", 3_i64),
            ("[\"car\",\"ed\",\"x\"]", 6_i64),
            (
                "{\"a\":\"car\",\"b\":[\"ed\",\"x\",{\"k\":\"milk\"}],\"c\":{\"d\":\"z\"}}",
                11_i64,
            ),
            ("[{\"x\":\"aa\"},{\"y\":[\"bbb\",\"cccc\"]},\"d\"]", 10_i64),
            (
                "[\n  \"a\",\n  {\"b\": [\"cd\", \"efg\"]},\n  \"hij\"\n]",
                9_i64,
            ),
        ];

        for (input, expected) in cases {
            let eval_value = run_compiled_main(
                &compiled,
                "main_string_leaf_chars",
                &[Value::String(input.to_string())],
            )
            .expect("json parser adt string leaf harness should run in evaluator")
            .to_json_string();
            let wasm_value = run_wasm_artifact(&artifact, &[Value::String(input.to_string())])
                .expect("json parser adt string leaf harness should run in wasm")
                .to_json_string();
            assert_eq!(
                eval_value, wasm_value,
                "string leaf harness parity mismatch for input: {input}"
            );
            let total = eval_value
                .parse::<i64>()
                .expect("string leaf harness output should be numeric i64");
            assert_eq!(
                total, expected,
                "string leaf harness total mismatch for input: {input}"
            );
        }
    }

    #[test]
    fn wasm_matches_recursive_enum_list_length() {
        let src = "enum List a =\n  | Nil\n  | Cons a (List a)\n\nlen : List i64 -> i64\nlen Nil = 0\nlen (Cons _ xs) = 1 + len xs\n\nmain : i64\nmain = len (Cons 1 (Cons 2 Nil))\n";
        assert_eq!(wasm_run(src, "main", "[]"), "2");
    }

    #[test]
    fn wasm_returns_recursive_enum_nil_debug() {
        let src = "enum List a =\n  | Nil\n  | Cons a (List a)\n\nmain : List i64\nmain = Nil\n";
        assert_eq!(
            wasm_run(src, "main", "[]"),
            "{\"$enum\":\"Nil\",\"fields\":[]}"
        );
    }

    #[test]
    fn wasm_returns_recursive_enum_cons_debug() {
        let src =
            "enum List a =\n  | Nil\n  | Cons a (List a)\n\nmain : List i64\nmain = Cons 1 Nil\n";
        let value = run_wasm_main(src, "main", "[]").expect("cons debug should run");
        assert_eq!(
            value.to_json_string(),
            "{\"$enum\":\"Cons\",\"fields\":[1,{\"$enum\":\"Nil\",\"fields\":[]}]}"
        );
    }

    #[test]
    fn wasm_matches_recursive_enum_tree_sum() {
        let src = "enum Tree a =\n  | Leaf a\n  | Bin (Tree a) (Tree a)\n\nsum : Tree i64 -> i64\nsum (Leaf x) = x\nsum (Bin l r) = sum l + sum r\n\nmain : i64\nmain = sum (Bin (Leaf 2) (Bin (Leaf 3) (Leaf 4)))\n";
        assert_eq!(wasm_run(src, "main", "[]"), "9");
    }

    #[test]
    fn wasm_matches_recursive_enum_tree_sum_pair() {
        let src = "enum Tree a =\n  | Leaf a\n  | Bin (Tree a) (Tree a)\n\nsum : Tree i64 -> i64\nsum (Leaf x) = x\nsum (Bin l r) = sum l + sum r\n\nmain : i64\nmain = sum (Bin (Leaf 2) (Leaf 3))\n";
        assert_eq!(wasm_run(src, "main", "[]"), "5");
    }

    #[test]
    fn wasm_matches_recursive_enum_tree_right_subtree_sum() {
        let src = "enum Tree a =\n  | Leaf a\n  | Bin (Tree a) (Tree a)\n\nright : Tree i64 -> Tree i64\nright (Bin _ r) = r\nright t = t\n\nsum : Tree i64 -> i64\nsum (Leaf x) = x\nsum (Bin l r) = sum l + sum r\n\nmain : i64\nmain = sum (right (Bin (Leaf 2) (Bin (Leaf 3) (Leaf 4))))\n";
        assert_eq!(wasm_run(src, "main", "[]"), "7");
    }

    #[test]
    fn wasm_matches_recursive_enum_tree_right_subtree_sum_after_left_sum() {
        let src = "enum Tree a =\n  | Leaf a\n  | Bin (Tree a) (Tree a)\n\nleft : Tree i64 -> Tree i64\nleft (Bin l _) = l\nleft t = t\n\nright : Tree i64 -> Tree i64\nright (Bin _ r) = r\nright t = t\n\nsum : Tree i64 -> i64\nsum (Leaf x) = x\nsum (Bin l r) = sum l + sum r\n\nprobe : Tree i64 -> i64\nprobe t = let x = sum (left t) in sum (right t)\n\nmain : i64\nmain = probe (Bin (Leaf 2) (Bin (Leaf 3) (Leaf 4)))\n";
        assert_eq!(wasm_run(src, "main", "[]"), "7");
    }

    #[test]
    fn wasm_matches_recursive_enum_tree_sum_with_lets() {
        let src = "enum Tree a =\n  | Leaf a\n  | Bin (Tree a) (Tree a)\n\nsum : Tree i64 -> i64\nsum (Leaf x) = x\nsum (Bin l r) = let a = sum l in let b = sum r in a + b\n\nmain : i64\nmain = sum (Bin (Leaf 2) (Bin (Leaf 3) (Leaf 4)))\n";
        assert_eq!(wasm_run(src, "main", "[]"), "9");
    }

    #[test]
    fn wasm_matches_recursive_enum_clause_bound_right_subtree_sum_after_left_sum() {
        let src = "enum Tree a =\n  | Leaf a\n  | Bin (Tree a) (Tree a)\n\nsum : Tree i64 -> i64\nsum (Leaf x) = x\nsum (Bin l r) = sum l + sum r\n\nprobe : Tree i64 -> i64\nprobe (Leaf x) = x\nprobe (Bin l r) = let a = sum l in sum r\n\nmain : i64\nmain = probe (Bin (Leaf 2) (Bin (Leaf 3) (Leaf 4)))\n";
        assert_eq!(wasm_run(src, "main", "[]"), "7");
    }

    #[test]
    fn wasm_returns_recursive_enum_bin_debug() {
        let src = "enum Tree a =\n  | Leaf a\n  | Bin (Tree a) (Tree a)\n\nmain : Tree i64\nmain = Bin (Leaf 2) (Leaf 3)\n";
        assert_eq!(
            wasm_run(src, "main", "[]"),
            "{\"$enum\":\"Bin\",\"fields\":[{\"$enum\":\"Leaf\",\"fields\":[2]},{\"$enum\":\"Leaf\",\"fields\":[3]}]}"
        );
    }

    #[test]
    fn wasm_returns_recursive_enum_nested_bin_debug() {
        let src = "enum Tree a =\n  | Leaf a\n  | Bin (Tree a) (Tree a)\n\nmain : Tree i64\nmain = Bin (Leaf 2) (Bin (Leaf 3) (Leaf 4))\n";
        assert_eq!(
            wasm_run(src, "main", "[]"),
            "{\"$enum\":\"Bin\",\"fields\":[{\"$enum\":\"Leaf\",\"fields\":[2]},{\"$enum\":\"Bin\",\"fields\":[{\"$enum\":\"Leaf\",\"fields\":[3]},{\"$enum\":\"Leaf\",\"fields\":[4]}]}]}"
        );
    }

    #[test]
    fn wasm_returns_recursive_enum_leaf_debug() {
        let src = "enum Tree a =\n  | Leaf a\n  | Bin (Tree a) (Tree a)\n\nmain : Tree i64\nmain = Leaf 2\n";
        assert_eq!(
            wasm_run(src, "main", "[]"),
            "{\"$enum\":\"Leaf\",\"fields\":[2]}"
        );
    }

    #[test]
    fn wasm_returns_binary_recursive_enum_without_payloads() {
        let src = "enum T =\n  | A\n  | B T T\n\nmain : T\nmain = B A A\n";
        assert_eq!(
            wasm_run(src, "main", "[]"),
            "{\"$enum\":\"B\",\"fields\":[{\"$enum\":\"A\",\"fields\":[]},{\"$enum\":\"A\",\"fields\":[]}]}"
        );
    }

    #[test]
    fn wasm_matches_enum_payload_row_with_nested_enum_field() {
        let src = "enum Json =\n  | JNull\n\nenum ParseResult a =\n  | POk a i64\n  | PErr i64\n\nunwrap : ParseResult Json -> i64\nunwrap (POk JNull _) = 10\nunwrap _ = 0 - 999\n\nmain : i64\nmain = unwrap (POk JNull 0)\n";
        assert_eq!(wasm_run(src, "main", "[]"), "10");
    }

    #[test]
    fn wasm_matches_enum_payload_row_with_nested_enum_and_string_fields() {
        let src = "enum Json =\n  | JNull\n\nenum ParseResult a =\n  | POk a string\n  | PErr i64\n\nunwrap : ParseResult Json -> i64\nunwrap (POk JNull rest) = 10\nunwrap _ = 0 - 999\n\nmain : string -> i64\nmain s = unwrap (POk JNull s)\n";
        assert_eq!(wasm_run(src, "main", "[\"car\"]"), "10");
    }

    #[test]
    fn wasm_matches_enum_payload_row_string_field_as_empty_slice() {
        let src = "enum Json =\n  | JNull\n\nenum ParseResult a =\n  | POk a string\n  | PErr i64\n\nrest_score : string -> i64\nrest_score [] = 10\nrest_score _ = 0 - 999\n\nunwrap : ParseResult Json -> i64\nunwrap (POk JNull rest) = rest_score rest\nunwrap _ = 0 - 999\n\nmain : string -> i64\nmain s = unwrap (POk JNull s)\n";
        assert_eq!(wasm_run(src, "main", "[\"\"]"), "10");
    }

    #[test]
    fn wasm_returns_tuple_json_array() {
        let src = "main : (i64, i64)\nmain = (1, 2)\n";
        assert_eq!(wasm_run(src, "main", "[]"), "[1,2]");
    }

    #[test]
    fn wasm_matches_tuple_argument_destructuring() {
        let src = "fst : (i64, i64) -> i64\nfst (x, y) = x\nmain : i64\nmain = fst (4, 9)\n";
        assert_eq!(wasm_run(src, "main", "[]"), "4");
    }

    #[test]
    fn wasm_matches_tuple_projection() {
        let src = "main : i64\nmain = (1, true).0\n";
        assert_eq!(wasm_run(src, "main", "[]"), "1");
    }

    #[test]
    fn wasm_accepts_tuple_json_array_argument() {
        let src = "swap : (i64, i64) -> (i64, i64)\nswap (x, y) = (y, x)\nmain : (i64, i64) -> (i64, i64)\nmain p = swap p\n";
        assert_eq!(wasm_run(src, "main", "[[3,4]]"), "[4,3]");
    }

    #[test]
    fn wasm_matches_record_bulk_entry() {
        let src = "bump : {x:i64,y:i64} -> {x:i64,y:i64}\nbump p = p { x = p.x + 1, y = p.y + 2 }\nmain : {x:i64,y:i64}[n] -> {x:i64,y:i64}[n]\nmain ps = bump ps\n";
        assert_eq!(
            wasm_run(src, "main", "[[{\"x\":1,\"y\":10},{\"x\":2,\"y\":20}]]"),
            "[{\"x\":2,\"y\":12},{\"x\":3,\"y\":22}]"
        );
    }

    #[test]
    fn wasm_matches_scalar_plus_bulk_record_entry() {
        let src = "axpy2 : i64 -> {x:i64,y:i64} -> {x:i64,y:i64} -> {x:i64,y:i64}\naxpy2 a u v = { x = a * u.x + v.x, y = a * u.y + v.y }\nmain : i64 -> {x:i64,y:i64}[n] -> {x:i64,y:i64}[n] -> {x:i64,y:i64}[n]\nmain a us vs = axpy2 a us vs\n";
        assert_eq!(
            wasm_run(
                src,
                "main",
                "[2,[{\"x\":1,\"y\":3},{\"x\":2,\"y\":4}],[{\"x\":10,\"y\":20},{\"x\":30,\"y\":40}]]"
            ),
            "[{\"x\":12,\"y\":26},{\"x\":34,\"y\":48}]"
        );
    }

    #[test]
    fn grouped_record_kernel_fuses_helper_call_leaves() {
        let src = "axpy2 : i64 -> {x:i64,y:i64} -> {x:i64,y:i64} -> {x:i64,y:i64}\naxpy2 a u v = { x = a * u.x + v.x, y = a * u.y + v.y }\nmain : i64 -> {x:i64,y:i64}[n] -> {x:i64,y:i64}[n] -> {x:i64,y:i64}[n]\nmain a us vs = axpy2 a us vs\n";
        let compiled = compile_source(src).unwrap();
        let plan = build_wasm_plan(&compiled.checked, "main").unwrap();
        let lowered_map = compiled
            .lowered
            .functions
            .iter()
            .map(|function| (function.name.clone(), function))
            .collect::<BTreeMap<_, _>>();
        let scalar_indices = compiled
            .lowered
            .functions
            .iter()
            .enumerate()
            .filter_map(|(index, function)| match function.kind {
                LoweredKind::Scalar { .. } | LoweredKind::Structural { .. } => {
                    Some((function.name.clone(), index as u32))
                }
                LoweredKind::Kernel { .. } => None,
            })
            .collect::<BTreeMap<_, _>>();
        let signatures = plan
            .checked
            .functions
            .iter()
            .map(|function| (function.name.clone(), function))
            .collect::<BTreeMap<_, _>>();

        let group = compiled
            .grouped
            .functions
            .iter()
            .find(|group| group.source_name == "main" && group.leaf_paths.len() == 2)
            .expect("expected grouped record kernel for main");
        let compiled = compile_grouped_kernel_function(
            group,
            &lowered_map,
            &scalar_indices,
            &signatures,
            IntentClass::GroupedMap,
        )
        .unwrap();
        assert!(compiled.is_some());
    }

    #[test]
    fn wasm_marks_unary_record_leaf_buffers_as_reusable() {
        let src = "bump : {x:i64,y:i64} -> {x:i64,y:i64}\nbump p = p { x = p.x + 1, y = p.y + 2 }\nmain : {x:i64,y:i64}[n] -> {x:i64,y:i64}[n]\nmain ps = bump ps\n";
        let artifact = compile_wasm_main(src, "main").expect("record program should compile");
        let leaves = artifact
            .leaf_exports
            .iter()
            .map(|leaf| (leaf.export_name.clone(), leaf.reusable_param_leaf))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(leaves.get("main$x"), Some(&Some(0)));
        assert_eq!(leaves.get("main$y"), Some(&Some(1)));
    }

    #[test]
    fn wasm_skips_reuse_for_double_used_leaf_buffer() {
        let src = "square : {x:i64,y:i64} -> {x:i64,y:i64}\nsquare p = { x = p.x * p.x, y = p.y * p.y }\nmain : {x:i64,y:i64}[n] -> {x:i64,y:i64}[n]\nmain ps = square ps\n";
        let artifact = compile_wasm_main(src, "main").expect("record program should compile");
        let leaves = artifact
            .leaf_exports
            .iter()
            .map(|leaf| (leaf.export_name.clone(), leaf.reusable_param_leaf))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(leaves.get("main$x"), Some(&None));
        assert_eq!(leaves.get("main$y"), Some(&None));
    }

    #[test]
    fn structural_seq_fold_detects_inlineable_enum_matcher() {
        let src = "enum Json =\n  | JNull\n  | JNum i64\n\njson_weight : Json -> i64\njson_weight JNull = 10\njson_weight (JNum n) = 100 + n\njson_weight _ = 0\n\narray_seq_weight : Json[*] -> i64\narray_seq_weight [] = 0\narray_seq_weight [value, ...rest] = json_weight value + array_seq_weight rest\n\nmain : i64\nmain = array_seq_weight [JNum 1, JNum 2]\n";
        let compiled = compile_source(src).expect("source should compile");
        let lowered_map = compiled
            .lowered
            .functions
            .iter()
            .map(|function| (function.name.clone(), function))
            .collect::<BTreeMap<_, _>>();
        let signatures = compiled
            .checked
            .functions
            .iter()
            .map(|function| (function.name.clone(), function))
            .collect::<BTreeMap<_, _>>();
        let matcher = detect_inlineable_enum_matcher(
            &lowered_map,
            &signatures,
            &compiled.checked.enum_ctors,
            "json_weight",
        )
        .expect("json_weight should be an inlineable enum matcher");
        assert_eq!(matcher.clauses.len(), 3);
    }

    #[test]
    fn wasm_parity_matches_evaluator() {
        let src = "axpy : i64 -> i64 -> i64 -> i64\naxpy a x y = a * x + y\nmain : i64 -> i64[n] -> i64[n] -> i64[n]\nmain a xs ys = axpy a xs ys\n";
        let eval = run_main(src, "main", "[3,[1,2,3],[4,5,6]]")
            .unwrap()
            .to_json_string();
        let wasm = run_wasm_main(src, "main", "[3,[1,2,3],[4,5,6]]")
            .unwrap()
            .to_json_string();
        assert_eq!(eval, wasm);
    }

    #[test]
    fn wasm_function_profiler_reports_calls_and_time() {
        let source = r#"
bump : i64 -> i64
bump x = x + 1

twice : i64 -> i64
twice x = bump (bump x)

main : i64 -> i64
main x = twice x
"#;
        let profile =
            run_wasm_profile_fns_main(source, "main", "[7]").expect("Wasm function profile");
        assert_eq!(profile.result_json, "9");
        let rows = profile
            .functions
            .iter()
            .map(|row| (row.name.as_str(), (row.calls, row.total_us)))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(rows.get("main").map(|(calls, _)| *calls), Some(1));
        assert!(rows.get("main").is_some_and(|(_, total)| *total > 0));
        assert!(rows.contains_key("bump"));
        assert!(rows.contains_key("twice"));
    }

    #[test]
    fn wasm_function_profiler_includes_structural_functions() {
        let source = r#"
enum List a =
  | Nil
  | Cons a (List a)

len : List i64 -> i64
len Nil = 0
len (Cons _ xs) = 1 + len xs

main : i64
main = len (Cons 1 (Cons 2 Nil))
"#;
        let profile = run_wasm_profile_fns_main(source, "main", "[]")
            .expect("Wasm structural function profile");
        let len_row = profile
            .functions
            .iter()
            .find(|row| row.name == "len")
            .expect("len should appear in Wasm profile");
        assert!(len_row.calls >= 1);
        assert!(len_row.total_us > 0);
    }
}
