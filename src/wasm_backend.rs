use super::*;
use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::rc::Rc;

use wasm_encoder::{
    BlockType, CodeSection, ExportKind, ExportSection, Function, FunctionSection, Instruction,
    MemArg, MemorySection, MemoryType, Module, TypeSection, ValType,
};
use wasmtime::{
    Config, Engine, Func, Instance, Memory, Module as WasmtimeModule, OptLevel, Store, Val,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmArtifact {
    pub bytes: Vec<u8>,
    pub export_name: String,
    pub params: Vec<WasmParamAbi>,
    pub result: WasmResultAbi,
    pub grouped_export: Option<WasmGroupedExport>,
    pub leaf_exports: Vec<WasmLeafExport>,
    pub optimizer_reports: Vec<WasmOptimizationReport>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WasmOptimizationReport {
    pub function: String,
    pub intent: IntentClass,
    pub vectorizable: bool,
    pub vector_unroll: usize,
    pub fallback_reason: Option<String>,
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
    Bulk { prim: Prim },
}

pub fn compile_wasm_main(source: &str, main: &str) -> Result<WasmArtifact> {
    let (_surface, _module, checked) = compile_frontend(source)?;
    compile_wasm_artifact_checked(&checked, main)
}

pub fn prepare_wasm_main(source: &str, main: &str) -> Result<PreparedWasmMain> {
    let (_surface, _module, checked) = compile_frontend(source)?;
    let plan = build_wasm_plan(&checked, main)?;
    let artifact = compile_wasm_artifact_checked(&checked, main)?;
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
    let args = parse_host_args(args_json, &checked, main)?;
    let artifact = compile_wasm_artifact_checked(&checked, main)?;
    prepare_wasm_artifact(artifact)?.run(&args)
}

pub fn wasm_command(path: &str, main: &str, out: Option<&str>) -> Result<String> {
    let source = read_source_file(path)?;
    let artifact = compile_wasm_main(&source, main)?;
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
        "{}\n{}",
        render_optimizer_report_comments(&artifact.optimizer_reports),
        wat
    ))
}

pub fn wat_command(path: &str, main: &str) -> Result<String> {
    let source = read_source_file(path)?;
    wat_main(&source, main)
}

pub fn run_wasm_command(path: &str, main: &str, args_json: &str) -> Result<String> {
    let source = read_source_file(path)?;
    let value = run_wasm_main(&source, main, args_json)?;
    Ok(value.to_json_string())
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
        let fallback = report
            .fallback_reason
            .as_ref()
            .map(|reason| format!(" fallback={reason}"))
            .unwrap_or_default();
        lines.push(format!(
            ";; - fn={} intent={:?} plan={}{}",
            report.function, report.intent, plan, fallback
        ));
    }
    lines.join("\n")
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
        }
    }
    Ok(())
}

fn run_prepared_bound(bound: &mut BoundPreparedRun) -> Result<()> {
    let input_args = build_prepared_input_wasm_args(bound)?;
    let mut scalar_updates = Vec::<(usize, ScalarValue)>::new();
    let mut runtime = bound.runtime.borrow_mut();
    let mut scratch_results = vec![Val::I64(0)];
    match &bound.call_plan {
        PreparedCallPlan::DirectScalar {
            func_name,
            output_slot,
            prim,
        } => {
            let func = runtime_func(&runtime, func_name)?;
            func.call(&mut runtime.store, &input_args, &mut scratch_results)
                .map_err(|error| {
                    SimdError::new(format!("Wasm prepared execution failed: {error}"))
                })?;
            let value = scratch_results.first().ok_or_else(|| {
                SimdError::new("Wasm prepared scalar call did not return a value")
            })?;
            let scalar = wasmtime_to_scalar(value.clone(), *prim)?;
            scalar_updates.push((*output_slot, scalar));
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
        let ptr = align_to(self.arena_pinned_end, 16);
        let end = ptr
            .checked_add(bytes)
            .ok_or_else(|| SimdError::new("Wasm arena pinned allocation overflowed"))?;
        ensure_memory_size(&self.memory, &mut self.store, end)?;
        self.arena_pinned_end = end;
        self.arena_cursor = self.arena_cursor.max(end);
        Ok(ptr)
    }
}

fn compile_wasm_artifact_checked(
    checked_program: &CheckedProgram,
    main: &str,
) -> Result<WasmArtifact> {
    let mut plan = build_wasm_plan(checked_program, main)?;
    let normalized = normalize_records(&plan.checked)?;
    let lowered_program = optimize_lowered_program(&lower_program(&normalized)?);
    let grouped_program = group_lowered_program(&normalized, &lowered_program)?;
    let intent_analysis = analyze_intents(&grouped_program);
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
    let mut function_indices = BTreeMap::<String, u32>::new();
    let mut scalar_indices = BTreeMap::<String, u32>::new();
    for (index, function) in plan.checked.functions.iter().enumerate() {
        function_indices.insert(function.name.clone(), index as u32);
        if matches!(
            lowered_map
                .get(&function.name)
                .map(|function| &function.kind),
            Some(LoweredKind::Scalar { .. })
        ) {
            scalar_indices.insert(function.name.clone(), index as u32);
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
    let mut function_section = FunctionSection::new();
    let mut memory_section = MemorySection::new();
    let mut export_section = ExportSection::new();
    let mut code_section = CodeSection::new();
    let mut optimizer_reports = Vec::<WasmOptimizationReport>::new();

    for function in &plan.checked.functions {
        let lowered = lowered_map.get(&function.name).copied().ok_or_else(|| {
            SimdError::new(format!("missing lowered function '{}'", function.name))
        })?;
        let (params, result) = match lowered.kind {
            LoweredKind::Scalar { .. } => scalar_signature(function)?,
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
        let intent = intent_by_leaf
            .get(&function.name)
            .map(|report| report.intent.clone())
            .unwrap_or(IntentClass::Fallback);
        let wasm_function = match lowered.kind {
            LoweredKind::Scalar { .. } => {
                optimizer_reports.push(WasmOptimizationReport {
                    function: function.name.clone(),
                    intent: if lowered.tail_loop.is_some() {
                        IntentClass::ScalarTailRec
                    } else {
                        intent
                    },
                    vectorizable: false,
                    vector_unroll: 0,
                    fallback_reason: None,
                });
                compile_scalar_function(lowered, function, &scalar_indices, &signatures)?
            }
            LoweredKind::Kernel { .. } => {
                let compiled = compile_kernel_entry(
                    lowered,
                    function,
                    &lowered_map,
                    &scalar_indices,
                    &signatures,
                    intent,
                )?;
                optimizer_reports.push(compiled.report);
                compiled.function
            }
        };
        code_section.function(&wasm_function);
    }

    let mut next_function_index = plan.checked.functions.len() as u32;
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
    module.section(&function_section);
    module.section(&memory_section);
    module.section(&export_section);
    module.section(&code_section);

    Ok(WasmArtifact {
        bytes: module.finish(),
        export_name: main.to_string(),
        params: plan.params,
        result: plan.result,
        grouped_export,
        leaf_exports: plan.leaf_exports,
        optimizer_reports,
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
        Prim::I32 | Prim::F32 => weighted_ops >= 18 && total_streams <= 8,
        Prim::I64 | Prim::F64 => weighted_ops >= 28 && total_streams <= 5,
    };
    let vec2_ready = match result_prim {
        Prim::I32 | Prim::F32 => weighted_ops >= 8 || op_count >= 6,
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
            LoweredKind::Scalar { .. } => None,
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
        let prim = pattern.ty.prim().ok_or_else(|| {
            SimdError::new("grouped kernel parameter pattern did not have a primitive type")
        })?;
        match access {
            AccessKind::Same => {
                abi_params.push(KernelParam::Same {
                    prim,
                    value_local: wasm_param_index,
                });
                wasm_params.push(wasm_val_type(prim));
                wasm_param_index += 1;
            }
            AccessKind::Lane => {
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
            KernelParam::Same { .. } => {
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

    let first_len_local = abi_params
        .iter()
        .find_map(|param| match param {
            KernelParam::Lane { len_local, .. } => Some(*len_local),
            KernelParam::Same { .. } => None,
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
    let vector_hoists = vector_clause
        .as_ref()
        .map(|vector_clause| {
            if vector_clause.clauses.len() == 1 {
                collect_hoisted_exprs(
                    vector_clause.clauses[0].bodies.iter().copied(),
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
    let param_prims = reference_clause
        .patterns
        .iter()
        .map(|pattern| match &pattern.ty {
            Type::Scalar(prim) => Ok(*prim),
            other => Err(SimdError::new(format!(
                "grouped kernel clause pattern has non-scalar lowered type {:?}",
                other
            ))),
        })
        .collect::<Result<Vec<_>>>()?;

    for clause_index in 0..reference.clauses.len() {
        let reference_clause = &reference.clauses[clause_index];
        let clause_key = reference_clause as *const _ as usize;
        let locals = scalar_locals.get(&clause_key).cloned().unwrap_or_default();
        if clause_has_condition(&reference_clause.patterns) {
            emit_matching_if(
                &mut function,
                &reference_clause.patterns,
                &param_prims,
                Some(&locals),
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
            vectorizable: !matches!(vector_plan, VectorPlan::ScalarOnly),
            vector_unroll: vector_plan_unroll(vector_plan),
            fallback_reason,
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
    if clause_has_condition(&reference.clauses.last()?.patterns) {
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
        LoweredKind::Scalar { clauses } | LoweredKind::Kernel { clauses, .. } => clauses,
    };
    if clauses.len() != 1 {
        return Ok(None);
    }
    let clause = &clauses[0];
    if clause.patterns.len() != args.len()
        || clause
            .patterns
            .iter()
            .any(|pattern| matches!(pattern.pattern, Pattern::Int(_) | Pattern::Float(_)))
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
    let params = params
        .into_iter()
        .map(|ty| match ty {
            Type::Scalar(prim) => Ok(wasm_val_type(prim)),
            Type::Record(_) => Err(SimdError::new(format!(
                "scalar Wasm function '{}' cannot accept record parameters",
                function.name
            ))),
            other => Err(SimdError::new(format!(
                "scalar Wasm function '{}' cannot accept non-scalar parameter {:?}",
                function.name, other
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    let result = match result {
        Type::Scalar(prim) => Some(wasm_val_type(prim)),
        Type::Record(_) => {
            return Err(SimdError::new(format!(
                "scalar Wasm function '{}' cannot return records",
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
    Ok((params, result))
}

fn entry_signature(function: &CheckedFunction) -> Result<(Vec<ValType>, Option<ValType>)> {
    let (params, result) = function.signature.ty.fun_parts();
    let mut lowered = Vec::new();
    let mut saw_bulk = None::<usize>;
    for (index, ty) in params.into_iter().enumerate() {
        match ty {
            Type::Scalar(prim) => lowered.push(wasm_val_type(prim)),
            Type::Bulk(_, _) => {
                saw_bulk.get_or_insert(index);
                lowered.push(ValType::I32);
                lowered.push(ValType::I32);
            }
            Type::Record(_) => {
                return Err(SimdError::new(
                    "Wasm backend does not yet support record entry parameters",
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
        Type::Record(_) => {
            return Err(SimdError::new(
                "Wasm backend does not yet support record entry results",
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
        Prim::I64 => ValType::I64,
        Prim::F32 => ValType::F32,
        Prim::F64 => ValType::F64,
    }
}

fn compile_scalar_function(
    lowered: &LoweredFunction,
    checked: &CheckedFunction,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
) -> Result<Function> {
    let (param_types, result_ty) = checked.signature.ty.fun_parts();
    let _result_prim = match result_ty {
        Type::Scalar(prim) => prim,
        other => {
            return Err(SimdError::new(format!(
                "scalar function '{}' has unsupported result {:?}",
                checked.name, other
            )));
        }
    };
    let mut locals = Vec::<(u32, ValType)>::new();
    let temp_locals = if lowered.tail_loop.is_some() {
        let mut temp = Vec::new();
        for prim in param_types.iter().map(|ty| match ty {
            Type::Scalar(prim) => Ok(*prim),
            Type::Record(_) => Err(SimdError::new(format!(
                "scalar function '{}' has unsupported record parameters",
                checked.name
            ))),
            other => Err(SimdError::new(format!(
                "scalar function '{}' has unsupported parameter {:?}",
                checked.name, other
            ))),
        }) {
            let prim = prim?;
            temp.push((param_types.len() + temp.len()) as u32);
            locals.push((1, wasm_val_type(prim)));
        }
        temp
    } else {
        Vec::new()
    };
    let result_local = if lowered.tail_loop.is_some() {
        let local = (param_types.len() + temp_locals.len()) as u32;
        locals.push((1, wasm_val_type(_result_prim)));
        Some(local)
    } else {
        None
    };
    let mut function = Function::new(locals);
    let local_map = checked
        .signature
        .ty
        .fun_parts()
        .0
        .into_iter()
        .enumerate()
        .map(|(index, ty)| match ty {
            Type::Scalar(prim) => Ok((index as u32, prim)),
            Type::Record(_) => Err(SimdError::new(format!(
                "scalar function '{}' has unsupported record parameters",
                checked.name
            ))),
            other => Err(SimdError::new(format!(
                "scalar function '{}' has unsupported parameter {:?}",
                checked.name, other
            ))),
        })
        .collect::<Result<Vec<_>>>()?;
    let param_prims = local_map.iter().map(|(_, prim)| *prim).collect::<Vec<_>>();

    match (&lowered.kind, &lowered.tail_loop) {
        (LoweredKind::Scalar { clauses: _ }, Some(tail_loop)) => {
            let result_local =
                result_local.ok_or_else(|| SimdError::new("missing tail-loop result local"))?;
            function.instruction(&Instruction::Block(BlockType::Empty));
            function.instruction(&Instruction::Loop(BlockType::Empty));
            for clause in &tail_loop.clauses {
                emit_matching_if(&mut function, &clause.patterns, &param_prims, None)?;
                match &clause.action {
                    TailAction::Continue { args } => {
                        for (temp_local, arg) in temp_locals.iter().zip(args) {
                            compile_scalar_ir_expr(
                                &mut function,
                                arg,
                                &pattern_local_map(&clause.patterns),
                                scalar_indices,
                                signatures,
                            )?;
                            function.instruction(&Instruction::LocalSet(*temp_local));
                        }
                        for (index, temp_local) in temp_locals.iter().enumerate() {
                            function.instruction(&Instruction::LocalGet(*temp_local));
                            function.instruction(&Instruction::LocalSet(index as u32));
                        }
                        function.instruction(&Instruction::Br(1));
                    }
                    TailAction::Return { expr } => {
                        emit_tail_position_scalar_return(
                            &mut function,
                            expr,
                            &pattern_local_map(&clause.patterns),
                            scalar_indices,
                            signatures,
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
        (LoweredKind::Scalar { clauses }, None) => {
            for clause in clauses {
                let locals = pattern_local_map(&clause.patterns);
                if clause_has_condition(&clause.patterns) {
                    emit_matching_if(&mut function, &clause.patterns, &param_prims, None)?;
                    emit_tail_position_scalar_return(
                        &mut function,
                        &clause.body,
                        &locals,
                        scalar_indices,
                        signatures,
                    )?;
                    function.instruction(&Instruction::End);
                } else {
                    emit_tail_position_scalar_return(
                        &mut function,
                        &clause.body,
                        &locals,
                        scalar_indices,
                        signatures,
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

    function.instruction(&Instruction::End);
    Ok(function)
}

fn compile_kernel_entry(
    lowered: &LoweredFunction,
    checked: &CheckedFunction,
    lowered_map: &BTreeMap<String, &LoweredFunction>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    intent: IntentClass,
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
            Type::Bulk(prim, _) => {
                abi_params.push(KernelParam::Lane {
                    prim: *prim,
                    ptr_local: wasm_param_index,
                    len_local: wasm_param_index + 1,
                });
                wasm_param_index += 2;
            }
            Type::Record(_) => {
                return Err(SimdError::new(
                    "Wasm backend does not yet support record kernel parameters",
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
            KernelParam::Same { .. } => {
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

    let first_len_local = abi_params
        .iter()
        .find_map(|param| match param {
            KernelParam::Lane { len_local, .. } => Some(*len_local),
            KernelParam::Same { .. } => None,
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
    let vector_hoists = vector_clause
        .as_ref()
        .map(|vector_clause| {
            if vector_clause.clauses.len() == 1 {
                collect_hoisted_exprs(
                    std::iter::once(vector_clause.clauses[0].body),
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
    for clause in &clauses {
        let clause_key = clause as *const _ as usize;
        let locals = scalar_locals.get(&clause_key).cloned().unwrap_or_default();
        if clause_has_condition(&clause.patterns) {
            emit_matching_if(
                &mut function,
                &clause.patterns,
                &clause
                    .patterns
                    .iter()
                    .map(|pattern| match &pattern.ty {
                        Type::Scalar(prim) => Ok(*prim),
                        other => Err(SimdError::new(format!(
                            "kernel clause pattern has non-scalar lowered type {:?}",
                            other
                        ))),
                    })
                    .collect::<Result<Vec<_>>>()?,
                Some(&locals),
            )?;
            function.instruction(&Instruction::LocalGet(output_ptr_loop_local));
            compile_scalar_ir_expr_with_hoists(
                &mut function,
                &clause.body,
                &locals,
                scalar_indices,
                signatures,
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
    function.instruction(&Instruction::End);
    Ok(CompiledKernelEntry {
        function,
        report: WasmOptimizationReport {
            function: checked.name.clone(),
            intent,
            vectorizable: !matches!(vector_plan, VectorPlan::ScalarOnly),
            vector_unroll: vector_plan_unroll(vector_plan),
            fallback_reason,
        },
    })
}

#[derive(Debug, Clone, Copy)]
enum KernelParam {
    Same {
        prim: Prim,
        value_local: u32,
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

fn kernel_same_locals(clauses: &[LoweredClause], params: &[KernelParam]) -> BTreeMap<String, u32> {
    let mut locals = BTreeMap::new();
    for clause in clauses {
        for (pattern, param) in clause.patterns.iter().zip(params) {
            if let (KernelParam::Same { value_local, .. }, Pattern::Name(name)) =
                (param, &pattern.pattern)
            {
                locals.insert(name.clone(), *value_local);
            }
        }
    }
    locals
}

fn collect_hoisted_exprs<'a, I>(
    roots: I,
    variant_locals: &BTreeSet<String>,
    mode: HoistMode,
) -> Vec<HoistedExpr<'a>>
where
    I: IntoIterator<Item = &'a IrExpr>,
{
    let mut seen = BTreeSet::new();
    let mut hoisted = Vec::new();
    for root in roots {
        collect_hoisted_expr(root, variant_locals, mode, &mut seen, &mut hoisted);
    }
    hoisted
}

fn collect_hoisted_expr<'a>(
    expr: &'a IrExpr,
    variant_locals: &BTreeSet<String>,
    mode: HoistMode,
    seen: &mut BTreeSet<HoistExprKey>,
    out: &mut Vec<HoistedExpr<'a>>,
) -> bool {
    let invariant = match &expr.kind {
        IrExprKind::Local(name) => !variant_locals.contains(name),
        IrExprKind::Int(_, _) | IrExprKind::Float(_, _) => true,
        IrExprKind::Let { bindings, body } => {
            let mut invariant = true;
            for binding in bindings {
                invariant &= collect_hoisted_expr(&binding.expr, variant_locals, mode, seen, out);
            }
            invariant &= collect_hoisted_expr(body, variant_locals, mode, seen, out);
            invariant
        }
        IrExprKind::Call { args, .. } => {
            let mut invariant = true;
            for arg in args {
                invariant &= collect_hoisted_expr(arg, variant_locals, mode, seen, out);
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
    }
}

fn prim_op_key(op: PrimOp) -> PrimOpKey {
    match op {
        PrimOp::Add => PrimOpKey::Add,
        PrimOp::Sub => PrimOpKey::Sub,
        PrimOp::Mul => PrimOpKey::Mul,
        PrimOp::Div => PrimOpKey::Div,
        PrimOp::Mod => PrimOpKey::Mod,
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
            &emitted,
            &inline_bindings,
        )?;
        function.instruction(&Instruction::LocalSet(local));
        emitted.insert(hoisted_expr.key.clone(), local);
    }
    Ok(())
}

fn clause_has_condition(patterns: &[TypedPattern]) -> bool {
    patterns
        .iter()
        .any(|pattern| matches!(pattern.pattern, Pattern::Int(_) | Pattern::Float(_)))
}

fn pattern_local_map(patterns: &[TypedPattern]) -> BTreeMap<String, u32> {
    patterns
        .iter()
        .enumerate()
        .filter_map(|(index, pattern)| match &pattern.pattern {
            Pattern::Name(name) => Some((name.clone(), index as u32)),
            Pattern::Wildcard | Pattern::Int(_) | Pattern::Float(_) => None,
        })
        .collect()
}

fn emit_matching_if(
    function: &mut Function,
    patterns: &[TypedPattern],
    param_prims: &[Prim],
    locals: Option<&BTreeMap<String, u32>>,
) -> Result<()> {
    emit_clause_condition(function, patterns, param_prims, locals)?;
    function.instruction(&Instruction::If(BlockType::Empty));
    Ok(())
}

fn emit_clause_condition(
    function: &mut Function,
    patterns: &[TypedPattern],
    param_prims: &[Prim],
    locals: Option<&BTreeMap<String, u32>>,
) -> Result<()> {
    let mut emitted = false;
    for (index, pattern) in patterns.iter().enumerate() {
        let prim = *param_prims
            .get(index)
            .ok_or_else(|| SimdError::new("pattern arity mismatch in Wasm codegen"))?;
        match &pattern.pattern {
            Pattern::Wildcard | Pattern::Name(_) => {}
            Pattern::Int(expected) => {
                emitted = true;
                emit_pattern_value(function, pattern, index as u32, locals)?;
                emit_int_const(function, prim, *expected)?;
                emit_int_eq(function, prim);
                if index > 0 {
                    function.instruction(&Instruction::I32And);
                }
            }
            Pattern::Float(expected) => {
                emitted = true;
                emit_pattern_value(function, pattern, index as u32, locals)?;
                emit_float_bits_eq(function, prim, *expected)?;
                if index > 0 {
                    function.instruction(&Instruction::I32And);
                }
            }
        }
    }
    if !emitted {
        function.instruction(&Instruction::I32Const(1));
    }
    Ok(())
}

fn emit_pattern_value(
    function: &mut Function,
    pattern: &TypedPattern,
    fallback_local: u32,
    locals: Option<&BTreeMap<String, u32>>,
) -> Result<()> {
    match &pattern.pattern {
        Pattern::Name(name) => {
            let local = locals
                .and_then(|locals| locals.get(name))
                .copied()
                .unwrap_or(fallback_local);
            function.instruction(&Instruction::LocalGet(local));
        }
        Pattern::Wildcard | Pattern::Int(_) | Pattern::Float(_) => {
            function.instruction(&Instruction::LocalGet(fallback_local));
        }
    }
    Ok(())
}

fn emit_int_const(function: &mut Function, prim: Prim, value: i64) -> Result<()> {
    match prim {
        Prim::I32 => {
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

fn emit_int_eq(function: &mut Function, prim: Prim) {
    match prim {
        Prim::I32 => {
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
        Prim::I32 | Prim::I64 => Err(SimdError::new(
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
) -> Result<()> {
    let inline_bindings = BTreeMap::<String, IrExpr>::new();
    compile_scalar_ir_expr_with_hoists(
        function,
        expr,
        locals,
        scalar_indices,
        signatures,
        &BTreeMap::new(),
        &inline_bindings,
    )
}

fn emit_tail_position_scalar_return(
    function: &mut Function,
    expr: &IrExpr,
    locals: &BTreeMap<String, u32>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
) -> Result<()> {
    let inline_bindings = BTreeMap::<String, IrExpr>::new();
    emit_tail_position_scalar_return_with_bindings(
        function,
        expr,
        locals,
        scalar_indices,
        signatures,
        &inline_bindings,
    )
}

fn emit_tail_position_scalar_return_with_bindings(
    function: &mut Function,
    expr: &IrExpr,
    locals: &BTreeMap<String, u32>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
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
        if result != expr.ty {
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
                &BTreeMap::new(),
                inline_bindings,
            )?;
            if &arg.ty != ty {
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
        function.instruction(&Instruction::ReturnCall(index));
        return Ok(());
    }

    compile_scalar_ir_expr_with_hoists(
        function,
        expr,
        locals,
        scalar_indices,
        signatures,
        &BTreeMap::new(),
        inline_bindings,
    )?;
    function.instruction(&Instruction::Return);
    Ok(())
}

fn compile_scalar_ir_expr_with_hoists(
    function: &mut Function,
    expr: &IrExpr,
    locals: &BTreeMap<String, u32>,
    scalar_indices: &BTreeMap<String, u32>,
    signatures: &BTreeMap<String, &CheckedFunction>,
    hoisted_locals: &BTreeMap<HoistExprKey, u32>,
    inline_bindings: &BTreeMap<String, IrExpr>,
) -> Result<()> {
    if let Some(local) = hoisted_locals.get(&hoist_expr_key(expr)) {
        function.instruction(&Instruction::LocalGet(*local));
        return Ok(());
    }
    match &expr.kind {
        IrExprKind::Local(name) => {
            if let Some(index) = locals.get(name).copied() {
                function.instruction(&Instruction::LocalGet(index));
            } else if let Some(inline_expr) = inline_bindings.get(name) {
                compile_scalar_ir_expr_with_hoists(
                    function,
                    inline_expr,
                    locals,
                    scalar_indices,
                    signatures,
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
            Prim::I32 | Prim::I64 => {
                return Err(SimdError::new(format!(
                    "float literal cannot inhabit integer primitive {:?}",
                    prim
                )));
            }
        },
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
                    hoisted_locals,
                    inline_bindings,
                )?;
                compile_scalar_ir_expr_with_hoists(
                    function,
                    &args[1],
                    locals,
                    scalar_indices,
                    signatures,
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
                        hoisted_locals,
                        inline_bindings,
                    )?;
                    if arg.ty != ty {
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
            }
        },
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
        (PrimOp::Add, Prim::I32) => {
            function.instruction(&Instruction::I32Add);
        }
        (PrimOp::Sub, Prim::I32) => {
            function.instruction(&Instruction::I32Sub);
        }
        (PrimOp::Mul, Prim::I32) => {
            function.instruction(&Instruction::I32Mul);
        }
        (PrimOp::Div, Prim::I32) => {
            function.instruction(&Instruction::I32DivS);
        }
        (PrimOp::Mod, Prim::I32) => {
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
        (PrimOp::Eq, Prim::I32) => {
            function.instruction(&Instruction::I32Eq);
            if result_prim == Prim::I64 {
                function.instruction(&Instruction::I64ExtendI32U);
            }
        }
        (PrimOp::Lt, Prim::I32) => {
            function.instruction(&Instruction::I32LtS);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Gt, Prim::I32) => {
            function.instruction(&Instruction::I32GtS);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Le, Prim::I32) => {
            function.instruction(&Instruction::I32LeS);
            function.instruction(&Instruction::I64ExtendI32U);
        }
        (PrimOp::Ge, Prim::I32) => {
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
    if clauses.is_empty() {
        return None;
    }
    if clause_has_condition(&clauses.last()?.patterns) {
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
        IrExprKind::Let { bindings, body } => {
            bindings
                .iter()
                .all(|binding| is_vectorizable_expr(&binding.expr, locals, result_prim))
                && is_vectorizable_expr(body, locals, result_prim)
        }
        IrExprKind::Call { callee, args } => match callee {
            Callee::Function(_) => false,
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
                    result_prim == Prim::I64 && matches!(operand_prim, Prim::I64 | Prim::F64)
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
        (Prim::I32, false) => {
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
        if !clause_has_condition(clause.patterns) {
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
        if !clause_has_condition(clause.patterns) {
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
        IrExprKind::Local(name) => {
            if let Some(slot) = locals.get(name) {
                match params[*slot] {
                    KernelParam::Same { prim, value_local } => {
                        emit_vector_splat_local(function, prim, value_local)
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
        },
    }
    Ok(())
}

fn emit_vector_splat_local(function: &mut Function, prim: Prim, local: u32) {
    function.instruction(&Instruction::LocalGet(local));
    match prim {
        Prim::I32 => {
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
        Prim::I32 => {
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
        Prim::I32 | Prim::I64 => {
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
        (PrimOp::Add, Prim::I32) => {
            function.instruction(&Instruction::I32x4Add);
        }
        (PrimOp::Sub, Prim::I32) => {
            function.instruction(&Instruction::I32x4Sub);
        }
        (PrimOp::Mul, Prim::I32) => {
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
        Prim::I32 => {
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
        Prim::I32 => {
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
        Prim::I32 | Prim::F32 => 4,
        Prim::I64 | Prim::F64 => 8,
    }
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
        if let Pattern::Name(name) = &pattern.pattern {
            local_types.insert(name.clone(), ty.clone());
            let mut leaf_names = BTreeMap::new();
            for leaf in flatten_type_leaves(ty) {
                leaf_names.insert(leaf.path.clone(), leaf_symbol_name(name, &leaf.path));
            }
            local_leaf_names.insert(name.clone(), leaf_names);
        }
    }
    (local_types, local_leaf_names)
}

fn flatten_clause_patterns(
    patterns: &[TypedPattern],
    param_types: &[Type],
) -> Result<Vec<TypedPattern>> {
    let mut flattened = Vec::new();
    for (pattern, ty) in patterns.iter().zip(param_types) {
        match ty {
            Type::Scalar(_) | Type::Bulk(_, _) => flattened.push(TypedPattern {
                pattern: pattern.pattern.clone(),
                ty: ty.clone(),
            }),
            Type::Record(_) => {
                for leaf in flatten_type_leaves(ty) {
                    let leaf_pattern = match &pattern.pattern {
                        Pattern::Name(name) => Pattern::Name(leaf_symbol_name(name, &leaf.path)),
                        Pattern::Wildcard => Pattern::Wildcard,
                        Pattern::Int(_) | Pattern::Float(_) => {
                            return Err(SimdError::new(
                                "record parameters cannot use literal patterns",
                            ));
                        }
                    };
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
    let leaf_ty = type_at_leaf_path(&expr.ty, leaf_path)?;
    match &expr.kind {
        TypedExprKind::Local(name) => {
            let local_ty = local_types.get(name).ok_or_else(|| {
                SimdError::new(format!("unknown local '{}' in Wasm normalization", name))
            })?;
            let local_name = local_leaf_names
                .get(name)
                .and_then(|names| names.get(leaf_path))
                .cloned()
                .unwrap_or_else(|| name.clone());
            let _ = local_ty;
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
            &leaf_path.prepend(field),
            local_types,
            local_leaf_names,
            checked_map,
            result_leaf_names,
        ),
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

fn wasm_param_abi_from_type(ty: &[Type]) -> Result<Vec<WasmParamAbi>> {
    ty.iter().map(wasm_param_abi_from_single_type).collect()
}

fn wasm_param_abi_from_single_type(ty: &Type) -> Result<WasmParamAbi> {
    match ty {
        Type::Scalar(prim) => Ok(WasmParamAbi::Scalar { prim: *prim }),
        Type::Bulk(prim, _) => Ok(WasmParamAbi::Bulk { prim: *prim }),
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
    }
}

fn wasm_result_abi_from_type(ty: &Type, param_types: &[Type]) -> Result<WasmResultAbi> {
    match ty {
        Type::Scalar(prim) => Ok(WasmResultAbi::Scalar { prim: *prim }),
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
    }
}

fn type_contains_bulk_leaf(ty: &Type) -> bool {
    match ty {
        Type::Bulk(_, _) => true,
        Type::Record(fields) => fields
            .iter()
            .any(|(_, field_ty)| type_contains_bulk_leaf(field_ty)),
        Type::Scalar(_) | Type::Fun(_, _) => false,
    }
}

fn wasm_leaf_result_abi_from_type(ty: &Type) -> Result<WasmLeafResultAbi> {
    match ty {
        Type::Scalar(prim) => Ok(WasmLeafResultAbi::Scalar { prim: *prim }),
        Type::Bulk(prim, _) => Ok(WasmLeafResultAbi::Bulk { prim: *prim }),
        Type::Record(_) => Err(SimdError::new("leaf result ABI cannot contain records")),
        Type::Fun(_, _) => Err(SimdError::new("leaf result ABI cannot contain functions")),
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
            Type::Record(_) => Err(SimdError::new("record type requires a non-empty leaf path")),
            Type::Fun(_, _) => Err(SimdError::new(
                "function types cannot be used as leaf values",
            )),
        };
    }
    let Some((head, tail)) = leaf_path.split_first() else {
        unreachable!("non-root leaf path must have a head");
    };
    match ty {
        Type::Record(fields) => {
            let field = fields.get(head).ok_or_else(|| {
                SimdError::new(format!("record type is missing field '{}'", head))
            })?;
            type_at_leaf_path(field, &tail)
        }
        _ => Err(SimdError::new(format!(
            "cannot select field '{}' from non-record type {:?}",
            head, ty
        ))),
    }
}

fn build_engine() -> Result<Engine> {
    let mut config = Config::new();
    config.wasm_simd(true);
    config.wasm_tail_call(true);
    config.cranelift_opt_level(OptLevel::Speed);
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
        LoweredKind::Scalar { clauses } | LoweredKind::Kernel { clauses, .. } => clauses,
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
            WasmLeafResultAbi::Scalar { .. } => vec![Val::I64(0)],
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
        }
    }

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
    match (value, abi) {
        (Value::Scalar(_), WasmParamAbi::Scalar { .. }) => out.push(value.clone()),
        (Value::Bulk(bulk), WasmParamAbi::Bulk { prim }) => {
            if bulk.prim != *prim {
                return Err(SimdError::new(format!(
                    "Wasm bulk argument expected {:?}, found {:?}",
                    prim, bulk.prim
                )));
            }
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
            out.push(value.clone());
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
        (value, WasmParamAbi::Record { .. }) => {
            return Err(SimdError::new(format!(
                "record Wasm ABI expected a record value, found {:?}",
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
        WasmResultAbi::Scalar { .. } | WasmResultAbi::Bulk { .. } => leaves
            .get(prefix)
            .cloned()
            .ok_or_else(|| SimdError::new(format!("missing leaf result for path {:?}", prefix))),
        WasmResultAbi::Record { fields } => {
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

fn leaf_result_prim(result: &WasmLeafResultAbi) -> Prim {
    match result {
        WasmLeafResultAbi::Scalar { prim } | WasmLeafResultAbi::Bulk { prim } => *prim,
    }
}

fn build_runtime(
    engine: &Engine,
    module: &WasmtimeModule,
    artifact: &WasmArtifact,
) -> Result<WasmRuntime> {
    let mut store = Store::new(engine, ());
    let instance = Instance::new(&mut store, module, &[])
        .map_err(|error| SimdError::new(format!("failed to instantiate Wasm module: {error}")))?;
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

    for (index, (abi, value)) in artifact.params.iter().zip(args).enumerate() {
        match (abi, value) {
            (WasmParamAbi::Scalar { prim }, Value::Scalar(value)) => {
                wasm_args.push(scalar_to_wasmtime(value, *prim)?);
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
        let Value::Bulk(shape_value) = shape_source else {
            return Err(SimdError::new(
                "Wasm bulk result shape must come from a bulk input argument",
            ));
        };
        let len = shape_value.elements.len();
        let bulk = BulkShapeInfo {
            shape: shape_value.shape.clone(),
            len,
        };
        let out_ptr = ensure_output_bulk_buffer(runtime, &LeafPath::root(), *prim, &bulk)?;
        wasm_args.push(Val::I32(i32::try_from(out_ptr).map_err(|_| {
            SimdError::new("bulk output pointer does not fit in i32")
        })?));
        wasm_args.push(Val::I32(i32::try_from(len).map_err(|_| {
            SimdError::new("bulk output length does not fit in i32")
        })?));
        output_shape = Some(shape_value.shape.clone());
        output_ptr = Some(out_ptr);
        output_len = Some(len);
    }

    let mut results = if matches!(artifact.result, WasmResultAbi::Scalar { .. }) {
        vec![Val::I64(0)]
    } else {
        Vec::new()
    };
    func.call(&mut runtime.store, &wasm_args, &mut results)
        .map_err(|error| SimdError::new(format!("Wasm execution failed: {error}")))?;

    match &artifact.result {
        WasmResultAbi::Scalar { prim } => {
            let value = results
                .into_iter()
                .next()
                .ok_or_else(|| SimdError::new("Wasm scalar entry did not produce a result"))?;
            Ok(Value::Scalar(wasmtime_to_scalar(value, *prim)?))
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
    }
}

fn read_scalar_bytes(data: &[u8], cursor: &mut usize, prim: Prim) -> Result<ScalarValue> {
    match prim {
        Prim::I32 => Ok(ScalarValue::I32(i32::from_le_bytes(read_bytes::<4>(
            data, cursor,
        )?))),
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
            grouped_export: None,
            leaf_exports: vec![WasmLeafExport {
                leaf_path: LeafPath::root(),
                export_name: "alias".to_string(),
                result: WasmLeafResultAbi::Scalar { prim: Prim::I64 },
                used_param_leaves: Vec::new(),
                reusable_param_leaf: None,
            }],
            optimizer_reports: Vec::new(),
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
                LoweredKind::Scalar { .. } => Some((function.name.clone(), index as u32)),
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
}
