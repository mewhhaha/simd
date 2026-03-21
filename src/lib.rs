use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::rc::Rc;

mod benchmarks;
mod formatter;
mod lsp;
mod wasm_backend;

pub use benchmarks::bench_command;
pub use benchmarks::{BenchOptions, bench_command_with_options};
pub use formatter::{fmt_command, format_program, format_source_text};
pub use lsp::lsp_command;
pub use wasm_backend::{
    BoundPreparedRun, PreparedLayout, PreparedSlotKind, PreparedSlotMetadata, PreparedSlotRole,
    PreparedWasmMain, WasmArtifact, WasmExecutable, WasmHigherOrderReport, WasmParamAbi,
    WasmResultAbi, compile_wasm_main, prepare_wasm_artifact, prepare_wasm_main, run_wasm_artifact,
    run_wasm_command, run_wasm_main, wasm_command, wat_command, wat_main,
};

pub type Result<T> = std::result::Result<T, SimdError>;

#[derive(Debug, Clone, PartialEq)]
pub struct SimdError {
    message: String,
}

impl SimdError {
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for SimdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SimdError {}

#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceProgram {
    pub decls: Vec<Decl>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decl {
    Import(ImportDecl),
    Signature(Signature),
    Clause(Clause),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImportDecl {
    pub path: String,
    pub alias: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Signature {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Clause {
    pub name: String,
    pub patterns: Vec<Pattern>,
    pub body: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LetBinding {
    pub name: String,
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefExpr {
    pub alias: Option<String>,
    pub name: String,
    pub type_args: Vec<TypeArg>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeArg {
    Prim(Prim),
    Name(String),
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Int(i64),
    Float(f64),
    Name(String),
    Wildcard,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Ref(RefExpr),
    Int(i64),
    Float(f64),
    App(Box<Expr>, Box<Expr>),
    Lambda {
        param: String,
        body: Box<Expr>,
    },
    Let {
        bindings: Vec<LetBinding>,
        body: Box<Expr>,
    },
    Record(Vec<(String, Expr)>),
    Project(Box<Expr>, String),
    RecordUpdate {
        base: Box<Expr>,
        fields: Vec<(String, Expr)>,
    },
    Infix {
        op: PrimOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimOp {
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

impl PrimOp {
    fn from_token(token: &TokenKind) -> Option<Self> {
        match token {
            TokenKind::Plus => Some(Self::Add),
            TokenKind::Minus => Some(Self::Sub),
            TokenKind::Star => Some(Self::Mul),
            TokenKind::Slash => Some(Self::Div),
            TokenKind::Percent => Some(Self::Mod),
            TokenKind::EqEq => Some(Self::Eq),
            TokenKind::Lt => Some(Self::Lt),
            TokenKind::Gt => Some(Self::Gt),
            TokenKind::Le => Some(Self::Le),
            TokenKind::Ge => Some(Self::Ge),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Type {
    Scalar(Prim),
    Bulk(Prim, Shape),
    Record(BTreeMap<String, Type>),
    Var(String),
    Infer(u32),
    Fun(Vec<Type>, Box<Type>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Prim {
    I32,
    I64,
    F32,
    F64,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct Shape(pub Vec<Dim>);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Dim {
    Const(usize),
    Var(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub imports: Vec<ImportDecl>,
    pub functions: Vec<FunctionDef>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub signature: Signature,
    pub clauses: Vec<Clause>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CheckedProgram {
    pub functions: Vec<CheckedFunction>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct LeafPath(pub Vec<String>);

#[derive(Debug, Clone, PartialEq)]
pub struct TypeLeaf {
    pub path: LeafPath,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedProgram {
    pub functions: Vec<NormalizedFunction>,
    pub entries: Vec<NormalizedEntry>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedEntry {
    pub source_name: String,
    pub source_signature: Signature,
    pub param_leaves: Vec<Vec<TypeLeaf>>,
    pub result_leaves: Vec<TypeLeaf>,
    pub leaf_functions: BTreeMap<LeafPath, String>,
    pub pointwise: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizedFunction {
    pub name: String,
    pub source_name: String,
    pub leaf_path: LeafPath,
    pub signature: Signature,
    pub clauses: Vec<TypedClause>,
    pub pointwise: bool,
    pub tailrec: TailRecInfo,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CheckedFunction {
    pub name: String,
    pub signature: Signature,
    pub clauses: Vec<TypedClause>,
    pub pointwise: bool,
    pub tailrec: TailRecInfo,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedClause {
    pub patterns: Vec<TypedPattern>,
    pub body: TypedExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedLetBinding {
    pub name: String,
    pub expr: TypedExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedPattern {
    pub pattern: Pattern,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedExpr {
    pub ty: Type,
    pub kind: TypedExprKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TypedExprKind {
    Local(String),
    FunctionRef {
        name: String,
    },
    Int(i64, Prim),
    Float(f64, Prim),
    Lambda {
        param: String,
        body: Box<TypedExpr>,
    },
    Let {
        bindings: Vec<TypedLetBinding>,
        body: Box<TypedExpr>,
    },
    Record(BTreeMap<String, TypedExpr>),
    Project {
        base: Box<TypedExpr>,
        field: String,
    },
    RecordUpdate {
        base: Box<TypedExpr>,
        fields: BTreeMap<String, TypedExpr>,
    },
    Call {
        callee: Callee,
        args: Vec<TypedArg>,
        lifted_shape: Option<Shape>,
    },
    Apply {
        callee: Box<TypedExpr>,
        arg: Box<TypedExpr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypedArg {
    pub mode: AccessKind,
    pub expr: Box<TypedExpr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Callee {
    Function(String),
    Prim(PrimOp),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AccessKind {
    Same,
    Lane,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TailRecInfo {
    pub recursive: bool,
    pub loop_lowerable: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoweredProgram {
    pub functions: Vec<LoweredFunction>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroupedLoweredProgram {
    pub functions: Vec<GroupedLoweredFunction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IntentAnalysis {
    pub reports: Vec<KernelIntentReport>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelIntentReport {
    pub source_name: String,
    pub leaf_paths: Vec<LeafPath>,
    pub intent: IntentClass,
    pub features: KernelFeatures,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntentClass {
    MapUnary,
    MapBinaryBroadcast,
    MapTernaryBroadcast,
    GroupedMap,
    ScalarTailRec,
    Fallback,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KernelFeatures {
    pub op_count: usize,
    pub load_streams: usize,
    pub store_streams: usize,
    pub scalar_broadcast_count: usize,
    pub record_leaf_count: usize,
    pub rank: usize,
    pub shape_size: Option<usize>,
    pub primitive_width_bytes: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroupedLoweredFunction {
    pub source_name: String,
    pub leaf_paths: Vec<LeafPath>,
    pub result_leaves: Vec<TypeLeaf>,
    pub param_access: Vec<AccessKind>,
    pub kind: GroupedLoweredKind,
    pub tail_loop: Option<TailLoop>,
    pub leaves: Vec<LoweredFunction>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum GroupedLoweredKind {
    Scalar {
        clauses: Vec<LoweredClause>,
    },
    Kernel {
        shape: Shape,
        vector_width: usize,
        cleanup: bool,
        clauses: Vec<LoweredClause>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoweredFunction {
    pub name: String,
    pub param_access: Vec<AccessKind>,
    pub result: Type,
    pub kind: LoweredKind,
    pub tail_loop: Option<TailLoop>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum LoweredKind {
    Scalar {
        clauses: Vec<LoweredClause>,
    },
    Kernel {
        shape: Shape,
        vector_width: usize,
        cleanup: bool,
        clauses: Vec<LoweredClause>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct LoweredClause {
    pub patterns: Vec<TypedPattern>,
    pub body: IrExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TailLoop {
    pub clauses: Vec<TailLoopClause>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TailLoopClause {
    pub patterns: Vec<TypedPattern>,
    pub action: TailAction,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TailAction {
    Continue { args: Vec<IrExpr> },
    Return { expr: IrExpr },
}

#[derive(Debug, Clone, PartialEq)]
pub struct IrLetBinding {
    pub name: String,
    pub expr: IrExpr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct IrExpr {
    pub ty: Type,
    pub kind: IrExprKind,
}

#[derive(Debug, Clone, PartialEq)]
pub enum IrExprKind {
    Local(String),
    Int(i64, Prim),
    Float(f64, Prim),
    Let {
        bindings: Vec<IrLetBinding>,
        body: Box<IrExpr>,
    },
    Call {
        callee: Callee,
        args: Vec<IrExpr>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct CompiledProgram {
    pub surface: SurfaceProgram,
    pub module: Module,
    pub checked: CheckedProgram,
    pub normalized: NormalizedProgram,
    pub lowered: LoweredProgram,
    pub grouped: GroupedLoweredProgram,
    pub intents: IntentAnalysis,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Scalar(ScalarValue),
    Bulk(BulkValue),
    Record(BTreeMap<String, Value>),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScalarValue {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub struct BulkValue {
    pub prim: Prim,
    pub shape: Vec<usize>,
    pub elements: Vec<ScalarValue>,
}

impl Type {
    pub fn arity(&self) -> usize {
        match self {
            Self::Fun(args, _) => args.len(),
            _ => 0,
        }
    }

    pub fn fun_parts(&self) -> (Vec<Type>, Type) {
        match self {
            Self::Fun(args, ret) => (args.clone(), ret.as_ref().clone()),
            other => (Vec::new(), other.clone()),
        }
    }

    fn prim(&self) -> Option<Prim> {
        match self {
            Self::Scalar(prim) | Self::Bulk(prim, _) => Some(*prim),
            Self::Record(_) | Self::Var(_) | Self::Infer(_) | Self::Fun(_, _) => None,
        }
    }
}

impl Prim {
    fn parse(name: &str) -> Option<Self> {
        match name {
            "i32" => Some(Self::I32),
            "i64" => Some(Self::I64),
            "f32" => Some(Self::F32),
            "f64" => Some(Self::F64),
            _ => None,
        }
    }

    fn is_int(self) -> bool {
        matches!(self, Self::I32 | Self::I64)
    }

    fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    fn lane_width(self) -> usize {
        match self {
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 2,
        }
    }

    fn byte_width(self) -> usize {
        match self {
            Self::I32 | Self::F32 => 4,
            Self::I64 | Self::F64 => 8,
        }
    }
}

impl BulkValue {
    fn scalar_at(&self, index: usize) -> ScalarValue {
        self.elements[index].clone()
    }
}

impl LeafPath {
    fn root() -> Self {
        Self(Vec::new())
    }

    fn is_root(&self) -> bool {
        self.0.is_empty()
    }

    fn prepend(&self, field: &str) -> Self {
        let mut path = Vec::with_capacity(self.0.len() + 1);
        path.push(field.to_string());
        path.extend(self.0.iter().cloned());
        Self(path)
    }

    fn child(&self, field: &str) -> Self {
        let mut path = self.0.clone();
        path.push(field.to_string());
        Self(path)
    }

    fn suffix(&self) -> String {
        self.0.join("$")
    }

    fn split_first(&self) -> Option<(&str, LeafPath)> {
        self.0
            .split_first()
            .map(|(head, tail)| (head.as_str(), LeafPath(tail.to_vec())))
    }
}

impl Value {
    pub fn ty(&self) -> Type {
        match self {
            Self::Scalar(value) => Type::Scalar(value.prim()),
            Self::Bulk(value) => Type::Bulk(
                value.prim,
                Shape(value.shape.iter().copied().map(Dim::Const).collect()),
            ),
            Self::Record(fields) => Type::Record(
                fields
                    .iter()
                    .map(|(name, value)| (name.clone(), value.ty()))
                    .collect(),
            ),
        }
    }

    pub fn to_json_string(&self) -> String {
        match self {
            Self::Scalar(value) => value.to_json_string(),
            Self::Bulk(value) => render_bulk_json(&value.elements, &value.shape, 0),
            Self::Record(fields) => render_record_json(fields),
        }
    }
}

impl ScalarValue {
    fn prim(&self) -> Prim {
        match self {
            Self::I32(_) => Prim::I32,
            Self::I64(_) => Prim::I64,
            Self::F32(_) => Prim::F32,
            Self::F64(_) => Prim::F64,
        }
    }

    fn to_json_string(&self) -> String {
        match self {
            Self::I32(value) => value.to_string(),
            Self::I64(value) => value.to_string(),
            Self::F32(value) => format_float(*value as f64),
            Self::F64(value) => format_float(*value),
        }
    }
}

fn render_bulk_json(elements: &[ScalarValue], shape: &[usize], offset: usize) -> String {
    if shape.is_empty() {
        return elements[offset].to_json_string();
    }
    if shape.len() == 1 {
        let mut parts = Vec::with_capacity(shape[0]);
        for index in 0..shape[0] {
            parts.push(elements[offset + index].to_json_string());
        }
        return format!("[{}]", parts.join(","));
    }
    let stride = shape[1..].iter().product::<usize>();
    let mut parts = Vec::with_capacity(shape[0]);
    for index in 0..shape[0] {
        parts.push(render_bulk_json(
            elements,
            &shape[1..],
            offset + (index * stride),
        ));
    }
    format!("[{}]", parts.join(","))
}

fn render_record_json(fields: &BTreeMap<String, Value>) -> String {
    let record = Value::Record(fields.clone());
    if let Some(shape) = value_lift_shape(&record) {
        return render_lifted_value_json(&record, &shape, 0);
    }
    let mut parts = Vec::with_capacity(fields.len());
    for (name, value) in fields {
        parts.push(format!("\"{}\":{}", name, value.to_json_string()));
    }
    format!("{{{}}}", parts.join(","))
}

fn render_lifted_value_json(value: &Value, shape: &[usize], offset: usize) -> String {
    if shape.len() == 1 {
        let mut parts = Vec::with_capacity(shape[0]);
        for index in 0..shape[0] {
            let lane = extract_lifted_lane(value, offset + index)
                .expect("lifted record JSON rendering should only visit valid lanes");
            parts.push(lane.to_json_string());
        }
        return format!("[{}]", parts.join(","));
    }
    let stride = shape[1..].iter().product::<usize>();
    let mut parts = Vec::with_capacity(shape[0]);
    for index in 0..shape[0] {
        parts.push(render_lifted_value_json(
            value,
            &shape[1..],
            offset + (index * stride),
        ));
    }
    format!("[{}]", parts.join(","))
}

fn value_lift_shape(value: &Value) -> Option<Vec<usize>> {
    match value {
        Value::Scalar(_) => None,
        Value::Bulk(bulk) => Some(bulk.shape.clone()),
        Value::Record(fields) => {
            let mut shape = None::<Vec<usize>>;
            for value in fields.values() {
                match value_lift_shape(value) {
                    Some(field_shape) => match &shape {
                        None => shape = Some(field_shape),
                        Some(existing) if existing == &field_shape => {}
                        Some(_) => return None,
                    },
                    None => {}
                }
            }
            shape
        }
    }
}

fn value_leaf_prim(value: &Value) -> Option<Prim> {
    match value {
        Value::Scalar(value) => Some(value.prim()),
        Value::Bulk(value) => Some(value.prim),
        Value::Record(fields) => fields.values().find_map(value_leaf_prim),
    }
}

fn extract_lifted_lane(value: &Value, index: usize) -> Result<Value> {
    match value {
        Value::Bulk(bulk) => Ok(Value::Scalar(bulk.scalar_at(index))),
        Value::Record(fields) => Ok(Value::Record(
            fields
                .iter()
                .map(|(name, value)| Ok((name.clone(), extract_lifted_lane(value, index)?)))
                .collect::<Result<BTreeMap<_, _>>>()?,
        )),
        Value::Scalar(_) => Err(SimdError::new(
            "cannot extract a lifted lane from a scalar value",
        )),
    }
}

fn collect_lifted_value(values: &[Value], scalar_ty: &Type, shape: &[usize]) -> Result<Value> {
    match scalar_ty {
        Type::Scalar(prim) => Ok(Value::Bulk(BulkValue {
            prim: *prim,
            shape: shape.to_vec(),
            elements: values
                .iter()
                .map(|value| match value {
                    Value::Scalar(value) if value.prim() == *prim => Ok(value.clone()),
                    other => Err(SimdError::new(format!(
                        "lifted lane result expected scalar {:?}, found {:?}",
                        prim, other
                    ))),
                })
                .collect::<Result<Vec<_>>>()?,
        })),
        Type::Record(fields) => {
            let mut lifted_fields = BTreeMap::new();
            for (name, field_ty) in fields {
                let mut field_values = Vec::with_capacity(values.len());
                for value in values {
                    let Value::Record(record) = value else {
                        return Err(SimdError::new(
                            "lifted record result expected record lane values",
                        ));
                    };
                    field_values.push(record.get(name).cloned().ok_or_else(|| {
                        SimdError::new(format!("lifted record lane is missing field '{}'", name))
                    })?);
                }
                lifted_fields.insert(
                    name.clone(),
                    collect_lifted_value(&field_values, field_ty, shape)?,
                );
            }
            Ok(Value::Record(lifted_fields))
        }
        Type::Bulk(_, _) => Err(SimdError::new(
            "lifted result type unexpectedly contained bulk leaves already",
        )),
        Type::Var(_) | Type::Infer(_) => Err(SimdError::new(
            "lifted result type unexpectedly contained an unresolved type variable",
        )),
        Type::Fun(_, _) => Err(SimdError::new(
            "lifted result type unexpectedly contained a function",
        )),
    }
}

pub fn flatten_type_leaves(ty: &Type) -> Vec<TypeLeaf> {
    let mut leaves = Vec::new();
    collect_type_leaves(ty, &LeafPath::root(), &mut leaves);
    leaves
}

fn collect_type_leaves(ty: &Type, prefix: &LeafPath, leaves: &mut Vec<TypeLeaf>) {
    match ty {
        Type::Scalar(_) | Type::Bulk(_, _) => leaves.push(TypeLeaf {
            path: prefix.clone(),
            ty: ty.clone(),
        }),
        Type::Record(fields) => {
            for (name, field_ty) in fields {
                collect_type_leaves(field_ty, &prefix.child(name), leaves);
            }
        }
        Type::Var(_) | Type::Infer(_) | Type::Fun(_, _) => {}
    }
}

pub fn flatten_value_leaves(value: &Value, ty: &Type) -> Result<Vec<(LeafPath, Value)>> {
    match (value, ty) {
        (Value::Scalar(_), Type::Scalar(_)) | (Value::Bulk(_), Type::Bulk(_, _)) => {
            Ok(vec![(LeafPath::root(), value.clone())])
        }
        (Value::Record(fields), Type::Record(field_types)) => {
            let mut leaves = Vec::new();
            for (name, field_ty) in field_types {
                let field_value = fields.get(name).ok_or_else(|| {
                    SimdError::new(format!("record value is missing field '{}'", name))
                })?;
                for (path, value) in flatten_value_leaves(field_value, field_ty)? {
                    leaves.push((
                        LeafPath(std::iter::once(name.clone()).chain(path.0).collect()),
                        value,
                    ));
                }
            }
            Ok(leaves)
        }
        _ => Err(SimdError::new(format!(
            "cannot flatten value {:?} against type {:?}",
            value, ty
        ))),
    }
}

pub fn rebuild_value_from_leaves(ty: &Type, leaves: &BTreeMap<LeafPath, Value>) -> Result<Value> {
    match ty {
        Type::Scalar(_) | Type::Bulk(_, _) => leaves
            .get(&LeafPath::root())
            .cloned()
            .ok_or_else(|| SimdError::new(format!("missing root leaf for type {:?}", ty))),
        Type::Record(fields) => {
            let mut record = BTreeMap::new();
            for (name, field_ty) in fields {
                let field_leaves = leaves
                    .iter()
                    .filter_map(|(path, value)| {
                        path.0.split_first().and_then(|(head, tail)| {
                            (head == name).then(|| (LeafPath(tail.to_vec()), value.clone()))
                        })
                    })
                    .collect::<BTreeMap<_, _>>();
                record.insert(
                    name.clone(),
                    rebuild_value_from_leaves(field_ty, &field_leaves)?,
                );
            }
            Ok(Value::Record(record))
        }
        Type::Var(_) | Type::Infer(_) => {
            Err(SimdError::new("cannot rebuild values for unresolved types"))
        }
        Type::Fun(_, _) => Err(SimdError::new(
            "cannot rebuild function values from flattened leaves",
        )),
    }
}

fn leaf_value_at<'a>(leaves: &'a [(LeafPath, Value)], path: &LeafPath) -> Option<&'a Value> {
    leaves
        .iter()
        .find_map(|(leaf_path, value)| (leaf_path == path).then_some(value))
}

fn format_float(value: f64) -> String {
    let text = format!("{}", value);
    if text.contains('.') || text.contains('e') || text.contains('E') {
        text
    } else {
        format!("{text}.0")
    }
}

#[derive(Debug, Clone, PartialEq)]
struct Token {
    kind: TokenKind,
    text: String,
    leading_space: bool,
}

#[derive(Debug, Clone, PartialEq)]
enum TokenKind {
    Ident,
    Import,
    As,
    Let,
    In,
    Int,
    Float,
    Newline,
    Semicolon,
    Colon,
    Arrow,
    Eq,
    EqEq,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    Lt,
    Gt,
    Le,
    Ge,
    Backslash,
    Dot,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
}

fn lex(source: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = source.chars().collect();
    let mut index = 0;
    while index < chars.len() {
        let ch = chars[index];
        if ch == '\n' {
            tokens.push(Token {
                kind: TokenKind::Newline,
                text: "\\n".to_string(),
                leading_space: false,
            });
            index += 1;
            continue;
        }
        let mut leading_space = false;
        if ch.is_whitespace() {
            leading_space = true;
            index += 1;
            while index < chars.len() && chars[index].is_whitespace() && chars[index] != '\n' {
                index += 1;
            }
            if index >= chars.len() {
                break;
            }
        }
        let ch = chars[index];
        if ch == '\n' {
            tokens.push(Token {
                kind: TokenKind::Newline,
                text: "\\n".to_string(),
                leading_space: false,
            });
            index += 1;
            continue;
        }
        if ch.is_ascii_lowercase() {
            let start = index;
            index += 1;
            while index < chars.len()
                && (chars[index].is_ascii_alphanumeric() || chars[index] == '_')
            {
                index += 1;
            }
            let text: String = chars[start..index].iter().collect();
            let kind = match text.as_str() {
                "import" => TokenKind::Import,
                "as" => TokenKind::As,
                "let" => TokenKind::Let,
                "in" => TokenKind::In,
                _ => TokenKind::Ident,
            };
            tokens.push(Token {
                kind,
                text,
                leading_space,
            });
            continue;
        }
        if ch == '_' {
            tokens.push(Token {
                kind: TokenKind::Ident,
                text: "_".to_string(),
                leading_space,
            });
            index += 1;
            continue;
        }
        if ch == '\\' {
            tokens.push(Token {
                kind: TokenKind::Backslash,
                text: "\\".to_string(),
                leading_space,
            });
            index += 1;
            continue;
        }
        if ch.is_ascii_digit() {
            let start = index;
            index += 1;
            while index < chars.len() && chars[index].is_ascii_digit() {
                index += 1;
            }
            let mut kind = TokenKind::Int;
            if index < chars.len() && chars[index] == '.' {
                kind = TokenKind::Float;
                index += 1;
                if index >= chars.len() || !chars[index].is_ascii_digit() {
                    return Err(SimdError::new("float literal must have digits after '.'"));
                }
                while index < chars.len() && chars[index].is_ascii_digit() {
                    index += 1;
                }
            }
            let text: String = chars[start..index].iter().collect();
            tokens.push(Token {
                kind,
                text,
                leading_space,
            });
            continue;
        }
        let (kind, width) = match ch {
            ':' => (TokenKind::Colon, 1),
            '=' => {
                if chars.get(index + 1) == Some(&'=') {
                    (TokenKind::EqEq, 2)
                } else {
                    (TokenKind::Eq, 1)
                }
            }
            '+' => (TokenKind::Plus, 1),
            '-' => {
                if chars.get(index + 1) == Some(&'>') {
                    (TokenKind::Arrow, 2)
                } else {
                    (TokenKind::Minus, 1)
                }
            }
            '*' => (TokenKind::Star, 1),
            '/' => (TokenKind::Slash, 1),
            '%' => (TokenKind::Percent, 1),
            '<' => {
                if chars.get(index + 1) == Some(&'=') {
                    (TokenKind::Le, 2)
                } else {
                    (TokenKind::Lt, 1)
                }
            }
            '>' => {
                if chars.get(index + 1) == Some(&'=') {
                    (TokenKind::Ge, 2)
                } else {
                    (TokenKind::Gt, 1)
                }
            }
            '(' => (TokenKind::LParen, 1),
            ')' => (TokenKind::RParen, 1),
            '[' => (TokenKind::LBracket, 1),
            ']' => (TokenKind::RBracket, 1),
            '{' => (TokenKind::LBrace, 1),
            '}' => (TokenKind::RBrace, 1),
            '.' => (TokenKind::Dot, 1),
            ',' => (TokenKind::Comma, 1),
            ';' => (TokenKind::Semicolon, 1),
            _ => {
                return Err(SimdError::new(format!(
                    "unexpected character '{}' in input",
                    ch
                )));
            }
        };
        let text: String = chars[index..index + width].iter().collect();
        tokens.push(Token {
            kind,
            text,
            leading_space,
        });
        index += width;
    }
    Ok(tokens)
}

struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    import_aliases: BTreeMap<String, String>,
    saw_non_import_decl: bool,
}

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            pos: 0,
            import_aliases: BTreeMap::new(),
            saw_non_import_decl: false,
        }
    }

    fn parse_program(&mut self) -> Result<SurfaceProgram> {
        let mut decls = Vec::new();
        self.skip_newlines();
        while !self.is_eof() {
            decls.push(self.parse_decl()?);
            self.skip_newlines();
        }
        Ok(SurfaceProgram { decls })
    }

    fn parse_decl(&mut self) -> Result<Decl> {
        if self.peek_is(TokenKind::Import) {
            if self.saw_non_import_decl {
                return Err(SimdError::new(
                    "imports must appear before function signatures and clauses",
                ));
            }
            return self.parse_import_decl();
        }
        self.saw_non_import_decl = true;
        let name = self.expect_ident()?;
        if self.eat(TokenKind::Colon) {
            let ty = self.parse_type()?;
            return Ok(Decl::Signature(Signature { name, ty }));
        }
        let mut patterns = Vec::new();
        while !self.eat(TokenKind::Eq) {
            patterns.push(self.parse_pattern()?);
        }
        let body = self.parse_expr(0)?;
        Ok(Decl::Clause(Clause {
            name,
            patterns,
            body,
        }))
    }

    fn parse_import_decl(&mut self) -> Result<Decl> {
        self.expect(TokenKind::Import)?;
        let mut segments = vec![self.expect_ident()?];
        while self.eat(TokenKind::Slash) {
            segments.push(self.expect_ident()?);
        }
        let path = segments.join("/");
        let default_alias = segments
            .last()
            .cloned()
            .ok_or_else(|| SimdError::new("import path cannot be empty"))?;
        let alias = if self.eat(TokenKind::As) {
            self.expect_ident()?
        } else {
            default_alias
        };
        if alias == "_" {
            return Err(SimdError::new("import alias '_' is not allowed"));
        }
        if self.import_aliases.contains_key(&alias) {
            return Err(SimdError::new(format!(
                "import alias '{}' is declared more than once",
                alias
            )));
        }
        self.import_aliases.insert(alias.clone(), path.clone());
        Ok(Decl::Import(ImportDecl { path, alias }))
    }

    fn parse_pattern(&mut self) -> Result<Pattern> {
        let token = self
            .peek()
            .cloned()
            .ok_or_else(|| SimdError::new("unexpected end of input in pattern"))?;
        match token.kind {
            TokenKind::Ident => {
                self.pos += 1;
                if token.text == "_" {
                    Ok(Pattern::Wildcard)
                } else {
                    Ok(Pattern::Name(token.text))
                }
            }
            TokenKind::Int => {
                self.pos += 1;
                Ok(Pattern::Int(parse_int(&token.text)?))
            }
            TokenKind::Float => {
                self.pos += 1;
                Ok(Pattern::Float(parse_float(&token.text)?))
            }
            _ => Err(SimdError::new(format!(
                "unexpected token '{}' in pattern",
                token.text
            ))),
        }
    }

    fn parse_type(&mut self) -> Result<Type> {
        let head = self.parse_nonfun_type()?;
        if self.eat(TokenKind::Arrow) {
            let tail = self.parse_type()?;
            let (mut args, ret) = tail.fun_parts();
            args.insert(0, head);
            Ok(Type::Fun(args, Box::new(ret)))
        } else {
            Ok(head)
        }
    }

    fn parse_nonfun_type(&mut self) -> Result<Type> {
        let mut head = if self.eat(TokenKind::LParen) {
            let ty = self.parse_type()?;
            self.expect(TokenKind::RParen)?;
            ty
        } else if self.eat(TokenKind::LBrace) {
            let mut fields = BTreeMap::new();
            if !self.eat(TokenKind::RBrace) {
                loop {
                    let name = self.expect_ident()?;
                    self.expect(TokenKind::Colon)?;
                    let ty = self.parse_type()?;
                    if fields.insert(name.clone(), ty).is_some() {
                        return Err(SimdError::new(format!(
                            "record type field '{}' is declared more than once",
                            name
                        )));
                    }
                    if self.eat(TokenKind::Comma) {
                        continue;
                    }
                    self.expect(TokenKind::RBrace)?;
                    break;
                }
            }
            Type::Record(fields)
        } else {
            let name = self.expect_ident()?;
            match Prim::parse(&name) {
                Some(prim) => Type::Scalar(prim),
                None if is_implicit_type_var_name(&name) => Type::Var(name),
                None => {
                    return Err(SimdError::new(format!("unknown primitive type '{name}'")));
                }
            }
        };

        while self.eat(TokenKind::LBracket) {
            let mut dims = Vec::new();
            loop {
                dims.push(self.parse_dim()?);
                if self.eat(TokenKind::Comma) {
                    continue;
                }
                self.expect(TokenKind::RBracket)?;
                break;
            }
            head = lift_type_over_shape(&head, &Shape(dims))?;
        }
        Ok(head)
    }

    fn parse_dim(&mut self) -> Result<Dim> {
        let token = self
            .peek()
            .cloned()
            .ok_or_else(|| SimdError::new("unexpected end of input in shape"))?;
        match token.kind {
            TokenKind::Int => {
                self.pos += 1;
                Ok(Dim::Const(parse_nat(&token.text)?))
            }
            TokenKind::Ident => {
                self.pos += 1;
                Ok(Dim::Var(token.text))
            }
            _ => Err(SimdError::new(format!(
                "unexpected token '{}' in shape",
                token.text
            ))),
        }
    }

    fn parse_expr(&mut self, min_bp: u8) -> Result<Expr> {
        let mut lhs = self.parse_atom()?;
        loop {
            while self.peek_is(TokenKind::Dot) || self.peek_is(TokenKind::LBrace) {
                lhs = self.parse_postfix(lhs)?;
            }
            if matches!(
                self.peek().map(|token| &token.kind),
                Some(TokenKind::Newline | TokenKind::Semicolon | TokenKind::In)
            ) {
                break;
            }
            if self.next_starts_atom() && min_bp <= 70 {
                let rhs = self.parse_atom()?;
                lhs = Expr::App(Box::new(lhs), Box::new(rhs));
                continue;
            }
            let token = match self.peek() {
                Some(token) => token.clone(),
                None => break,
            };
            let op = match PrimOp::from_token(&token.kind) {
                Some(op) => op,
                None => break,
            };
            let (lbp, rbp) = infix_binding_power(op);
            if lbp < min_bp {
                break;
            }
            self.pos += 1;
            let rhs = self.parse_expr(rbp)?;
            lhs = Expr::Infix {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_atom(&mut self) -> Result<Expr> {
        let token = self
            .peek()
            .cloned()
            .ok_or_else(|| SimdError::new("unexpected end of input in expression"))?;
        match token.kind {
            TokenKind::Let => self.parse_let_expr(),
            TokenKind::Backslash => self.parse_lambda_expr(),
            TokenKind::Ident => {
                self.pos += 1;
                Ok(Expr::Ref(self.parse_ref_expr(token.text)?))
            }
            TokenKind::Int => {
                self.pos += 1;
                Ok(Expr::Int(parse_int(&token.text)?))
            }
            TokenKind::Float => {
                self.pos += 1;
                Ok(Expr::Float(parse_float(&token.text)?))
            }
            TokenKind::LParen => {
                self.pos += 1;
                let expr = self.parse_expr(0)?;
                self.expect(TokenKind::RParen)?;
                Ok(expr)
            }
            TokenKind::LBrace => {
                self.pos += 1;
                Ok(Expr::Record(self.parse_record_expr_fields()?))
            }
            _ => Err(SimdError::new(format!(
                "unexpected token '{}' in expression",
                token.text
            ))),
        }
    }

    fn parse_postfix(&mut self, lhs: Expr) -> Result<Expr> {
        if self.eat(TokenKind::Dot) {
            return Ok(Expr::Project(Box::new(lhs), self.expect_ident()?));
        }
        if self.eat(TokenKind::LBrace) {
            return Ok(Expr::RecordUpdate {
                base: Box::new(lhs),
                fields: self.parse_record_expr_fields()?,
            });
        }
        Ok(lhs)
    }

    fn parse_record_expr_fields(&mut self) -> Result<Vec<(String, Expr)>> {
        let mut fields = Vec::new();
        if self.eat(TokenKind::RBrace) {
            return Ok(fields);
        }
        loop {
            let name = self.expect_ident()?;
            self.expect(TokenKind::Eq)?;
            let expr = self.parse_expr(0)?;
            fields.push((name, expr));
            if self.eat(TokenKind::Comma) {
                continue;
            }
            self.expect(TokenKind::RBrace)?;
            break;
        }
        Ok(fields)
    }

    fn next_starts_atom(&self) -> bool {
        matches!(
            self.peek().map(|token| &token.kind),
            Some(
                TokenKind::Ident
                    | TokenKind::Backslash
                    | TokenKind::Let
                    | TokenKind::Int
                    | TokenKind::Float
                    | TokenKind::LParen
            )
        )
    }

    fn parse_lambda_expr(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Backslash)?;
        let param = self.expect_ident()?;
        self.expect(TokenKind::Arrow)?;
        let body = self.parse_expr(0)?;
        Ok(Expr::Lambda {
            param,
            body: Box::new(body),
        })
    }

    fn parse_ref_expr(&mut self, head: String) -> Result<RefExpr> {
        let mut segments = vec![head];
        while self.can_parse_backslash_chain() {
            self.pos += 1;
            segments.push(self.expect_ident()?);
        }
        let (alias, name, type_args) =
            if self.import_aliases.contains_key(&segments[0]) && segments.len() >= 2 {
                let alias = Some(segments[0].clone());
                let name = segments[1].clone();
                let type_args = segments[2..]
                    .iter()
                    .map(|segment| parse_type_arg(segment))
                    .collect::<Result<Vec<_>>>()?;
                (alias, name, type_args)
            } else {
                let name = segments[0].clone();
                let type_args = segments[1..]
                    .iter()
                    .map(|segment| parse_type_arg(segment))
                    .collect::<Result<Vec<_>>>()?;
                (None, name, type_args)
            };
        Ok(RefExpr {
            alias,
            name,
            type_args,
        })
    }

    fn parse_let_expr(&mut self) -> Result<Expr> {
        self.expect(TokenKind::Let)?;
        self.skip_newlines();
        let mut bindings = Vec::new();
        loop {
            let name = match self
                .peek()
                .cloned()
                .ok_or_else(|| SimdError::new("unexpected end of input in let binding"))?
            {
                Token {
                    kind: TokenKind::Ident,
                    text,
                    ..
                } => {
                    self.pos += 1;
                    text
                }
                Token {
                    kind: TokenKind::Let | TokenKind::In,
                    text,
                    ..
                } => {
                    return Err(SimdError::new(format!(
                        "unexpected keyword '{}' in let binding",
                        text
                    )));
                }
                token => {
                    return Err(SimdError::new(format!(
                        "expected let binding name, found '{}'",
                        token.text
                    )));
                }
            };
            self.expect(TokenKind::Eq)?;
            let expr = self.parse_expr(0)?;
            bindings.push(LetBinding { name, expr });
            self.skip_newlines();
            if self.eat(TokenKind::Semicolon) {
                self.skip_newlines();
                if self.peek_is(TokenKind::In) {
                    break;
                }
                continue;
            }
            if self.peek_is(TokenKind::In) {
                break;
            }
            if self.peek_is(TokenKind::Ident) {
                continue;
            }
            break;
        }
        self.skip_newlines();
        self.expect(TokenKind::In)?;
        self.skip_newlines();
        let body = self.parse_expr(0)?;
        Ok(Expr::Let {
            bindings,
            body: Box::new(body),
        })
    }

    fn expect_ident(&mut self) -> Result<String> {
        let token = self
            .peek()
            .cloned()
            .ok_or_else(|| SimdError::new("unexpected end of input"))?;
        if token.kind != TokenKind::Ident {
            return Err(SimdError::new(format!(
                "expected identifier, found '{}'",
                token.text
            )));
        }
        self.pos += 1;
        Ok(token.text)
    }

    fn expect(&mut self, kind: TokenKind) -> Result<()> {
        let token = self
            .peek()
            .cloned()
            .ok_or_else(|| SimdError::new("unexpected end of input"))?;
        if token.kind != kind {
            return Err(SimdError::new(format!(
                "expected {:?}, found '{}'",
                kind, token.text
            )));
        }
        self.pos += 1;
        Ok(())
    }

    fn eat(&mut self, kind: TokenKind) -> bool {
        if self.peek().map(|token| token.kind.clone()) == Some(kind) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn peek_is(&self, kind: TokenKind) -> bool {
        self.peek().map(|token| token.kind.clone()) == Some(kind)
    }

    fn can_parse_backslash_chain(&self) -> bool {
        let Some(backslash) = self.peek() else {
            return false;
        };
        let Some(segment) = self.tokens.get(self.pos + 1) else {
            return false;
        };
        backslash.kind == TokenKind::Backslash
            && segment.kind == TokenKind::Ident
            && !backslash.leading_space
            && !segment.leading_space
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    fn is_eof(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn skip_newlines(&mut self) {
        while self.eat(TokenKind::Newline) {}
    }
}

fn infix_binding_power(op: PrimOp) -> (u8, u8) {
    match op {
        PrimOp::Mul | PrimOp::Div | PrimOp::Mod => (50, 51),
        PrimOp::Add | PrimOp::Sub => (40, 41),
        PrimOp::Eq | PrimOp::Lt | PrimOp::Gt | PrimOp::Le | PrimOp::Ge => (30, 31),
    }
}

fn parse_int(text: &str) -> Result<i64> {
    text.parse::<i64>()
        .map_err(|_| SimdError::new(format!("invalid integer literal '{text}'")))
}

fn parse_nat(text: &str) -> Result<usize> {
    text.parse::<usize>()
        .map_err(|_| SimdError::new(format!("invalid natural literal '{text}'")))
}

fn is_implicit_type_var_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|ch| ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_')
}

fn parse_type_arg(segment: &str) -> Result<TypeArg> {
    match Prim::parse(segment) {
        Some(prim) => Ok(TypeArg::Prim(prim)),
        None if is_implicit_type_var_name(segment) => Ok(TypeArg::Name(segment.to_string())),
        None => Err(SimdError::new(format!(
            "explicit specialization segment '{}' must be a primitive type or in-scope type variable",
            segment
        ))),
    }
}

fn parse_float(text: &str) -> Result<f64> {
    text.parse::<f64>()
        .map_err(|_| SimdError::new(format!("invalid float literal '{text}'")))
}

pub fn parse_source(source: &str) -> Result<SurfaceProgram> {
    let tokens = lex(source)?;
    Parser::new(tokens).parse_program()
}

pub fn read_source_file(path: &str) -> Result<String> {
    fs::read_to_string(path)
        .map_err(|error| SimdError::new(format!("failed to read '{path}': {error}")))
}

pub fn compile_source(source: &str) -> Result<CompiledProgram> {
    let (surface, module, checked) = compile_frontend(source)?;
    let normalized = normalize_records(&checked)?;
    let lowered = optimize_lowered_program(&lower_program(&normalized)?);
    let grouped = group_lowered_program(&normalized, &lowered)?;
    let intents = analyze_intents(&grouped);
    Ok(CompiledProgram {
        surface,
        module,
        checked,
        normalized,
        lowered,
        grouped,
        intents,
    })
}

pub(crate) fn compile_frontend(source: &str) -> Result<(SurfaceProgram, Module, CheckedProgram)> {
    let surface = parse_source(source)?;
    let module = group_program(&surface)?;
    let pointwise = analyze_pointwise(&module)?;
    let checked = check_program(&module, &pointwise)?;
    Ok((surface, module, checked))
}

fn group_program(surface: &SurfaceProgram) -> Result<Module> {
    let mut imports = Vec::<ImportDecl>::new();
    let mut import_aliases = BTreeSet::<String>::new();
    let mut order = Vec::<String>::new();
    let mut signatures = BTreeMap::<String, Signature>::new();
    let mut clauses = BTreeMap::<String, Vec<Clause>>::new();

    for decl in &surface.decls {
        match decl {
            Decl::Import(import_decl) => {
                if !import_aliases.insert(import_decl.alias.clone()) {
                    return Err(SimdError::new(format!(
                        "import alias '{}' is declared more than once",
                        import_decl.alias
                    )));
                }
                imports.push(import_decl.clone());
            }
            Decl::Signature(signature) => {
                if !order.contains(&signature.name) {
                    order.push(signature.name.clone());
                }
                if signatures
                    .insert(signature.name.clone(), signature.clone())
                    .is_some()
                {
                    return Err(SimdError::new(format!(
                        "function '{}' has multiple signatures",
                        signature.name
                    )));
                }
            }
            Decl::Clause(clause) => {
                if !order.contains(&clause.name) {
                    order.push(clause.name.clone());
                }
                clauses
                    .entry(clause.name.clone())
                    .or_default()
                    .push(clause.clone());
            }
        }
    }

    let mut functions = Vec::new();
    for name in order {
        let signature = signatures
            .remove(&name)
            .ok_or_else(|| SimdError::new(format!("function '{name}' is missing a signature")))?;
        let mut function_clauses = clauses.remove(&name).unwrap_or_default();
        if function_clauses.is_empty() {
            return Err(SimdError::new(format!("function '{name}' has no clauses")));
        }
        let arity = signature.ty.arity();
        for clause in &function_clauses {
            if clause.patterns.len() != arity {
                return Err(SimdError::new(format!(
                    "function '{}' clause arity {} does not match signature arity {}",
                    name,
                    clause.patterns.len(),
                    arity
                )));
            }
        }
        validate_signature_shape_contract(&signature)?;
        functions.push(FunctionDef {
            name,
            signature,
            clauses: std::mem::take(&mut function_clauses),
        });
    }

    if let Some((name, _)) = clauses.into_iter().next() {
        return Err(SimdError::new(format!(
            "function '{}' has clauses but no signature",
            name
        )));
    }

    Ok(Module { imports, functions })
}

fn validate_signature_shape_contract(signature: &Signature) -> Result<()> {
    let _ = signature;
    Ok(())
}

fn analyze_pointwise(module: &Module) -> Result<BTreeMap<String, bool>> {
    let known: BTreeMap<String, usize> = module
        .functions
        .iter()
        .map(|function| (function.name.clone(), function.signature.ty.arity()))
        .collect();

    for function in &module.functions {
        for clause in &function.clauses {
            validate_pointwise_expr(&clause.body, &known)?;
        }
    }

    Ok(module
        .functions
        .iter()
        .map(|function| (function.name.clone(), true))
        .collect())
}

fn validate_pointwise_expr(expr: &Expr, known: &BTreeMap<String, usize>) -> Result<()> {
    match expr {
        Expr::Ref(_) | Expr::Int(_) | Expr::Float(_) => Ok(()),
        Expr::Lambda { body, .. } => validate_pointwise_expr(body, known),
        Expr::Let { bindings, body } => {
            for binding in bindings {
                validate_pointwise_expr(&binding.expr, known)?;
            }
            validate_pointwise_expr(body, known)
        }
        Expr::App(_, _) => {
            let (head, _args) = flatten_apps(expr);
            if let Expr::Ref(reference) = head {
                if reference.alias.is_none()
                    && reference.type_args.is_empty()
                    && known.contains_key(&reference.name)
                {
                    let (_, args) = flatten_apps(expr);
                    for arg in args {
                        validate_pointwise_expr(arg, known)?;
                    }
                    Ok(())
                } else {
                    validate_pointwise_expr(head, known)
                }
            } else {
                validate_pointwise_expr(head, known)
            }
        }
        Expr::Infix { lhs, rhs, .. } => {
            validate_pointwise_expr(lhs, known)?;
            validate_pointwise_expr(rhs, known)
        }
        Expr::Record(fields) => {
            for (_, value) in fields {
                validate_pointwise_expr(value, known)?;
            }
            Ok(())
        }
        Expr::Project(base, _) => validate_pointwise_expr(base, known),
        Expr::RecordUpdate { base, fields } => {
            validate_pointwise_expr(base, known)?;
            for (_, value) in fields {
                validate_pointwise_expr(value, known)?;
            }
            Ok(())
        }
    }
}

fn check_program(module: &Module, pointwise: &BTreeMap<String, bool>) -> Result<CheckedProgram> {
    let signatures: BTreeMap<String, Signature> = module
        .functions
        .iter()
        .map(|function| (function.name.clone(), function.signature.clone()))
        .collect();
    let import_aliases = module
        .imports
        .iter()
        .map(|import| (import.alias.clone(), import.path.clone()))
        .collect::<BTreeMap<_, _>>();

    let mut checked_functions = Vec::new();
    for function in &module.functions {
        let (arg_types, ret_type) = function.signature.ty.fun_parts();
        let mut clauses = Vec::new();
        for clause in &function.clauses {
            let mut locals = BTreeMap::<String, Type>::new();
            let mut patterns = Vec::new();
            for (pattern, ty) in clause.patterns.iter().zip(&arg_types) {
                check_pattern(pattern, ty, &mut locals)?;
                patterns.push(TypedPattern {
                    pattern: pattern.clone(),
                    ty: ty.clone(),
                });
            }
            let context = TypeContext {
                locals: &locals,
                signatures: &signatures,
                imports: &import_aliases,
                pointwise,
            };
            let body = infer_expr(&clause.body, &context, Some(&ret_type))?;
            if body.ty != ret_type {
                return Err(SimdError::new(format!(
                    "function '{}' clause body has type {:?}, expected {:?}",
                    function.name, body.ty, ret_type
                )));
            }
            clauses.push(TypedClause { patterns, body });
        }
        let tailrec = analyze_tailrec(&function.name, &clauses);
        checked_functions.push(CheckedFunction {
            name: function.name.clone(),
            signature: function.signature.clone(),
            clauses,
            pointwise: *pointwise.get(&function.name).unwrap_or(&false),
            tailrec,
        });
    }
    Ok(CheckedProgram {
        functions: checked_functions,
    })
}

fn check_pattern(pattern: &Pattern, ty: &Type, locals: &mut BTreeMap<String, Type>) -> Result<()> {
    match pattern {
        Pattern::Wildcard => Ok(()),
        Pattern::Name(name) => {
            if name == "_" {
                return Err(SimdError::new("wildcard '_' cannot bind a name"));
            }
            if locals.insert(name.clone(), ty.clone()).is_some() {
                return Err(SimdError::new(format!(
                    "pattern name '{}' is bound more than once in one clause",
                    name
                )));
            }
            Ok(())
        }
        Pattern::Int(_) => match ty {
            Type::Scalar(prim) if prim.is_int() => Ok(()),
            _ => Err(SimdError::new(format!(
                "integer pattern is only valid against scalar integer types, found {:?}",
                ty
            ))),
        },
        Pattern::Float(_) => match ty {
            Type::Scalar(prim) if prim.is_float() => Ok(()),
            _ => Err(SimdError::new(format!(
                "float pattern is only valid against scalar float types, found {:?}",
                ty
            ))),
        },
    }
}

struct TypeContext<'a> {
    locals: &'a BTreeMap<String, Type>,
    signatures: &'a BTreeMap<String, Signature>,
    imports: &'a BTreeMap<String, String>,
    pointwise: &'a BTreeMap<String, bool>,
}

fn infer_expr(
    expr: &Expr,
    context: &TypeContext<'_>,
    expected: Option<&Type>,
) -> Result<TypedExpr> {
    match expr {
        Expr::Ref(reference) => infer_ref_expr(reference, context, expected),
        Expr::Int(value) => {
            let prim = match expected.and_then(Type::prim) {
                Some(prim) if prim.is_int() => prim,
                Some(prim) if prim.is_float() => {
                    return Err(SimdError::new(format!(
                        "integer literal '{}' used where {:?} was expected",
                        value, prim
                    )));
                }
                _ => Prim::I64,
            };
            Ok(TypedExpr {
                ty: Type::Scalar(prim),
                kind: TypedExprKind::Int(*value, prim),
            })
        }
        Expr::Float(value) => {
            let prim = match expected.and_then(Type::prim) {
                Some(prim) if prim.is_float() => prim,
                Some(prim) if prim.is_int() => {
                    return Err(SimdError::new(format!(
                        "float literal '{}' used where {:?} was expected",
                        value, prim
                    )));
                }
                _ => Prim::F64,
            };
            Ok(TypedExpr {
                ty: Type::Scalar(prim),
                kind: TypedExprKind::Float(*value, prim),
            })
        }
        Expr::Lambda { param, body } => infer_lambda_expr(param, body, context, expected),
        Expr::Let { bindings, body } => infer_let_expr(bindings, body, context, expected),
        Expr::Record(fields) => infer_record_expr(fields, context, expected),
        Expr::Project(base, field) => infer_projection_expr(base, field, context),
        Expr::RecordUpdate { base, fields } => infer_record_update_expr(base, fields, context),
        Expr::Infix { op, lhs, rhs } => infer_primitive_call(*op, lhs, rhs, context, expected),
        Expr::App(_, _) => infer_apply_expr(expr, context, expected),
    }
}

fn infer_ref_expr(
    reference: &RefExpr,
    context: &TypeContext<'_>,
    expected: Option<&Type>,
) -> Result<TypedExpr> {
    if reference.alias.is_none() && reference.type_args.is_empty() {
        if let Some(ty) = context.locals.get(&reference.name) {
            return Ok(TypedExpr {
                ty: ty.clone(),
                kind: TypedExprKind::Local(reference.name.clone()),
            });
        }
    }
    let resolved_name = resolve_ref_name(reference, context).ok_or_else(|| {
        SimdError::new(format!(
            "unknown function reference '{}'{}",
            render_ref_expr(reference),
            render_ref_resolution_suffix(reference, context)
        ))
    })?;
    let signature = context
        .signatures
        .get(&resolved_name)
        .ok_or_else(|| SimdError::new(format!("unknown function '{}'", resolved_name)))?;
    let ty = instantiate_ref_type(&signature.ty, &reference.type_args)?;
    if reference.type_args.is_empty() && type_contains_var(&ty) {
        return Err(SimdError::new(format!(
            "polymorphic function '{}' requires explicit specialization in value position",
            render_ref_expr(reference)
        )));
    }
    if let Some(expected_ty) = expected {
        if &ty != expected_ty {
            return Err(SimdError::new(format!(
                "function reference '{}' has type {:?}, expected {:?}",
                render_ref_expr(reference),
                ty,
                expected_ty
            )));
        }
    }
    Ok(TypedExpr {
        ty,
        kind: TypedExprKind::FunctionRef {
            name: resolved_name,
        },
    })
}

fn infer_lambda_expr(
    param: &str,
    body: &Expr,
    context: &TypeContext<'_>,
    expected: Option<&Type>,
) -> Result<TypedExpr> {
    let expected_fun = expected.and_then(|ty| match ty {
        Type::Fun(args, ret) if args.len() == 1 => Some((args[0].clone(), ret.as_ref().clone())),
        _ => None,
    });
    let Some((param_ty, ret_ty)) = expected_fun else {
        return Err(SimdError::new(
            "lambda expressions currently require an expected unary function type",
        ));
    };
    let mut locals = context.locals.clone();
    locals.insert(param.to_string(), param_ty.clone());
    let body_context = TypeContext {
        locals: &locals,
        signatures: context.signatures,
        imports: context.imports,
        pointwise: context.pointwise,
    };
    let typed_body = infer_expr(body, &body_context, Some(&ret_ty))?;
    if typed_body.ty != ret_ty {
        return Err(SimdError::new(format!(
            "lambda body has type {:?}, expected {:?}",
            typed_body.ty, ret_ty
        )));
    }
    Ok(TypedExpr {
        ty: Type::Fun(vec![param_ty], Box::new(ret_ty)),
        kind: TypedExprKind::Lambda {
            param: param.to_string(),
            body: Box::new(typed_body),
        },
    })
}

fn infer_let_expr(
    bindings: &[LetBinding],
    body: &Expr,
    context: &TypeContext<'_>,
    expected: Option<&Type>,
) -> Result<TypedExpr> {
    let order = topo_sort_let_bindings(bindings)?;
    let mut let_locals = BTreeMap::<String, Type>::new();
    let mut typed_bindings = Vec::with_capacity(bindings.len());

    for index in order {
        let binding = &bindings[index];
        let locals = combined_locals(context.locals, &let_locals);
        let binding_context = TypeContext {
            locals: &locals,
            signatures: context.signatures,
            imports: context.imports,
            pointwise: context.pointwise,
        };
        let typed = infer_expr(&binding.expr, &binding_context, None)?;
        if binding.name != "_" {
            let_locals.insert(binding.name.clone(), typed.ty.clone());
        }
        typed_bindings.push(TypedLetBinding {
            name: binding.name.clone(),
            expr: typed,
        });
    }

    let locals = combined_locals(context.locals, &let_locals);
    let body_context = TypeContext {
        locals: &locals,
        signatures: context.signatures,
        imports: context.imports,
        pointwise: context.pointwise,
    };
    let typed_body = infer_expr(body, &body_context, expected)?;
    Ok(TypedExpr {
        ty: typed_body.ty.clone(),
        kind: TypedExprKind::Let {
            bindings: typed_bindings,
            body: Box::new(typed_body),
        },
    })
}

fn topo_sort_let_bindings(bindings: &[LetBinding]) -> Result<Vec<usize>> {
    let mut binding_names = BTreeMap::<String, usize>::new();
    for (index, binding) in bindings.iter().enumerate() {
        if binding.name == "_" {
            continue;
        }
        if binding_names.insert(binding.name.clone(), index).is_some() {
            return Err(SimdError::new(format!(
                "let binding '{}' is defined more than once",
                binding.name
            )));
        }
    }

    let dependencies = bindings
        .iter()
        .map(|binding| {
            collect_expr_local_names(&binding.expr)
                .into_iter()
                .filter_map(|name| binding_names.get(&name).copied())
                .collect::<BTreeSet<_>>()
        })
        .collect::<Vec<_>>();

    let mut ordered = Vec::with_capacity(bindings.len());
    let mut visiting = vec![false; bindings.len()];
    let mut visited = vec![false; bindings.len()];

    fn visit(
        index: usize,
        bindings: &[LetBinding],
        dependencies: &[BTreeSet<usize>],
        visiting: &mut [bool],
        visited: &mut [bool],
        ordered: &mut Vec<usize>,
    ) -> Result<()> {
        if visited[index] {
            return Ok(());
        }
        if visiting[index] {
            return Err(SimdError::new(format!(
                "cyclic let binding involving '{}'",
                bindings[index].name
            )));
        }
        visiting[index] = true;
        for dependency in &dependencies[index] {
            visit(
                *dependency,
                bindings,
                dependencies,
                visiting,
                visited,
                ordered,
            )?;
        }
        visiting[index] = false;
        visited[index] = true;
        ordered.push(index);
        Ok(())
    }

    for index in 0..bindings.len() {
        visit(
            index,
            bindings,
            &dependencies,
            &mut visiting,
            &mut visited,
            &mut ordered,
        )?;
    }
    Ok(ordered)
}

fn collect_expr_local_names(expr: &Expr) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    collect_expr_local_names_into(expr, &mut names);
    names
}

fn collect_expr_local_names_into(expr: &Expr, names: &mut BTreeSet<String>) {
    match expr {
        Expr::Ref(reference) => {
            if reference.alias.is_none() && reference.type_args.is_empty() {
                names.insert(reference.name.clone());
            }
        }
        Expr::Int(_) | Expr::Float(_) => {}
        Expr::Lambda { body, .. } => {
            collect_expr_local_names_into(body, names);
        }
        Expr::App(fun, arg) => {
            collect_expr_local_names_into(fun, names);
            collect_expr_local_names_into(arg, names);
        }
        Expr::Let { bindings, body } => {
            for binding in bindings {
                collect_expr_local_names_into(&binding.expr, names);
            }
            collect_expr_local_names_into(body, names);
        }
        Expr::Record(fields) => {
            for (_, expr) in fields {
                collect_expr_local_names_into(expr, names);
            }
        }
        Expr::Project(base, _) => collect_expr_local_names_into(base, names),
        Expr::RecordUpdate { base, fields } => {
            collect_expr_local_names_into(base, names);
            for (_, expr) in fields {
                collect_expr_local_names_into(expr, names);
            }
        }
        Expr::Infix { lhs, rhs, .. } => {
            collect_expr_local_names_into(lhs, names);
            collect_expr_local_names_into(rhs, names);
        }
    }
}

fn combined_locals(
    outer: &BTreeMap<String, Type>,
    inner: &BTreeMap<String, Type>,
) -> BTreeMap<String, Type> {
    let mut locals = outer.clone();
    for (name, ty) in inner {
        locals.insert(name.clone(), ty.clone());
    }
    locals
}

fn type_contains_var(ty: &Type) -> bool {
    match ty {
        Type::Var(_) | Type::Infer(_) => true,
        Type::Scalar(_) | Type::Bulk(_, _) => false,
        Type::Record(fields) => fields.values().any(type_contains_var),
        Type::Fun(args, ret) => args.iter().any(type_contains_var) || type_contains_var(ret),
    }
}

fn collect_type_vars_in_order(ty: &Type) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut vars = Vec::new();
    collect_type_vars_in_order_into(ty, &mut seen, &mut vars);
    vars
}

fn collect_type_vars_in_order_into(ty: &Type, seen: &mut BTreeSet<String>, vars: &mut Vec<String>) {
    match ty {
        Type::Var(name) => {
            if seen.insert(name.clone()) {
                vars.push(name.clone());
            }
        }
        Type::Scalar(_) | Type::Bulk(_, _) | Type::Infer(_) => {}
        Type::Record(fields) => {
            for field_ty in fields.values() {
                collect_type_vars_in_order_into(field_ty, seen, vars);
            }
        }
        Type::Fun(args, ret) => {
            for arg in args {
                collect_type_vars_in_order_into(arg, seen, vars);
            }
            collect_type_vars_in_order_into(ret, seen, vars);
        }
    }
}

fn apply_type_subst(ty: &Type, subst: &BTreeMap<String, Type>) -> Type {
    match ty {
        Type::Var(name) => subst.get(name).cloned().unwrap_or_else(|| ty.clone()),
        Type::Scalar(_) | Type::Bulk(_, _) | Type::Infer(_) => ty.clone(),
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

fn unify_type_template(
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
                "expected scalar {:?}, found {:?}",
                left, actual
            ))),
        },
        Type::Bulk(left_prim, left_shape) => match actual {
            Type::Bulk(right_prim, right_shape)
                if left_prim == right_prim && left_shape == right_shape =>
            {
                Ok(())
            }
            _ => Err(SimdError::new(format!(
                "expected bulk {:?}, found {:?}",
                template, actual
            ))),
        },
        Type::Record(left_fields) => match actual {
            Type::Record(right_fields) if left_fields.len() == right_fields.len() => {
                for (name, left_ty) in left_fields {
                    let right_ty = right_fields.get(name).ok_or_else(|| {
                        SimdError::new(format!("record field '{}' is missing", name))
                    })?;
                    unify_type_template(left_ty, right_ty, subst)?;
                }
                Ok(())
            }
            _ => Err(SimdError::new(format!(
                "expected record {:?}, found {:?}",
                template, actual
            ))),
        },
        Type::Fun(left_args, left_ret) => match actual {
            Type::Fun(right_args, right_ret) if left_args.len() == right_args.len() => {
                for (left_arg, right_arg) in left_args.iter().zip(right_args) {
                    unify_type_template(left_arg, right_arg, subst)?;
                }
                unify_type_template(left_ret, right_ret, subst)
            }
            _ => Err(SimdError::new(format!(
                "expected function {:?}, found {:?}",
                template, actual
            ))),
        },
        Type::Infer(_) => Ok(()),
    }
}

fn infer_record_expr(
    fields: &[(String, Expr)],
    context: &TypeContext<'_>,
    expected: Option<&Type>,
) -> Result<TypedExpr> {
    let expected_fields = match expected {
        Some(Type::Record(fields)) => Some(fields),
        Some(other) => {
            return Err(SimdError::new(format!(
                "record literal used where {:?} was expected",
                other
            )));
        }
        None => None,
    };

    let mut typed_fields = BTreeMap::new();
    for (name, expr) in fields {
        let field_expected = expected_fields.and_then(|items| items.get(name));
        let typed = infer_expr(expr, context, field_expected)?;
        if let Some(field_expected) = field_expected {
            if typed.ty != *field_expected {
                return Err(SimdError::new(format!(
                    "record field '{}' has type {:?}, expected {:?}",
                    name, typed.ty, field_expected
                )));
            }
        }
        if typed_fields.insert(name.clone(), typed).is_some() {
            return Err(SimdError::new(format!(
                "record field '{}' is set more than once",
                name
            )));
        }
    }

    if let Some(expected_fields) = expected_fields {
        if typed_fields.len() != expected_fields.len()
            || expected_fields
                .keys()
                .any(|name| !typed_fields.contains_key(name))
        {
            return Err(SimdError::new(
                "record literal fields do not match the expected closed record type",
            ));
        }
    }

    let ty = Type::Record(
        typed_fields
            .iter()
            .map(|(name, expr)| (name.clone(), expr.ty.clone()))
            .collect(),
    );
    Ok(TypedExpr {
        ty,
        kind: TypedExprKind::Record(typed_fields),
    })
}

fn infer_projection_expr(base: &Expr, field: &str, context: &TypeContext<'_>) -> Result<TypedExpr> {
    let base = infer_expr(base, context, None)?;
    let Type::Record(fields) = &base.ty else {
        return Err(SimdError::new(format!(
            "field projection '.{}' requires a record value, found {:?}",
            field, base.ty
        )));
    };
    let field_ty = fields
        .get(field)
        .cloned()
        .ok_or_else(|| SimdError::new(format!("record field '{}' does not exist", field)))?;
    Ok(TypedExpr {
        ty: field_ty,
        kind: TypedExprKind::Project {
            base: Box::new(base),
            field: field.to_string(),
        },
    })
}

fn infer_record_update_expr(
    base: &Expr,
    fields: &[(String, Expr)],
    context: &TypeContext<'_>,
) -> Result<TypedExpr> {
    let base = infer_expr(base, context, None)?;
    let Type::Record(base_fields) = &base.ty else {
        return Err(SimdError::new(format!(
            "record update requires a record base value, found {:?}",
            base.ty
        )));
    };
    let mut typed_fields = BTreeMap::new();
    for (name, expr) in fields {
        let expected_ty = base_fields
            .get(name)
            .ok_or_else(|| SimdError::new(format!("record field '{}' does not exist", name)))?;
        let typed = infer_expr(expr, context, Some(expected_ty))?;
        if typed.ty != *expected_ty {
            return Err(SimdError::new(format!(
                "record update field '{}' has type {:?}, expected {:?}",
                name, typed.ty, expected_ty
            )));
        }
        if typed_fields.insert(name.clone(), typed).is_some() {
            return Err(SimdError::new(format!(
                "record update field '{}' is set more than once",
                name
            )));
        }
    }
    Ok(TypedExpr {
        ty: base.ty.clone(),
        kind: TypedExprKind::RecordUpdate {
            base: Box::new(base),
            fields: typed_fields,
        },
    })
}

fn infer_apply_expr(
    expr: &Expr,
    context: &TypeContext<'_>,
    expected: Option<&Type>,
) -> Result<TypedExpr> {
    let (head, args_exprs) = flatten_apps(expr);
    if let Some(direct_call) = infer_direct_function_call(head, &args_exprs, context, expected)? {
        return Ok(direct_call);
    }
    let mut typed = infer_expr(head, context, None)?;
    for arg_expr in args_exprs {
        let (param_ty, rest_ty) = match &typed.ty {
            Type::Fun(params, ret) if !params.is_empty() => {
                let param_ty = params[0].clone();
                let rest_ty = if params.len() == 1 {
                    ret.as_ref().clone()
                } else {
                    Type::Fun(params[1..].to_vec(), ret.clone())
                };
                (param_ty, rest_ty)
            }
            other => {
                return Err(SimdError::new(format!(
                    "cannot apply non-function value of type {:?}",
                    other
                )));
            }
        };
        let typed_arg = infer_expr(arg_expr, context, Some(&param_ty))?;
        if typed_arg.ty != param_ty {
            return Err(SimdError::new(format!(
                "application argument has type {:?}, expected {:?}",
                typed_arg.ty, param_ty
            )));
        }
        typed = TypedExpr {
            ty: rest_ty,
            kind: TypedExprKind::Apply {
                callee: Box::new(typed),
                arg: Box::new(typed_arg),
            },
        };
    }
    if let Some(expected_ty) = expected {
        if &typed.ty != expected_ty {
            return Err(SimdError::new(format!(
                "application has type {:?}, expected {:?}",
                typed.ty, expected_ty
            )));
        }
    }
    Ok(typed)
}

fn infer_direct_function_call(
    head: &Expr,
    args_exprs: &[&Expr],
    context: &TypeContext<'_>,
    expected: Option<&Type>,
) -> Result<Option<TypedExpr>> {
    let Expr::Ref(reference) = head else {
        return Ok(None);
    };
    let Some(name) = resolve_ref_name(reference, context) else {
        return Ok(None);
    };
    let signature = context
        .signatures
        .get(&name)
        .ok_or_else(|| SimdError::new(format!("unknown function '{}'", name)))?;
    let (param_types, ret_type) = instantiate_call_signature(
        signature,
        &reference.type_args,
        args_exprs,
        context,
        expected,
    )?;
    if args_exprs.len() > param_types.len() {
        return Err(SimdError::new(format!(
            "function '{}' expects {} arguments, found {}",
            render_ref_expr(reference),
            param_types.len(),
            args_exprs.len()
        )));
    }
    if args_exprs.len() < param_types.len() {
        let mut typed = infer_ref_expr(reference, context, None)?;
        for (expr_arg, param_ty) in args_exprs.iter().zip(param_types.iter()) {
            let typed_arg = infer_expr(expr_arg, context, Some(param_ty))?;
            typed = TypedExpr {
                ty: match &typed.ty {
                    Type::Fun(params, ret) if !params.is_empty() => {
                        if typed_arg.ty != params[0] {
                            return Err(SimdError::new(format!(
                                "application argument has type {:?}, expected {:?}",
                                typed_arg.ty, params[0]
                            )));
                        }
                        if params.len() == 1 {
                            ret.as_ref().clone()
                        } else {
                            Type::Fun(params[1..].to_vec(), ret.clone())
                        }
                    }
                    other => {
                        return Err(SimdError::new(format!(
                            "cannot apply non-function value of type {:?}",
                            other
                        )));
                    }
                },
                kind: TypedExprKind::Apply {
                    callee: Box::new(typed),
                    arg: Box::new(typed_arg),
                },
            };
        }
        return Ok(Some(typed));
    }
    let mut dim_subst = BTreeMap::<String, Dim>::new();
    let mut lifted_shape: Option<Shape> = None;
    let mut args = Vec::new();
    for (expr_arg, param_ty) in args_exprs.iter().zip(&param_types) {
        let typed_arg = infer_expr(expr_arg, context, Some(param_ty))?;
        let mut exact_subst = dim_subst.clone();
        let mode = if unify_param_type(param_ty, &typed_arg.ty, &mut exact_subst).is_ok() {
            dim_subst = exact_subst;
            AccessKind::Same
        } else {
            let mut trial_shape = lifted_shape.clone();
            if *context.pointwise.get(&name).unwrap_or(&false) {
                match unify_lifted_type(&mut trial_shape, param_ty, &typed_arg.ty) {
                    Ok(()) => {
                        lifted_shape = trial_shape;
                        AccessKind::Lane
                    }
                    Err(error) => return Err(error),
                }
            } else {
                return Err(SimdError::new(format!(
                    "cannot pass argument of type {:?} to parameter {:?} in '{}'",
                    typed_arg.ty, param_ty, name
                )));
            }
        };
        args.push(TypedArg {
            mode,
            expr: Box::new(typed_arg),
        });
    }
    let instantiated_ret = apply_dim_subst(&ret_type, &dim_subst);
    let ty = if let Some(shape) = &lifted_shape {
        lift_type_over_shape(&instantiated_ret, shape)?
    } else {
        instantiated_ret
    };
    if let Some(expected_ty) = expected {
        if &ty != expected_ty {
            return Err(SimdError::new(format!(
                "call to '{}' has type {:?}, expected {:?}",
                name, ty, expected_ty
            )));
        }
    }
    Ok(Some(TypedExpr {
        ty,
        kind: TypedExprKind::Call {
            callee: Callee::Function(name),
            args,
            lifted_shape,
        },
    }))
}

fn resolve_ref_name(reference: &RefExpr, context: &TypeContext<'_>) -> Option<String> {
    if reference.alias.is_none() && context.signatures.contains_key(&reference.name) {
        return Some(reference.name.clone());
    }
    let alias = reference.alias.as_ref()?;
    let path = context.imports.get(alias)?;
    let qualified = format!("{}/{}", path, reference.name);
    if context.signatures.contains_key(&qualified) {
        return Some(qualified.clone());
    }
    if context.signatures.contains_key(&reference.name) {
        return Some(reference.name.clone());
    }
    None
}

fn render_ref_resolution_suffix(reference: &RefExpr, context: &TypeContext<'_>) -> String {
    let Some(alias) = &reference.alias else {
        return String::new();
    };
    let Some(path) = context.imports.get(alias) else {
        return String::new();
    };
    format!(
        " (import '{}' resolves to '{}/{}')",
        alias, path, reference.name
    )
}

fn render_ref_expr(reference: &RefExpr) -> String {
    let mut out = String::new();
    if let Some(alias) = &reference.alias {
        out.push_str(alias);
        out.push('\\');
    }
    out.push_str(&reference.name);
    for type_arg in &reference.type_args {
        out.push('\\');
        match type_arg {
            TypeArg::Prim(prim) => out.push_str(match prim {
                Prim::I32 => "i32",
                Prim::I64 => "i64",
                Prim::F32 => "f32",
                Prim::F64 => "f64",
            }),
            TypeArg::Name(name) => out.push_str(name),
        }
    }
    out
}

fn instantiate_ref_type(ty: &Type, type_args: &[TypeArg]) -> Result<Type> {
    let vars = collect_type_vars_in_order(ty);
    if type_args.is_empty() {
        return Ok(ty.clone());
    }
    if vars.len() != type_args.len() {
        return Err(SimdError::new(format!(
            "explicit specialization expects {} type arguments, found {}",
            vars.len(),
            type_args.len()
        )));
    }
    let subst = vars
        .into_iter()
        .zip(type_args.iter())
        .map(|(name, arg)| {
            let ty = match arg {
                TypeArg::Prim(prim) => Type::Scalar(*prim),
                TypeArg::Name(name) => Type::Var(name.clone()),
            };
            (name, ty)
        })
        .collect::<BTreeMap<_, _>>();
    Ok(apply_type_subst(ty, &subst))
}

fn instantiate_call_signature(
    signature: &Signature,
    explicit_type_args: &[TypeArg],
    args_exprs: &[&Expr],
    context: &TypeContext<'_>,
    expected: Option<&Type>,
) -> Result<(Vec<Type>, Type)> {
    let (mut params, mut ret) = signature.ty.fun_parts();
    if !explicit_type_args.is_empty() {
        let specialized = instantiate_ref_type(&signature.ty, explicit_type_args)?;
        return Ok(specialized.fun_parts());
    }
    let vars = collect_type_vars_in_order(&signature.ty);
    if vars.is_empty() {
        return Ok((params, ret));
    }
    let mut subst = BTreeMap::<String, Type>::new();
    for (expr_arg, param_ty) in args_exprs.iter().zip(params.iter()) {
        let typed_arg = infer_expr(expr_arg, context, None)?;
        unify_type_template(param_ty, &typed_arg.ty, &mut subst)?;
    }
    if let Some(expected_ty) = expected {
        unify_type_template(&ret, expected_ty, &mut subst)?;
    }
    params = params
        .into_iter()
        .map(|ty| apply_type_subst(&ty, &subst))
        .collect();
    ret = apply_type_subst(&ret, &subst);
    Ok((params, ret))
}

fn infer_primitive_call(
    op: PrimOp,
    lhs: &Expr,
    rhs: &Expr,
    context: &TypeContext<'_>,
    expected: Option<&Type>,
) -> Result<TypedExpr> {
    let mut left = infer_expr(lhs, context, expected)?;
    let mut right =
        infer_expr(rhs, context, Some(&left.ty)).or_else(|_| infer_expr(rhs, context, None))?;

    if !primitive_operands_compatible(op, &left.ty, &right.ty) {
        if let Some(prim) = right.ty.prim() {
            if let Some(retargeted) = retarget_literal(&left, prim) {
                left = retargeted;
            }
        }
    }
    if !primitive_operands_compatible(op, &left.ty, &right.ty) {
        if let Some(prim) = left.ty.prim() {
            if let Some(retargeted) = retarget_literal(&right, prim) {
                right = retargeted;
            }
        }
    }
    let (left_mode, right_mode, result_ty) = infer_primitive_signature(op, &left.ty, &right.ty)?;
    let call_shape = match (&left_mode, &right_mode) {
        (AccessKind::Lane, _) => left.ty.bulk_shape(),
        (_, AccessKind::Lane) => right.ty.bulk_shape(),
        _ => None,
    };
    Ok(TypedExpr {
        ty: result_ty,
        kind: TypedExprKind::Call {
            callee: Callee::Prim(op),
            args: vec![
                TypedArg {
                    mode: left_mode,
                    expr: Box::new(left),
                },
                TypedArg {
                    mode: right_mode,
                    expr: Box::new(right),
                },
            ],
            lifted_shape: call_shape,
        },
    })
}

trait TypeExt {
    fn bulk_shape(&self) -> Option<Shape>;
}

impl TypeExt for Type {
    fn bulk_shape(&self) -> Option<Shape> {
        match self {
            Type::Bulk(_, shape) => Some(shape.clone()),
            _ => None,
        }
    }
}

fn primitive_operands_compatible(op: PrimOp, left: &Type, right: &Type) -> bool {
    infer_primitive_signature(op, left, right).is_ok()
}

fn infer_primitive_signature(
    op: PrimOp,
    left: &Type,
    right: &Type,
) -> Result<(AccessKind, AccessKind, Type)> {
    match (left, right) {
        (Type::Var(left_name), Type::Var(right_name)) if left_name == right_name => Ok((
            AccessKind::Same,
            AccessKind::Same,
            match op {
                PrimOp::Eq | PrimOp::Lt | PrimOp::Gt | PrimOp::Le | PrimOp::Ge => {
                    Type::Scalar(Prim::I64)
                }
                _ => Type::Var(left_name.clone()),
            },
        )),
        (Type::Scalar(left_prim), Type::Scalar(right_prim)) if left_prim == right_prim => {
            ensure_valid_primitive_prim(op, *left_prim)?;
            let result = primitive_result_type(op, *left_prim);
            Ok((AccessKind::Same, AccessKind::Same, result))
        }
        (Type::Scalar(left_prim), Type::Bulk(right_prim, shape)) if left_prim == right_prim => {
            ensure_valid_primitive_prim(op, *left_prim)?;
            let result = lift_result_type(op, *left_prim, shape.clone());
            Ok((AccessKind::Same, AccessKind::Lane, result))
        }
        (Type::Bulk(left_prim, shape), Type::Scalar(right_prim)) if left_prim == right_prim => {
            ensure_valid_primitive_prim(op, *left_prim)?;
            let result = lift_result_type(op, *left_prim, shape.clone());
            Ok((AccessKind::Lane, AccessKind::Same, result))
        }
        (Type::Bulk(left_prim, left_shape), Type::Bulk(right_prim, right_shape))
            if left_prim == right_prim && left_shape == right_shape =>
        {
            ensure_valid_primitive_prim(op, *left_prim)?;
            let result = lift_result_type(op, *left_prim, left_shape.clone());
            Ok((AccessKind::Lane, AccessKind::Lane, result))
        }
        _ => Err(SimdError::new(format!(
            "primitive {:?} cannot operate on {:?} and {:?}",
            op, left, right
        ))),
    }
}

fn ensure_valid_primitive_prim(op: PrimOp, prim: Prim) -> Result<()> {
    if matches!(op, PrimOp::Mod) && prim.is_float() {
        return Err(SimdError::new(
            "modulo is only defined on integer primitives in v0",
        ));
    }
    Ok(())
}

fn primitive_result_type(op: PrimOp, prim: Prim) -> Type {
    match op {
        PrimOp::Eq | PrimOp::Lt | PrimOp::Gt | PrimOp::Le | PrimOp::Ge => Type::Scalar(Prim::I64),
        _ => Type::Scalar(prim),
    }
}

fn lift_result_type(op: PrimOp, prim: Prim, shape: Shape) -> Type {
    match op {
        PrimOp::Eq | PrimOp::Lt | PrimOp::Gt | PrimOp::Le | PrimOp::Ge => {
            Type::Bulk(Prim::I64, shape)
        }
        _ => Type::Bulk(prim, shape),
    }
}

fn retarget_literal(expr: &TypedExpr, prim: Prim) -> Option<TypedExpr> {
    match &expr.kind {
        TypedExprKind::Int(value, _) if prim.is_int() => Some(TypedExpr {
            ty: Type::Scalar(prim),
            kind: TypedExprKind::Int(*value, prim),
        }),
        TypedExprKind::Float(value, _) if prim.is_float() => Some(TypedExpr {
            ty: Type::Scalar(prim),
            kind: TypedExprKind::Float(*value, prim),
        }),
        _ => None,
    }
}

fn unify_lifted_shape(slot: &mut Option<Shape>, shape: &Shape) -> Result<()> {
    match slot {
        None => {
            *slot = Some(shape.clone());
            Ok(())
        }
        Some(existing) if existing == shape => Ok(()),
        Some(existing) => Err(SimdError::new(format!(
            "lifted bulk arguments have incompatible shapes {:?} and {:?}",
            existing, shape
        ))),
    }
}

fn lift_type_over_shape(ty: &Type, shape: &Shape) -> Result<Type> {
    match ty {
        Type::Scalar(prim) => Ok(Type::Bulk(*prim, shape.clone())),
        Type::Var(name) => Ok(Type::Var(name.clone())),
        Type::Infer(index) => Ok(Type::Infer(*index)),
        Type::Record(fields) => Ok(Type::Record(
            fields
                .iter()
                .map(|(name, field_ty)| Ok((name.clone(), lift_type_over_shape(field_ty, shape)?)))
                .collect::<Result<BTreeMap<_, _>>>()?,
        )),
        Type::Bulk(_, _) => Err(SimdError::new(
            "bulk postfix cannot be applied to a type that is already bulk",
        )),
        Type::Fun(_, _) => Err(SimdError::new(
            "bulk postfix cannot be applied to function types",
        )),
    }
}

fn unify_lifted_type(slot: &mut Option<Shape>, param: &Type, actual: &Type) -> Result<()> {
    match (param, actual) {
        (Type::Var(_), _) | (Type::Infer(_), _) => Ok(()),
        (Type::Scalar(left), Type::Bulk(right, shape)) if left == right => {
            unify_lifted_shape(slot, shape)
        }
        (Type::Record(left), Type::Record(right)) if left.len() == right.len() => {
            for (name, left_ty) in left {
                let right_ty = right.get(name).ok_or_else(|| {
                    SimdError::new(format!(
                        "record field '{}' is missing from lifted argument",
                        name
                    ))
                })?;
                unify_lifted_type(slot, left_ty, right_ty)?;
            }
            Ok(())
        }
        _ => Err(SimdError::new(format!(
            "expected lifted argument matching {:?}, found {:?}",
            param, actual
        ))),
    }
}

fn unify_param_type(
    param: &Type,
    actual: &Type,
    dim_subst: &mut BTreeMap<String, Dim>,
) -> Result<()> {
    match (param, actual) {
        (Type::Var(_), _) => Ok(()),
        (Type::Infer(_), _) => Ok(()),
        (Type::Scalar(left), Type::Scalar(right)) if left == right => Ok(()),
        (Type::Bulk(left_prim, left_shape), Type::Bulk(right_prim, right_shape))
            if left_prim == right_prim && left_shape.0.len() == right_shape.0.len() =>
        {
            for (left_dim, right_dim) in left_shape.0.iter().zip(&right_shape.0) {
                match left_dim {
                    Dim::Const(value) => {
                        if right_dim != &Dim::Const(*value) {
                            return Err(SimdError::new(format!(
                                "shape mismatch: expected {:?}, found {:?}",
                                left_shape, right_shape
                            )));
                        }
                    }
                    Dim::Var(name) => match dim_subst.get(name) {
                        Some(bound) if bound == right_dim => {}
                        Some(bound) => {
                            return Err(SimdError::new(format!(
                                "shape variable '{}' was bound to {:?}, cannot also be {:?}",
                                name, bound, right_dim
                            )));
                        }
                        None => {
                            dim_subst.insert(name.clone(), right_dim.clone());
                        }
                    },
                }
            }
            Ok(())
        }
        (Type::Record(left), Type::Record(right)) if left.len() == right.len() => {
            for (name, left_ty) in left {
                let right_ty = right
                    .get(name)
                    .ok_or_else(|| SimdError::new(format!("record field '{}' is missing", name)))?;
                unify_param_type(left_ty, right_ty, dim_subst)?;
            }
            Ok(())
        }
        (Type::Fun(left_args, left_ret), Type::Fun(right_args, right_ret))
            if left_args.len() == right_args.len() =>
        {
            for (left_arg, right_arg) in left_args.iter().zip(right_args) {
                unify_param_type(left_arg, right_arg, dim_subst)?;
            }
            unify_param_type(left_ret, right_ret, dim_subst)
        }
        _ => Err(SimdError::new(format!(
            "expected argument type {:?}, found {:?}",
            param, actual
        ))),
    }
}

fn apply_dim_subst(ty: &Type, subst: &BTreeMap<String, Dim>) -> Type {
    match ty {
        Type::Scalar(_) | Type::Var(_) | Type::Infer(_) => ty.clone(),
        Type::Bulk(prim, shape) => Type::Bulk(
            *prim,
            Shape(
                shape
                    .0
                    .iter()
                    .map(|dim| match dim {
                        Dim::Const(value) => Dim::Const(*value),
                        Dim::Var(name) => subst
                            .get(name)
                            .cloned()
                            .unwrap_or_else(|| Dim::Var(name.clone())),
                    })
                    .collect(),
            ),
        ),
        Type::Record(fields) => Type::Record(
            fields
                .iter()
                .map(|(name, field_ty)| (name.clone(), apply_dim_subst(field_ty, subst)))
                .collect(),
        ),
        Type::Fun(args, ret) => Type::Fun(
            args.iter().map(|arg| apply_dim_subst(arg, subst)).collect(),
            Box::new(apply_dim_subst(ret, subst)),
        ),
    }
}

fn flatten_apps<'a>(expr: &'a Expr) -> (&'a Expr, Vec<&'a Expr>) {
    let mut args = Vec::new();
    let mut current = expr;
    while let Expr::App(fun, arg) = current {
        args.push(arg.as_ref());
        current = fun.as_ref();
    }
    args.reverse();
    (current, args)
}

fn analyze_tailrec(name: &str, clauses: &[TypedClause]) -> TailRecInfo {
    let mut recursive = false;
    let mut valid = true;
    for clause in clauses {
        visit_tail_calls(&clause.body, name, true, &mut recursive, &mut valid);
    }
    TailRecInfo {
        recursive,
        loop_lowerable: recursive && valid,
    }
}

fn visit_tail_calls(
    expr: &TypedExpr,
    self_name: &str,
    in_tail: bool,
    recursive: &mut bool,
    valid: &mut bool,
) {
    match &expr.kind {
        TypedExprKind::Local(_)
        | TypedExprKind::FunctionRef { .. }
        | TypedExprKind::Int(_, _)
        | TypedExprKind::Float(_, _) => {}
        TypedExprKind::Lambda { body, .. } => {
            visit_tail_calls(body, self_name, false, recursive, valid);
        }
        TypedExprKind::Let { bindings, body } => {
            for binding in bindings {
                visit_tail_calls(&binding.expr, self_name, false, recursive, valid);
            }
            visit_tail_calls(body, self_name, in_tail, recursive, valid);
        }
        TypedExprKind::Record(fields) => {
            for field in fields.values() {
                visit_tail_calls(field, self_name, false, recursive, valid);
            }
        }
        TypedExprKind::Project { base, .. } => {
            visit_tail_calls(base, self_name, false, recursive, valid);
        }
        TypedExprKind::RecordUpdate { base, fields } => {
            visit_tail_calls(base, self_name, false, recursive, valid);
            for field in fields.values() {
                visit_tail_calls(field, self_name, false, recursive, valid);
            }
        }
        TypedExprKind::Call { callee, args, .. } => {
            if callee == &Callee::Function(self_name.to_string()) {
                *recursive = true;
                if !in_tail {
                    *valid = false;
                }
            }
            for arg in args {
                visit_tail_calls(&arg.expr, self_name, false, recursive, valid);
            }
        }
        TypedExprKind::Apply { callee, arg } => {
            visit_tail_calls(callee, self_name, false, recursive, valid);
            visit_tail_calls(arg, self_name, false, recursive, valid);
            *valid = false;
        }
    }
}

#[derive(Debug, Clone)]
struct LocalLeafBinding {
    path: LeafPath,
    local_name: String,
    ty: Type,
}

fn normalize_records(checked: &CheckedProgram) -> Result<NormalizedProgram> {
    let mut entries = Vec::new();
    for function in &checked.functions {
        let (params, ret) = function.signature.ty.fun_parts();
        let param_leaves = params.iter().map(flatten_type_leaves).collect::<Vec<_>>();
        let result_leaves = flatten_type_leaves(&ret);
        let leaf_functions = result_leaves
            .iter()
            .map(|leaf| {
                (
                    leaf.path.clone(),
                    normalized_leaf_function_name(&function.name, &leaf.path),
                )
            })
            .collect();
        entries.push(NormalizedEntry {
            source_name: function.name.clone(),
            source_signature: function.signature.clone(),
            param_leaves,
            result_leaves,
            leaf_functions,
            pointwise: function.pointwise,
        });
    }

    let entry_map = entries
        .iter()
        .map(|entry| (entry.source_name.clone(), entry.clone()))
        .collect::<BTreeMap<_, _>>();

    let mut functions = Vec::new();
    for function in &checked.functions {
        let entry = entry_map.get(&function.name).ok_or_else(|| {
            SimdError::new(format!(
                "missing normalization entry for '{}'",
                function.name
            ))
        })?;
        for result_leaf in &entry.result_leaves {
            functions.push(normalize_function_leaf(
                function,
                entry,
                &entry_map,
                result_leaf,
            )?);
        }
    }

    Ok(NormalizedProgram { functions, entries })
}

fn normalize_function_leaf(
    function: &CheckedFunction,
    entry: &NormalizedEntry,
    entry_map: &BTreeMap<String, NormalizedEntry>,
    result_leaf: &TypeLeaf,
) -> Result<NormalizedFunction> {
    let mut param_types = Vec::new();
    for leaves in &entry.param_leaves {
        for leaf in leaves {
            param_types.push(leaf.ty.clone());
        }
    }

    let mut clauses = Vec::new();
    for clause in &function.clauses {
        let mut patterns = Vec::new();
        let mut locals = BTreeMap::<String, Vec<LocalLeafBinding>>::new();
        for (typed_pattern, leaves) in clause.patterns.iter().zip(&entry.param_leaves) {
            match &typed_pattern.pattern {
                Pattern::Name(name) => {
                    for leaf in leaves {
                        let local_name = normalized_local_name(name, &leaf.path);
                        patterns.push(TypedPattern {
                            pattern: Pattern::Name(local_name.clone()),
                            ty: leaf.ty.clone(),
                        });
                        locals
                            .entry(name.clone())
                            .or_default()
                            .push(LocalLeafBinding {
                                path: leaf.path.clone(),
                                local_name,
                                ty: leaf.ty.clone(),
                            });
                    }
                }
                Pattern::Wildcard => {
                    for leaf in leaves {
                        patterns.push(TypedPattern {
                            pattern: Pattern::Wildcard,
                            ty: leaf.ty.clone(),
                        });
                    }
                }
                Pattern::Int(value) => {
                    if leaves.len() != 1 {
                        return Err(SimdError::new(format!(
                            "record parameter in '{}' cannot use literal pattern {}",
                            function.name, value
                        )));
                    }
                    patterns.push(TypedPattern {
                        pattern: Pattern::Int(*value),
                        ty: leaves[0].ty.clone(),
                    });
                }
                Pattern::Float(value) => {
                    if leaves.len() != 1 {
                        return Err(SimdError::new(format!(
                            "record parameter in '{}' cannot use literal pattern {}",
                            function.name, value
                        )));
                    }
                    patterns.push(TypedPattern {
                        pattern: Pattern::Float(*value),
                        ty: leaves[0].ty.clone(),
                    });
                }
            }
        }

        let body = normalize_expr_to_leaf(&clause.body, &result_leaf.path, &locals, entry_map)?;
        clauses.push(TypedClause { patterns, body });
    }

    let leaf_name = entry
        .leaf_functions
        .get(&result_leaf.path)
        .cloned()
        .ok_or_else(|| {
            SimdError::new(format!(
                "missing leaf function name for '{}'",
                function.name
            ))
        })?;
    let signature = Signature {
        name: leaf_name.clone(),
        ty: Type::Fun(param_types, Box::new(result_leaf.ty.clone())),
    };
    let tailrec = analyze_tailrec(&leaf_name, &clauses);
    Ok(NormalizedFunction {
        name: leaf_name,
        source_name: function.name.clone(),
        leaf_path: result_leaf.path.clone(),
        signature,
        clauses,
        pointwise: function.pointwise,
        tailrec,
    })
}

fn normalize_expr_to_leaf(
    expr: &TypedExpr,
    requested: &LeafPath,
    locals: &BTreeMap<String, Vec<LocalLeafBinding>>,
    entries: &BTreeMap<String, NormalizedEntry>,
) -> Result<TypedExpr> {
    let leaf_ty = lookup_leaf_type(&expr.ty, requested)?;
    match &expr.kind {
        TypedExprKind::Local(name) => {
            let leaves = locals.get(name).ok_or_else(|| {
                SimdError::new(format!("unknown local '{}' during normalization", name))
            })?;
            let binding = leaves
                .iter()
                .find(|leaf| leaf.path == *requested)
                .ok_or_else(|| {
                    SimdError::new(format!(
                        "local '{}' does not contain leaf path {:?}",
                        name, requested
                    ))
                })?;
            Ok(TypedExpr {
                ty: binding.ty.clone(),
                kind: TypedExprKind::Local(binding.local_name.clone()),
            })
        }
        TypedExprKind::Int(value, prim) => {
            if !requested.is_root() {
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
            if !requested.is_root() {
                return Err(SimdError::new(
                    "float literal cannot be projected into a record leaf",
                ));
            }
            Ok(TypedExpr {
                ty: leaf_ty,
                kind: TypedExprKind::Float(*value, *prim),
            })
        }
        TypedExprKind::FunctionRef { .. }
        | TypedExprKind::Lambda { .. }
        | TypedExprKind::Apply { .. } => {
            return Err(SimdError::new(
                "record normalization does not yet support higher-order expressions",
            ));
        }
        TypedExprKind::Let { bindings, body } => {
            let mut normalized_bindings = Vec::<TypedLetBinding>::new();
            let mut local_leaves = locals.clone();
            for binding in bindings {
                if binding.name == "_" {
                    continue;
                }
                let binding_leaves = flatten_type_leaves(&binding.expr.ty);
                for leaf in &binding_leaves {
                    let local_name = normalized_local_name(&binding.name, &leaf.path);
                    let expr =
                        normalize_expr_to_leaf(&binding.expr, &leaf.path, &local_leaves, entries)?;
                    normalized_bindings.push(TypedLetBinding {
                        name: local_name.clone(),
                        expr: expr.clone(),
                    });
                    local_leaves
                        .entry(binding.name.clone())
                        .or_default()
                        .push(LocalLeafBinding {
                            path: leaf.path.clone(),
                            local_name,
                            ty: leaf.ty.clone(),
                        });
                }
            }
            let normalized_body = normalize_expr_to_leaf(body, requested, &local_leaves, entries)?;
            let demanded = demanded_typed_locals(&normalized_body);
            let kept = prune_typed_let_bindings(normalized_bindings, demanded);
            if kept.is_empty() {
                Ok(normalized_body)
            } else {
                Ok(TypedExpr {
                    ty: leaf_ty,
                    kind: TypedExprKind::Let {
                        bindings: kept,
                        body: Box::new(normalized_body),
                    },
                })
            }
        }
        TypedExprKind::Record(fields) => {
            let (field, rest) = requested.split_first().ok_or_else(|| {
                SimdError::new("record expression must normalize to a concrete leaf path")
            })?;
            let field_expr = fields.get(field).ok_or_else(|| {
                SimdError::new(format!(
                    "record field '{}' is missing during normalization",
                    field
                ))
            })?;
            normalize_expr_to_leaf(field_expr, &rest, locals, entries)
        }
        TypedExprKind::Project { base, field } => {
            normalize_expr_to_leaf(base, &requested.prepend(field), locals, entries)
        }
        TypedExprKind::RecordUpdate { base, fields } => {
            let (field, rest) = requested.split_first().ok_or_else(|| {
                SimdError::new("record update must normalize to a concrete leaf path")
            })?;
            if let Some(update) = fields.get(field) {
                normalize_expr_to_leaf(update, &rest, locals, entries)
            } else {
                normalize_expr_to_leaf(base, requested, locals, entries)
            }
        }
        TypedExprKind::Call {
            callee: Callee::Prim(op),
            args,
            lifted_shape,
        } => {
            if !requested.is_root() {
                return Err(SimdError::new(format!(
                    "primitive {:?} cannot normalize to nested record leaf {:?}",
                    op, requested
                )));
            }
            let args = args
                .iter()
                .map(|arg| {
                    Ok(TypedArg {
                        mode: arg.mode,
                        expr: Box::new(normalize_expr_to_leaf(
                            &arg.expr,
                            &LeafPath::root(),
                            locals,
                            entries,
                        )?),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(TypedExpr {
                ty: leaf_ty,
                kind: TypedExprKind::Call {
                    callee: Callee::Prim(*op),
                    args,
                    lifted_shape: lifted_shape.clone(),
                },
            })
        }
        TypedExprKind::Call {
            callee: Callee::Function(name),
            args,
            lifted_shape,
        } => {
            let entry = entries.get(name).ok_or_else(|| {
                SimdError::new(format!("missing normalization entry for callee '{}'", name))
            })?;
            let leaf_name = entry
                .leaf_functions
                .get(requested)
                .cloned()
                .ok_or_else(|| {
                    SimdError::new(format!(
                        "callee '{}' does not expose normalized leaf {:?}",
                        name, requested
                    ))
                })?;
            let mut normalized_args = Vec::new();
            for (arg, param_leaves) in args.iter().zip(&entry.param_leaves) {
                for leaf in param_leaves {
                    normalized_args.push(TypedArg {
                        mode: arg.mode,
                        expr: Box::new(normalize_expr_to_leaf(
                            &arg.expr, &leaf.path, locals, entries,
                        )?),
                    });
                }
            }
            Ok(TypedExpr {
                ty: leaf_ty,
                kind: TypedExprKind::Call {
                    callee: Callee::Function(leaf_name),
                    args: normalized_args,
                    lifted_shape: lifted_shape.clone(),
                },
            })
        }
    }
}

fn lookup_leaf_type(ty: &Type, path: &LeafPath) -> Result<Type> {
    if path.is_root() {
        return match ty {
            Type::Scalar(_) | Type::Bulk(_, _) => Ok(ty.clone()),
            Type::Record(_) => Err(SimdError::new(format!(
                "record type {:?} requires a concrete leaf path",
                ty
            ))),
            Type::Var(_) | Type::Infer(_) => Err(SimdError::new(
                "unresolved type variables cannot appear as normalized leaves",
            )),
            Type::Fun(_, _) => Err(SimdError::new(
                "function type cannot appear as a normalized leaf",
            )),
        };
    }
    let (field, rest) = path
        .split_first()
        .ok_or_else(|| SimdError::new("invalid empty leaf path"))?;
    match ty {
        Type::Record(fields) => {
            let field_ty = fields.get(field).ok_or_else(|| {
                SimdError::new(format!(
                    "record field '{}' does not exist in {:?}",
                    field, ty
                ))
            })?;
            lookup_leaf_type(field_ty, &rest)
        }
        _ => Err(SimdError::new(format!(
            "cannot select record leaf {:?} from non-record type {:?}",
            path, ty
        ))),
    }
}

fn demanded_typed_locals(expr: &TypedExpr) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    collect_typed_local_names(expr, &mut names);
    names
}

fn collect_typed_local_names(expr: &TypedExpr, names: &mut BTreeSet<String>) {
    match &expr.kind {
        TypedExprKind::Local(name) => {
            names.insert(name.clone());
        }
        TypedExprKind::FunctionRef { .. }
        | TypedExprKind::Int(_, _)
        | TypedExprKind::Float(_, _) => {}
        TypedExprKind::Lambda { body, .. } => collect_typed_local_names(body, names),
        TypedExprKind::Let { bindings, body } => {
            for binding in bindings {
                collect_typed_local_names(&binding.expr, names);
            }
            collect_typed_local_names(body, names);
        }
        TypedExprKind::Record(fields) => {
            for expr in fields.values() {
                collect_typed_local_names(expr, names);
            }
        }
        TypedExprKind::Project { base, .. } => collect_typed_local_names(base, names),
        TypedExprKind::RecordUpdate { base, fields } => {
            collect_typed_local_names(base, names);
            for expr in fields.values() {
                collect_typed_local_names(expr, names);
            }
        }
        TypedExprKind::Call { args, .. } => {
            for arg in args {
                collect_typed_local_names(&arg.expr, names);
            }
        }
        TypedExprKind::Apply { callee, arg } => {
            collect_typed_local_names(callee, names);
            collect_typed_local_names(arg, names);
        }
    }
}

fn prune_typed_let_bindings(
    bindings: Vec<TypedLetBinding>,
    mut demanded: BTreeSet<String>,
) -> Vec<TypedLetBinding> {
    let mut kept = Vec::new();
    for binding in bindings.into_iter().rev() {
        if demanded.remove(&binding.name) {
            collect_typed_local_names(&binding.expr, &mut demanded);
            kept.push(binding);
        }
    }
    kept.reverse();
    kept
}

fn normalized_leaf_function_name(source_name: &str, path: &LeafPath) -> String {
    if path.is_root() {
        source_name.to_string()
    } else {
        format!("{}{}{}", source_name, '$', path.suffix())
    }
}

fn normalized_local_name(source_name: &str, path: &LeafPath) -> String {
    if path.is_root() {
        source_name.to_string()
    } else {
        format!("{}{}{}", source_name, '$', path.suffix())
    }
}

pub(crate) fn lower_program(checked: &NormalizedProgram) -> Result<LoweredProgram> {
    let mut functions = Vec::new();
    for function in &checked.functions {
        functions.push(lower_function(function)?);
    }
    Ok(LoweredProgram { functions })
}

fn group_lowered_program(
    normalized: &NormalizedProgram,
    lowered: &LoweredProgram,
) -> Result<GroupedLoweredProgram> {
    if normalized.functions.len() != lowered.functions.len() {
        return Err(SimdError::new(
            "normalized and lowered programs have different function counts",
        ));
    }

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    enum GroupedLoweredKindKey {
        Scalar {
            result_prim: Prim,
            clause_count: usize,
            clause_patterns: Vec<String>,
            tail_loop: bool,
        },
        Kernel {
            result_prim: Prim,
            shape: Shape,
            vector_width: usize,
            cleanup: bool,
            clause_count: usize,
            clause_patterns: Vec<String>,
            tail_loop: bool,
        },
    }

    #[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
    struct GroupKey {
        source_name: String,
        param_access: Vec<AccessKind>,
        kind: GroupedLoweredKindKey,
    }

    #[derive(Debug, Clone)]
    struct GroupBuilder {
        source_name: String,
        leaf_paths: Vec<LeafPath>,
        result_leaves: Vec<TypeLeaf>,
        param_access: Vec<AccessKind>,
        kind: GroupedLoweredKind,
        tail_loop: Option<TailLoop>,
        leaves: Vec<LoweredFunction>,
    }

    let mut groups = Vec::<(GroupKey, GroupBuilder)>::new();

    for (normalized_function, lowered_function) in
        normalized.functions.iter().zip(&lowered.functions)
    {
        if normalized_function.name != lowered_function.name {
            return Err(SimdError::new(format!(
                "normalized function '{}' does not match lowered function '{}'",
                normalized_function.name, lowered_function.name
            )));
        }

        let result_prim = lowered_function.result.prim().ok_or_else(|| {
            SimdError::new(format!(
                "lowered function '{}' does not have a primitive leaf result",
                lowered_function.name
            ))
        })?;
        let clause_count = match &lowered_function.kind {
            LoweredKind::Scalar { clauses } => clauses.len(),
            LoweredKind::Kernel { clauses, .. } => clauses.len(),
        };
        let clause_patterns = match &lowered_function.kind {
            LoweredKind::Scalar { clauses } | LoweredKind::Kernel { clauses, .. } => clauses
                .iter()
                .map(|clause| format!("{:?}", clause.patterns))
                .collect::<Vec<_>>(),
        };
        let kind_key = match &lowered_function.kind {
            LoweredKind::Scalar { .. } => GroupedLoweredKindKey::Scalar {
                result_prim,
                clause_count,
                clause_patterns,
                tail_loop: lowered_function.tail_loop.is_some(),
            },
            LoweredKind::Kernel {
                shape,
                vector_width,
                cleanup,
                ..
            } => GroupedLoweredKindKey::Kernel {
                result_prim,
                shape: shape.clone(),
                vector_width: *vector_width,
                cleanup: *cleanup,
                clause_count,
                clause_patterns,
                tail_loop: lowered_function.tail_loop.is_some(),
            },
        };
        let key = GroupKey {
            source_name: normalized_function.source_name.clone(),
            param_access: lowered_function.param_access.clone(),
            kind: kind_key,
        };
        let result_leaf = TypeLeaf {
            path: normalized_function.leaf_path.clone(),
            ty: normalized_function.signature.ty.fun_parts().1,
        };

        if let Some((_, builder)) = groups
            .iter_mut()
            .find(|(existing_key, _)| *existing_key == key)
        {
            builder.leaf_paths.push(result_leaf.path.clone());
            builder.result_leaves.push(result_leaf);
            builder.leaves.push(lowered_function.clone());
        } else {
            let kind = match &lowered_function.kind {
                LoweredKind::Scalar { clauses } => GroupedLoweredKind::Scalar {
                    clauses: clauses.clone(),
                },
                LoweredKind::Kernel {
                    shape,
                    vector_width,
                    cleanup,
                    clauses,
                } => GroupedLoweredKind::Kernel {
                    shape: shape.clone(),
                    vector_width: *vector_width,
                    cleanup: *cleanup,
                    clauses: clauses.clone(),
                },
            };
            let tail_loop = lowered_function.tail_loop.clone();
            groups.push((
                key,
                GroupBuilder {
                    source_name: normalized_function.source_name.clone(),
                    leaf_paths: vec![result_leaf.path.clone()],
                    result_leaves: vec![result_leaf],
                    param_access: lowered_function.param_access.clone(),
                    kind,
                    tail_loop,
                    leaves: vec![lowered_function.clone()],
                },
            ));
        }
    }

    Ok(GroupedLoweredProgram {
        functions: groups
            .into_iter()
            .map(|(_, builder)| GroupedLoweredFunction {
                source_name: builder.source_name,
                leaf_paths: builder.leaf_paths,
                result_leaves: builder.result_leaves,
                param_access: builder.param_access,
                kind: builder.kind,
                tail_loop: builder.tail_loop,
                leaves: builder.leaves,
            })
            .collect(),
    })
}

fn optimize_lowered_program(program: &LoweredProgram) -> LoweredProgram {
    LoweredProgram {
        functions: program
            .functions
            .iter()
            .map(optimize_lowered_function)
            .collect(),
    }
}

fn optimize_lowered_function(function: &LoweredFunction) -> LoweredFunction {
    let kind = match &function.kind {
        LoweredKind::Scalar { clauses } => LoweredKind::Scalar {
            clauses: clauses.iter().map(optimize_lowered_clause).collect(),
        },
        LoweredKind::Kernel {
            shape,
            vector_width,
            cleanup,
            clauses,
        } => LoweredKind::Kernel {
            shape: shape.clone(),
            vector_width: *vector_width,
            cleanup: *cleanup,
            clauses: clauses.iter().map(optimize_lowered_clause).collect(),
        },
    };
    let tail_loop = function.tail_loop.as_ref().map(optimize_tail_loop);
    LoweredFunction {
        name: function.name.clone(),
        param_access: function.param_access.clone(),
        result: function.result.clone(),
        kind,
        tail_loop,
    }
}

fn optimize_tail_loop(tail_loop: &TailLoop) -> TailLoop {
    TailLoop {
        clauses: tail_loop
            .clauses
            .iter()
            .map(|clause| TailLoopClause {
                patterns: clause.patterns.clone(),
                action: match &clause.action {
                    TailAction::Continue { args } => TailAction::Continue {
                        args: args.iter().map(optimize_ir_expr).collect(),
                    },
                    TailAction::Return { expr } => TailAction::Return {
                        expr: optimize_ir_expr(expr),
                    },
                },
            })
            .collect(),
    }
}

fn optimize_lowered_clause(clause: &LoweredClause) -> LoweredClause {
    LoweredClause {
        patterns: clause.patterns.clone(),
        body: optimize_ir_expr(&clause.body),
    }
}

fn optimize_ir_expr(expr: &IrExpr) -> IrExpr {
    let kind = match &expr.kind {
        IrExprKind::Local(name) => IrExprKind::Local(name.clone()),
        IrExprKind::Int(value, prim) => IrExprKind::Int(*value, *prim),
        IrExprKind::Float(value, prim) => IrExprKind::Float(*value, *prim),
        IrExprKind::Call { callee, args } => {
            let args = args.iter().map(optimize_ir_expr).collect::<Vec<_>>();
            fold_int_prim_call(expr.ty.clone(), callee.clone(), args)
        }
        IrExprKind::Let { bindings, body } => {
            let mut optimized_bindings = Vec::<IrLetBinding>::new();
            let mut alias = BTreeMap::<String, String>::new();
            for binding in bindings {
                let mut binding_expr = optimize_ir_expr(&binding.expr);
                rewrite_locals(&mut binding_expr, &alias);
                if let Some(existing) = optimized_bindings
                    .iter()
                    .find(|existing| existing.expr == binding_expr)
                    .map(|existing| existing.name.clone())
                {
                    alias.insert(binding.name.clone(), existing);
                    continue;
                }
                optimized_bindings.push(IrLetBinding {
                    name: binding.name.clone(),
                    expr: binding_expr,
                });
            }
            let mut optimized_body = optimize_ir_expr(body);
            rewrite_locals(&mut optimized_body, &alias);
            let pruned = prune_ir_let_bindings(optimized_bindings, &optimized_body);
            if pruned.is_empty() {
                return optimized_body;
            }
            IrExprKind::Let {
                bindings: pruned,
                body: Box::new(optimized_body),
            }
        }
    };
    IrExpr {
        ty: expr.ty.clone(),
        kind,
    }
}

fn fold_int_prim_call(ty: Type, callee: Callee, args: Vec<IrExpr>) -> IrExprKind {
    let Callee::Prim(op) = callee else {
        return IrExprKind::Call {
            callee: callee.clone(),
            args,
        };
    };
    if args.len() != 2 {
        return IrExprKind::Call {
            callee: Callee::Prim(op),
            args,
        };
    }
    let lhs_int = match args[0].kind {
        IrExprKind::Int(value, prim) if prim.is_int() => Some((value, prim)),
        _ => None,
    };
    let rhs_int = match args[1].kind {
        IrExprKind::Int(value, prim) if prim.is_int() => Some((value, prim)),
        _ => None,
    };
    if let (Some((lhs, lhs_prim)), Some((rhs, rhs_prim))) = (lhs_int, rhs_int)
        && lhs_prim == rhs_prim
    {
        let folded = match op {
            PrimOp::Add => lhs.checked_add(rhs),
            PrimOp::Sub => lhs.checked_sub(rhs),
            PrimOp::Mul => lhs.checked_mul(rhs),
            PrimOp::Div => {
                if rhs == 0 {
                    None
                } else {
                    lhs.checked_div(rhs)
                }
            }
            PrimOp::Mod => {
                if rhs == 0 {
                    None
                } else {
                    lhs.checked_rem(rhs)
                }
            }
            PrimOp::Eq => Some((lhs == rhs) as i64),
            PrimOp::Lt => Some((lhs < rhs) as i64),
            PrimOp::Gt => Some((lhs > rhs) as i64),
            PrimOp::Le => Some((lhs <= rhs) as i64),
            PrimOp::Ge => Some((lhs >= rhs) as i64),
        };
        if let Some(value) = folded {
            let result_prim = ty.prim().unwrap_or(lhs_prim);
            return IrExprKind::Int(value, result_prim);
        }
    }
    IrExprKind::Call {
        callee: Callee::Prim(op),
        args,
    }
}

fn rewrite_locals(expr: &mut IrExpr, alias: &BTreeMap<String, String>) {
    match &mut expr.kind {
        IrExprKind::Local(name) => {
            if let Some(target) = alias.get(name) {
                *name = target.clone();
            }
        }
        IrExprKind::Let { bindings, body } => {
            for binding in bindings {
                rewrite_locals(&mut binding.expr, alias);
            }
            rewrite_locals(body, alias);
        }
        IrExprKind::Call { args, .. } => {
            for arg in args {
                rewrite_locals(arg, alias);
            }
        }
        IrExprKind::Int(_, _) | IrExprKind::Float(_, _) => {}
    }
}

fn prune_ir_let_bindings(bindings: Vec<IrLetBinding>, body: &IrExpr) -> Vec<IrLetBinding> {
    let mut demanded = collect_ir_local_names(body);
    let mut kept = Vec::<IrLetBinding>::new();
    for binding in bindings.into_iter().rev() {
        if demanded.remove(&binding.name) {
            demanded.extend(collect_ir_local_names(&binding.expr));
            kept.push(binding);
        }
    }
    kept.reverse();
    kept
}

fn collect_ir_local_names(expr: &IrExpr) -> BTreeSet<String> {
    let mut names = BTreeSet::<String>::new();
    collect_ir_local_names_into(expr, &mut names);
    names
}

fn collect_ir_local_names_into(expr: &IrExpr, names: &mut BTreeSet<String>) {
    match &expr.kind {
        IrExprKind::Local(name) => {
            names.insert(name.clone());
        }
        IrExprKind::Let { bindings, body } => {
            for binding in bindings {
                collect_ir_local_names_into(&binding.expr, names);
            }
            collect_ir_local_names_into(body, names);
        }
        IrExprKind::Call { args, .. } => {
            for arg in args {
                collect_ir_local_names_into(arg, names);
            }
        }
        IrExprKind::Int(_, _) | IrExprKind::Float(_, _) => {}
    }
}

fn analyze_intents(grouped: &GroupedLoweredProgram) -> IntentAnalysis {
    IntentAnalysis {
        reports: grouped
            .functions
            .iter()
            .map(analyze_grouped_function_intent)
            .collect(),
    }
}

fn analyze_grouped_function_intent(function: &GroupedLoweredFunction) -> KernelIntentReport {
    let lane_inputs = function
        .param_access
        .iter()
        .filter(|access| **access == AccessKind::Lane)
        .count();
    let same_inputs = function
        .param_access
        .iter()
        .filter(|access| **access == AccessKind::Same)
        .count();
    let record_leaf_count = function.leaf_paths.len();
    let intent = match &function.kind {
        GroupedLoweredKind::Scalar { .. } => {
            if function.tail_loop.is_some() {
                IntentClass::ScalarTailRec
            } else {
                IntentClass::Fallback
            }
        }
        GroupedLoweredKind::Kernel { .. } if record_leaf_count > 1 => IntentClass::GroupedMap,
        GroupedLoweredKind::Kernel { .. } if lane_inputs == 1 && same_inputs == 0 => {
            IntentClass::MapUnary
        }
        GroupedLoweredKind::Kernel { .. } if lane_inputs == 1 && same_inputs == 1 => {
            IntentClass::MapBinaryBroadcast
        }
        GroupedLoweredKind::Kernel { .. } if lane_inputs == 2 && same_inputs == 1 => {
            IntentClass::MapTernaryBroadcast
        }
        GroupedLoweredKind::Kernel { .. } => IntentClass::Fallback,
    };
    let (rank, shape_size) = match &function.kind {
        GroupedLoweredKind::Kernel { shape, .. } => {
            let rank = shape.0.len();
            let shape_size = shape.0.iter().try_fold(1usize, |acc, dim| match dim {
                Dim::Const(value) => acc.checked_mul(*value),
                Dim::Var(_) => None,
            });
            (rank, shape_size)
        }
        GroupedLoweredKind::Scalar { .. } => (0, None),
    };
    let op_count = function
        .leaves
        .iter()
        .flat_map(|leaf| match &leaf.kind {
            LoweredKind::Scalar { clauses } | LoweredKind::Kernel { clauses, .. } => clauses,
        })
        .map(|clause| count_primitive_ops(&clause.body))
        .sum::<usize>();
    let primitive_width_bytes = function
        .result_leaves
        .first()
        .and_then(|leaf| leaf.ty.prim())
        .map(|prim| prim.byte_width())
        .unwrap_or(8);
    KernelIntentReport {
        source_name: function.source_name.clone(),
        leaf_paths: function.leaf_paths.clone(),
        intent,
        features: KernelFeatures {
            op_count,
            load_streams: lane_inputs,
            store_streams: record_leaf_count.max(1),
            scalar_broadcast_count: same_inputs,
            record_leaf_count,
            rank,
            shape_size,
            primitive_width_bytes,
        },
    }
}

fn count_primitive_ops(expr: &IrExpr) -> usize {
    match &expr.kind {
        IrExprKind::Call {
            callee: Callee::Prim(_),
            args,
        } => 1 + args.iter().map(count_primitive_ops).sum::<usize>(),
        IrExprKind::Call {
            callee: Callee::Function(_),
            args,
        } => args.iter().map(count_primitive_ops).sum(),
        IrExprKind::Let { bindings, body } => {
            bindings
                .iter()
                .map(|binding| count_primitive_ops(&binding.expr))
                .sum::<usize>()
                + count_primitive_ops(body)
        }
        IrExprKind::Local(_) | IrExprKind::Int(_, _) | IrExprKind::Float(_, _) => 0,
    }
}

fn lower_function(function: &NormalizedFunction) -> Result<LoweredFunction> {
    let (params, ret) = function.signature.ty.fun_parts();
    let param_access: Vec<AccessKind> = params
        .iter()
        .map(|ty| match ty {
            Type::Scalar(_) => AccessKind::Same,
            Type::Bulk(_, _) => AccessKind::Lane,
            Type::Record(_) => AccessKind::Same,
            Type::Var(_) | Type::Infer(_) => AccessKind::Same,
            Type::Fun(_, _) => AccessKind::Same,
        })
        .collect();

    let kind = match &ret {
        Type::Bulk(prim, shape) => {
            let clauses = function
                .clauses
                .iter()
                .map(|clause| lower_clause(clause, &param_access, Some(shape)))
                .collect::<Result<Vec<_>>>()?;
            LoweredKind::Kernel {
                shape: shape.clone(),
                vector_width: prim.lane_width(),
                cleanup: true,
                clauses,
            }
        }
        Type::Scalar(_) => {
            let clauses = function
                .clauses
                .iter()
                .map(|clause| lower_clause(clause, &param_access, None))
                .collect::<Result<Vec<_>>>()?;
            LoweredKind::Scalar { clauses }
        }
        Type::Record(_) => {
            return Err(SimdError::new(
                "normalized lowering encountered an unexpected record result",
            ));
        }
        Type::Var(_) | Type::Infer(_) => {
            return Err(SimdError::new(format!(
                "function '{}' still contains unresolved polymorphic types during lowering",
                function.name
            )));
        }
        Type::Fun(_, _) => {
            return Err(SimdError::new(format!(
                "function '{}' lowers to unsupported higher-order result",
                function.name
            )));
        }
    };

    let tail_loop = if function.tailrec.loop_lowerable {
        Some(lower_tail_loop(function)?)
    } else {
        None
    };

    Ok(LoweredFunction {
        name: function.name.clone(),
        param_access,
        result: ret,
        kind,
        tail_loop,
    })
}

fn lower_clause(
    clause: &TypedClause,
    param_access: &[AccessKind],
    kernel_shape: Option<&Shape>,
) -> Result<LoweredClause> {
    let mut lane_bindings = BTreeMap::<String, Type>::new();
    let mut patterns = Vec::new();
    for (typed_pattern, access) in clause.patterns.iter().zip(param_access.iter()) {
        let lowered_ty = match (access, &typed_pattern.ty) {
            (AccessKind::Lane, Type::Bulk(prim, _)) => Type::Scalar(*prim),
            _ => typed_pattern.ty.clone(),
        };
        if let Pattern::Name(name) = &typed_pattern.pattern {
            lane_bindings.insert(name.clone(), lowered_ty.clone());
        }
        patterns.push(TypedPattern {
            pattern: typed_pattern.pattern.clone(),
            ty: lowered_ty,
        });
    }
    let body = lower_expr_to_ir(&clause.body, kernel_shape, &lane_bindings)?;
    Ok(LoweredClause { patterns, body })
}

fn lower_tail_loop(function: &NormalizedFunction) -> Result<TailLoop> {
    let clauses = function
        .clauses
        .iter()
        .map(|clause| {
            let body = lower_expr_to_ir(&clause.body, None, &collect_scalar_locals(clause))?;
            let action = lower_tail_action(&body, &function.name);
            Ok(TailLoopClause {
                patterns: clause.patterns.clone(),
                action,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(TailLoop { clauses })
}

fn lower_tail_action(body: &IrExpr, self_name: &str) -> TailAction {
    match &body.kind {
        IrExprKind::Call {
            callee: Callee::Function(name),
            args,
        } if name == self_name => TailAction::Continue { args: args.clone() },
        IrExprKind::Let {
            bindings,
            body: inner,
        } => match lower_tail_action(inner, self_name) {
            TailAction::Continue { args } => TailAction::Continue {
                args: args
                    .into_iter()
                    .map(|arg| IrExpr {
                        ty: arg.ty.clone(),
                        kind: IrExprKind::Let {
                            bindings: bindings.clone(),
                            body: Box::new(arg),
                        },
                    })
                    .collect(),
            },
            TailAction::Return { expr: _ } => TailAction::Return { expr: body.clone() },
        },
        _ => TailAction::Return { expr: body.clone() },
    }
}

fn collect_scalar_locals(clause: &TypedClause) -> BTreeMap<String, Type> {
    let mut locals = BTreeMap::new();
    for pattern in &clause.patterns {
        if let Pattern::Name(name) = &pattern.pattern {
            locals.insert(name.clone(), pattern.ty.clone());
        }
    }
    locals
}

fn lower_expr_to_ir(
    expr: &TypedExpr,
    kernel_shape: Option<&Shape>,
    locals: &BTreeMap<String, Type>,
) -> Result<IrExpr> {
    match &expr.kind {
        TypedExprKind::Local(name) => {
            let lowered_ty = match (&expr.ty, kernel_shape) {
                (Type::Bulk(prim, shape), Some(kernel_shape)) if shape == kernel_shape => {
                    Type::Scalar(*prim)
                }
                _ => expr.ty.clone(),
            };
            if matches!(lowered_ty, Type::Bulk(_, _)) {
                return Err(SimdError::new(format!(
                    "cannot lower bulk local '{}' into scalar loop IR directly",
                    name
                )));
            }
            if !locals.contains_key(name) {
                return Err(SimdError::new(format!(
                    "unknown local '{}' during lowering",
                    name
                )));
            }
            Ok(IrExpr {
                ty: lowered_ty,
                kind: IrExprKind::Local(name.clone()),
            })
        }
        TypedExprKind::Int(value, prim) => Ok(IrExpr {
            ty: Type::Scalar(*prim),
            kind: IrExprKind::Int(*value, *prim),
        }),
        TypedExprKind::Float(value, prim) => Ok(IrExpr {
            ty: Type::Scalar(*prim),
            kind: IrExprKind::Float(*value, *prim),
        }),
        TypedExprKind::FunctionRef { .. }
        | TypedExprKind::Lambda { .. }
        | TypedExprKind::Apply { .. } => Err(SimdError::new(
            "higher-order expressions are not supported by loop lowering",
        )),
        TypedExprKind::Let { bindings, body } => {
            let mut lowered_bindings = Vec::new();
            let mut extended_locals = locals.clone();
            for binding in bindings {
                let expr = lower_expr_to_ir(&binding.expr, kernel_shape, &extended_locals)?;
                extended_locals.insert(binding.name.clone(), expr.ty.clone());
                lowered_bindings.push(IrLetBinding {
                    name: binding.name.clone(),
                    expr,
                });
            }
            let body = lower_expr_to_ir(body, kernel_shape, &extended_locals)?;
            Ok(IrExpr {
                ty: body.ty.clone(),
                kind: IrExprKind::Let {
                    bindings: lowered_bindings,
                    body: Box::new(body),
                },
            })
        }
        TypedExprKind::Record(_)
        | TypedExprKind::Project { .. }
        | TypedExprKind::RecordUpdate { .. } => Err(SimdError::new(
            "record expressions are not yet supported by loop lowering",
        )),
        TypedExprKind::Call {
            callee,
            args,
            lifted_shape,
        } => {
            if let Some(shape) = lifted_shape {
                if Some(shape) != kernel_shape {
                    return Err(SimdError::new(
                        "nested bulk kernels with different shapes are not supported in v0 lowering",
                    ));
                }
            }
            let mut lowered_args = Vec::new();
            for arg in args {
                lowered_args.push(lower_expr_to_ir(&arg.expr, kernel_shape, locals)?);
            }
            let lowered_ty = match (&expr.ty, kernel_shape) {
                (Type::Bulk(prim, shape), Some(kernel_shape)) if shape == kernel_shape => {
                    Type::Scalar(*prim)
                }
                _ => expr.ty.clone(),
            };
            Ok(IrExpr {
                ty: lowered_ty,
                kind: IrExprKind::Call {
                    callee: callee.clone(),
                    args: lowered_args,
                },
            })
        }
    }
}

#[derive(Clone)]
enum Binding<'a> {
    Ready(EvalValue<'a>),
    Thunk {
        expr: &'a TypedExpr,
        env: Rc<Env<'a>>,
    },
}

#[derive(Clone)]
enum EvalValue<'a> {
    Host(Value),
    Closure(Closure<'a>),
}

#[derive(Clone)]
enum Closure<'a> {
    Function {
        name: String,
        bound_args: Vec<Binding<'a>>,
    },
    Lambda {
        param: String,
        body: &'a TypedExpr,
        env: Rc<Env<'a>>,
    },
}

type Env<'a> = BTreeMap<String, Binding<'a>>;

pub fn run_main(source: &str, main: &str, args_json: &str) -> Result<Value> {
    let (_surface, _module, checked) = compile_frontend(source)?;
    let args = parse_host_args(args_json, &checked, main)?;
    Evaluator::new(&checked).run_function(main, args)
}

pub fn run_compiled_main(compiled: &CompiledProgram, main: &str, args: &[Value]) -> Result<Value> {
    Evaluator::new(&compiled.checked).run_function(main, args.to_vec())
}

fn parse_host_args(args_json: &str, checked: &CheckedProgram, main: &str) -> Result<Vec<Value>> {
    let function = checked
        .functions
        .iter()
        .find(|function| function.name == main)
        .ok_or_else(|| SimdError::new(format!("unknown entry function '{}'", main)))?;
    let (params, _) = function.signature.ty.fun_parts();
    let json = JsonParser::new(args_json).parse()?;
    let JsonValue::Array(items) = json else {
        return Err(SimdError::new("run --args expects a top-level JSON array"));
    };
    if items.len() != params.len() {
        return Err(SimdError::new(format!(
            "entry function '{}' expects {} arguments, found {} in JSON",
            main,
            params.len(),
            items.len()
        )));
    }
    let mut shape_env = BTreeMap::<String, usize>::new();
    let mut values = Vec::new();
    for (item, ty) in items.iter().zip(&params) {
        values.push(value_from_json(item, ty, &mut shape_env)?);
    }
    Ok(values)
}

struct Evaluator<'a> {
    functions: BTreeMap<String, &'a CheckedFunction>,
}

impl<'a> Evaluator<'a> {
    fn new(program: &'a CheckedProgram) -> Self {
        let functions = program
            .functions
            .iter()
            .map(|function| (function.name.clone(), function))
            .collect();
        Self { functions }
    }

    fn run_function(&self, name: &str, args: Vec<Value>) -> Result<Value> {
        let function = self
            .functions
            .get(name)
            .copied()
            .ok_or_else(|| SimdError::new(format!("unknown function '{}'", name)))?;
        if args.len() != function.signature.ty.arity() {
            return Err(SimdError::new(format!(
                "function '{}' expects {} arguments, found {}",
                name,
                function.signature.ty.arity(),
                args.len()
            )));
        }
        let bindings = args
            .into_iter()
            .map(|value| Binding::Ready(EvalValue::Host(value)))
            .collect();
        self.expect_host_value(self.call_function(function, bindings)?)
    }

    fn call_function(
        &self,
        function: &'a CheckedFunction,
        args: Vec<Binding<'a>>,
    ) -> Result<EvalValue<'a>> {
        for clause in &function.clauses {
            if let Some(env) = self.match_clause(clause, &args)? {
                let env = Rc::new(env);
                return self.eval_expr(&clause.body, &env);
            }
        }
        Err(SimdError::new(format!(
            "no clause of '{}' matched the provided arguments",
            function.name
        )))
    }

    fn match_clause(
        &self,
        clause: &'a TypedClause,
        args: &[Binding<'a>],
    ) -> Result<Option<Env<'a>>> {
        let mut env = BTreeMap::new();
        for (pattern, arg) in clause.patterns.iter().zip(args) {
            if !self.match_pattern(pattern, arg, &mut env)? {
                return Ok(None);
            }
        }
        Ok(Some(env))
    }

    fn match_pattern(
        &self,
        pattern: &TypedPattern,
        binding: &Binding<'a>,
        env: &mut Env<'a>,
    ) -> Result<bool> {
        match &pattern.pattern {
            Pattern::Wildcard => Ok(true),
            Pattern::Name(name) => {
                env.insert(name.clone(), binding.clone());
                Ok(true)
            }
            Pattern::Int(expected) => {
                let Value::Scalar(value) = self.expect_host_value(self.force_binding(binding)?)?
                else {
                    return Err(SimdError::new("literal pattern cannot match bulk input"));
                };
                Ok(matches_int_pattern(*expected, &value))
            }
            Pattern::Float(expected) => {
                let Value::Scalar(value) = self.expect_host_value(self.force_binding(binding)?)?
                else {
                    return Err(SimdError::new("literal pattern cannot match bulk input"));
                };
                Ok(matches_float_pattern(*expected, &value))
            }
        }
    }

    fn force_binding(&self, binding: &Binding<'a>) -> Result<EvalValue<'a>> {
        match binding {
            Binding::Ready(value) => Ok(value.clone()),
            Binding::Thunk { expr, env } => self.eval_expr(expr, env),
        }
    }

    fn expect_host_value(&self, value: EvalValue<'a>) -> Result<Value> {
        match value {
            EvalValue::Host(value) => Ok(value),
            EvalValue::Closure(_) => Err(SimdError::new(
                "function values cannot cross the host evaluation boundary",
            )),
        }
    }

    fn eval_expr(&self, expr: &'a TypedExpr, env: &Rc<Env<'a>>) -> Result<EvalValue<'a>> {
        match &expr.kind {
            TypedExprKind::Local(name) => {
                let binding = env
                    .get(name)
                    .ok_or_else(|| SimdError::new(format!("unknown local '{}'", name)))?;
                self.force_binding(binding)
            }
            TypedExprKind::FunctionRef { name } => Ok(EvalValue::Closure(Closure::Function {
                name: name.clone(),
                bound_args: Vec::new(),
            })),
            TypedExprKind::Int(value, prim) => Ok(EvalValue::Host(Value::Scalar(make_int_value(
                *value, *prim,
            )?))),
            TypedExprKind::Float(value, prim) => Ok(EvalValue::Host(Value::Scalar(
                make_float_value(*value, *prim),
            ))),
            TypedExprKind::Lambda { param, body } => Ok(EvalValue::Closure(Closure::Lambda {
                param: param.clone(),
                body,
                env: env.clone(),
            })),
            TypedExprKind::Let { bindings, body } => {
                let mut scope = (**env).clone();
                for binding in bindings {
                    if binding.name == "_" {
                        continue;
                    }
                    let snapshot = Rc::new(scope.clone());
                    scope.insert(
                        binding.name.clone(),
                        Binding::Thunk {
                            expr: &binding.expr,
                            env: snapshot,
                        },
                    );
                }
                self.eval_expr(body, &Rc::new(scope))
            }
            TypedExprKind::Record(fields) => Ok(EvalValue::Host(Value::Record(
                fields
                    .iter()
                    .map(|(name, expr)| {
                        Ok((
                            name.clone(),
                            self.expect_host_value(self.eval_expr(expr, env)?)?,
                        ))
                    })
                    .collect::<Result<BTreeMap<_, _>>>()?,
            ))),
            TypedExprKind::Project { base, field } => {
                let Value::Record(fields) = self.expect_host_value(self.eval_expr(base, env)?)?
                else {
                    return Err(SimdError::new(
                        "record projection evaluated a non-record base",
                    ));
                };
                fields
                    .get(field)
                    .cloned()
                    .map(EvalValue::Host)
                    .ok_or_else(|| {
                        SimdError::new(format!("record field '{}' does not exist", field))
                    })
            }
            TypedExprKind::RecordUpdate { base, fields } => {
                let Value::Record(mut base_fields) =
                    self.expect_host_value(self.eval_expr(base, env)?)?
                else {
                    return Err(SimdError::new("record update evaluated a non-record base"));
                };
                for (name, expr) in fields {
                    base_fields.insert(
                        name.clone(),
                        self.expect_host_value(self.eval_expr(expr, env)?)?,
                    );
                }
                Ok(EvalValue::Host(Value::Record(base_fields)))
            }
            TypedExprKind::Apply { callee, arg } => {
                let closure = match self.eval_expr(callee, env)? {
                    EvalValue::Closure(closure) => closure,
                    EvalValue::Host(value) => {
                        return Err(SimdError::new(format!(
                            "cannot apply non-function runtime value {:?}",
                            value
                        )));
                    }
                };
                self.apply_closure(
                    closure,
                    Binding::Thunk {
                        expr: arg,
                        env: env.clone(),
                    },
                )
            }
            TypedExprKind::Call {
                callee,
                args,
                lifted_shape,
            } => match lifted_shape {
                Some(_) => self.eval_lifted_call(callee, args, env),
                None => self.eval_scalar_call(callee, args, env),
            },
        }
    }

    fn apply_closure(&self, closure: Closure<'a>, arg: Binding<'a>) -> Result<EvalValue<'a>> {
        match closure {
            Closure::Function {
                name,
                mut bound_args,
            } => {
                bound_args.push(arg);
                let function = self
                    .functions
                    .get(&name)
                    .copied()
                    .ok_or_else(|| SimdError::new(format!("unknown function '{}'", name)))?;
                if bound_args.len() == function.signature.ty.arity() {
                    self.call_function(function, bound_args)
                } else {
                    Ok(EvalValue::Closure(Closure::Function { name, bound_args }))
                }
            }
            Closure::Lambda { param, body, env } => {
                let mut scope = (*env).clone();
                scope.insert(param, arg);
                self.eval_expr(body, &Rc::new(scope))
            }
        }
    }

    fn eval_scalar_call(
        &self,
        callee: &Callee,
        args: &'a [TypedArg],
        env: &Rc<Env<'a>>,
    ) -> Result<EvalValue<'a>> {
        match callee {
            Callee::Prim(op) => {
                let mut values = Vec::new();
                for arg in args {
                    let value = self.expect_host_value(self.eval_expr(&arg.expr, env)?)?;
                    let Value::Scalar(value) = value else {
                        return Err(SimdError::new("primitive scalar call received bulk input"));
                    };
                    values.push(value);
                }
                Ok(EvalValue::Host(Value::Scalar(apply_primitive(
                    *op, &values[0], &values[1],
                )?)))
            }
            Callee::Function(name) => {
                let function = self
                    .functions
                    .get(name)
                    .copied()
                    .ok_or_else(|| SimdError::new(format!("unknown function '{}'", name)))?;
                let bindings = args
                    .iter()
                    .map(|arg| Binding::Thunk {
                        expr: &arg.expr,
                        env: env.clone(),
                    })
                    .collect();
                self.call_function(function, bindings)
            }
        }
    }

    fn eval_lifted_call(
        &self,
        callee: &Callee,
        args: &'a [TypedArg],
        env: &Rc<Env<'a>>,
    ) -> Result<EvalValue<'a>> {
        let mut same_args = Vec::<Value>::new();
        let mut lane_args = Vec::<Value>::new();
        for arg in args {
            let value = self.expect_host_value(self.eval_expr(&arg.expr, env)?)?;
            match (arg.mode, value_lift_shape(&value), value) {
                (AccessKind::Same, None, value) => same_args.push(value),
                (AccessKind::Lane, Some(_), value) => lane_args.push(value),
                (AccessKind::Same, Some(_), _) => {
                    return Err(SimdError::new(
                        "broadcast argument unexpectedly evaluated to bulk",
                    ));
                }
                (AccessKind::Lane, None, _) => {
                    return Err(SimdError::new(
                        "lane argument unexpectedly evaluated to scalar",
                    ));
                }
            }
        }
        let shape = lane_args
            .first()
            .and_then(value_lift_shape)
            .ok_or_else(|| SimdError::new("lifted call requires at least one bulk argument"))?;
        for bulk in &lane_args {
            if value_lift_shape(bulk) != Some(shape.clone()) {
                return Err(SimdError::new(
                    "lifted call received incompatible runtime bulk shapes",
                ));
            }
        }

        let scalar_result_ty = match callee {
            Callee::Prim(op) => {
                let lane_prim = value_leaf_prim(&lane_args[0]).ok_or_else(|| {
                    SimdError::new("lifted primitive arguments did not contain primitive leaves")
                })?;
                Type::Scalar(primitive_result_prim(*op, lane_prim))
            }
            Callee::Function(name) => {
                let function = self
                    .functions
                    .get(name)
                    .copied()
                    .ok_or_else(|| SimdError::new(format!("unknown function '{}'", name)))?;
                let (_, ret) = function.signature.ty.fun_parts();
                ret
            }
        };

        let len = shape.iter().product::<usize>();
        let mut lane_results = Vec::with_capacity(len);
        for offset in 0..len {
            let mut scalar_args = Vec::new();
            let mut same_index = 0usize;
            let mut lane_index = 0usize;
            for arg in args {
                match arg.mode {
                    AccessKind::Same => {
                        scalar_args.push(Binding::Ready(EvalValue::Host(
                            same_args[same_index].clone(),
                        )));
                        same_index += 1;
                    }
                    AccessKind::Lane => {
                        scalar_args.push(Binding::Ready(EvalValue::Host(extract_lifted_lane(
                            &lane_args[lane_index],
                            offset,
                        )?)));
                        lane_index += 1;
                    }
                }
            }
            let value = match callee {
                Callee::Prim(op) => {
                    let scalars = scalar_args
                        .into_iter()
                        .map(|binding| match binding {
                            Binding::Ready(EvalValue::Host(Value::Scalar(value))) => value,
                            _ => unreachable!(),
                        })
                        .collect::<Vec<_>>();
                    Value::Scalar(apply_primitive(*op, &scalars[0], &scalars[1])?)
                }
                Callee::Function(name) => {
                    let function =
                        self.functions.get(name).copied().ok_or_else(|| {
                            SimdError::new(format!("unknown function '{}'", name))
                        })?;
                    self.expect_host_value(self.call_function(function, scalar_args)?)?
                }
            };
            lane_results.push(value);
        }

        Ok(EvalValue::Host(collect_lifted_value(
            &lane_results,
            &scalar_result_ty,
            &shape,
        )?))
    }
}

fn matches_int_pattern(expected: i64, actual: &ScalarValue) -> bool {
    match actual {
        ScalarValue::I32(value) => i64::from(*value) == expected,
        ScalarValue::I64(value) => *value == expected,
        ScalarValue::F32(_) | ScalarValue::F64(_) => false,
    }
}

fn matches_float_pattern(expected: f64, actual: &ScalarValue) -> bool {
    match actual {
        ScalarValue::F32(value) => (*value as f64).to_bits() == expected.to_bits(),
        ScalarValue::F64(value) => value.to_bits() == expected.to_bits(),
        ScalarValue::I32(_) | ScalarValue::I64(_) => false,
    }
}

fn make_int_value(value: i64, prim: Prim) -> Result<ScalarValue> {
    match prim {
        Prim::I32 => Ok(ScalarValue::I32(i32::try_from(value).map_err(|_| {
            SimdError::new(format!("integer literal '{}' does not fit in i32", value))
        })?)),
        Prim::I64 => Ok(ScalarValue::I64(value)),
        Prim::F32 | Prim::F64 => Err(SimdError::new("integer literal cannot inhabit float type")),
    }
}

fn make_float_value(value: f64, prim: Prim) -> ScalarValue {
    match prim {
        Prim::F32 => ScalarValue::F32(value as f32),
        Prim::F64 => ScalarValue::F64(value),
        Prim::I32 | Prim::I64 => unreachable!(),
    }
}

fn primitive_result_prim(op: PrimOp, prim: Prim) -> Prim {
    match op {
        PrimOp::Eq | PrimOp::Lt | PrimOp::Gt | PrimOp::Le | PrimOp::Ge => Prim::I64,
        _ => prim,
    }
}

fn apply_primitive(op: PrimOp, left: &ScalarValue, right: &ScalarValue) -> Result<ScalarValue> {
    match (left, right) {
        (ScalarValue::I32(left), ScalarValue::I32(right)) => {
            apply_int_primitive_i64(op, i64::from(*left), i64::from(*right))
                .and_then(|value| make_int_value(value, primitive_result_prim(op, Prim::I32)))
        }
        (ScalarValue::I64(left), ScalarValue::I64(right)) => {
            apply_int_primitive_i64(op, *left, *right)
                .and_then(|value| make_int_value(value, primitive_result_prim(op, Prim::I64)))
        }
        (ScalarValue::F32(left), ScalarValue::F32(right)) => {
            apply_float_primitive(op, f64::from(*left), f64::from(*right)).map(|value| {
                match primitive_result_prim(op, Prim::F32) {
                    Prim::F32 => ScalarValue::F32(value as f32),
                    Prim::I64 => ScalarValue::I64(value as i64),
                    _ => unreachable!(),
                }
            })
        }
        (ScalarValue::F64(left), ScalarValue::F64(right)) => {
            apply_float_primitive(op, *left, *right).map(|value| {
                match primitive_result_prim(op, Prim::F64) {
                    Prim::F64 => ScalarValue::F64(value),
                    Prim::I64 => ScalarValue::I64(value as i64),
                    _ => unreachable!(),
                }
            })
        }
        _ => Err(SimdError::new(format!(
            "primitive {:?} received mismatched scalar operands {:?} and {:?}",
            op, left, right
        ))),
    }
}

fn apply_int_primitive_i64(op: PrimOp, left: i64, right: i64) -> Result<i64> {
    let value = match op {
        PrimOp::Add => left + right,
        PrimOp::Sub => left - right,
        PrimOp::Mul => left * right,
        PrimOp::Div => left / right,
        PrimOp::Mod => left % right,
        PrimOp::Eq => return Ok(if left == right { 1 } else { 0 }),
        PrimOp::Lt => return Ok(if left < right { 1 } else { 0 }),
        PrimOp::Gt => return Ok(if left > right { 1 } else { 0 }),
        PrimOp::Le => return Ok(if left <= right { 1 } else { 0 }),
        PrimOp::Ge => return Ok(if left >= right { 1 } else { 0 }),
    };
    Ok(value)
}

fn apply_float_primitive(op: PrimOp, left: f64, right: f64) -> Result<f64> {
    let value = match op {
        PrimOp::Add => left + right,
        PrimOp::Sub => left - right,
        PrimOp::Mul => left * right,
        PrimOp::Div => left / right,
        PrimOp::Mod => {
            return Err(SimdError::new(
                "modulo is not defined for floats in v0 primitive semantics",
            ));
        }
        PrimOp::Eq => {
            return Ok(if left.to_bits() == right.to_bits() {
                1.0
            } else {
                0.0
            });
        }
        PrimOp::Lt => return Ok(if left < right { 1.0 } else { 0.0 }),
        PrimOp::Gt => return Ok(if left > right { 1.0 } else { 0.0 }),
        PrimOp::Le => return Ok(if left <= right { 1.0 } else { 0.0 }),
        PrimOp::Ge => return Ok(if left >= right { 1.0 } else { 0.0 }),
    };
    Ok(value)
}

pub fn run_lowered_main(source: &str, main: &str, args_json: &str) -> Result<Value> {
    let compiled = compile_source(source)?;
    let args = parse_host_args(args_json, &compiled.checked, main)?;
    run_compiled_lowered_main(&compiled, main, &args)
}

pub fn run_compiled_lowered_main(
    compiled: &CompiledProgram,
    main: &str,
    args: &[Value],
) -> Result<Value> {
    let entry = compiled
        .normalized
        .entries
        .iter()
        .find(|entry| entry.source_name == main)
        .ok_or_else(|| SimdError::new(format!("unknown entry function '{}'", main)))?;
    let (param_types, result_ty) = entry.source_signature.ty.fun_parts();
    if args.len() != param_types.len() {
        return Err(SimdError::new(format!(
            "entry function '{}' expects {} arguments, found {}",
            main,
            param_types.len(),
            args.len()
        )));
    }

    let mut flattened_args = Vec::new();
    for ((arg, param_ty), leaves) in args.iter().zip(&param_types).zip(&entry.param_leaves) {
        let flat = flatten_value_leaves(arg, param_ty)?;
        for leaf in leaves {
            flattened_args.push(leaf_value_at(&flat, &leaf.path).cloned().ok_or_else(|| {
                SimdError::new(format!(
                    "entry argument is missing normalized leaf {:?}",
                    leaf.path
                ))
            })?);
        }
    }

    let evaluator = LoweredEvaluator::new(&compiled.lowered);
    let mut result_leaves = BTreeMap::new();
    for result_leaf in &entry.result_leaves {
        let leaf_function = entry.leaf_functions.get(&result_leaf.path).ok_or_else(|| {
            SimdError::new(format!("missing lowered leaf function for '{}'", main))
        })?;
        let value = evaluator.run_function(leaf_function, flattened_args.clone())?;
        result_leaves.insert(result_leaf.path.clone(), value);
    }
    rebuild_value_from_leaves(&result_ty, &result_leaves)
}

struct LoweredEvaluator<'a> {
    functions: BTreeMap<String, &'a LoweredFunction>,
}

impl<'a> LoweredEvaluator<'a> {
    fn new(program: &'a LoweredProgram) -> Self {
        let functions = program
            .functions
            .iter()
            .map(|function| (function.name.clone(), function))
            .collect();
        Self { functions }
    }

    fn run_function(&self, name: &str, args: Vec<Value>) -> Result<Value> {
        let function = self
            .functions
            .get(name)
            .copied()
            .ok_or_else(|| SimdError::new(format!("unknown lowered function '{}'", name)))?;
        match &function.kind {
            LoweredKind::Scalar { clauses } => {
                let scalar_args = args
                    .into_iter()
                    .map(|value| match value {
                        Value::Scalar(value) => Ok(value),
                        Value::Bulk(_) => Err(SimdError::new(format!(
                            "scalar lowered function '{}' received bulk input",
                            name
                        ))),
                        Value::Record(_) => Err(SimdError::new(format!(
                            "scalar lowered function '{}' received record input",
                            name
                        ))),
                    })
                    .collect::<Result<Vec<_>>>()?;
                self.run_scalar(function, clauses, scalar_args)
            }
            LoweredKind::Kernel {
                shape: _,
                vector_width: _,
                cleanup: _,
                clauses,
            } => self.run_kernel(function, clauses, args),
        }
    }

    fn run_scalar(
        &self,
        function: &LoweredFunction,
        clauses: &[LoweredClause],
        args: Vec<ScalarValue>,
    ) -> Result<Value> {
        if let Some(tail_loop) = &function.tail_loop {
            return self.run_tail_loop(tail_loop, args);
        }
        for clause in clauses {
            if let Some(env) = self.match_lowered_clause(&clause.patterns, &args)? {
                return Ok(Value::Scalar(self.eval_ir_expr(&clause.body, &env)?));
            }
        }
        Err(SimdError::new(format!(
            "no lowered clause of '{}' matched the provided arguments",
            function.name
        )))
    }

    fn run_tail_loop(&self, tail_loop: &TailLoop, mut state: Vec<ScalarValue>) -> Result<Value> {
        loop {
            let mut advanced = false;
            for clause in &tail_loop.clauses {
                if let Some(env) = self.match_lowered_clause(&clause.patterns, &state)? {
                    match &clause.action {
                        TailAction::Continue { args } => {
                            state = args
                                .iter()
                                .map(|arg| self.eval_ir_expr(arg, &env))
                                .collect::<Result<Vec<_>>>()?;
                            advanced = true;
                            break;
                        }
                        TailAction::Return { expr } => {
                            return Ok(Value::Scalar(self.eval_ir_expr(expr, &env)?));
                        }
                    }
                }
            }
            if !advanced {
                return Err(SimdError::new("tail loop found no matching clause"));
            }
        }
    }

    fn run_kernel(
        &self,
        function: &LoweredFunction,
        clauses: &[LoweredClause],
        args: Vec<Value>,
    ) -> Result<Value> {
        let lane_shape = args
            .iter()
            .zip(&function.param_access)
            .find_map(|(value, access)| match (value, access) {
                (Value::Bulk(bulk), AccessKind::Lane) => Some(bulk.shape.clone()),
                _ => None,
            })
            .ok_or_else(|| SimdError::new("kernel requires at least one lane input"))?;
        let prim = match &function.result {
            Type::Bulk(prim, _) => *prim,
            _ => return Err(SimdError::new("kernel result must be bulk")),
        };
        let len = lane_shape.iter().product::<usize>();
        let mut elements = Vec::with_capacity(len);
        for index in 0..len {
            let mut lane_args = Vec::new();
            for (value, access) in args.iter().zip(&function.param_access) {
                match (value, access) {
                    (Value::Scalar(value), AccessKind::Same) => lane_args.push(value.clone()),
                    (Value::Bulk(bulk), AccessKind::Lane) => lane_args.push(bulk.scalar_at(index)),
                    (Value::Scalar(_), AccessKind::Lane) => {
                        return Err(SimdError::new(
                            "kernel lane parameter received scalar input",
                        ));
                    }
                    (Value::Bulk(_), AccessKind::Same) => {
                        return Err(SimdError::new("kernel same parameter received bulk input"));
                    }
                    (Value::Record(_), _) => {
                        return Err(SimdError::new(
                            "kernel lowering does not yet support record runtime values",
                        ));
                    }
                }
            }
            let value = self
                .run_scalar(function, clauses, lane_args)?
                .expect_scalar()?;
            elements.push(value);
        }
        Ok(Value::Bulk(BulkValue {
            prim,
            shape: lane_shape,
            elements,
        }))
    }

    fn match_lowered_clause(
        &self,
        patterns: &[TypedPattern],
        args: &[ScalarValue],
    ) -> Result<Option<BTreeMap<String, ScalarValue>>> {
        let mut env = BTreeMap::new();
        for (pattern, value) in patterns.iter().zip(args) {
            match &pattern.pattern {
                Pattern::Wildcard => {}
                Pattern::Name(name) => {
                    env.insert(name.clone(), value.clone());
                }
                Pattern::Int(expected) if matches_int_pattern(*expected, value) => {}
                Pattern::Float(expected) if matches_float_pattern(*expected, value) => {}
                Pattern::Int(_) | Pattern::Float(_) => return Ok(None),
            }
        }
        Ok(Some(env))
    }

    fn eval_ir_expr(
        &self,
        expr: &IrExpr,
        env: &BTreeMap<String, ScalarValue>,
    ) -> Result<ScalarValue> {
        match &expr.kind {
            IrExprKind::Local(name) => env
                .get(name)
                .cloned()
                .ok_or_else(|| SimdError::new(format!("unknown lowered local '{}'", name))),
            IrExprKind::Int(value, prim) => make_int_value(*value, *prim),
            IrExprKind::Float(value, prim) => Ok(make_float_value(*value, *prim)),
            IrExprKind::Let { bindings, body } => {
                let mut scope = env.clone();
                for binding in bindings {
                    let value = self.eval_ir_expr(&binding.expr, &scope)?;
                    scope.insert(binding.name.clone(), value);
                }
                self.eval_ir_expr(body, &scope)
            }
            IrExprKind::Call { callee, args } => {
                let values = args
                    .iter()
                    .map(|arg| self.eval_ir_expr(arg, env))
                    .collect::<Result<Vec<_>>>()?;
                match callee {
                    Callee::Prim(op) => apply_primitive(*op, &values[0], &values[1]),
                    Callee::Function(name) => self
                        .run_function(
                            name,
                            values.into_iter().map(Value::Scalar).collect::<Vec<_>>(),
                        )?
                        .expect_scalar(),
                }
            }
        }
    }
}

trait ValueExt {
    fn expect_scalar(self) -> Result<ScalarValue>;
}

impl ValueExt for Value {
    fn expect_scalar(self) -> Result<ScalarValue> {
        match self {
            Value::Scalar(value) => Ok(value),
            Value::Bulk(_) => Err(SimdError::new("expected scalar value, found bulk")),
            Value::Record(_) => Err(SimdError::new("expected scalar value, found record")),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
enum JsonValue {
    Number(String),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

struct JsonParser<'a> {
    chars: Vec<char>,
    pos: usize,
    source: &'a str,
}

impl<'a> JsonParser<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            chars: source.chars().collect(),
            pos: 0,
            source,
        }
    }

    fn parse(mut self) -> Result<JsonValue> {
        let value = self.parse_value()?;
        self.skip_ws();
        if self.pos != self.chars.len() {
            return Err(SimdError::new(format!(
                "unexpected trailing JSON content near '{}'",
                &self.source[self.pos..]
            )));
        }
        Ok(value)
    }

    fn parse_value(&mut self) -> Result<JsonValue> {
        self.skip_ws();
        let Some(ch) = self.chars.get(self.pos).copied() else {
            return Err(SimdError::new("unexpected end of JSON input"));
        };
        match ch {
            '[' => self.parse_array(),
            '{' => self.parse_object(),
            '"' => self.parse_string().map(JsonValue::String),
            '-' | '0'..='9' => self.parse_number(),
            _ => Err(SimdError::new(format!(
                "unsupported JSON token '{}' in run arguments",
                ch
            ))),
        }
    }

    fn parse_array(&mut self) -> Result<JsonValue> {
        self.pos += 1;
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            if self.chars.get(self.pos) == Some(&']') {
                self.pos += 1;
                break;
            }
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.chars.get(self.pos) {
                Some(',') => {
                    self.pos += 1;
                }
                Some(']') => {
                    self.pos += 1;
                    break;
                }
                Some(ch) => {
                    return Err(SimdError::new(format!(
                        "expected ',' or ']', found '{}'",
                        ch
                    )));
                }
                None => return Err(SimdError::new("unterminated JSON array")),
            }
        }
        Ok(JsonValue::Array(items))
    }

    fn parse_object(&mut self) -> Result<JsonValue> {
        self.pos += 1;
        let mut fields = BTreeMap::new();
        loop {
            self.skip_ws();
            if self.chars.get(self.pos) == Some(&'}') {
                self.pos += 1;
                break;
            }
            let key = self.parse_string()?;
            self.skip_ws();
            if self.chars.get(self.pos) != Some(&':') {
                return Err(SimdError::new("expected ':' after JSON object key"));
            }
            self.pos += 1;
            let value = self.parse_value()?;
            fields.insert(key, value);
            self.skip_ws();
            match self.chars.get(self.pos) {
                Some(',') => self.pos += 1,
                Some('}') => {
                    self.pos += 1;
                    break;
                }
                Some(ch) => {
                    return Err(SimdError::new(format!(
                        "expected ',' or '}}', found '{}'",
                        ch
                    )));
                }
                None => return Err(SimdError::new("unterminated JSON object")),
            }
        }
        Ok(JsonValue::Object(fields))
    }

    fn parse_string(&mut self) -> Result<String> {
        if self.chars.get(self.pos) != Some(&'"') {
            return Err(SimdError::new("expected JSON string"));
        }
        self.pos += 1;
        let start = self.pos;
        while let Some(ch) = self.chars.get(self.pos) {
            match ch {
                '"' => {
                    let text: String = self.chars[start..self.pos].iter().collect();
                    self.pos += 1;
                    return Ok(text);
                }
                '\\' => {
                    return Err(SimdError::new(
                        "JSON string escapes are not supported in run arguments",
                    ));
                }
                _ => self.pos += 1,
            }
        }
        Err(SimdError::new("unterminated JSON string"))
    }

    fn parse_number(&mut self) -> Result<JsonValue> {
        let start = self.pos;
        if self.chars.get(self.pos) == Some(&'-') {
            self.pos += 1;
        }
        while self
            .chars
            .get(self.pos)
            .is_some_and(|ch| ch.is_ascii_digit())
        {
            self.pos += 1;
        }
        if self.chars.get(self.pos) == Some(&'.') {
            self.pos += 1;
            while self
                .chars
                .get(self.pos)
                .is_some_and(|ch| ch.is_ascii_digit())
            {
                self.pos += 1;
            }
        }
        Ok(JsonValue::Number(
            self.chars[start..self.pos].iter().collect(),
        ))
    }

    fn skip_ws(&mut self) {
        while self
            .chars
            .get(self.pos)
            .is_some_and(|ch| ch.is_whitespace())
        {
            self.pos += 1;
        }
    }
}

fn value_from_json(
    json: &JsonValue,
    ty: &Type,
    shape_env: &mut BTreeMap<String, usize>,
) -> Result<Value> {
    match ty {
        Type::Scalar(prim) => Ok(Value::Scalar(scalar_from_json_number(json, *prim)?)),
        Type::Bulk(prim, expected_shape) => {
            let (shape, elements) = collect_bulk_values(json, *prim)?;
            if shape.len() != expected_shape.0.len() {
                return Err(SimdError::new(format!(
                    "bulk rank mismatch: runtime shape {:?}, expected {:?}",
                    shape, expected_shape
                )));
            }
            for (runtime_dim, expected_dim) in shape.iter().zip(&expected_shape.0) {
                match expected_dim {
                    Dim::Const(expected) if runtime_dim != expected => {
                        return Err(SimdError::new(format!(
                            "runtime shape {:?} does not match expected constant dimension {}",
                            shape, expected
                        )));
                    }
                    Dim::Const(_) => {}
                    Dim::Var(name) => match shape_env.get(name) {
                        Some(bound) if bound == runtime_dim => {}
                        Some(bound) => {
                            return Err(SimdError::new(format!(
                                "shape variable '{}' was bound to {}, cannot also be {}",
                                name, bound, runtime_dim
                            )));
                        }
                        None => {
                            shape_env.insert(name.clone(), *runtime_dim);
                        }
                    },
                }
            }
            Ok(Value::Bulk(BulkValue {
                prim: *prim,
                shape,
                elements,
            }))
        }
        Type::Record(fields) => {
            let values = match json {
                JsonValue::Object(values) => values.clone(),
                JsonValue::Array(_) => fields
                    .keys()
                    .map(|name| Ok((name.clone(), project_record_field_json(json, name)?)))
                    .collect::<Result<BTreeMap<_, _>>>()?,
                _ => {
                    return Err(SimdError::new(
                        "record runtime argument must be a JSON object or array of record objects",
                    ));
                }
            };
            let mut record = BTreeMap::new();
            for (name, ty) in fields {
                let value = values.get(name).ok_or_else(|| {
                    SimdError::new(format!(
                        "record runtime argument is missing field '{}'",
                        name
                    ))
                })?;
                record.insert(name.clone(), value_from_json(value, ty, shape_env)?);
            }
            if values.len() != fields.len() || values.keys().any(|name| !fields.contains_key(name))
            {
                return Err(SimdError::new(
                    "record runtime argument has extra fields not present in the type",
                ));
            }
            Ok(Value::Record(record))
        }
        Type::Var(_) | Type::Infer(_) => Err(SimdError::new(
            "host JSON cannot provide values for unresolved polymorphic types",
        )),
        Type::Fun(_, _) => Err(SimdError::new("host JSON cannot provide function values")),
    }
}

fn scalar_from_json_number(json: &JsonValue, prim: Prim) -> Result<ScalarValue> {
    let JsonValue::Number(number) = json else {
        return Err(SimdError::new(
            "scalar runtime argument must be a JSON number",
        ));
    };
    if prim.is_int() {
        if number.contains('.') {
            return Err(SimdError::new(format!(
                "integer parameter received non-integer JSON number '{}'",
                number
            )));
        }
        return make_int_value(parse_int(number)?, prim);
    }
    Ok(make_float_value(parse_float(number)?, prim))
}

fn project_record_field_json(json: &JsonValue, field: &str) -> Result<JsonValue> {
    match json {
        JsonValue::Object(values) => values.get(field).cloned().ok_or_else(|| {
            SimdError::new(format!("record JSON object is missing field '{}'", field))
        }),
        JsonValue::Array(items) => Ok(JsonValue::Array(
            items
                .iter()
                .map(|item| project_record_field_json(item, field))
                .collect::<Result<Vec<_>>>()?,
        )),
        JsonValue::Number(_) | JsonValue::String(_) => Err(SimdError::new(format!(
            "record JSON field projection expected object/array structure for field '{}'",
            field
        ))),
    }
}

fn collect_bulk_values(json: &JsonValue, prim: Prim) -> Result<(Vec<usize>, Vec<ScalarValue>)> {
    match json {
        JsonValue::Array(items) => {
            let mut all_shapes = Vec::<Vec<usize>>::new();
            let mut elements = Vec::<ScalarValue>::new();
            for item in items {
                let (child_shape, child_elements) = collect_bulk_values(item, prim)?;
                all_shapes.push(child_shape);
                elements.extend(child_elements);
            }
            if let Some(first) = all_shapes.first() {
                if all_shapes.iter().any(|shape| shape != first) {
                    return Err(SimdError::new(
                        "ragged JSON arrays are not valid bulk values",
                    ));
                }
                let mut shape = vec![items.len()];
                shape.extend(first.clone());
                Ok((shape, elements))
            } else {
                Ok((vec![0], elements))
            }
        }
        JsonValue::Number(_) => Ok((Vec::new(), vec![scalar_from_json_number(json, prim)?])),
        JsonValue::String(_) | JsonValue::Object(_) => Err(SimdError::new(
            "bulk runtime argument must be a JSON array or number",
        )),
    }
}

pub fn parse_command(path: &str) -> Result<String> {
    let source = read_source_file(path)?;
    let program = parse_source(&source)?;
    Ok(format!("{:#?}", program))
}

pub fn check_command(path: &str) -> Result<String> {
    let source = read_source_file(path)?;
    let (_surface, _module, checked) = compile_frontend(&source)?;
    match normalize_records(&checked).and_then(|normalized| {
        lower_program(&normalized)
            .map(|lowered| optimize_lowered_program(&lowered))
            .map(|lowered| {
                let grouped = group_lowered_program(&normalized, &lowered).ok();
                let intents = grouped.as_ref().map(analyze_intents);
                (normalized, lowered, grouped, intents)
            })
    }) {
        Ok((normalized, lowered, grouped, intents)) => Ok(format!(
            "Checked Program\n{:#?}\n\nNormalized Program\n{:#?}\n\nLowered Program\n{:#?}\n\nGrouped Program\n{:#?}\n\nIntent Analysis\n{:#?}",
            checked,
            normalized,
            lowered,
            grouped.unwrap_or(GroupedLoweredProgram { functions: vec![] }),
            intents.unwrap_or(IntentAnalysis { reports: vec![] }),
        )),
        Err(error) => Ok(format!(
            "Checked Program\n{:#?}\n\nNormalized Program / Lowered Program\n<unavailable: {}>",
            checked, error
        )),
    }
}

pub fn run_command(path: &str, main: &str, args_json: &str) -> Result<String> {
    let source = read_source_file(path)?;
    let value = run_main(&source, main, args_json)?;
    Ok(value.to_json_string())
}

pub fn inspect_html_command(out: Option<&str>) -> Result<String> {
    let html = bundled_examples_inspector_html()?;
    let out_path = out.unwrap_or("docs/inspector.html");
    if let Some(parent) = std::path::Path::new(out_path).parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|error| {
            SimdError::new(format!(
                "failed to create directory '{}': {}",
                parent.display(),
                error
            ))
        })?;
    }
    fs::write(out_path, html)
        .map_err(|error| SimdError::new(format!("failed to write '{}': {}", out_path, error)))?;
    Ok(format!("wrote {}", out_path))
}

fn bundled_examples_inspector_html() -> Result<String> {
    let presets = [
        (
            "inc-i64",
            "inc-i64",
            "examples/inc_i64.simd",
            "main",
            include_str!("../examples/inc_i64.simd"),
        ),
        (
            "square-f32",
            "square-f32",
            "examples/square_f32.simd",
            "main",
            include_str!("../examples/square_f32.simd"),
        ),
        (
            "axpy-i64",
            "axpy-i64",
            "examples/axpy_i64.simd",
            "main",
            include_str!("../examples/axpy_i64.simd"),
        ),
        (
            "axpy2-record-i64",
            "axpy2-record-i64",
            "examples/axpy2_record_i64.simd",
            "main",
            include_str!("../examples/axpy2_record_i64.simd"),
        ),
        (
            "pow2-i64",
            "pow2-i64",
            "examples/pow2_i64.simd",
            "main",
            include_str!("../examples/pow2_i64.simd"),
        ),
        (
            "mouse-glow-f32",
            "mouse-glow-f32",
            "examples/mouse_glow_f32.simd",
            "main",
            include_str!("../examples/mouse_glow_f32.simd"),
        ),
        (
            "mouse-rings-f32",
            "mouse-rings-f32",
            "examples/mouse_rings_f32.simd",
            "main",
            include_str!("../examples/mouse_rings_f32.simd"),
        ),
    ];

    let mut preset_json = String::from("[\n");
    for (index, (id, label, file, main, source)) in presets.iter().enumerate() {
        let wat = wat_main(source, main)?;
        if index > 0 {
            preset_json.push_str(",\n");
        }
        preset_json.push_str("  {");
        preset_json.push_str(&format!(
            "\"id\":{},\"label\":{},\"file\":{},\"main\":{},\"source\":{},\"wat\":{}",
            json_string(id),
            json_string(label),
            json_string(file),
            json_string(main),
            json_string(source),
            json_string(&wat),
        ));
        preset_json.push('}');
    }
    preset_json.push_str("\n]");

    Ok(format!(
        r#"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>simd wat inspector</title>
    <style>
      :root {{
        color-scheme: dark;
        --bg: #0b1020;
        --panel: #11182d;
        --panel-alt: #162038;
        --border: #27324d;
        --text: #e5edf7;
        --muted: #96a4bf;
        --accent: #7dd3fc;
      }}

      * {{
        box-sizing: border-box;
      }}

      body {{
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: var(--text);
        background: var(--bg);
      }}

      .app {{
        min-height: 100vh;
        display: grid;
        grid-template-rows: auto 1fr;
      }}

      .toolbar {{
        display: flex;
        gap: 12px;
        align-items: center;
        padding: 12px 16px;
        border-bottom: 1px solid var(--border);
        background: #0f172b;
      }}

      .toolbar strong {{
        font-size: 14px;
        color: var(--text);
      }}

      .toolbar label,
      .meta {{
        color: var(--muted);
        font-size: 13px;
      }}

      select {{
        min-width: 240px;
        padding: 8px 10px;
        border: 1px solid var(--border);
        background: var(--panel);
        color: var(--text);
        font: inherit;
        border-radius: 8px;
        outline: none;
      }}

      select:focus {{
        border-color: var(--accent);
        box-shadow: 0 0 0 2px rgba(125, 211, 252, 0.18);
      }}

      .panes {{
        display: grid;
        grid-template-columns: 1fr 1fr;
        min-height: 0;
      }}

      .pane {{
        min-width: 0;
        display: grid;
        grid-template-rows: auto 1fr;
      }}

      .pane + .pane {{
        border-left: 1px solid var(--border);
      }}

      .pane-header {{
        display: flex;
        justify-content: space-between;
        gap: 12px;
        padding: 10px 14px;
        border-bottom: 1px solid var(--border);
        background: var(--panel-alt);
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--muted);
      }}

      pre {{
        margin: 0;
        padding: 16px;
        overflow: auto;
        white-space: pre;
        font-size: 13px;
        line-height: 1.5;
        font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
        background: var(--panel);
        color: var(--text);
      }}

      @media (max-width: 900px) {{
        .panes {{
          grid-template-columns: 1fr;
        }}

        .pane + .pane {{
          border-left: 0;
          border-top: 1px solid var(--border);
        }}
      }}
    </style>
  </head>
  <body>
    <main class="app">
      <section class="toolbar">
        <strong>simd wat inspector</strong>
        <label for="preset">preset</label>
        <select id="preset"></select>
        <span id="summary" class="meta"></span>
      </section>
      <section class="panes">
        <article class="pane">
          <div class="pane-header">
            <span>Source</span>
            <span id="source-meta"></span>
          </div>
          <pre id="source"></pre>
        </article>
        <article class="pane">
          <div class="pane-header">
            <span>WAT</span>
            <span id="wat-meta"></span>
          </div>
          <pre id="wat"></pre>
        </article>
      </section>
    </main>
    <script>
      const presets = {preset_json};
      const select = document.getElementById("preset");
      const source = document.getElementById("source");
      const wat = document.getElementById("wat");
      const summary = document.getElementById("summary");
      const sourceMeta = document.getElementById("source-meta");
      const watMeta = document.getElementById("wat-meta");

      for (const preset of presets) {{
        const option = document.createElement("option");
        option.value = preset.id;
        option.textContent = preset.label;
        select.appendChild(option);
      }}

      function render(id) {{
        const preset = presets.find((item) => item.id === id) ?? presets[0];
        select.value = preset.id;
        source.textContent = preset.source;
        wat.textContent = preset.wat;
        summary.textContent = `${{preset.file}} • main=${{preset.main}}`;
        sourceMeta.textContent = preset.file;
        watMeta.textContent = `${{preset.wat.split("\n").length}} lines`;
      }}

      select.addEventListener("change", (event) => render(event.target.value));
      render(presets[0]?.id);
    </script>
  </body>
</html>
"#
    ))
}

fn json_string(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '<' => out.push_str("\\u003c"),
            '>' => out.push_str("\\u003e"),
            '&' => out.push_str("\\u0026"),
            c if c.is_control() => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn compile(src: &str) -> CompiledProgram {
        compile_source(src).expect("program should compile")
    }

    fn run(src: &str, main: &str, args_json: &str) -> String {
        run_main(src, main, args_json)
            .expect("program should run")
            .to_json_string()
    }

    #[test]
    fn parses_application_tighter_than_infix() {
        let program = parse_source(
            "f : i64 -> i64 -> i64\ng : i64 -> i64\nmain : i64 -> i64\nmain x = f (g x) x\n",
        )
        .unwrap();
        assert_eq!(program.decls.len(), 4);
    }

    #[test]
    fn parses_let_expression() {
        let program =
            parse_source("main : i64 -> i64\nmain x = let y = x + 1; z = y * 2 in z\n").unwrap();
        assert_eq!(program.decls.len(), 2);
    }

    #[test]
    fn parses_import_and_qualified_call() {
        let program = parse_source(
            "import math/scalar as scalar\naxpy : i64 -> i64 -> i64 -> i64\naxpy a x y = a * x + y\nmain : i64 -> i64[n] -> i64[n] -> i64[n]\nmain a xs ys = scalar\\axpy a xs ys\n",
        )
        .unwrap();
        assert_eq!(program.decls.len(), 5);
        assert!(matches!(program.decls[0], Decl::Import(_)));
    }

    #[test]
    fn parses_lambda_and_specialized_refs() {
        let program = parse_source(
            "main : i64 -> i64\nmain x = apply (id\\i64) (\\y -> y) x\napply : (i64 -> i64) -> i64 -> i64\napply f x = f x\nid : t -> t\nid y = y\n",
        )
        .unwrap();
        assert_eq!(program.decls.len(), 6);
    }

    #[test]
    fn rejects_import_after_function_declaration() {
        let error = parse_source("main : i64 -> i64\nmain x = x\nimport math/scalar as scalar\n")
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("imports must appear before function signatures and clauses")
        );
    }

    #[test]
    fn rejects_clause_arity_mismatch() {
        let error = compile_source("f : i64 -> i64\nf x = x\nf x y = y\n").unwrap_err();
        assert!(error.to_string().contains("arity"));
    }

    #[test]
    fn rejects_incompatible_lifted_shapes() {
        let error = compile_source(
            "axpy : i64 -> i64 -> i64 -> i64\naxpy a x y = a * x + y\nmain : i64 -> i64[n] -> i64[m] -> i64[n]\nmain a xs ys = axpy a xs ys\n",
        )
        .unwrap_err();
        assert!(error.to_string().contains("incompatible shapes"));
    }

    #[test]
    fn comparison_returns_i64() {
        let compiled = compile("lt : i64 -> i64 -> i64\nlt x y = x < y\n");
        let function = compiled
            .checked
            .functions
            .iter()
            .find(|function| function.name == "lt")
            .unwrap();
        let (_, ret) = function.signature.ty.fun_parts();
        assert_eq!(ret, Type::Scalar(Prim::I64));
    }

    #[test]
    fn evaluates_bulk_inc() {
        let src = "inc : i64 -> i64\ninc x = x + 1\nmain : i64[n] -> i64[n]\nmain xs = inc xs\n";
        assert_eq!(run(src, "main", "[[1,2,3]]"), "[2,3,4]");
    }

    #[test]
    fn evaluates_broadcast_axpy() {
        let src = "axpy : i64 -> i64 -> i64 -> i64\naxpy a x y = a * x + y\nmain : i64 -> i64[n] -> i64[n] -> i64[n]\nmain a xs ys = axpy a xs ys\n";
        assert_eq!(run(src, "main", "[2,[1,2,3],[10,20,30]]"), "[12,24,36]");
    }

    #[test]
    fn evaluates_qualified_alias_call() {
        let src = "import math/scalar as scalar\naxpy : i64 -> i64 -> i64 -> i64\naxpy a x y = a * x + y\nmain : i64 -> i64[n] -> i64[n] -> i64[n]\nmain a xs ys = scalar\\axpy a xs ys\n";
        assert_eq!(run(src, "main", "[2,[1,2,3],[10,20,30]]"), "[12,24,36]");
    }

    #[test]
    fn spaced_slash_stays_division_even_with_import_alias() {
        let src =
            "import math/scalar as scalar\nmain : i64 -> i64 -> i64\nmain scalar x = scalar / x\n";
        assert_eq!(run(src, "main", "[8,2]"), "4");
    }

    #[test]
    fn evaluates_clause_selection_per_lane() {
        let src = "pred : i64 -> i64\npred 0 = 0\npred n = n - 1\nmain : i64[n] -> i64[n]\nmain xs = pred xs\n";
        assert_eq!(run(src, "main", "[[0,1,4]]"), "[0,0,3]");
    }

    #[test]
    fn wildcard_does_not_force_demand() {
        let src = "trap : i64 -> i64\ntrap 0 = 0\nleft : i64 -> i64 -> i64\nleft x _ = x\nmain : i64 -> i64\nmain x = left x (trap 1)\n";
        assert_eq!(run(src, "main", "[7]"), "7");
    }

    #[test]
    fn let_bindings_are_dependency_sorted() {
        let src = "main : i64 -> i64\nmain x = let y = z + 1; z = x * 2 in y\n";
        assert_eq!(run(src, "main", "[7]"), "15");
    }

    #[test]
    fn cyclic_let_bindings_are_rejected() {
        let error =
            compile_source("main : i64 -> i64\nmain x = let y = z; z = y in y\n").unwrap_err();
        assert!(error.to_string().contains("cyclic let binding"));
    }

    #[test]
    fn rejects_unspecialized_polymorphic_value_ref() {
        let error = run_main(
            "id : t -> t\nid x = x\nmain : i64 -> i64\nmain x = let f = id in f x\n",
            "main",
            "[7]",
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("requires explicit specialization in value position")
        );
    }

    #[test]
    fn let_wildcard_has_zero_demand() {
        let src =
            "trap : i64 -> i64\ntrap 0 = 0\nmain : i64 -> i64\nmain x = let _ = trap 1 in x\n";
        assert_eq!(run(src, "main", "[7]"), "7");
    }

    #[test]
    fn evaluates_lambda_through_higher_order_argument() {
        let src = "apply : (i64 -> i64) -> i64 -> i64\napply f x = f x\nmain : i64 -> i64\nmain x = apply (\\y -> y + 1) x\n";
        assert_eq!(run(src, "main", "[7]"), "8");
    }

    #[test]
    fn evaluates_partial_application_of_top_level_function() {
        let src = "add : i64 -> i64 -> i64\nadd x y = x + y\napply : (i64 -> i64) -> i64 -> i64\napply f x = f x\nmain : i64 -> i64\nmain x = let inc = add 1 in apply inc x\n";
        assert_eq!(run(src, "main", "[7]"), "8");
    }

    #[test]
    fn evaluates_inferred_top_level_polymorphism() {
        let src = "square : t -> t\nsquare x = x * x\nmain : i64 -> i64\nmain x = square x\n";
        assert_eq!(run(src, "main", "[7]"), "49");
    }

    #[test]
    fn evaluates_explicit_specialization_in_value_position() {
        let src = "id : t -> t\nid x = x\nmain : i64 -> i64\nmain x = let f = id\\i64 in f x\n";
        assert_eq!(run(src, "main", "[7]"), "7");
    }

    #[test]
    fn evaluates_record_projection_and_update() {
        let src = "bump : {x:i64,y:i64} -> {x:i64,y:i64}\nbump p = p { x = p.x + 1 }\nmain : {x:i64,y:i64} -> {x:i64,y:i64}\nmain p = bump p\n";
        assert_eq!(run(src, "main", "[{\"x\":1,\"y\":2}]"), "{\"x\":2,\"y\":2}");
    }

    #[test]
    fn evaluates_lifted_record_fields() {
        let src = "bump : {x:i64,y:i64} -> {x:i64,y:i64}\nbump p = p { x = p.x + 1, y = p.y + 2 }\nmain : {x:i64,y:i64}[n] -> {x:i64,y:i64}[n]\nmain ps = bump ps\n";
        assert_eq!(
            run(src, "main", "[{\"x\":[1,2],\"y\":[10,20]}]"),
            "[{\"x\":2,\"y\":12},{\"x\":3,\"y\":22}]"
        );
    }

    #[test]
    fn accepts_aos_bulk_record_json_input() {
        let src = "bump : {x:i64,y:i64} -> {x:i64,y:i64}\nbump p = p { x = p.x + 1, y = p.y + 2 }\nmain : {x:i64,y:i64}[n] -> {x:i64,y:i64}[n]\nmain ps = bump ps\n";
        assert_eq!(
            run(src, "main", "[[{\"x\":1,\"y\":10},{\"x\":2,\"y\":20}]]"),
            "[{\"x\":2,\"y\":12},{\"x\":3,\"y\":22}]"
        );
    }

    #[test]
    fn normalizes_record_function_into_leaf_functions() {
        let compiled =
            compile("bump : {x:i64,y:i64} -> {x:i64,y:i64}\nbump p = p { x = p.x + 1 }\n");
        let names = compiled
            .normalized
            .functions
            .iter()
            .map(|function| function.name.clone())
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["bump$x".to_string(), "bump$y".to_string()]);
    }

    #[test]
    fn groups_compatible_record_kernel_leaves() {
        let compiled = compile(
            "main : {x:i64,y:i64}[n] -> {x:i64,y:i64}[n]\nmain p = p { x = p.x + 1, y = p.y + 2 }\n",
        );
        let group = compiled
            .grouped
            .functions
            .iter()
            .find(|group| group.source_name == "main")
            .expect("grouped lowering should contain main");
        assert_eq!(
            group.leaf_paths,
            vec![
                LeafPath(vec!["x".to_string()]),
                LeafPath(vec!["y".to_string()])
            ]
        );
        assert!(matches!(group.kind, GroupedLoweredKind::Kernel { .. }));
        assert_eq!(group.result_leaves.len(), 2);
    }

    #[test]
    fn splits_mixed_primitive_record_leaves_into_separate_groups() {
        let compiled = compile(
            "main : {x:i64,y:f32}[n] -> {x:i64,y:f32}[n]\nmain p = p { x = p.x + 1, y = p.y + 1.0 }\n",
        );
        let groups = compiled
            .grouped
            .functions
            .iter()
            .filter(|group| group.source_name == "main")
            .collect::<Vec<_>>();
        assert_eq!(groups.len(), 2);
        assert!(
            groups
                .iter()
                .any(|group| group.leaf_paths == vec![LeafPath(vec!["x".to_string()])])
        );
        assert!(
            groups
                .iter()
                .any(|group| group.leaf_paths == vec![LeafPath(vec!["y".to_string()])])
        );
    }

    #[test]
    fn lowered_record_execution_matches_checked_execution() {
        let src = "bump : {x:i64,y:i64} -> {x:i64,y:i64}\nbump p = p { x = p.x + 1, y = p.y + 2 }\nmain : {x:i64,y:i64}[n] -> {x:i64,y:i64}[n]\nmain ps = bump ps\n";
        let checked = run_main(src, "main", "[[{\"x\":1,\"y\":10},{\"x\":2,\"y\":20}]]")
            .unwrap()
            .to_json_string();
        let lowered = run_lowered_main(src, "main", "[[{\"x\":1,\"y\":10},{\"x\":2,\"y\":20}]]")
            .unwrap()
            .to_json_string();
        assert_eq!(checked, lowered);
    }

    #[test]
    fn tail_recursive_function_lowers_to_loop() {
        let compiled =
            compile("pow2 : i64 -> i64 -> i64\npow2 0 x = x\npow2 n x = pow2 (n - 1) (x * 2)\n");
        let function = compiled
            .lowered
            .functions
            .iter()
            .find(|function| function.name == "pow2")
            .unwrap();
        assert!(function.tail_loop.is_some());
    }

    #[test]
    fn non_tail_recursive_function_is_not_loop_lowered() {
        let compiled = compile("bad : i64 -> i64\nbad 0 = 0\nbad n = 1 + bad (n - 1)\n");
        let function = compiled
            .lowered
            .functions
            .iter()
            .find(|function| function.name == "bad")
            .unwrap();
        assert!(function.tail_loop.is_none());
    }

    #[test]
    fn lowered_and_checked_execution_agree() {
        let src = "pow2 : i64 -> i64 -> i64\npow2 0 x = x\npow2 n x = pow2 (n - 1) (x * 2)\nmain : i64[n] -> i64[n]\nmain xs = pow2 3 xs\n";
        let checked = run_main(src, "main", "[[1,2,3]]").unwrap().to_json_string();
        let lowered = run_lowered_main(src, "main", "[[1,2,3]]")
            .unwrap()
            .to_json_string();
        assert_eq!(checked, lowered);
    }

    #[test]
    fn lowered_and_checked_execution_agree_with_record_let() {
        let src = "bump : {x:i64,y:i64} -> {x:i64,y:i64}\nbump p = let q = p { x = p.x + 1 }; r = q { y = q.y + 2 } in r\nmain : {x:i64,y:i64}[n] -> {x:i64,y:i64}[n]\nmain ps = bump ps\n";
        let checked = run_main(src, "main", "[[{\"x\":1,\"y\":10},{\"x\":2,\"y\":20}]]")
            .unwrap()
            .to_json_string();
        let lowered = run_lowered_main(src, "main", "[[{\"x\":1,\"y\":10},{\"x\":2,\"y\":20}]]")
            .unwrap()
            .to_json_string();
        assert_eq!(checked, lowered);
    }

    #[test]
    fn intent_analysis_classifies_broadcast_kernel() {
        let compiled = compile(
            "axpy : i64 -> i64 -> i64 -> i64\naxpy a x y = a * x + y\nmain : i64 -> i64[n] -> i64[n] -> i64[n]\nmain a xs ys = axpy a xs ys\n",
        );
        let report = compiled
            .intents
            .reports
            .iter()
            .find(|report| report.source_name == "main")
            .expect("intent report for main");
        assert_eq!(report.intent, IntentClass::MapTernaryBroadcast);
    }

    #[test]
    fn intent_analysis_classifies_scalar_tailrec() {
        let compiled =
            compile("pow2 : i64 -> i64 -> i64\npow2 0 x = x\npow2 n x = pow2 (n - 1) (x * 2)\n");
        let report = compiled
            .intents
            .reports
            .iter()
            .find(|report| report.source_name == "pow2")
            .expect("intent report for pow2");
        assert_eq!(report.intent, IntentClass::ScalarTailRec);
    }

    #[test]
    fn inspector_html_embeds_examples_and_wat() {
        let html = bundled_examples_inspector_html().unwrap();
        assert!(html.contains("simd wat inspector"));
        assert!(html.contains("axpy2-record-i64"));
        assert!(html.contains("mouse-glow-f32"));
        assert!(html.contains("mouse-rings-f32"));
        assert!(html.contains("(module"));
        assert!(html.contains("color-scheme: dark"));
        assert!(html.contains("--bg: #0b1020"));
        assert!(!html.contains("color-scheme: light"));
    }

    #[test]
    fn prelude_example_compiles() {
        let source = include_str!("../examples/prelude.simd");
        let compiled = compile_source(source).expect("prelude should compile");
        assert!(!compiled.checked.functions.is_empty());
    }

    #[test]
    fn prelude_step_behaves_for_natural_numbers() {
        let prelude = include_str!("../examples/prelude.simd");
        let source = format!("{prelude}\nmain : i64 -> i64 -> i64\nmain edge x = step edge x\n");
        assert_eq!(
            run_main(&source, "main", "[3,2]")
                .expect("step should run")
                .to_json_string(),
            "0"
        );
        assert_eq!(
            run_main(&source, "main", "[3,3]")
                .expect("step should run")
                .to_json_string(),
            "1"
        );
        assert_eq!(
            run_main(&source, "main", "[3,9]")
                .expect("step should run")
                .to_json_string(),
            "1"
        );
    }
}
