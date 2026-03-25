use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use tokio::sync::RwLock;
use tower_lsp::jsonrpc::Result as LspResult;
use tower_lsp::lsp_types::{
    Diagnostic, DiagnosticSeverity, DidChangeTextDocumentParams, DidOpenTextDocumentParams, Hover,
    HoverContents, HoverParams, InitializeParams, InitializeResult, InlayHint, InlayHintKind,
    InlayHintLabel, InlayHintParams, MarkupContent, MarkupKind, MessageType, Position, Range,
    ServerCapabilities, TextDocumentContentChangeEvent, TextDocumentSyncCapability,
    TextDocumentSyncKind, TextEdit, Url,
};
use tower_lsp::{Client, LanguageServer, LspService, Server};

use crate::formatter::format_type_for_display;
use crate::{
    Pattern, Prim, PrimOp, Result, SimdError, Type, TypedExpr, TypedExprKind, compile_frontend,
    format_source_text, parse_source,
};

#[derive(Default)]
struct State {
    documents: HashMap<Url, String>,
}

struct Backend {
    client: Client,
    state: Arc<RwLock<State>>,
}

impl Backend {
    fn new(client: Client) -> Self {
        Self {
            client,
            state: Arc::new(RwLock::new(State::default())),
        }
    }

    async fn publish_diagnostics(&self, uri: Url, source: &str) {
        let diagnostics = diagnostics_for_source(source);
        self.client
            .publish_diagnostics(uri, diagnostics, None)
            .await;
    }
}

#[tower_lsp::async_trait]
impl LanguageServer for Backend {
    async fn initialize(&self, _: InitializeParams) -> LspResult<InitializeResult> {
        Ok(InitializeResult {
            capabilities: ServerCapabilities {
                text_document_sync: Some(TextDocumentSyncCapability::Kind(
                    TextDocumentSyncKind::FULL,
                )),
                hover_provider: Some(tower_lsp::lsp_types::HoverProviderCapability::Simple(true)),
                document_formatting_provider: Some(tower_lsp::lsp_types::OneOf::Left(true)),
                inlay_hint_provider: Some(tower_lsp::lsp_types::OneOf::Left(true)),
                ..ServerCapabilities::default()
            },
            server_info: Some(tower_lsp::lsp_types::ServerInfo {
                name: "simd".to_string(),
                version: Some(env!("CARGO_PKG_VERSION").to_string()),
            }),
        })
    }

    async fn initialized(&self, _: tower_lsp::lsp_types::InitializedParams) {
        self.client
            .log_message(
                MessageType::INFO,
                "simd language server initialized (tower-lsp)",
            )
            .await;
    }

    async fn shutdown(&self) -> LspResult<()> {
        Ok(())
    }

    async fn did_open(&self, params: DidOpenTextDocumentParams) {
        let uri = params.text_document.uri;
        let text = params.text_document.text;
        {
            let mut state = self.state.write().await;
            state.documents.insert(uri.clone(), text.clone());
        }
        self.publish_diagnostics(uri, &text).await;
    }

    async fn did_change(&self, params: DidChangeTextDocumentParams) {
        let uri = params.text_document.uri;
        let Some(text) = latest_change_text(&params.content_changes) else {
            return;
        };
        {
            let mut state = self.state.write().await;
            state.documents.insert(uri.clone(), text.clone());
        }
        self.publish_diagnostics(uri, &text).await;
        let _ = self.client.inlay_hint_refresh().await;
    }

    async fn formatting(
        &self,
        params: tower_lsp::lsp_types::DocumentFormattingParams,
    ) -> LspResult<Option<Vec<TextEdit>>> {
        let uri = params.text_document.uri;
        let source = {
            let state = self.state.read().await;
            state.documents.get(&uri).cloned()
        };
        let Some(source) = source else {
            return Ok(None);
        };
        let formatted = match format_source_text(&source) {
            Ok(formatted) => formatted,
            Err(error) => {
                self.client
                    .log_message(MessageType::ERROR, format!("format failed: {}", error))
                    .await;
                return Ok(None);
            }
        };
        {
            // Keep server-side document state in sync immediately so a
            // near-immediate inlay-hint request uses the formatted text.
            let mut state = self.state.write().await;
            state.documents.insert(uri.clone(), formatted.clone());
        }
        self.publish_diagnostics(uri.clone(), &formatted).await;
        let _ = self.client.inlay_hint_refresh().await;
        let edit = TextEdit {
            range: full_document_range(&source),
            new_text: formatted,
        };
        Ok(Some(vec![edit]))
    }

    async fn hover(&self, params: HoverParams) -> LspResult<Option<Hover>> {
        let uri = params.text_document_position_params.text_document.uri;
        let position = params.text_document_position_params.position;
        let source = {
            let state = self.state.read().await;
            state.documents.get(&uri).cloned()
        };
        let Some(source) = source else {
            return Ok(None);
        };
        Ok(hover_for_position(&source, position))
    }

    async fn inlay_hint(&self, params: InlayHintParams) -> LspResult<Option<Vec<InlayHint>>> {
        let uri = params.text_document.uri;
        let source = {
            let state = self.state.read().await;
            state.documents.get(&uri).cloned()
        };
        let Some(source) = source else {
            return Ok(Some(Vec::new()));
        };
        Ok(Some(inlay_hints_for_source(&source, Some(params.range))))
    }
}

pub fn lsp_command() -> Result<()> {
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .map_err(|error| SimdError::new(format!("failed to start tokio runtime: {}", error)))?;
    runtime.block_on(async {
        let stdin = tokio::io::stdin();
        let stdout = tokio::io::stdout();
        let (service, socket) = LspService::new(Backend::new);
        Server::new(stdin, stdout, socket).serve(service).await;
    });
    Ok(())
}

fn latest_change_text(changes: &[TextDocumentContentChangeEvent]) -> Option<String> {
    changes.last().map(|change| change.text.clone())
}

fn diagnostics_for_source(source: &str) -> Vec<Diagnostic> {
    if let Err(error) = parse_source(source) {
        return vec![basic_error_diagnostic(format!("syntax error: {}", error))];
    }
    if let Err(error) = compile_frontend(source) {
        return vec![basic_error_diagnostic(error.to_string())];
    }
    Vec::new()
}

fn basic_error_diagnostic(message: String) -> Diagnostic {
    Diagnostic {
        range: Range {
            start: Position::new(0, 0),
            end: Position::new(0, 1),
        },
        severity: Some(DiagnosticSeverity::ERROR),
        source: Some("simd".to_string()),
        message,
        ..Diagnostic::default()
    }
}

fn hover_for_position(source: &str, position: Position) -> Option<Hover> {
    let ident = identifier_at_position(source, position)?;
    if let Some(hover) = hover_for_witness_token(source, &ident) {
        return Some(hover);
    }
    let (_surface, module, checked) = compile_frontend(source).ok()?;

    if let Some(alias) = module
        .type_aliases
        .iter()
        .find(|alias| alias.name == ident.name)
    {
        let params = if alias.params.is_empty() {
            String::new()
        } else {
            format!(" {}", alias.params.join(" "))
        };
        let rendered = format!(
            "```simd\ntype {}{} = {}\n```",
            alias.name,
            params,
            format_type_for_display(&alias.body)
        );
        return Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: rendered,
            }),
            range: Some(ident.range),
        });
    }

    if let Some(enum_decl) = module.enums.iter().find(|enum_decl| enum_decl.name == ident.name) {
        let params = if enum_decl.params.is_empty() {
            String::new()
        } else {
            format!(" {}", enum_decl.params.join(" "))
        };
        let rendered = format!("```simd\nenum {}{}\n```", enum_decl.name, params);
        return Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: rendered,
            }),
            range: Some(ident.range),
        });
    }

    if let Some((enum_decl, ctor)) = module.enums.iter().find_map(|enum_decl| {
        enum_decl
            .ctors
            .iter()
            .find(|ctor| ctor.name == ident.name)
            .map(|ctor| (enum_decl, ctor))
    }) {
        let enum_result = if enum_decl.params.is_empty() {
            enum_decl.name.clone()
        } else {
            format!("{} {}", enum_decl.name, enum_decl.params.join(" "))
        };
        let mut parts = ctor
            .fields
            .iter()
            .map(format_type_for_display)
            .collect::<Vec<_>>();
        parts.push(enum_result);
        let rendered = format!("```simd\n{} : {}\n```", ctor.name, parts.join(" -> "));
        return Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: rendered,
            }),
            range: Some(ident.range),
        });
    }

    if let Some(family) = module
        .families
        .iter()
        .find(|family| family.name == ident.name)
    {
        let head = match family.op {
            Some(op) => {
                let params = if family.params.is_empty() {
                    String::new()
                } else {
                    family
                        .params
                        .iter()
                        .map(|param| format!("\\{}", param))
                        .collect::<String>()
                };
                format!("({}){}", format_prim_op(op), params)
            }
            None => {
                let params = if family.params.is_empty() {
                    String::new()
                } else {
                    family
                        .params
                        .iter()
                        .map(|param| format!("\\{}", param))
                        .collect::<String>()
                };
                format!("{}{}", family.name, params)
            }
        };
        return Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: format!(
                    "```simd\nfamily {} : {}\n```",
                    head,
                    format_type_for_display(&family.ty)
                ),
            }),
            range: Some(ident.range),
        });
    }

    if let Some(function) = module
        .functions
        .iter()
        .find(|function| function.name == ident.name)
    {
        let rendered = format!(
            "```simd\n{} : {}\n```",
            function.name,
            format_type_for_display(&function.signature.ty)
        );
        return Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: rendered,
            }),
            range: Some(ident.range),
        });
    }

    let context_function = function_name_at_line(source, ident.line)?;
    let function = checked
        .functions
        .iter()
        .find(|function| function.name == context_function)?;
    let constructor_names = checked.enum_ctors.keys().cloned().collect::<BTreeSet<_>>();
    let local_types = collect_local_types(function, &constructor_names);
    let candidates = local_types.get(&ident.name)?;
    if candidates.is_empty() {
        return None;
    }
    let mut rendered_types = candidates
        .iter()
        .map(format_type_for_display)
        .collect::<Vec<_>>();
    rendered_types.sort();
    rendered_types.dedup();
    let signature = if rendered_types.len() == 1 {
        format!("{} : {}", ident.name, rendered_types[0])
    } else {
        format!("{} : {}", ident.name, rendered_types.join(" | "))
    };
    Some(Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: format!("```simd\n{}\n```", signature),
        }),
        range: Some(ident.range),
    })
}

fn hover_for_witness_token(source: &str, ident: &IdentifierAtPosition) -> Option<Hover> {
    if ident.name == "Type" {
        return Some(markup_hover(
            ident.range.clone(),
            "type witness keyword".to_string(),
        ));
    }
    if Prim::parse(&ident.name).is_some() && witness_token_context(source, ident) {
        return Some(markup_hover(
            ident.range.clone(),
            format!("primitive witness `{}`", ident.name),
        ));
    }
    None
}

fn witness_token_context(source: &str, ident: &IdentifierAtPosition) -> bool {
    let Some(line) = source.lines().nth(ident.line) else {
        return false;
    };
    let Some(start) = utf16_col_to_byte_col(line, ident.range.start.character as usize) else {
        return false;
    };
    let prefix = &line[..start];
    prefix.contains(':') || prefix.contains('\\') || prefix.contains("Type")
}

fn markup_hover(range: Range, contents: String) -> Hover {
    Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: format!("```simd\n{}\n```", contents),
        }),
        range: Some(range),
    }
}

#[derive(Debug, Clone)]
struct IdentifierAtPosition {
    name: String,
    range: Range,
    line: usize,
}

fn identifier_at_position(source: &str, position: Position) -> Option<IdentifierAtPosition> {
    let line_index = usize::try_from(position.line).ok()?;
    let line = source.lines().nth(line_index)?;
    let utf16_col = usize::try_from(position.character).ok()?;
    let byte_col = utf16_col_to_byte_col(line, utf16_col)?;

    let mut spans = Vec::<(usize, usize)>::new();
    let bytes = line.as_bytes();
    let mut index = 0usize;
    while index < bytes.len() {
        if is_ident_char(bytes[index]) {
            let start = index;
            index += 1;
            while index < bytes.len() && is_ident_char(bytes[index]) {
                index += 1;
            }
            spans.push((start, index));
        } else {
            index += 1;
        }
    }

    let (start, end) = spans
        .into_iter()
        .find(|(start, end)| *start <= byte_col && byte_col <= *end)?;
    let name = line[start..end].to_string();
    let start_col = byte_col_to_utf16_col(line, start)?;
    let end_col = byte_col_to_utf16_col(line, end)?;
    Some(IdentifierAtPosition {
        name,
        range: Range {
            start: Position::new(position.line, start_col),
            end: Position::new(position.line, end_col),
        },
        line: line_index,
    })
}

fn is_ident_char(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_'
}

fn utf16_col_to_byte_col(line: &str, utf16_col: usize) -> Option<usize> {
    let mut utf16_index = 0usize;
    for (byte_index, ch) in line.char_indices() {
        if utf16_index == utf16_col {
            return Some(byte_index);
        }
        utf16_index += ch.len_utf16();
        if utf16_index > utf16_col {
            return Some(byte_index);
        }
    }
    if utf16_index == utf16_col {
        Some(line.len())
    } else {
        None
    }
}

fn byte_col_to_utf16_col(line: &str, byte_col: usize) -> Option<u32> {
    if byte_col > line.len() {
        return None;
    }
    let mut utf16 = 0usize;
    for (index, ch) in line.char_indices() {
        if index >= byte_col {
            break;
        }
        utf16 += ch.len_utf16();
    }
    u32::try_from(utf16).ok()
}

fn function_name_at_line(source: &str, line: usize) -> Option<String> {
    let mut current = None::<String>;
    for (index, raw_line) in source.lines().enumerate() {
        if index > line {
            break;
        }
        let trimmed_prefix = raw_line.trim_start();
        if trimmed_prefix.starts_with("family ")
            || trimmed_prefix.starts_with("family(")
            || trimmed_prefix.starts_with("family\t")
        {
            continue;
        }
        if raw_line
            .chars()
            .next()
            .is_some_and(|character| character.is_whitespace())
        {
            continue;
        }
        let trimmed = raw_line.trim();
        if trimmed.is_empty() || trimmed.starts_with("--") || trimmed.starts_with('#') {
            continue;
        }
        if trimmed.starts_with("enum ") || trimmed.starts_with('|') {
            continue;
        }
        let Some((name, rest)) = parse_decl_head_and_rest(trimmed) else {
            continue;
        };
        if rest.contains(':') || rest.contains('=') {
            current = Some(name);
        }
    }
    current
}

fn parse_decl_head_and_rest(trimmed: &str) -> Option<(String, &str)> {
    if let Some((name, rest)) = parse_operator_decl_head(trimmed) {
        return Some((name, rest));
    }
    let bytes = trimmed.as_bytes();
    let mut index = 0usize;
    while bytes.get(index).is_some_and(|byte| is_ident_char(*byte)) {
        index += 1;
    }
    if index == 0 {
        return None;
    }
    let head = trimmed[..index].to_string();
    let mut segments = Vec::<String>::new();
    while bytes.get(index) == Some(&b'\\') {
        index += 1;
        let segment_start = index;
        while bytes.get(index).is_some_and(|byte| is_ident_char(*byte)) {
            index += 1;
        }
        if index == segment_start {
            return None;
        }
        segments.push(trimmed[segment_start..index].to_string());
    }
    if segments.is_empty() {
        return Some((head, &trimmed[index..]));
    }
    let mut name_parts = Vec::with_capacity(segments.len() + 2);
    name_parts.push("__fam".to_string());
    name_parts.push(head);
    name_parts.extend(segments);
    Some((name_parts.join("$"), &trimmed[index..]))
}

fn parse_operator_decl_head(trimmed: &str) -> Option<(String, &str)> {
    let bytes = trimmed.as_bytes();
    if bytes.first() != Some(&b'(') {
        return None;
    }
    let close = trimmed.find(')')?;
    let operator = &trimmed[1..close];
    let key = operator_key_for_lsp(operator)?;
    let mut index = close + 1;
    let mut segments = Vec::<String>::new();
    while bytes.get(index) == Some(&b'\\') {
        index += 1;
        let start = index;
        while bytes.get(index).is_some_and(|byte| is_ident_char(*byte)) {
            index += 1;
        }
        if index == start {
            return None;
        }
        segments.push(trimmed[start..index].to_string());
    }
    if segments.is_empty() {
        return None;
    }
    let mut name_parts = Vec::with_capacity(segments.len() + 2);
    name_parts.push("__fam".to_string());
    name_parts.push(key.to_string());
    name_parts.extend(segments);
    Some((name_parts.join("$"), &trimmed[index..]))
}

fn operator_key_for_lsp(operator: &str) -> Option<&'static str> {
    match operator {
        "+" => Some("add"),
        "-" => Some("sub"),
        "*" => Some("mul"),
        "/" => Some("div"),
        "%" => Some("mod"),
        "==" => Some("eq"),
        "<" => Some("lt"),
        ">" => Some("gt"),
        "<=" => Some("le"),
        ">=" => Some("ge"),
        _ => None,
    }
}

fn format_prim_op(op: PrimOp) -> &'static str {
    match op {
        PrimOp::Add => "+",
        PrimOp::Sub => "-",
        PrimOp::Mul => "*",
        PrimOp::Div => "/",
        PrimOp::Mod => "%",
        PrimOp::And => "&&",
        PrimOp::Or => "||",
        PrimOp::Eq => "==",
        PrimOp::Lt => "<",
        PrimOp::Gt => ">",
        PrimOp::Le => "<=",
        PrimOp::Ge => ">=",
    }
}

fn collect_local_types(
    function: &crate::CheckedFunction,
    constructor_names: &BTreeSet<String>,
) -> BTreeMap<String, Vec<Type>> {
    let mut map = BTreeMap::<String, Vec<Type>>::new();
    let (args, _) = function.signature.ty.fun_parts();
    for clause in &function.clauses {
        for (pattern, arg_ty) in clause.patterns.iter().zip(&args) {
            if let Pattern::Name(name) = &pattern.pattern
                && name != "_"
                && !constructor_names.contains(name)
            {
                insert_type(&mut map, name, arg_ty.clone());
            }
        }
        collect_expr_local_types(&clause.body, &mut map);
    }
    map
}

fn collect_expr_local_types(expr: &TypedExpr, map: &mut BTreeMap<String, Vec<Type>>) {
    match &expr.kind {
        TypedExprKind::Local(name) => {
            if name != "_" {
                insert_type(map, name, expr.ty.clone());
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
        TypedExprKind::Lambda { body, .. } => collect_expr_local_types(body, map),
        TypedExprKind::Let { bindings, body } => {
            for binding in bindings {
                if binding.name != "_" {
                    insert_type(map, &binding.name, binding.expr.ty.clone());
                }
                collect_expr_local_types(&binding.expr, map);
            }
            collect_expr_local_types(body, map);
        }
        TypedExprKind::Record(fields) => {
            for value in fields.values() {
                collect_expr_local_types(value, map);
            }
        }
        TypedExprKind::Project { base, .. } => collect_expr_local_types(base, map),
        TypedExprKind::RecordUpdate { base, fields } => {
            collect_expr_local_types(base, map);
            for value in fields.values() {
                collect_expr_local_types(value, map);
            }
        }
        TypedExprKind::Call { args, .. } => {
            for arg in args {
                collect_expr_local_types(&arg.expr, map);
            }
        }
        TypedExprKind::Apply { callee, arg } => {
            collect_expr_local_types(callee, map);
            collect_expr_local_types(arg, map);
        }
    }
}

fn insert_type(map: &mut BTreeMap<String, Vec<Type>>, name: &str, ty: Type) {
    let entry = map.entry(name.to_string()).or_default();
    if !entry.contains(&ty) {
        entry.push(ty);
    }
}

fn full_document_range(source: &str) -> Range {
    let mut lines = source.lines().collect::<Vec<_>>();
    if lines.is_empty() {
        lines.push("");
    }
    let line_index = lines.len().saturating_sub(1);
    let character = lines[line_index]
        .chars()
        .map(char::len_utf16)
        .sum::<usize>();
    Range {
        start: Position::new(0, 0),
        end: Position::new(line_index as u32, character as u32),
    }
}

fn inlay_hints_for_source(source: &str, range: Option<Range>) -> Vec<InlayHint> {
    let mut hints = Vec::<InlayHint>::new();
    let mut inside_vertical_let = false;
    for (line_index, line) in source.lines().enumerate() {
        let trimmed = line.trim_start();
        let indent = line.len().saturating_sub(trimmed.len());
        if trimmed == "let" || trimmed.ends_with("= let") || trimmed.ends_with(" let") {
            inside_vertical_let = true;
            continue;
        }
        if inside_vertical_let {
            if trimmed == "in" || trimmed.starts_with("in ") {
                inside_vertical_let = false;
                continue;
            }
            if let Some(binding_col) = binding_name_column(trimmed) {
                let position = Position::new(line_index as u32, (indent + binding_col) as u32);
                if range
                    .as_ref()
                    .is_some_and(|range| !position_in_range(position, range))
                {
                    continue;
                }
                hints.push(InlayHint {
                    position,
                    label: InlayHintLabel::String("let ".to_string()),
                    kind: Some(InlayHintKind::TYPE),
                    text_edits: None,
                    tooltip: None,
                    padding_left: Some(false),
                    padding_right: Some(false),
                    data: None,
                });
            }
        }
    }
    hints
}

fn binding_name_column(trimmed_line: &str) -> Option<usize> {
    let bytes = trimmed_line.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    if !(bytes[0].is_ascii_lowercase() || bytes[0] == b'_') {
        return None;
    }
    let mut index = 1usize;
    while index < bytes.len() && is_ident_char(bytes[index]) {
        index += 1;
    }
    let rest = trimmed_line[index..].trim_start();
    if rest.starts_with('=') { Some(0) } else { None }
}

fn position_in_range(position: Position, range: &Range) -> bool {
    (position.line > range.start.line
        || (position.line == range.start.line && position.character >= range.start.character))
        && (position.line < range.end.line
            || (position.line == range.end.line && position.character <= range.end.character))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_identifier_under_cursor() {
        let source = "main : i64[n] -> i64[n]\nmain xs = inc xs\n";
        let ident = identifier_at_position(source, Position::new(0, 1)).expect("identifier");
        assert_eq!(ident.name, "main");
        assert_eq!(ident.range.start.character, 0);
        assert_eq!(ident.range.end.character, 4);
    }

    #[test]
    fn hover_shows_function_type() {
        let source = "inc : i64 -> i64\ninc x = x + 1\n";
        let hover = hover_for_position(source, Position::new(0, 1)).expect("hover");
        let HoverContents::Markup(content) = hover.contents else {
            panic!("expected markdown hover content");
        };
        assert!(content.value.contains("inc : i64 -> i64"));
    }

    #[test]
    fn hover_shows_type_witness_keyword() {
        let source = "my_func : Type t -> t -> t\nmy_func i64 x = x + 1\n";
        let hover = hover_for_position(source, Position::new(0, 10)).expect("hover");
        let HoverContents::Markup(content) = hover.contents else {
            panic!("expected markdown hover content");
        };
        assert!(content.value.contains("type witness keyword"));
    }

    #[test]
    fn hover_shows_primitive_witness_token() {
        let source = "id : t -> t\nid x = x\nmain : i64 -> i64\nmain x = id\\i64 x\n";
        let hover = hover_for_position(source, Position::new(3, 12)).expect("hover");
        let HoverContents::Markup(content) = hover.contents else {
            panic!("expected markdown hover content");
        };
        assert!(content.value.contains("primitive witness"));
        assert!(content.value.contains("i64"));
    }

    #[test]
    fn hover_shows_local_value_type() {
        let source = "inc : i64 -> i64\ninc x = let y = x + 1 in y\n";
        let hover = hover_for_position(source, Position::new(1, 12)).expect("hover");
        let HoverContents::Markup(content) = hover.contents else {
            panic!("expected markdown hover content");
        };
        assert!(content.value.contains("y : i64"));
    }

    #[test]
    fn hover_shows_type_alias_definition() {
        let source = "type v3 a = {x:a,y:a,z:a}\nmain : v3 i64 -> i64\nmain p = p.x\n";
        let hover = hover_for_position(source, Position::new(0, 6)).expect("hover");
        let HoverContents::Markup(content) = hover.contents else {
            panic!("expected markdown hover content");
        };
        assert!(content.value.contains("type v3 a = {x:a,y:a,z:a}"));
    }

    #[test]
    fn hover_shows_family_declaration() {
        let source = "family Eq : i64 -> i64 -> i64\nmain : i64 -> i64\nmain x = x + 1\n";
        let hover = hover_for_position(source, Position::new(0, 8)).expect("hover");
        let HoverContents::Markup(content) = hover.contents else {
            panic!("expected markdown hover content");
        };
        assert!(content.value.contains("family Eq : i64 -> i64 -> i64"));
    }

    #[test]
    fn hover_shows_family_declaration_with_parameters() {
        let source = "family my_func\\a\\b : a -> b -> a\nmain : i64 -> i64\nmain x = x\n";
        let hover = hover_for_position(source, Position::new(0, 8)).expect("hover");
        let HoverContents::Markup(content) = hover.contents else {
            panic!("expected markdown hover content");
        };
        assert!(content.value.contains("family my_func\\a\\b : a -> b -> a"));
    }

    #[test]
    fn hover_uses_function_context_after_family_declaration() {
        let source = "family Eq : i64 -> i64 -> i64\nmain : i64 -> i64\nmain x = x + 1\n";
        let hover = hover_for_position(source, Position::new(2, 9)).expect("hover");
        let HoverContents::Markup(content) = hover.contents else {
            panic!("expected markdown hover content");
        };
        assert!(content.value.contains("x : i64"));
    }

    #[test]
    fn hover_shows_local_type_inside_operator_instance_clause() {
        let source = "type v3 a = {x:a,y:a,z:a}\n(*)\\v3\\i64 : v3 i64 -> v3 i64 -> i64\n(*)\\v3\\i64 lhs rhs = let p = lhs.x * rhs.x in p\n";
        let hover = hover_for_position(source, Position::new(2, 25)).expect("hover");
        let HoverContents::Markup(content) = hover.contents else {
            panic!("expected markdown hover content");
        };
        assert!(content.value.contains("p : i64"));
    }

    #[test]
    fn diagnostics_report_syntax_errors() {
        let diagnostics = diagnostics_for_source("main : i64 -> i64\nmain x = let y = in x\n");
        assert_eq!(diagnostics.len(), 1);
        assert!(diagnostics[0].message.contains("syntax error"));
    }

    #[test]
    fn diagnostics_empty_for_valid_program() {
        let diagnostics = diagnostics_for_source("main : i64 -> i64\nmain x = x + 1\n");
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn diagnostics_empty_for_type_witness_program() {
        let diagnostics =
            diagnostics_for_source("my_func : Type t -> t -> t\nmy_func i64 x = x + 1\n");
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn diagnostics_empty_for_specialization_chain() {
        let diagnostics = diagnostics_for_source(
            "id : t -> t\nid x = x\nmain : i64 -> i64\nmain x = id\\i64 x\n",
        );
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn diagnostics_empty_for_type_alias_operator_instance_program() {
        let diagnostics = diagnostics_for_source(
            "type v3 a = {x:a,y:a,z:a}\n(*)\\v3\\a : v3 a -> v3 a -> v3 a\n(*)\\v3\\a lhs rhs = lhs\nmain : v3 i64 -> v3 i64\nmain x = x * x\n",
        );
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn diagnostics_empty_for_family_and_string_program() {
        let diagnostics = diagnostics_for_source(
            "concat\\string : string -> string -> string\nconcat\\string x y = y\n(==)\\string : string -> string -> bool\n(==)\\string a b = contains a b\nmain : bool\nmain = \"a\" `concat` \"b\" == \"ab\"\n",
        );
        assert!(diagnostics.is_empty());
    }

    #[test]
    fn inlay_hints_for_vertical_let_bindings() {
        let source = "main : i64 -> i64\nmain x = let\n  y = x + 1\n  z = y * 2\nin z\n";
        let hints = inlay_hints_for_source(source, None);
        assert_eq!(hints.len(), 2);
        assert_eq!(hints[0].position.line, 2);
        assert_eq!(hints[1].position.line, 3);
        assert_eq!(hints[0].position.character, 2);
        assert_eq!(hints[1].position.character, 2);
        let InlayHintLabel::String(label) = &hints[0].label else {
            panic!("expected plain string inlay label");
        };
        assert_eq!(label, "let ");
    }

    #[test]
    fn inlay_hints_cover_mouse_glow_vertical_bindings() {
        let source = include_str!("../examples/mouse_glow_f32.simd");
        let hints = inlay_hints_for_source(source, None);
        assert_eq!(hints.len(), 8);
        for hint in hints {
            let InlayHintLabel::String(label) = hint.label else {
                panic!("expected plain string inlay label");
            };
            assert_eq!(label, "let ");
        }
    }
}
