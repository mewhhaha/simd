use std::fs;

use crate::{
    Clause, Decl, Expr, FamilyDecl, FamilyInstanceSpec, ImportDecl, LetBinding,
    OperatorInstanceSpec, Pattern, Prim, PrimOp, RefExpr, Result, Shape, Signature, SimdError,
    SurfaceProgram, Type, TypeAliasDecl, parse_source, read_source_file,
};

const PREC_LET: u8 = 0;
const PREC_CMP: u8 = 30;
const PREC_ADD: u8 = 40;
const PREC_MUL: u8 = 50;
const PREC_APP: u8 = 70;
const PREC_POSTFIX: u8 = 80;
const PREC_ATOM: u8 = 90;

pub fn fmt_command(path: &str, check: bool) -> Result<String> {
    let source = read_source_file(path)?;
    let formatted = format_source_text(&source)?;
    if check {
        if normalize_for_compare(&source) == formatted {
            return Ok(format!("already formatted {}", path));
        }
        return Err(SimdError::new(format!("needs formatting {}", path)));
    }
    if normalize_for_compare(&source) == formatted {
        return Ok(format!("unchanged {}", path));
    }
    fs::write(path, formatted)
        .map_err(|error| SimdError::new(format!("failed to write '{}': {}", path, error)))?;
    Ok(format!("formatted {}", path))
}

pub fn format_source_text(source: &str) -> Result<String> {
    let program = parse_source(source)?;
    Ok(format_program(&program))
}

pub fn format_program(program: &SurfaceProgram) -> String {
    if program.decls.is_empty() {
        return String::new();
    }
    let mut out = String::new();
    for (index, decl) in program.decls.iter().enumerate() {
        if index > 0 {
            out.push_str(match (&program.decls[index - 1], decl) {
                (Decl::Import(_), Decl::Import(_)) => "\n",
                (Decl::Import(_), _) => "\n\n",
                (_, Decl::Import(_)) => "\n\n",
                (Decl::Family(_), Decl::Family(_)) => "\n",
                (Decl::Family(_), _) => "\n\n",
                (_, Decl::Family(_)) => "\n\n",
                (Decl::TypeAlias(_), Decl::TypeAlias(_)) => "\n",
                (Decl::TypeAlias(_), _) => "\n\n",
                (_, Decl::TypeAlias(_)) => "\n\n",
                (Decl::Signature(left), Decl::Clause(right)) if left.name == right.name => "\n",
                (Decl::Clause(left), Decl::Clause(right)) if left.name == right.name => "\n",
                _ => "\n\n",
            });
        }
        out.push_str(&format_decl(decl));
    }
    out.push('\n');
    out
}

fn normalize_for_compare(source: &str) -> String {
    let mut text = source.replace("\r\n", "\n");
    while text.ends_with('\n') {
        text.pop();
    }
    text.push('\n');
    text
}

fn format_decl(decl: &Decl) -> String {
    match decl {
        Decl::Import(import_decl) => format_import(import_decl),
        Decl::Family(family) => format_family_decl(family),
        Decl::TypeAlias(alias) => format_type_alias(alias),
        Decl::Enum(enum_decl) => format_enum_decl(enum_decl),
        Decl::Signature(signature) => format_signature(signature),
        Decl::Clause(clause) => format_clause(clause),
    }
}

fn format_enum_decl(enum_decl: &crate::EnumDecl) -> String {
    let mut out = String::new();
    out.push_str("enum ");
    out.push_str(&enum_decl.name);
    if !enum_decl.params.is_empty() {
        out.push(' ');
        out.push_str(&enum_decl.params.join(" "));
    }
    out.push_str(" =\n");
    for ctor in &enum_decl.ctors {
        out.push_str("  | ");
        out.push_str(&ctor.name);
        for field in &ctor.fields {
            out.push(' ');
            out.push_str(&format_type(field));
        }
        out.push('\n');
    }
    out
}

fn format_import(import_decl: &ImportDecl) -> String {
    format!("import {} as {}", import_decl.path, import_decl.alias)
}

fn format_family_decl(family: &FamilyDecl) -> String {
    let params = if family.params.is_empty() {
        String::new()
    } else {
        family.params.iter().fold(String::new(), |mut out, param| {
            out.push('\\');
            out.push_str(param);
            out
        })
    };
    let head = match family.op {
        Some(op) => format!("({}){}", format_prim_op(op), params),
        None => format!("{}{}", family.name, params),
    };
    format!("family {} : {}", head, format_type(&family.ty))
}

fn format_type_alias(alias: &TypeAliasDecl) -> String {
    if alias.params.is_empty() {
        format!("type {} = {}", alias.name, format_type(&alias.body))
    } else {
        format!(
            "type {} {} = {}",
            alias.name,
            alias.params.join(" "),
            format_type(&alias.body)
        )
    }
}

fn format_signature(signature: &Signature) -> String {
    format!(
        "{} : {}",
        format_decl_head(
            &signature.name,
            &signature.operator_instance,
            &signature.family_instance,
        ),
        format_type(&signature.ty)
    )
}

fn format_decl_head(
    name: &str,
    operator: &Option<OperatorInstanceSpec>,
    family: &Option<FamilyInstanceSpec>,
) -> String {
    if let Some(spec) = operator {
        return format_operator_instance_spec(spec);
    }
    if let Some(spec) = family {
        return format_family_instance_spec(spec);
    }
    name.to_string()
}

fn format_clause(clause: &Clause) -> String {
    let mut out = format_decl_head(
        &clause.name,
        &clause.operator_instance,
        &clause.family_instance,
    );
    for pattern in &clause.patterns {
        out.push(' ');
        out.push_str(&format_pattern(pattern));
    }
    out.push_str(" = ");
    out.push_str(&format_expr(&clause.body, PREC_LET));
    out
}

pub fn format_type_for_display(ty: &Type) -> String {
    format_type(ty)
}

fn format_type(ty: &Type) -> String {
    match ty {
        Type::Fun(args, ret) => {
            let mut parts = Vec::with_capacity(args.len() + 1);
            for arg in args {
                parts.push(format_type_atom(arg));
            }
            parts.push(format_type(ret));
            parts.join(" -> ")
        }
        _ => format_type_atom(ty),
    }
}

fn format_type_atom(ty: &Type) -> String {
    match ty {
        Type::Scalar(prim) => format_prim(*prim).to_string(),
        Type::Bulk(prim, shape) => format!("{}{}", format_prim(*prim), format_shape(shape)),
        Type::TypeToken(inner) => format!("Type {}", format_type_atom(inner)),
        Type::Record(fields) => {
            let mut parts = Vec::with_capacity(fields.len());
            for (name, field_ty) in fields {
                parts.push(format!("{}:{}", name, format_type(field_ty)));
            }
            format!("{{{}}}", parts.join(","))
        }
        Type::Named(name, args) => {
            if args.is_empty() {
                name.clone()
            } else {
                let mut parts = Vec::with_capacity(args.len() + 1);
                parts.push(name.clone());
                parts.extend(args.iter().map(format_type_atom));
                parts.join(" ")
            }
        }
        Type::Var(name) => name.clone(),
        Type::Infer(index) => format!("?{}", index),
        Type::Fun(_, _) => format!("({})", format_type(ty)),
    }
}

fn format_shape(shape: &Shape) -> String {
    let dims = shape
        .0
        .iter()
        .map(|dim| match dim {
            crate::Dim::Const(value) => value.to_string(),
            crate::Dim::Var(name) => name.clone(),
        })
        .collect::<Vec<_>>();
    format!("[{}]", dims.join(","))
}

fn format_pattern(pattern: &Pattern) -> String {
    match pattern {
        Pattern::Int(value) => value.to_string(),
        Pattern::Float(value) => super::format_float(*value),
        Pattern::Bool(value) => value.to_string(),
        Pattern::Type(prim) => format_prim(*prim).to_string(),
        Pattern::Ctor(name, subpatterns) => {
            let mut out = name.to_string();
            for subpattern in subpatterns {
                out.push(' ');
                out.push_str(&format_pattern(subpattern));
            }
            if subpatterns.is_empty() {
                return out;
            }
            format!("({})", out)
        }
        Pattern::Name(name) => name.clone(),
        Pattern::Wildcard => "_".to_string(),
    }
}

fn format_expr(expr: &Expr, min_prec: u8) -> String {
    let rendered = match expr {
        Expr::Ref(reference) => format_ref_expr(reference),
        Expr::Int(value) => value.to_string(),
        Expr::Float(value) => super::format_float(*value),
        Expr::Bool(value) => value.to_string(),
        Expr::String(value) => format_string_literal(value),
        Expr::Lambda { param, body } => {
            format!("\\{} -> {}", param, format_expr(body, PREC_LET))
        }
        Expr::Let { bindings, body } => format_let_expr(bindings, body),
        Expr::Record(fields) => format_record_fields(fields),
        Expr::Project(base, field) => {
            format!("{}.{}", format_expr(base, PREC_POSTFIX), field)
        }
        Expr::RecordUpdate { base, fields } => format!(
            "{} {}",
            format_expr(base, PREC_POSTFIX),
            format_record_fields(fields)
        ),
        Expr::App(_, _) => format_application(expr),
        Expr::Infix { op, lhs, rhs } => {
            let (lbp, rbp) = infix_binding_power(*op);
            format!(
                "{} {} {}",
                format_expr(lhs, lbp),
                format_prim_op(*op),
                format_expr(rhs, rbp)
            )
        }
    };
    if expr_precedence(expr) < min_prec {
        format!("({rendered})")
    } else {
        rendered
    }
}

fn format_let_expr(bindings: &[LetBinding], body: &Expr) -> String {
    if bindings.is_empty() {
        return format!("let\n  in {}", format_expr(body, PREC_LET));
    }
    let mut out = String::new();
    out.push_str("let\n");
    for binding in bindings {
        out.push_str("  ");
        out.push_str(&binding.name);
        out.push_str(" = ");
        let rendered = format_expr(&binding.expr, PREC_LET);
        out.push_str(&indent_multiline_tail(&rendered, "    "));
        out.push('\n');
    }
    out.push_str("  in ");
    let body_rendered = format_expr(body, PREC_LET);
    out.push_str(&indent_multiline_tail(&body_rendered, "     "));
    out
}

fn format_record_fields(fields: &[(String, Expr)]) -> String {
    let mut parts = Vec::with_capacity(fields.len());
    for (name, expr) in fields {
        parts.push(format!("{} = {}", name, format_expr(expr, PREC_LET)));
    }
    format!("{{ {} }}", parts.join(", "))
}

fn format_application(expr: &Expr) -> String {
    let mut args = Vec::<&Expr>::new();
    let mut cursor = expr;
    while let Expr::App(callee, arg) = cursor {
        args.push(arg.as_ref());
        cursor = callee;
    }
    args.reverse();
    let mut out = format_expr(cursor, PREC_APP);
    for arg in args {
        out.push(' ');
        out.push_str(&format_expr(arg, PREC_APP + 1));
    }
    out
}

fn indent_multiline_tail(text: &str, continuation_indent: &str) -> String {
    let mut lines = text.lines();
    let Some(first) = lines.next() else {
        return String::new();
    };
    let mut out = first.to_string();
    for line in lines {
        out.push('\n');
        out.push_str(continuation_indent);
        out.push_str(line);
    }
    out
}

fn expr_precedence(expr: &Expr) -> u8 {
    match expr {
        Expr::Let { .. } => PREC_LET,
        Expr::Infix { op, .. } => match op {
            PrimOp::Or => 20,
            PrimOp::And => 25,
            PrimOp::Eq | PrimOp::Lt | PrimOp::Gt | PrimOp::Le | PrimOp::Ge => PREC_CMP,
            PrimOp::Add | PrimOp::Sub => PREC_ADD,
            PrimOp::Mul | PrimOp::Div | PrimOp::Mod => PREC_MUL,
        },
        Expr::App(_, _) => PREC_APP,
        Expr::Project(_, _) | Expr::RecordUpdate { .. } => PREC_POSTFIX,
        Expr::Lambda { .. } => PREC_LET,
        Expr::Ref(_)
        | Expr::Int(_)
        | Expr::Float(_)
        | Expr::Bool(_)
        | Expr::String(_)
        | Expr::Record(_) => PREC_ATOM,
    }
}

fn format_ref_expr(reference: &RefExpr) -> String {
    let mut out = String::new();
    if let Some(alias) = &reference.alias {
        out.push_str(alias);
        out.push('\\');
    }
    out.push_str(&reference.name);
    for type_arg in &reference.type_args {
        out.push('\\');
        match type_arg {
            crate::TypeArg::Prim(prim) => out.push_str(format_prim(*prim)),
            crate::TypeArg::Name(name) => out.push_str(name),
        }
    }
    out
}

fn format_string_literal(value: &str) -> String {
    let mut out = String::from("\"");
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            _ => out.push(ch),
        }
    }
    out.push('"');
    out
}

fn infix_binding_power(op: PrimOp) -> (u8, u8) {
    match op {
        PrimOp::Mul | PrimOp::Div | PrimOp::Mod => (PREC_MUL, PREC_MUL + 1),
        PrimOp::Add | PrimOp::Sub => (PREC_ADD, PREC_ADD + 1),
        PrimOp::And => (25, 26),
        PrimOp::Or => (20, 21),
        PrimOp::Eq | PrimOp::Lt | PrimOp::Gt | PrimOp::Le | PrimOp::Ge => (PREC_CMP, PREC_CMP + 1),
    }
}

fn format_prim(prim: Prim) -> &'static str {
    match prim {
        Prim::I32 => "i32",
        Prim::I64 => "i64",
        Prim::F32 => "f32",
        Prim::F64 => "f64",
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

fn format_operator_instance_spec(spec: &OperatorInstanceSpec) -> String {
    let mut out = format!("({})", format_prim_op(spec.op));
    for segment in &spec.segments {
        out.push('\\');
        out.push_str(segment);
    }
    out
}

fn format_family_instance_spec(spec: &FamilyInstanceSpec) -> String {
    let mut out = spec.family.clone();
    for segment in &spec.segments {
        out.push('\\');
        out.push_str(segment);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_source;

    #[test]
    fn formats_signatures_clauses_and_spacing() {
        let source = "inc:i64->i64\ninc x=x+1\nsquare:f32->f32\nsquare x=x*x\n";
        let formatted = format_source_text(source).expect("format should succeed");
        assert_eq!(
            formatted,
            "inc : i64 -> i64\ninc x = x + 1\n\nsquare : f32 -> f32\nsquare x = x * x\n"
        );
    }

    #[test]
    fn formats_let_records_projection_and_update() {
        let source = "main:{x:i64,y:i64}->i64\nmain p=let q=p{x=p.x+1};_=q.y in q.x\n";
        let formatted = format_source_text(source).expect("format should succeed");
        assert_eq!(
            formatted,
            "main : {x:i64,y:i64} -> i64\nmain p = let\n  q = p { x = p.x + 1 }\n  _ = q.y\n  in q.x\n"
        );
    }

    #[test]
    fn parser_accepts_multiline_let_bindings() {
        let source = "main : i64 -> i64\nmain x = let\n  y = x + 1\n  z = y * 2\nin z\n";
        let program = parse_source(source).expect("multiline let should parse");
        assert_eq!(program.decls.len(), 2);
    }

    #[test]
    fn formats_type_aliases_and_operator_instance_heads() {
        let source =
            "type v3 a={x:a,y:a,z:a}\n(*)\\v3\\a:v3 a->v3 a->v3 a\n(*)\\v3\\a lhs rhs=lhs\n";
        let formatted = format_source_text(source).expect("format should succeed");
        assert_eq!(
            formatted,
            "type v3 a = {x:a,y:a,z:a}\n\n(*)\\v3\\a : v3 a -> v3 a -> v3 a\n(*)\\v3\\a lhs rhs = lhs\n"
        );
    }

    #[test]
    fn formats_family_declarations_canonically() {
        let source = "family(+) : i64 -> i64\nfamily Eq : i64 -> i64 -> i64\n";
        let formatted = format_source_text(source).expect("format should succeed");
        assert_eq!(
            formatted,
            "family (+) : i64 -> i64\nfamily Eq : i64 -> i64 -> i64\n"
        );
    }

    #[test]
    fn formats_ordered_family_parameters() {
        let source = "family my_func\\a\\b:a->b->a\n";
        let formatted = format_source_text(source).expect("format should succeed");
        assert_eq!(formatted, "family my_func\\a\\b : a -> b -> a\n");
    }

    #[test]
    fn formats_operator_family_declaration_parameters() {
        let source = "family (+)\\a : a -> a -> a\n";
        let formatted = format_source_text(source).expect("format should succeed");
        assert_eq!(formatted, "family (+)\\a : a -> a -> a\n");
    }

    #[test]
    fn formats_family_declarations_roundtrip_with_explicit_params() {
        let source = "family my_func\\a\\b : a -> b -> a\n";
        let formatted = format_source_text(source).expect("format should succeed");
        let parsed_original = parse_source(source).expect("original parses");
        let parsed_formatted = parse_source(&formatted).expect("formatted parses");
        assert_eq!(parsed_original, parsed_formatted);
    }

    #[test]
    fn formats_family_instance_heads_without_internal_names() {
        let source = "concat\\string : string -> string -> string\nconcat\\string x y = y\n";
        let formatted = format_source_text(source).expect("format should succeed");
        assert_eq!(
            formatted,
            "concat\\string : string -> string -> string\nconcat\\string x y = y\n"
        );
        assert!(!formatted.contains("__fam$"));
    }

    #[test]
    fn formatter_is_parse_stable() {
        let source = "axpy:i64->i64->i64->i64\naxpy a x y=a*x+y\nmain:i64->i64[n]->i64[n]->i64[n]\nmain a xs ys=axpy a xs ys\n";
        let formatted = format_source_text(source).expect("format should succeed");
        let parsed_original = parse_source(source).expect("original parses");
        let parsed_formatted = parse_source(&formatted).expect("formatted parses");
        assert_eq!(parsed_original, parsed_formatted);
    }

    #[test]
    fn formatter_is_idempotent() {
        let source = "pow2 : i64 -> i64 -> i64\npow2 0 x = x\npow2 n x = pow2 (n - 1) (x * 2)\n";
        let once = format_source_text(source).expect("first format should succeed");
        let twice = format_source_text(&once).expect("second format should succeed");
        assert_eq!(once, twice);
    }

    #[test]
    fn formats_imports_and_qualified_names_without_division_spacing() {
        let source = "import math/scalar as scalar\naxpy:i64->i64->i64->i64\naxpy a x y=a*x+y\nmain:i64->i64[n]->i64[n]->i64[n]\nmain a xs ys=scalar\\axpy a xs ys\n";
        let formatted = format_source_text(source).expect("format should succeed");
        assert_eq!(
            formatted,
            "import math/scalar as scalar\n\naxpy : i64 -> i64 -> i64 -> i64\naxpy a x y = a * x + y\n\nmain : i64 -> i64[n] -> i64[n] -> i64[n]\nmain a xs ys = scalar\\axpy a xs ys\n"
        );
    }

    #[test]
    fn formats_lambda_and_specialized_refs() {
        let source = "my_func:Type t->t->t\nmy_func i64 x=x+1\nmain:i64->i64\nmain x=scalar\\axpy\\i64 x\nhelper:i64->i64\nhelper x=(\\y->y+1) x\n";
        let formatted = format_source_text(source).expect("format should succeed");
        assert_eq!(
            formatted,
            "my_func : Type t -> t -> t\nmy_func i64 x = x + 1\n\nmain : i64 -> i64\nmain x = scalar\\axpy\\i64 x\n\nhelper : i64 -> i64\nhelper x = (\\y -> y + 1) x\n"
        );
    }

    #[test]
    fn keeps_division_spaced() {
        let source = "main : f32 -> f32 -> f32\nmain x y = x/y\n";
        let formatted = format_source_text(source).expect("format should succeed");
        assert_eq!(formatted, "main : f32 -> f32 -> f32\nmain x y = x / y\n");
    }

    #[test]
    fn formats_string_literals_with_escapes() {
        let source = "main : i64 -> i64\nmain x = \"a\\n\\t\\\\\\\"\"\n";
        let formatted = format_source_text(source).expect("format should succeed");
        assert_eq!(
            formatted,
            "main : i64 -> i64\nmain x = \"a\\n\\t\\\\\\\"\"\n"
        );
        let parsed_original = parse_source(source).expect("original parses");
        let parsed_formatted = parse_source(&formatted).expect("formatted parses");
        assert_eq!(parsed_original, parsed_formatted);
    }

    #[test]
    fn formats_bool_literals_and_patterns() {
        let source = "flip true=false\nflip false=true\n";
        let formatted = format_source_text(source).expect("format should succeed");
        assert_eq!(formatted, "flip true = false\nflip false = true\n");
    }
}
