const PREC = {
  cmp: 30,
  add: 40,
  mul: 50,
  app: 70,
  postfix: 80,
};

module.exports = grammar({
  name: "simd",

  extras: ($) => [/[ \t\r\f]/, $.comment],

  word: ($) => $.identifier,

  rules: {
    source_file: ($) =>
      seq(
        repeat($._newline),
        optional(
          seq(
            $.declaration,
            repeat(seq(repeat1($._newline), $.declaration)),
            repeat($._newline),
          ),
        ),
      ),

    declaration: ($) => choice($.import_decl, $.type_alias, $.signature, $.clause),

    import_decl: ($) =>
      seq("import", field("path", $.module_path), "as", field("alias", $.identifier)),

    module_path: (_) => /[a-z][a-zA-Z0-9_]*(\/[a-z][a-zA-Z0-9_]*)*/,

    _newline: (_) => token(/\r?\n/),

    comment: (_) => token(seq("--", /[^\n]*/)),

    type_alias: ($) =>
      seq(
        "type",
        field("name", $.identifier),
        repeat(field("param", $.identifier)),
        "=",
        field("body", $.type),
      ),

    signature: ($) =>
      seq(field("head", $.decl_head), ":", field("type", $.type)),

    clause: ($) =>
      seq(
        field("head", $.decl_head),
        repeat(field("pattern", $.pattern)),
        "=",
        field("body", $.expr),
      ),

    decl_head: ($) => choice($.identifier, $.operator_head),

    operator_head: ($) =>
      seq(
        "(",
        field("operator", choice("+", "-", "*", "/", "%", "==", "<", ">", "<=", ">=")),
        ")",
        repeat1(seq("\\", field("segment", choice($.identifier, $.prim_type)))),
      ),

    pattern: ($) => choice($.wildcard, $.prim_type, $.identifier, $.int, $.float),

    type: ($) =>
      choice(
        prec.right(seq(field("arg", $.type_nonfun), "->", field("result", $.type))),
        $.type_nonfun,
      ),

    type_nonfun: ($) =>
      choice(
        prec.left(seq(field("constructor", $.type_atom), repeat1(field("argument", $.type_atom)))),
        $.type_atom,
      ),

    type_atom: ($) =>
      choice(
        prec.left(seq(field("base", $.type_base), repeat1($.shape_suffix))),
        $.type_base,
      ),

    type_base: ($) =>
      choice($.prim_type, $.type_witness, $.identifier, $.record_type, seq("(", $.type, ")")),

    prim_type: (_) => choice("i32", "i64", "f32", "f64"),

    type_witness: ($) => seq("Type", field("witness", choice($.prim_type, $.identifier))),

    record_type: ($) =>
      seq("{", optional(seq($.type_field, repeat(seq(",", $.type_field)))), "}"),

    type_field: ($) => seq(field("name", $.identifier), ":", field("type", $.type)),

    shape_suffix: ($) =>
      seq("[", $.dim, repeat(seq(",", $.dim)), "]"),

    dim: ($) => choice($.nat, $.identifier),

    expr: ($) => choice($.let_expr, $.lambda_expr, $.comparison_expr),

    lambda_expr: ($) =>
      prec.right(seq("\\", field("param", $.identifier), "->", field("body", $.expr))),

    let_expr: ($) =>
      seq(
        "let",
        $.let_binding,
        repeat(seq(";", $.let_binding)),
        "in",
        $.expr,
      ),

    let_binding: ($) =>
      seq(
        field("name", choice($.identifier, $.wildcard)),
        "=",
        field("value", $.expr),
      ),

    comparison_expr: ($) =>
      choice(
        prec.left(
          PREC.cmp,
          seq(
            field("left", $.comparison_expr),
            field("operator", choice("==", "<", ">", "<=", ">=")),
            field("right", $.add_expr),
          ),
        ),
        $.add_expr,
      ),

    add_expr: ($) =>
      choice(
        prec.left(
          PREC.add,
          seq(
            field("left", $.add_expr),
            field("operator", choice("+", "-")),
            field("right", $.mul_expr),
          ),
        ),
        $.mul_expr,
      ),

    mul_expr: ($) =>
      choice(
        prec.left(
          PREC.mul,
          seq(
            field("left", $.mul_expr),
            field("operator", choice("*", "/", "%")),
            field("right", $.application_expr),
          ),
        ),
        $.application_expr,
      ),

    application_expr: ($) =>
      choice(
        prec.left(
          PREC.app,
          seq(
            field("function", $.application_expr),
            field("argument", $.postfix_expr),
          ),
        ),
        $.postfix_expr,
      ),

    postfix_expr: ($) =>
      choice(
        prec.left(
          PREC.postfix,
          seq(field("base", $.postfix_expr), ".", field("field", $.identifier)),
        ),
        prec.left(
          PREC.postfix,
          seq(field("base", $.postfix_expr), field("update", $.record_update)),
        ),
        $.atom,
      ),

    record_update: ($) =>
      seq("{", optional(seq($.record_field, repeat(seq(",", $.record_field)))), "}"),

    atom: ($) =>
      choice(
        $.qualified_ref,
        $.prim_type,
        $.identifier,
        $.int,
        $.float,
        $.record_expr,
        seq("(", $.expr, ")"),
      ),

    qualified_ref: ($) =>
      seq(
        field("head", $.identifier),
        repeat1(seq("\\", field("segment", choice($.identifier, $.prim_type)))),
      ),

    record_expr: ($) =>
      seq("{", optional(seq($.record_field, repeat(seq(",", $.record_field)))), "}"),

    record_field: ($) =>
      seq(field("name", $.identifier), "=", field("value", $.expr)),

    wildcard: (_) => "_",
    identifier: (_) => /[a-z][a-zA-Z0-9_]*/,
    nat: (_) => /[0-9]+/,
    int: (_) => /[0-9]+/,
    float: (_) => /[0-9]+\.[0-9]+/,
  },
});
