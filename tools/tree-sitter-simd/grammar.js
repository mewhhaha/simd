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

    declaration: ($) =>
      choice(
        $.import_decl,
        $.type_alias,
        $.family_decl,
        $.signature,
        $.clause,
      ),

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

    family_decl: ($) =>
      seq(
        "family",
        field("head", $.family_head),
        ":",
        field("type", $.type),
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

    decl_head: ($) => choice($.identifier, $.operator_head, $.family_instance_head),

    family_head: ($) =>
      choice(
        $.identifier,
        seq("(", field("operator", $.prim_operator), ")"),
      ),

    family_instance_head: ($) =>
      seq(
        field("name", $.identifier),
        repeat1(seq("\\", field("segment", choice($.identifier, $.prim_type)))),
      ),

    operator_head: ($) =>
      seq(
        "(",
        field("operator", $.prim_operator),
        ")",
        repeat1(seq("\\", field("segment", choice($.identifier, $.prim_type)))),
      ),

    prim_operator: (_) => choice("+", "-", "*", "/", "%", "==", "<", ">", "<=", ">="),

    pattern: ($) =>
      choice($.wildcard, $.bool_literal, $.prim_type, $.identifier, $.int, $.float),

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
      choice(
        $.prim_type,
        $.bool_type,
        $.type_witness,
        $.identifier,
        $.string_type,
        $.record_type,
        seq("(", $.type, ")"),
      ),

    string_type: (_) => "string",
    bool_type: (_) => "bool",

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
        prec.left(
          PREC.add,
          seq(
            field("left", $.add_expr),
            field("operator", $.infix_function_operator),
            field("right", $.mul_expr),
          ),
        ),
        $.mul_expr,
      ),

    infix_function_operator: ($) =>
      seq(
        "`",
        field("function", choice($.qualified_ref, $.identifier)),
        "`",
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
        $.bool_literal,
        $.string,
        $.string_type,
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
    string: () =>
      token(
        seq(
          '"',
          repeat(choice(/[^"\\\n]/, seq("\\", /./))),
          '"',
        ),
      ),
    identifier: (_) => /[a-z][a-zA-Z0-9_]*/,
    nat: (_) => /[0-9]+/,
    int: (_) => /[0-9]+/,
    float: (_) => /[0-9]+\.[0-9]+/,
    bool_literal: (_) => choice("true", "false"),
  },
});
