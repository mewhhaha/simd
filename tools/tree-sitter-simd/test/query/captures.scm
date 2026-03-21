(signature
  head: (decl_head
    (identifier) @function)
  type: (type) @type)

(clause
  head: (decl_head
    (identifier) @function)
  [
    pattern: (pattern (identifier)) @variable.parameter
    pattern: (pattern (prim_type)) @type.builtin
  ])

(type_alias
  "type" @keyword
  name: (identifier) @type
  param: (identifier) @type.parameter)

(family_decl
  "family" @keyword
  head: (family_head
    (identifier) @type))

(family_decl
  head: (family_head
    (prim_operator) @operator))

(signature
  head: (decl_head
    (operator_head
      operator: (_) @operator
      segment: (identifier) @type)))

(signature
  head: (decl_head
    (family_instance_head
      name: (identifier) @function
      segment: (identifier) @type)))

(clause
  head: (decl_head
    (operator_head
      operator: (_) @operator
      segment: (identifier) @type)))

(clause
  head: (decl_head
    (family_instance_head
      name: (identifier) @function
      segment: (identifier) @type)))

(qualified_ref
  head: (identifier) @function
  segment: (identifier) @function)

(infix_function_operator
  function: (identifier) @function)

(infix_function_operator
  function: (qualified_ref
    head: (identifier) @function
    segment: (identifier) @function))

(let_expr
  "let" @keyword
  "in" @keyword)

(let_binding
  name: (identifier) @variable
  value: (expr) @expression)

(type_field
  name: (identifier) @property)

(record_field
  name: (identifier) @property
  value: (expr) @expression)

(postfix_expr
  field: (identifier) @property)

(application_expr
  function: (_) @function.call
  argument: (_) @expression)

(comparison_expr operator: "==" @operator)
(comparison_expr operator: "<" @operator)
(comparison_expr operator: ">" @operator)
(comparison_expr operator: "<=" @operator)
(comparison_expr operator: ">=" @operator)
(add_expr operator: "+" @operator)
(add_expr operator: "-" @operator)
(mul_expr operator: "*" @operator)
(mul_expr operator: "/" @operator)
(mul_expr operator: "%" @operator)

(string) @string

[
  "{"
  "}"
  "("
  ")"
  "["
  "]"
] @punctuation.bracket
