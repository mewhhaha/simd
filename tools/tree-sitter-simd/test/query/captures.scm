(signature
  name: (identifier) @function
  type: (type) @type)

(clause
  name: (identifier) @function
  pattern: (pattern (identifier) @variable.parameter))

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

[
  "{"
  "}"
  "("
  ")"
  "["
  "]"
] @punctuation.bracket
