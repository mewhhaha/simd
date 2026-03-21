[
  (int)
  (float)
] @constant.numeric

(prim_type) @type.builtin

(type_witness
  "Type" @keyword)

(qualified_ref
  head: (identifier) @function
  segment: (identifier) @function)

(type_alias
  "type" @keyword
  name: (identifier) @type
  param: (identifier) @type.parameter)

(signature
  head: (decl_head
    (identifier) @function)
  type: (type) @type)

(signature
  head: (decl_head
    (operator_head
      operator: (_) @operator
      segment: (identifier) @type)))

(clause
  head: (decl_head
    (identifier) @function)
  pattern: (pattern (identifier) @variable.parameter))

(clause
  head: (decl_head
    (operator_head
      operator: (_) @operator
      segment: (identifier) @type)))

(let_expr
  "let" @keyword
  "in" @keyword)

(let_binding
  name: (identifier) @variable)

(type_field
  name: (identifier) @property)

(record_field
  name: (identifier) @property)

(postfix_expr
  field: (identifier) @property)

[
  "=="
  "<"
  ">"
  "<="
  ">="
  "+"
  "-"
  "*"
  "/"
  "%"
] @operator

[
  "{"
  "}"
  "("
  ")"
  "["
  "]"
] @punctuation.bracket

[
  ":"
  "->"
  "="
  "."
  ","
  ";"
] @punctuation.delimiter
