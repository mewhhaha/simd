[
  (int)
  (float)
] @constant.numeric

(prim_type) @type.builtin
(bool_type) @type.builtin
(bool_literal) @constant.builtin

(type_witness
  "Type" @keyword)

(qualified_ref
  head: (identifier) @function
  segment: (identifier) @function)

(infix_function_operator
  function: (identifier) @function)

(infix_function_operator
  function: (qualified_ref
    head: (identifier) @function
    segment: (identifier) @function))

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

(signature
  head: (decl_head
    (family_instance_head
      name: (identifier) @function
      segment: (identifier) @type)))

(family_decl
  "family" @keyword)

(family_decl
  head: (family_head
    (identifier) @type))

(family_decl
  head: (family_head
    (prim_operator) @operator))

(clause
  head: (decl_head
    (identifier) @function))

(clause
  pattern: (pattern (identifier) @variable.parameter))

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

(string) @string

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

(string_type) @type.builtin

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
  "`"
] @punctuation.delimiter
