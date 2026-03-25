[
  (int)
  (float)
] @constant.numeric

(prim_type) @type.builtin
(bool_type) @type.builtin
(bool_literal) @constant.builtin
(char) @string

(enum_decl
  "enum" @keyword
  name: (identifier) @type)

(enum_decl
  param: (identifier) @type.parameter)

(enum_ctor
  "|" @punctuation.delimiter
  name: (identifier) @type)

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
    (identifier) @type
    (identifier) @type.parameter))

(family_decl
  head: (family_head
    (prim_operator) @operator
    (identifier) @type.parameter))

((clause
  head: (decl_head
    (identifier) @type))
  (#match? @type "^[A-Z]"))

((clause
  head: (decl_head
    (identifier) @function))
  (#match? @function "^[_a-z]"))

((clause
  pattern: (pattern (identifier) @variable.parameter))
  (#match? @variable.parameter "^[_a-z]"))

((pattern (identifier) @type)
  (#match? @type "^[A-Z]"))

(ctor_pattern
  constructor: (identifier) @type)

(slice_rest
  "..." @punctuation.delimiter)

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

(and_operator) @operator
(or_operator) @operator

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
  "|"
  "."
  "..."
  ","
  ";"
  "`"
] @punctuation.delimiter
