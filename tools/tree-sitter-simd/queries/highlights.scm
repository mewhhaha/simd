[
  (int)
  (float)
] @constant.numeric

(prim_type) @type.builtin
(type_constructor) @type

(type_atom
  constructor: (type_constructor) @keyword
  (#eq? @keyword "Type"))

(qualified_ref
  head: (identifier) @function
  segment: (identifier) @function)

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
