# Examples

## `json_parser_adt.simd`

This example is the current end-to-end structural harness for:

- recursive enums/ADTs
- `char` and `string`
- prefix/suffix slice-pattern matching
- evaluator parity
- Wasm parity

What it does:

- parses a JSON string into a recursive `Json` ADT
- uses flat `T[*]` child regions for both arrays and objects
- represents object entries as `JField key_len value` elements inside `JObject Json[*]`
- uses the general `reverse : T[*] -> T[*]` builtin to finalize parsed array/object order
- exercises enum payload rows carrying both nested enums and `string`
- includes `main_string_leaf_chars`, which walks the parsed ADT and sums string-leaf character counts

Useful commands:

```sh
cargo run -- run examples/json_parser_adt.simd --main main --args '["[1,2,3]"]'
cargo run -- run-wasm examples/json_parser_adt.simd --main main --args '["[1,2,3]"]'
cargo run -- run-wasm-prepared examples/json_parser_adt.simd --main main --args '["[1,2,3]"]' --iters 10
cargo run -- run examples/json_parser_adt.simd --main main_string_leaf_chars --args '["{\"a\":\"car\",\"b\":[\"ed\"]}"]'
cargo run -- run-wasm examples/json_parser_adt.simd --main main_string_leaf_chars --args '["{\"a\":\"car\",\"b\":[\"ed\"]}"]'
```

Current limitations:

- this is still a handwritten parser in the language, not a built-in JSON parser
- enum host JSON input is still unsupported in the generic runtime boundary
- mutual-recursive enums and dynamically recursive enum multiplicity are still out of scope

## `tuple_basics.simd`

This example is the small tuple feature smoke test for:

- tuple literals
- tuple patterns
- tuple projection with `.0`, `.1`, ...
- JSON-array boundary for tuple params/results

Useful commands:

```sh
cargo run -- run examples/tuple_basics.simd --main main --args '[[3,4]]'
cargo run -- run-wasm examples/tuple_basics.simd --main main --args '[[3,4]]'
cargo run -- run examples/tuple_basics.simd --main main_describe --args '[[7,true]]'
```

Expected results:

- `main [3,4]` returns `[4,3]`
- `main_describe [7,true]` returns `[7,true,17]`

## `star_seq_basics.simd`

This example is the first `T[*]` smoke test for:

- owned runtime-sized sequences
- sequence literals like `[1, 2, 3]`
- type-directed bracket patterns like `[x, ...xs]`
- the general `reverse` builtin over `T[*]`
- JSON-array boundary for `T[*]`

Useful commands:

```sh
cargo run -- run examples/star_seq_basics.simd --main main --args '[[4,5,6]]'
cargo run -- run examples/star_seq_basics.simd --main main_boxed --args '[[4,5,6]]'
cargo run -- run-wasm examples/star_seq_basics.simd --main main --args '[[4,5,6]]'
cargo run -- run-wasm examples/star_seq_basics.simd --main main_boxed --args '[[4,5,6]]'
cargo run -- run-wasm examples/star_seq_basics.simd --main sum_seq --args '[[4,5,6]]'
cargo run -- run-wasm-prepared examples/star_seq_basics.simd --main sum_seq --args '[[4,5,6]]' --iters 10
```

Expected results:

- `main [4,5,6]` returns `[15,4,5,6]`
- `main_boxed [4,5,6]` returns `15`

Current limitation:

- Wasm supports scalar-element `T[*]` consumers like `sum_seq`
- Wasm supports direct scalar-element `T[*]` results like `main`
- Wasm supports scalar-element `T[*]` enum payload fields like `Box i64[*]`
- `run-wasm-prepared` supports the same scalar-element `T[*]` consumer/result surface
- remaining unsupported cases are broader non-scalar top-level `T[*]` ABI shapes beyond the current recursive-ADT-safe subset

## `star_seq_tree.simd`

This example proves the recursive-ADT-safe subset for `T[*]`:

- enum fields of type `Self[*]`
- recursive structural consumers over `T[*]`
- enum JSON rendering with array-shaped `T[*]` payloads

Useful commands:

```sh
cargo run -- run examples/star_seq_tree.simd --main main --args '[]'
cargo run -- run examples/star_seq_tree.simd --main main_tree --args '[]'
cargo run -- run-wasm examples/star_seq_tree.simd --main main --args '[]'
cargo run -- run-wasm examples/star_seq_tree.simd --main main_tree --args '[]'
cargo run -- run-wasm-prepared examples/star_seq_tree.simd --main main_tree --args '[]' --iters 10
```

Expected results:

- `main` returns `10`
- `main_tree` returns an enum JSON object whose `fields` array contains nested JSON arrays for the `T[*]` children
- evaluator, direct Wasm, and prepared Wasm all support this `Tree[*]` shape in the current recursive-ADT-safe subset
