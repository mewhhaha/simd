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
