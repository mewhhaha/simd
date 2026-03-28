# simd

`simd` is a semantics-first functional SIMD language prototype with:

- scalar and bulk numerics
- records with internal SoA lowering
- native n-ary tuples
- owned runtime-sized sequences with `T[*]` in the evaluator/core runtime
- `char`, `string`, and slice-view clause patterns
- recursive fixed-arity enums/ADTs
- evaluator, lowered execution, and Wasm execution paths
- prepared hot-path execution and profiling commands

Current recursive enum status:

- recursive fixed-arity self-recursive enums work through the evaluator, structural lowering, and Wasm
- evaluator, lowered execution, and Wasm all use preorder interval-tape semantics for enum trees
- Wasm enum values now use native tape-backed handles rather than the older pointer-node layout
- host JSON enum input is still unsupported
- `T[*]` is available in the surface language and evaluator/core runtime
- Wasm now supports scalar-element `T[*]` consumers through the existing `(ptr,len)` bulk ABI
- Wasm now supports direct scalar-element `T[*]` results and scalar-element enum payload fields
- Wasm now supports the current recursive-ADT-safe `T[*]` subset, including recursive `Self[*]` enum payload fields
- `reverse : T[*] -> T[*]` is now a first-class builtin across evaluator and Wasm for the current Wasm-storable `T[*]` subset
- broader non-scalar top-level `T[*]` ABI shapes beyond that subset are still out of scope

Useful commands:

```sh
cargo run -- run examples/json_parser_adt.simd --main main --args '["{\"a\":[1,2,3]}"]'
cargo run -- run-wasm examples/json_parser_adt.simd --main main --args '["{\"a\":[1,2,3]}"]'
cargo run -- run-wasm-prepared examples/json_parser_adt.simd --main main --args '["{\"a\":[1,2,3]}"]' --iters 10
cargo run -- run-profile examples/json_parser_adt.simd --main main --args '["{\"a\":[1,2,3]}"]'
cargo run -- run-profile-fns examples/json_parser_adt.simd --main main --args '["{\"a\":[1,2,3]}"]'
cargo run -- run-wasm-profile examples/json_parser_adt.simd --main main --args '["{\"a\":[1,2,3]}"]'
cargo run -- run-wasm-profile-fns examples/json_parser_adt.simd --main main --args '["{\"a\":[1,2,3]}"]'
cargo run -- run-profile-fns examples/star_seq_basics.simd --main sum_seq --args '[[1,2,3,4]]' --json
```

Implementation notes:

- repeated Wasm CLI runs use a local Wasmtime compilation cache under `.tmp/wasmtime-cache`
- `run-wasm-profile` reports cold-vs-cached Wasm stage timings
- `run-profile` reports evaluator frontend/arg-parse/execute timings
- `run-profile-fns` and `run-wasm-profile-fns` report per-function inclusive time and call counts for named SIMd functions
- both per-function profilers accept `--json` for machine-readable output
- `examples/json_parser_adt.simd` now uses flat `T[*]` child regions for both arrays and objects:
  - `JArray Json[*]`
  - `JObject Json[*]`
  - object entries are represented as `JField key_len value` elements inside the object sequence
  - sequence reversal now uses the general `reverse` builtin instead of a handwritten accumulator helper

Example programs:

- [JSON parser ADT notes](./examples/README.md)
- [JSON parser ADT source](./examples/json_parser_adt.simd)
- [Tuple basics source](./examples/tuple_basics.simd)
- [Star sequence basics source](./examples/star_seq_basics.simd)
- [Star sequence tree source](./examples/star_seq_tree.simd)

## Benchmark Tracking

This file tracks benchmark snapshots so we can compare progress over time.
All numbers are machine-dependent and should be compared on the same host.

## Web Demos

- [Demo Explorer](./docs/demo_explorer.html)
- [WAT Inspector](./docs/inspector.html)
- [Mouse Shader Demo](./docs/canvas_mouse_demo.html)
- [Image Upload Shader Lab](./docs/image_upload_shader_demo.html)
- [Stipple Pulse Shader Demo](./docs/image_stipple_pulse_shader_demo.html)
- [3D Matrix Canvas Demo](./docs/canvas_3d_matrix_demo.html)
- [Shadertoy Logo Shader Demo](./docs/canvas_string_sdf_demo.html)

Run all demos locally with:

```sh
just demo
```

### Latest Snapshot (2026-03-20)

Commands used:

```sh
cargo run --release -- bench all --size 262144 --iters 3
cargo run --release -- bench matrix --iters 1
```

#### Core suite (`bench all`)

| Case | Rust ns/elem | wasm-hot ns/elem | wasm-hot slowdown | wasm-prepared-hot ns/elem | wasm-prepared-hot slowdown |
| --- | ---: | ---: | ---: | ---: | ---: |
| `inc-i64` | 9.64 | 9.34 | 0.97x | 0.24 | 0.03x |
| `square-f32` | 9.70 | 8.82 | 0.91x | 0.14 | 0.01x |
| `axpy-i64` | 1.17 | 11.10 | 9.45x | 0.37 | 0.31x |
| `axpy2-record-i64` | 2.42 | 24.64 | 10.17x | 0.64 | 0.26x |
| `pow2-i64` | 9.80 | 10.70 | 1.09x | 2.01 | 0.21x |

Core contract summary:

- `core-large gate: PASS (5/5 faster)` (gated on `wasm-prepared-hot`)
- `axpy2 per-leaf within 10% of axpy per-leaf at large: PASS` (ratio `0.80x`)

#### Matrix suite (`bench matrix`)

Contract summary:

- `large slowdown<1.0: PASS (27/27 pass)`
- `medium slowdown<=1.05: PASS (22/22 pass)`
- `small baseline-regression gate: SKIP` (baseline unavailable in run)
- `core-suite large gate: PASS (5/5 faster)`
- `axpy2 per-leaf within 10% of axpy per-leaf at large: PASS` (ratio `0.85x`)
- `overall: PASS`

Notable optimizer plans observed:

- `vec4` appears on grouped image blend leaves (`main$r`, `main$g`, `main$b`).
- Broadcast-heavy `axpy` kernels still choose `vec1` with current structural cost model.

## Rerun Benchmarks

```sh
cargo run --release -- bench all --size 262144 --iters 3
cargo run --release -- bench matrix --iters 1
```

## Type Witness Dispatch (`Type t`)

Use `Type t` parameters when you want compile-time style branching in pure source:

```simd
my_func : Type t -> t -> t
my_func i64 x = x + 1
my_func i32 x = x + 2
my_func _ x = x
```

Run through the evaluator path with JSON type witness args:

```sh
cargo run -- run examples/type_dispatch.simd --main main --args '["i64", 41]'
```

Current backend status:
- evaluator path supports `Type t` witness arguments and type-pattern clauses
- normalization/lowering/Wasm still reject `Type t` programs in this milestone

## Current Language Surface

The current source language includes:

- primitive numeric scalars: `i32`, `i64`, `f32`, `f64`
- `char`
- shaped bulks
- `string`
- tuples
- records
- recursive fixed-arity enums
- clause patterns for literals, constructors, strings, chars, and prefix/suffix slice views

Examples:

```simd
my_char : char -> i64
my_char 'a' = 1
my_char _ = 0

starts_car : string -> i64
starts_car ['c', 'a', 'r', ...rest] = len rest
starts_car _ = 0

enum List a =
  | Nil
  | Cons a (List a)

swap : (i64, i64) -> (i64, i64)
swap (x, y) = (y, x)
```
