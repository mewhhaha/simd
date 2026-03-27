# SIMD Language Paper (Implementation-Accurate, March 2026)

## 1. Goal

This project implements a tiny pure functional language where scalar-looking code is the default authoring model, and bulk/SIMD execution is a compiler/runtime consequence.

Design intent:

- Keep source programs small, explicit, and predictable.
- Preserve strict semantics (no fast-math semantic drift).
- Make pointwise bulk execution the easy path.
- Keep the evaluator as the semantic oracle.

## 2. Source Language (Current)

The current language supports:

- Top-level function signatures and clauses.
- Primitive numeric scalars: `i32`, `i64`, `f32`, `f64`.
- `char`.
- `string`.
- Shaped bulk types: `t[n]`, `t[n,m]`, ...
- Closed records with nested fields.
- Recursive fixed-arity enums.
- Expression-local `let ... in ...`.
- Record literals, projection, and functional record updates.
- Function application and infix numeric/comparison operators.
- Clause patterns: numeric literals, char literals, string literals, constructor patterns, names, `_`, and rank-1 slice views.

Not supported:

- `if`, `match` expressions, loops, lambdas, higher-order functions.
- Mutual-recursive enums, general graph identity/cycles, enum JSON host input, reductions/scans, shape-changing bulk operators.
- Top-level `let`.

## 3. Type System and Lifting

Core type model:

- `Type = Scalar(Prim) | Bulk(Prim, Shape) | Named(...) | Record(...) | Fun(...) | TypeToken(...)`
- `Shape = Vec<Dim>` with `Dim = Const | Var`

Core semantic choices:

- Comparisons produce `i64` (`0` false, nonzero true).
- Pointwise lifting is implicit for well-typed calls once certified.
- Lifted call arguments are tracked as `Same` (broadcast/scalar) vs `Lane` (bulk lane-varying).
- Bulk call arguments in one lifted call must unify shape.
- `string` remains a source-friendly boundary type while the implementation supports `char`, string literals, and slice-pattern matching.

## 4. `let` and Zero-Demand `_`

`let` semantics are expression-local and acyclic:

- A let-group is dependency-sorted.
- Cycles are rejected.
- Bindings can be written in source order or dependency order.

`_` semantics:

- Clause-pattern `_` is wildcard, non-binding.
- `let _ = expr in body` is a discard binding and does not introduce a local.
- `_` is treated as zero-demand in this model.

## 5. Records, Strings, and Structural Data

Records are source-level values, but lowering is record-free:

- A normalization pass eliminates records into deterministic leaf paths.
- Record params/results are flattened to leaf primitives (including bulk leaves).
- Lowering and Wasm codegen consume normalized leaf functions/expressions.
- Runtime reconstructs source-level records at outer interfaces.

Storage policy:

- Internal bulk layout is SoA (fieldwise contiguous buffers).
- JSON boundary remains source-friendly (AoS accepted/emitted; SoA compatibility accepted where supported).

Recursive enum policy:

- Recursive fixed-arity enums are represented as preorder interval tapes.
- Each enum value uses `tags`, `ends`, and `slots` plus constructor row slabs for non-recursive payload fields.
- Evaluator, lowered execution, and Wasm now share this tape semantics.
- Wasm uses tape-backed enum handles instead of the earlier pointer-node representation.

String/slice policy:

- `char` is a first-class scalar.
- `string` participates in string literals, char literals, and prefix/suffix slice-pattern matching.
- String-heavy programs, including the JSON ADT parser example, run through evaluator and Wasm.

## 6. Compiler Pipeline (Current)

Current conceptual pipeline:

1. Parse surface syntax.
2. Group signatures/clauses and validate arities/contracts.
3. Typecheck + pointwise certification.
4. Normalize records to leaf functions.
5. Lower normalized program to loop-oriented IR and structural enum/slice operations.
6. Group compatible record leaves into grouped kernels when legal.
7. Intent analysis + strict-safe IR rewrites.
8. Wasm lowering/codegen from lowered/grouped IR.

The evaluator path remains the semantic reference for parity checks.

## 7. Runtime and Execution Paths

Two execution lanes are implemented:

- Compatibility lane: `run` / `run-wasm` using JSON I/O.
- Fast lane: prepared execution API with fixed shape layout binding and reusable memory.
- Profiling lane: `run-profile` / `run-wasm-profile` for stage timing.

Prepared ABI v1 characteristics:

- Contiguous row-major bulk memory.
- Stable slot metadata (role, leaf path, type, shape).
- Safe typed scalar/bulk read/write methods.
- No raw pointer public API in v1.

## 8. Wasm Backend (Current Scope)

Implemented backend behavior:

- Real Wasm byte generation via `wasm-encoder`.
- Execution through Wasmtime.
- SIMD vector loops + scalar tail cleanup.
- Grouped record kernels for compatible leaves.
- Structural fallbacks for unsupported/incompatible forms.
- Pointer-induction style loop codegen and invariant hoisting.
- Prepared-runtime memory reuse under conservative safety checks.
- Recursive enum lowering and Wasm execution through tape-backed structural memory.
- `char`, `string`, and slice-pattern programs through evaluator and Wasm.
- Persistent Wasmtime compilation caching for repeated CLI runs.

Current target profile in this codebase:

- Single linear memory (`memory64: false`).
- SIMD enabled.
- Cranelift opt level set to speed.

## 9. Benchmarks (Current)

Benchmarking includes:

- Core examples (`inc`, `square`, `axpy`, `axpy2-record`, `pow2`).
- Expanded matrix suite across `i32/i64/f32/f64`.
- Rank-2 kernels and record/grouped kernels.
- Image-like RGB workloads (tone mapping and blend).
- Rust reference implementations.
- Generated Wasm and handwritten Wasm baselines.
- `wasm-e2e`, `wasm-hot`, `wasm-prepared-hot`, and `wasm-prepared-hot+read`.

## 10. Tooling (Current)

Single `simd` binary currently exposes:

- Compiler frontend commands: `parse`, `check`.
- Formatter: `fmt`.
- Evaluator runner: `run`.
- Evaluator profiler: `run-profile`.
- Wasm flows: `wasm`, `wat`, `run-wasm`, `run-wasm-prepared`, `run-wasm-profile`.
- Benchmarks: `bench`.
- Inspector generation: `inspect-html`.
- LSP server: `lsp` (tower-lsp based diagnostics/hover/format/inlay hints).

Project also includes:

- Tree-sitter grammar for SIMd source.
- Helix integration (highlights + rainbow bracket queries + local config).
- WAT inspector page generation for example presets.

## 11. Explicitly Out of Scope (Still)

The following remain intentionally out of scope for this milestone:

- Semantics-relaxing transforms (e.g., non-deterministic fast-math behavior).
- General higher-order language features.
- Full Wasm GC/language-runtime object model.
- Shape-changing bulk language constructs.
- Mutual-recursive enums and general recursive graph identity.
- Generic host JSON enum input.
- Lowering for `Type t` witness programs.

## 12. Practical Status Summary

The language is no longer parser/typechecker-only. It now has:

- Executable evaluator semantics.
- Executable Wasm backend.
- Record semantics with internal SoA lowering.
- Recursive enum/ADT execution through evaluator, lowered execution, and Wasm.
- `char`, `string`, and slice-pattern support across the pipeline.
- Prepared hot path with stable typed ABI.
- Profile commands for evaluator/Wasm stage timing.
- Broad benchmark matrix with generated-vs-Rust and generated-vs-handwritten comparisons.

In short: this is a working semantics-first functional SIMD language prototype with production-style benchmarking and tooling infrastructure, not just a draft grammar.
