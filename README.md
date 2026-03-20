# simd

## Benchmark Tracking

This file tracks benchmark snapshots so we can compare progress over time.
All numbers are machine-dependent and should be compared on the same host.

## Web Demos

- [WAT Inspector](./docs/inspector.html)
- [Mouse Shader Demo](./docs/canvas_mouse_demo.html)
- [Image Upload Shader Lab](./docs/image_upload_shader_demo.html)

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
