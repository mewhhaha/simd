set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

default:
	@{{just_executable()}} --list

install:
	{{just_executable()}} helix-grammar
	cargo install --path .
	{{just_executable()}} helix-register

check:
	cargo fmt --check
	cargo test -q
	{{just_executable()}} helix-grammar

inspect-html:
	cargo run -- inspect-html --out docs/inspector.html

demo:
	cargo run --quiet -- wasm examples/mouse_glow_f32.simd --main main --out docs/mouse_glow_f32.wasm >/dev/null
	cargo run --quiet -- wasm examples/mouse_rings_f32.simd --main main --out docs/mouse_rings_f32.wasm >/dev/null
	cargo run --quiet -- wasm examples/string_sdf_f32.simd --main main --out docs/string_sdf_f32.wasm >/dev/null
	cargo run --quiet -- wasm examples/image_stipple_field_f32.simd --main main --out docs/image_stipple_field_f32.wasm >/dev/null
	cargo run --quiet -- wasm examples/image_stipple_force_f32.simd --main main --out docs/image_stipple_force_f32.wasm >/dev/null
	deno run --allow-net --allow-read scripts/dev_server.ts 8000

fmt file:
	cargo run -- fmt {{file}}

fmt-check file:
	cargo run -- fmt {{file}} --check

lsp:
	cargo run -- lsp

helix-grammar:
	./scripts/helix_grammar.sh

helix-register:
	./scripts/setup_helix.sh

harness mode="quick":
	sh ./scripts/iterate_structural_harness.sh {{mode}}

star-seq-harness mode="quick":
	sh ./scripts/iterate_star_seq_harness.sh {{mode}}

harness-diff:
	@files=($$(ls -1t .tmp/harness/structural_harness_*.ndjson 2>/dev/null)); \
	if [ $${#files[@]} -lt 2 ]; then \
	  echo "need at least two harness logs in .tmp/harness"; \
	  exit 1; \
	fi; \
	echo "new: $${files[0]}"; \
	echo "old: $${files[1]}"; \
	diff -u "$${files[1]}" "$${files[0]}" || true
