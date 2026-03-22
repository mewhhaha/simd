# Flat Recursive ADTs

This note sketches a representation for recursive data in `simd` that stays aligned with the
current regular-bulk + normalized-leaf execution model:

- no boxed recursive runtime values
- recursive edges lower to flat-store handles
- JSON-like trees use a linear tape / jump-array layout
- bulk `[n]` keeps its current meaning: regular SIMD-shaped data

## Why not boxed trees?

`simd` is currently strongest when values are regular, shape-aware, and normalize into primitive
leaf buffers. Boxed recursive trees fight that model:

- one allocation per node
- pointer chasing instead of dense loads
- poor cache locality
- no natural fit for current bulk `[n]` semantics

## Core rule

Allow recursive ADTs in source, but define them as *store-backed* at runtime.

For example, surface syntax can stay conventional:

```simd
data List a =
    Nil
  | Cons a (List a)
```

but the runtime meaning is not an inline nested tree. The recursive edge compiles to a node handle
into a flat store:

```text
ListId      = u32

ListStore a = {
  tag  : u8[n],   // 0 = Nil, 1 = Cons
  head : a[n],
  tail : u32[n],
}
```

So a `List a` value is logically recursive, but physically represented by a root node id plus its
backing store.

## JSON should use a tape / jump-array layout

For JSON, a linear tape is usually a better fit than a generic pointer-like reference.

Example logical source type:

```simd
data Json =
    JNull
  | JBool bool
  | JNum f64
  | JStr Span
  | JArr Json[]
  | JObj { key : Span, value : Json }[]
```

Proposed physical representation:

```text
JsonId = u32

JsonStore = {
  tag        : u8[n],   // null / bool / num / str / arr / obj
  bool_v     : u8[n],
  num_v      : f64[n],
  str_off    : u32[n],
  str_len    : u32[n],
  child_off  : u32[n],
  child_len  : u32[n],

  obj_key_off : u32[m],
  obj_key_len : u32[m],
  obj_val     : u32[m],

  bytes      : u8[k]
}
```

The important point is that arrays/objects do not contain boxed child values. They contain jump
metadata into contiguous child / entry regions.

That makes a parsed value behave more like a subtree start index in a tape than like a heap pointer.

## Interaction with `[n]`

Bulk `[n]` should keep its current meaning:

- regular SIMD-shaped data
- shape participates in lifting / unification
- values are dense and lane-oriented

Recursive containers are different. Their local child counts are runtime properties of one node,
not global bulk shapes for a whole function.

So:

- `Json[n]` means a regular bulk of `n` JSON values
- `{ key : Span, value : Json }[n]` means a regular bulk of object-entry values
- an array/object node should still lower to jump metadata (`child_off`, `child_len`) into a flat
  backing region

This avoids overloading bulk `[n]` with irregular-container semantics.

## What happens when execution hits a recursive edge?

The handle itself is cheap: typically just a `u32`.

The cost appears when we *follow* it:

- indirect loads into node metadata
- possible tag divergence between lanes
- less predictable access than ordinary primitive bulk buffers

That suggests a good implementation split:

1. SIMD for raw byte scanning / parsing
2. flat store construction
3. bulk processing over dense columns (`tag`, offsets, lengths, payload side tables)
4. specialized batched traversal for arrays / objects

In other words, arithmetic on leaves stays in the regular bulk world; following recursive edges is a
separate indexed traversal mode over a flat store, not ordinary pointwise bulk lifting.

## Suggested direction

1. Keep source recursive ADT syntax ergonomic.
2. Define recursive ADTs as store-backed by construction.
3. For JSON specifically, prefer tape / jump-array layout over boxed trees.
4. Keep bulk `[n]` semantics unchanged.
5. Lower recursive edges to integer node ids plus jump metadata, not machine pointers.

This gives `simd` a path to JSON and other recursive trees without abandoning the regular,
leaf-oriented execution model that current bulk code depends on.
