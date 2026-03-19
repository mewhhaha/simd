Here is a solid v0 base you can bootstrap from.

---

# Project base: tiny functional bulk language

## 1. Goal

Build a language that is:

* functional
* tiny
* easy to compile into fast code
* good at SIMD by default
* written with scalar-looking functions
* lifted to bulk operations by the type system

The language should feel like:

* just functions
* just equations
* just primitive numbers
* recursion instead of loops

No `while`.
No `for`.
No user-visible `map`.
No user-visible SIMD intrinsics.

The compiler decides when a function is pointwise and can be lifted over bulk values.

---

## 2. Core idea

A function written for scalars can also be used on bulk values when it is pointwise.

Example:

```txt
inc : i64 -> i64
inc x = x + 1
```

This same function may also be used as:

```txt
inc : i64[n] -> i64[n]
inc : i64[n,m] -> i64[n,m]
```

with no separate `map`.

So the user writes scalar equations.
The type system and compiler lift them.

---

## 3. Surface language

### Values

Primitive scalars:

```txt
i32
i64
f32
f64
```

Bulk values:

```txt
i64[n]
i64[n,m]
f32[n]
f64[n,m,k]
```

Meaning:

* `i64` is one scalar
* `i64[n]` is a regular rank-1 bulk value
* `i64[n,m]` is a regular rank-2 bulk value

Brackets carry shape.

`n`, `m`, `k` are shape parameters, not runtime scalar values.

---

## 4. Core syntax

### Function definitions

```txt
name : type
name pattern1 pattern2 ... = expr
```

Multiple equations define branching.

Example:

```txt
pred : i64 -> i64
pred 0 = 0
pred n = n - 1
```

### Patterns

Allowed patterns in v0:

```txt
literal
name
_
```

Examples:

```txt
0
n
_
```

`_` means zero demand:

* it does not bind a name
* it does not demand that value

### Expressions

Allowed expressions in v0:

* literals
* names
* function application
* primitive arithmetic
* parentheses

Examples:

```txt
x
42
f x
f x y
x + 1
(x * 2) + y
```

No term-level `match`.
No `if`.
No lambdas.
No local `let` in v0.

Branching happens only by top-level function clauses.

---

## 5. Grammar

```ebnf
program ::= decl*

decl    ::= sig? clause+

sig     ::= name ":" type

clause  ::= name pat* "=" expr

pat     ::= intlit
          | floatlit
          | name
          | "_"

expr    ::= atom
          | expr atom
          | expr op expr
          | "(" expr ")"

atom    ::= name
          | intlit
          | floatlit
          | "(" expr ")"

op      ::= "+" | "-" | "*" | "/" | "%" | "==" | "<" | ">" | "<=" | ">="

type    ::= scalar
          | type "->" type
          | scalar "[" dims "]"

scalar  ::= "i32" | "i64" | "f32" | "f64"

dims    ::= dim
          | dim "," dims

dim     ::= nat
          | name
```

Examples of valid types:

```txt
i64
i64 -> i64
i64 -> i64 -> i64
i64[n]
i64[n,m]
f32[n,m,k] -> f32[n,m,k]
```

---

## 6. Meaning of bulk types

`i64[n]` means:

* exactly `n` values of type `i64`
* regular shape
* bulk value
* pointwise-liftable target

`i64[n,m]` means:

* exactly `n × m` values of type `i64`
* regular rank-2 shape

This is not a list type.
This is not a boxed array object in the language semantics.
This is a regular bulk field.

---

## 7. Function semantics

All functions are pure.

A function equation means mathematical replacement.

Example:

```txt
add : i64 -> i64 -> i64
add x y = x + y
```

Multiple equations mean pattern-based choice:

```txt
pow2 : i64 -> i64 -> i64
pow2 0 x = x
pow2 n x = pow2 (n - 1) (x * 2)
```

That is the only iteration model in the language:

* recursion
* ideally tail recursion for efficient lowering

---

## 8. Zero-demand `_`

`_` is part of the meaning of the language.

Example:

```txt
left : i64 -> i64 -> i64
left x _ = x
```

This means:

* second argument is not used
* second argument is not demanded
* the compiler may skip computing it when possible

That becomes more important later when richer data types are added, but it belongs in v0 from the start.

---

## 9. Implicit lifting

This is the central typing rule.

If a function is pointwise over scalars, it may be used over matching bulk shapes.

### Scalar form

```txt
inc : i64 -> i64
inc x = x + 1
```

### Lifted forms

```txt
inc : i64[n] -> i64[n]
inc : i64[n,m] -> i64[n,m]
```

### Mixed scalar and bulk arguments

```txt
axpy : i64 -> i64 -> i64 -> i64
axpy a x y = a * x + y
```

This may be used as:

```txt
axpy : i64 -> i64[n] -> i64[n] -> i64[n]
axpy : i64 -> i64[n,m] -> i64[n,m] -> i64[n,m]
```

Here `a` is treated as broadcast.

### Rule sketch

A function may lift when:

* it is pure
* it is pointwise
* it does not combine lanes
* it does not inspect bulk structure as shape
* it does not introduce cross-element dependencies

In v0, that means:

* primitive ops
* function application
* pointwise recursion
* no reductions yet

---

## 10. Recursion and lowering

The source language uses recursion only.

The compiler lowers eligible tail recursion to loops.

Example:

```txt
pow2 : i64 -> i64 -> i64
pow2 0 x = x
pow2 n x = pow2 (n - 1) (x * 2)
```

This is tail-recursive and easy to lower.

When used on bulk values:

```txt
main : i64[n] -> i64[n]
main xs = pow2 3 xs
```

the compiler may generate SIMD code pack-by-pack.

So the model is:

* source: recursion
* backend: loops
* surface: functional
* machine: efficient

---

## 11. Pointwise kernel rule

A function is pointwise if every output element depends only on the corresponding input element position, plus scalar/broadcast values.

Good examples:

```txt
inc x = x + 1
square x = x * x
axpy a x y = a * x + y
pow2 0 x = x
pow2 n x = pow2 (n - 1) (x * 2)
```

Not in v0:

* sum of all elements
* prefix scan
* neighboring index access
* filtering
* shape-changing operations

Those can come later.

---

## 12. Minimal standard prelude

These primitives are enough for v0:

Arithmetic:

```txt
+ - * / %
```

Comparison:

```txt
== < > <= >=
```

Types:

```txt
i32 i64 f32 f64
```

No builtin `map`.
No builtin `select`.
No builtin bulk constructors in surface syntax yet.

Bulk behavior comes from implicit lifting.

---

## 13. Example program

```txt
inc : i64 -> i64
inc x = x + 1

double : i64 -> i64
double x = x + x

axpy : i64 -> i64 -> i64 -> i64
axpy a x y = a * x + y

pow2 : i64 -> i64 -> i64
pow2 0 x = x
pow2 n x = pow2 (n - 1) (x * 2)

main1 : i64[n] -> i64[n]
main1 xs = inc xs

main2 : i64[n] -> i64[n]
main2 xs = pow2 3 xs

main3 : i64 -> i64[n] -> i64[n] -> i64[n]
main3 a xs ys = axpy a xs ys
```

This already demonstrates the whole language model.

---

## 14. Internal compiler model

The user does not see this, but the compiler will want these internal distinctions:

* scalar value
* bulk value with shape
* shared scalar for all lanes
* varying value per lane

A useful internal split is:

* `Same t`
* `Lane t`

That lets the compiler see:

* `a : Same i64`
* `x : Lane i64`
* `y : Lane i64`

for a lifted call like `axpy a x y`.

Then lowering to SIMD is straightforward.

---

## 15. Compilation model

A good first compiler pipeline:

### Parse

Read function equations and type signatures.

### Type check

Infer and verify:

* scalar types
* bulk shapes
* pointwise compatibility

### Lift

Turn scalar pointwise functions into bulk kernels when bulk arguments appear.

### Tail recursion analysis

Detect tail-recursive functions and mark them loop-lowerable.

### Bulk lowering

For `i64[n]` and `i64[n,m]`, generate regular loops over shape dimensions.

### SIMD lowering

Inside those loops:

* process packs of elements
* broadcast scalar arguments
* vectorize primitive ops
* handle leftovers with scalar cleanup

For Wasm SIMD:

* `i32` and `f32` usually pack 4 lanes in `v128`
* `i64` and `f64` usually pack 2 lanes in `v128`

---

## 16. What v0 should reject

To keep the first implementation small, reject:

* non-tail recursion in performance-critical lifted paths
* mixed incompatible shapes
* shape-changing functions
* reductions
* scans
* arbitrary pattern matching beyond literals, names, `_`
* user-defined algebraic data types
* local bindings
* lambdas
* higher-order functions

That keeps the compiler simple and the language crisp.

---

## 17. What to implement first

### Parser

Support:

* names
* numbers
* function clauses
* signatures
* simple infix operators
* bulk type syntax like `i64[n,m]`

### Type checker

Support:

* scalar types
* function arrows
* bulk types
* shape variables
* simple pointwise lifting

### Interpreter

Have a scalar evaluator first.

### Bulk evaluator

Add runtime bulk values for testing, even before SIMD.

### Codegen

Lower to:

* plain loops first
* SIMD later

That way the language semantics are stable before the fast backend is finished.

---

## 18. Minimal project statement

This is a good project description:

> A tiny pure functional language with scalar equations, implicit bulk lifting, and tail-recursive lowering to performant code. Surface syntax has only primitive numbers, function clauses, recursion, `_`, and shaped bulk types like `i64[n,m]`. Pointwise functions lift automatically over bulk values and compile to regular loops and SIMD.

---

## 19. Current v0 contract

This is the short version to pin to the repo:

* Functions only
* Primitive numeric types only
* Bulk types use postfix shape syntax: `i64[n]`, `i64[n,m]`
* No `if`, no `while`, no `for`
* Branching only by function clauses
* `_` means zero demand
* Pointwise scalar functions lift automatically over bulk types
* Tail recursion lowers to loops
* SIMD is a backend consequence, not a surface feature

---

## 20. Starter syntax summary

```txt
inc : i64 -> i64
inc x = x + 1

pred : i64 -> i64
pred 0 = 0
pred n = n - 1

pow2 : i64 -> i64 -> i64
pow2 0 x = x
pow2 n x = pow2 (n - 1) (x * 2)

axpy : i64 -> i64 -> i64 -> i64
axpy a x y = a * x + y

main : i64[n] -> i64[n]
main xs = pow2 3 xs
```

This is small enough to build, and strong enough to grow.

The next good artifact after this is a `SPEC.md` with the exact grammar and lifting rules.

