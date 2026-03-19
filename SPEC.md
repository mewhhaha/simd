Below is a `SPEC.md` you can drop into a repo and start from.

````md
# Tiny Bulk Functional Language — v0 Spec

## 0. Purpose

This language is a tiny pure functional language designed to compile into efficient bulk code.

The surface language is intentionally small:

- primitive numeric types
- function equations
- recursion
- `_` as zero-demand pattern
- shaped bulk types like `i64[n]` and `i64[n,m]`

There is no surface `map`, `for`, `while`, `if`, or term-level `match`.

The key idea is:

- users write scalar-looking functions
- the type system recognizes pointwise functions
- pointwise functions may be lifted automatically over bulk values
- eligible tail recursion lowers to loops
- bulk code lowers to regular loops and SIMD packs

---

## 1. Design Principles

1. **Pure values only**
   - no mutation
   - no exceptions
   - no hidden effects

2. **Recursion is the only iteration**
   - no `for`
   - no `while`

3. **Branching is by function clauses**
   - no term-level `if`
   - no term-level `match`

4. **Bulk is part of the type**
   - `i64[n]` means a regular bulk value of shape `n`
   - `i64[n,m]` means a regular bulk value of shape `n × m`

5. **Pointwise lifting is implicit**
   - scalar functions may be used on bulk values when certified pointwise

6. **`_` means zero demand**
   - it does not bind
   - it does not introduce demand for that value

---

## 2. Lexical Conventions

- Lowercase identifiers name functions, variables, and shape variables.
- Numeric literals are integer or floating literals.
- Primitive types are reserved: `i32`, `i64`, `f32`, `f64`.
- `_` is reserved as wildcard.
- `:` introduces a type annotation.
- `->` introduces a function type.
- `=` introduces an equation.

---

## 3. Grammar

## 3.1 Program

```ebnf
program ::= decl*

decl    ::= sig
          | clause
````

## 3.2 Signatures

```ebnf
sig     ::= name ":" type
```

## 3.3 Function Clauses

```ebnf
clause  ::= name pat* "=" expr
```

A function may have multiple clauses with the same name and arity.

## 3.4 Patterns

```ebnf
pat     ::= intlit
          | floatlit
          | name
          | "_"
```

Patterns in v0 are intentionally restricted.

## 3.5 Expressions

```ebnf
expr    ::= atom
          | expr atom
          | expr op expr
          | "(" expr ")"

atom    ::= name
          | intlit
          | floatlit
          | "(" expr ")"
```

## 3.6 Operators

```ebnf
op      ::= "+"
          | "-"
          | "*"
          | "/"
          | "%"
          | "=="
          | "<"
          | ">"
          | "<="
          | ">="
```

## 3.7 Types

```ebnf
type    ::= fun_type

fun_type ::= bulk_type
           | bulk_type "->" fun_type

bulk_type ::= prim_type
            | prim_type "[" dims "]"

prim_type ::= "i32"
            | "i64"
            | "f32"
            | "f64"

dims    ::= dim
          | dim "," dims

dim     ::= nat
          | name
```

Examples:

```txt
i64
i64 -> i64
i64 -> i64 -> i64
i64[n]
i64[n,m]
f32[h,w,c]
i64[n] -> i64[n]
i64 -> i64[n] -> i64[n]
```

---

## 4. Types and Shapes

## 4.1 Primitive Scalar Types

The primitive scalar types are:

* `i32`
* `i64`
* `f32`
* `f64`

These are scalar runtime values.

## 4.2 Bulk Types

A bulk type is written by attaching a shape to a primitive type.

Examples:

* `i64[n]`
* `i64[n,m]`
* `f32[h,w,c]`

Meaning:

* `i64[n]` = regular rank-1 bulk of `i64`
* `i64[n,m]` = regular rank-2 bulk of `i64`
* `f32[h,w,c]` = regular rank-3 bulk of `f32`

A bulk type is **not** a list type.
A bulk type is **not** an irregular container.
A bulk type is a regular value with known rank and shape parameters.

## 4.3 Shape Variables

Names inside shape brackets are shape variables.

In:

```txt
i64[n,m]
```

`n` and `m` are shape variables.

They are **not** ordinary runtime integers.
They live in the shape level of the type system.

v0 allows only:

* natural number literals
* shape variables

in shape positions.

Examples:

* valid: `i64[4]`
* valid: `i64[n]`
* valid: `i64[n,m]`
* invalid in v0: `i64[n+1]`

---

## 5. Function Definitions

Functions are defined by equations.

Example:

```txt
inc : i64 -> i64
inc x = x + 1
```

Multiple equations define branching by pattern matching on arguments.

Example:

```txt
pred : i64 -> i64
pred 0 = 0
pred n = n - 1
```

Clause selection is top-to-bottom, first match.

Functions are curried by default.

Example:

```txt
add : i64 -> i64 -> i64
add x y = x + y
```

is equivalent in meaning to a function taking `x` and returning a function taking `y`.

---

## 6. Wildcard `_`

`_` is a wildcard pattern.

It means:

* do not bind a name
* do not demand the value in this branch

Example:

```txt
left : i64 -> i64 -> i64
left x _ = x
```

In this clause, the second argument is not demanded.

This is semantically meaningful, not just syntactic sugar.
The compiler may use `_` for demand analysis and dead argument elimination.

---

## 7. Expression Semantics

Expressions are pure.

The language has no observable effects, so implementations may freely:

* reorder independent computations
* eliminate unused values
* specialize bulk lifting
* erase zero-demand values

subject to preserving the resulting value.

Evaluation order is not semantically observable in v0.

---

## 8. Primitive Operations

Primitive arithmetic and comparison are built in.

Arithmetic:

* `+`
* `-`
* `*`
* `/`
* `%`

Comparison:

* `==`
* `<`
* `>`
* `<=`
* `>=`

v0 assumes primitive ops are defined only on matching primitive scalar types.

Examples:

* `i64 + i64 -> i64`
* `f32 * f32 -> f32`
* `i64 < i64 -> i64` is **not** valid unless comparison result type is specified separately

### v0 comparison result

To keep the core tiny, there are two acceptable routes:

### Option A

Comparison returns `i64` where:

* `0` means false
* nonzero means true

### Option B

Introduce `bool` as a primitive later

For the smallest first implementation, **Option A is acceptable**.

---

## 9. Pointwise Functions

A function is **pointwise** if each output element depends only on the corresponding input element(s) and any scalar broadcast arguments.

Examples of pointwise functions:

```txt
inc x = x + 1
square x = x * x
axpy a x y = a * x + y
```

Examples that are not pointwise:

* summing all elements of a bulk value
* prefix scans
* taking neighboring elements
* shape-changing transformations

Only pointwise functions are implicitly liftable in v0.

---

## 10. Implicit Bulk Lifting

This is the central typing rule of the language.

If a function is pointwise over scalar values, it may be used over bulk values of matching shape.

## 10.1 Unary lifting

Given:

```txt
f : T -> U
```

and `f` is pointwise, then it may also be used as:

```txt
f : T[n] -> U[n]
f : T[n,m] -> U[n,m]
f : T[s1,...,sk] -> U[s1,...,sk]
```

## 10.2 Multi-argument lifting

Given:

```txt
f : A -> B -> C
```

and `f` is pointwise, then the compiler may accept:

```txt
f : A[n] -> B[n] -> C[n]
f : A -> B[n] -> C[n]
f : A[n] -> B -> C[n]
```

when all bulk arguments share the same shape.

Scalar arguments are treated as broadcast.

## 10.3 Shape agreement

All lifted bulk arguments in one call must have identical shape.

Examples:

```txt
axpy : i64 -> i64 -> i64 -> i64
axpy a x y = a * x + y
```

Valid lifted uses:

```txt
axpy : i64 -> i64[n] -> i64[n] -> i64[n]
axpy : i64 -> i64[n,m] -> i64[n,m] -> i64[n,m]
```

Invalid lifted use:

```txt
axpy a xs ys
```

where:

* `xs : i64[n]`
* `ys : i64[m]`

unless `n` and `m` unify.

---

## 11. Recursion

Recursion is the only iteration construct in the language.

Example:

```txt
pow2 : i64 -> i64 -> i64
pow2 0 x = x
pow2 n x = pow2 (n - 1) (x * 2)
```

## 11.1 Performance requirement

A recursive function is loop-lowerable when:

* recursion is in tail position
* state is explicit in arguments
* each recursive branch makes structural or numeric progress

Example of tail recursion:

```txt
pow2 0 x = x
pow2 n x = pow2 (n - 1) (x * 2)
```

Example of non-tail recursion:

```txt
bad 0 = 0
bad n = 1 + bad (n - 1)
```

`bad` may still be valid language code, but v0 may reject it from fast lifted paths or compile it without loop lowering.

---

## 12. Tail Recursion and Bulk

Tail-recursive pointwise functions may be lifted over bulk values just like non-recursive pointwise functions.

Example:

```txt
pow2 : i64 -> i64 -> i64
pow2 0 x = x
pow2 n x = pow2 (n - 1) (x * 2)

main : i64[n] -> i64[n]
main xs = pow2 3 xs
```

Meaning:

* `pow2 3` is a scalar function from `i64` to `i64`
* since it is pointwise, it can lift over `i64[n]`
* the compiler may lower it to a loop over SIMD packs

---

## 13. Typing Rules Sketch

## 13.1 Variables and literals

* If `x : T` is in context, then `x` has type `T`.
* Integer literals default according to context, or to `i64` if no better choice exists in v0.
* Float literals default according to context, or to `f64`.

## 13.2 Primitive operators

Arithmetic operators require matching scalar types.

Examples:

* if `x : i64` and `y : i64`, then `x + y : i64`
* if `x : f32` and `y : f32`, then `x * y : f32`

## 13.3 Function application

If:

```txt
f : A -> B
x : A
```

then:

```txt
f x : B
```

## 13.4 Lifted application

If:

```txt
f : A -> B
x : A[n]
```

and `f` is pointwise, then:

```txt
f x : B[n]
```

Likewise for higher rank shapes.

## 13.5 Multi-argument lifting

Lift argument-wise, provided all bulk arguments unify on shape.

---

## 14. What v0 Rejects

To keep the first implementation small, v0 rejects or postpones:

* user-defined algebraic data types
* term-level pattern matching
* `if`
* `while`
* `for`
* lambdas
* local `let`
* higher-order functions
* reductions
* scans
* stencils
* shape-changing operations
* non-regular data
* arbitrary shape arithmetic in types

---

## 15. Examples

## 15.1 Scalar functions

```txt
inc : i64 -> i64
inc x = x + 1

double : i64 -> i64
double x = x + x

pred : i64 -> i64
pred 0 = 0
pred n = n - 1
```

## 15.2 Tail-recursive pointwise function

```txt
pow2 : i64 -> i64 -> i64
pow2 0 x = x
pow2 n x = pow2 (n - 1) (x * 2)
```

## 15.3 Broadcast plus bulk

```txt
axpy : i64 -> i64 -> i64 -> i64
axpy a x y = a * x + y
```

Possible lifted uses:

```txt
axpy a xs ys
```

where:

* `a : i64`
* `xs : i64[n]`
* `ys : i64[n]`

result:

* `i64[n]`

## 15.4 Rank-2 bulk

```txt
main : i64[n,m] -> i64[n,m]
main xs = inc xs
```

This means apply `inc` pointwise to every cell in an `n × m` bulk value.

---

## 16. Implementation Notes

A good first compiler pipeline:

1. Parse source into function signatures and clauses.
2. Type-check scalar expressions.
3. Infer shape variables.
4. Identify pointwise functions.
5. Insert implicit lifting where bulk values appear.
6. Detect tail recursion for loop lowering.
7. Lower bulk shape to regular loops.
8. Lower inner loops to SIMD packs when profitable.

### Internal representation

Even if not exposed in the language, the compiler will likely benefit from internal categories like:

* scalar
* bulk
* same-across-lanes
* varying-by-lane

This helps code generation for broadcasts and vector operations.

---

## 17. Suggested v0 Project Layout

```txt
/spec
  SPEC.md

/src
  lexer
  parser
  ast
  types
  infer
  pointwise
  lift
  tailrec
  lower_loops
  lower_simd

/tests
  parse
  typing
  lifting
  recursion
  codegen
```

---

## 18. Minimal Reference Examples

```txt
inc : i64 -> i64
inc x = x + 1

square : i64 -> i64
square x = x * x

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

main4 : i64[n,m] -> i64[n,m]
main4 xs = square xs
```

---

## 19. Summary

This language is defined by:

* primitive numeric scalar types
* shaped bulk types `T[n,m,...]`
* function equations only
* recursion only
* `_` as zero-demand wildcard
* implicit lifting of pointwise scalar functions over bulk types
* tail recursion as the path to efficient loop lowering

Everything else can come later.

Start with this, and you already have a clean foundation for:

* scalar interpretation
* regular bulk lowering
* SIMD-oriented compilation
* later growth into richer types and more operations


