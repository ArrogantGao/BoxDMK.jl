# Hybrid Default Solver Design

## Goal

Make the standard validated BoxDMK workflow use Julia for tree construction and the vendored Fortran library for evaluation. The package should fail fast during initialization if the required Fortran solve library is missing.

## Motivation

The current codebase already validates that the practical fast-and-correct Laplace path is:

- `build_tree(...)` in Julia
- `bdmk_fortran(tree, fvals, ...)` for the solve

The remaining native Julia solver path is useful for debugging and for kernels that are not yet safe on the Fortran-backed public path. The repo also already depends on the vendored Fortran library for the validated hybrid reference path, so making that dependency explicit simplifies the public story.

## Scope

This change covers:

- package initialization behavior,
- public `bdmk(...)` dispatch,
- result normalization back to Julia tree ordering,
- README guidance,
- regression tests for the new default.

This change does not remove the native Julia solver internals. They remain available for internal debugging and benchmarking.

## Chosen Approach

### 1. Require the Fortran solve library at package load

During `__init__`, resolve the vendored Fortran solve library path and throw a clear, actionable error if it is missing. The error should instruct the user to run:

```bash
julia --project deps/build_fortran_ref.jl
```

This makes the runtime model explicit instead of silently falling back to a weaker path.

### 2. Make public `bdmk(...)` use the hybrid solve path by default

Replace the current narrow reference-only dispatch gate with a broader hybrid dispatch that routes validated public Laplace solves through `bdmk_fortran(tree, fvals, ...)`, while keeping the native Julia solver for kernels outside that safe slice.

The public API remains:

```julia
tree, fvals = build_tree(...)
result = bdmk(tree, fvals, kernel; ...)
```

but the default execution model becomes:

- Julia builds the adaptive tree and function samples,
- Fortran consumes that tree and sampled data for the solve.

### 3. Preserve Julia-facing output ordering

The Fortran packing layer reorders boxes into Fortran level order. Public `bdmk(...)` must continue to return arrays in Julia tree order. The wrapper therefore needs to restore order for:

- `pot`
- `grad`
- `hess`

Target outputs do not depend on box order and can be returned directly.

### 4. Keep native Julia solver internal

`_bdmk_native(...)` remains in the codebase for:

- debugging,
- stepwise parity tooling,
- future solver work,
- kernels that are not yet safe on the Fortran-backed public path.

It should no longer be the default public path.

## Alternatives Considered

### Keep the current narrow hybrid dispatch

Rejected because it does not make the hybrid path the default way to work and leaves too much user-visible behavior depending on configuration-specific special cases.

### Require users to call `bdmk_fortran(...)` directly

Rejected because it complicates the public API unnecessarily. The package should expose the preferred path through `bdmk(...)`.

### Keep package load permissive and error later

Rejected because it hides an actual hard dependency. A load-time error is clearer and avoids confusing partial behavior.

## Error Handling

If the solve library is missing at load time, the package should throw an informative error that includes:

- the expected library path,
- the build command,
- a short explanation that the public solver depends on the vendored Fortran backend.

## Testing Strategy

Tests should cover:

- public `bdmk(...)` using the hybrid path beyond the old narrow reference gate,
- output order restoration for `pot`, `grad`, and `hess`,
- init-time error messaging logic via a helper that can be tested without reloading Julia.

## README Changes

The README should state plainly that the default workflow is:

```julia
tree, fvals = build_tree(...)
result = bdmk(tree, fvals, kernel; ...)
```

and that `bdmk(...)` uses the vendored Fortran solve backend. It should also move the Fortran library build step earlier in setup, before normal usage examples.
