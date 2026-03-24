# BoxDMK.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://ArrogantGao.github.io/BoxDMK.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://ArrogantGao.github.io/BoxDMK.jl/dev/)
[![Build Status](https://github.com/ArrogantGao/BoxDMK.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ArrogantGao/BoxDMK.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/ArrogantGao/BoxDMK.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/ArrogantGao/BoxDMK.jl)

`BoxDMK.jl` evaluates volume potentials on adaptive box trees in 3D.

Given a source density `f` and a kernel `K`, it computes

```math
u(x) = \int_{\Omega} K(x, y) f(y)\,dy
```

on tensor-product box nodes, with optional interpolation to user targets and optional derivative recovery.

This repository now contains both:

- the Julia implementation,
- a vendored Fortran reference build used for parity, debugging, and hybrid execution.

## Current Status

The default workflow is now:

1. Build the tree in Julia with `build_tree(...)`
2. Solve through the public `bdmk(...)` API

For the validated Laplace workflow, `bdmk(...)` uses the vendored Fortran solve backend. That is the standard supported path for correctness and performance in this repo.

The direct Fortran entrypoints remain available for advanced use:

- `bdmk_fortran(tree, fvals, ...)` for parity/debug work
- `build_tree_fortran(...)` when you explicitly want the Fortran tree builder

In practice, if your source density is a Julia function, the recommended Laplace path is still:

- Julia tree construction
- Fortran evaluation

because `build_tree_fortran(...)` must callback from Fortran into Julia for RHS sampling.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/ArrogantGao/BoxDMK.jl")
```

Requires Julia `1.10+`.

## Build The Vendored Fortran Libraries

The public solver requires the vendored Fortran shared libraries at package load.

From the package root, build them with:

```bash
julia --project deps/build_fortran_ref.jl
```

This produces repo-local libraries under `deps/usr/lib/`.

If the solve library is missing, `using BoxDMK` will throw an error telling you to run that command.

## Quick Start

```julia
using BoxDMK

f(x) = [exp(-100 * sum((x .- 0.5) .^ 2))]

tree, fvals = build_tree(
    f,
    LaplaceKernel(),
    LegendreBasis();
    ndim = 3,
    norder = 8,
    eps = 1e-6,
    boxlen = 1.0,
    nd = 1,
)

result = bdmk(tree, fvals, LaplaceKernel(); eps = 1e-6)

size(result.pot)
```

The returned `SolverResult` contains:

- `pot`
- `grad` or `nothing`
- `hess` or `nothing`
- `target_pot` or `nothing`
- `target_grad` or `nothing`
- `target_hess` or `nothing`

`bdmk(...)` is the normal entrypoint. For Laplace solves, it uses the Fortran solve backend behind the public Julia API.

## Target Evaluation

```julia
targets = rand(3, 100)

result = bdmk(
    tree,
    fvals,
    LaplaceKernel();
    eps = 1e-6,
    targets = targets,
)

result.target_pot
```

For Julia-facing APIs, the physical domain is `[0, boxlen]^3`.

The raw Fortran reference uses a box centered at the origin. The wrappers handle that coordinate shift internally for `build_tree_fortran(...)` and `bdmk_fortran(...)`.

## Kernels And Bases

Supported kernels:

```julia
LaplaceKernel()
YukawaKernel(beta)
SqrtLaplaceKernel()
```

Supported bases:

```julia
LegendreBasis()
ChebyshevBasis()
```

## Fortran And Hybrid APIs

The default public workflow is:

```julia
tree, fvals = build_tree(...)
result = bdmk(tree, fvals, kernel; eps = 1e-6)
```

If you want to call the Fortran solve wrapper explicitly, you can still do:

```julia
result = bdmk_fortran(tree, fvals, LaplaceKernel(); eps = 1e-6)
```

For Laplace kernels, the public `bdmk(...)` path already uses the same Fortran solve backend. Calling `bdmk_fortran(...)` explicitly is mainly useful for:

- parity checks,
- benchmark/debug tooling,
- explicit wrapper tests.

You can also build the tree with Fortran:

```julia
ftree = build_tree_fortran(
    f,
    LaplaceKernel(),
    LegendreBasis();
    ndim = 3,
    norder = 8,
    eps = 1e-6,
    boxlen = 1.0,
    nd = 1,
)
```

Use that only when you specifically want the Fortran tree builder. For Julia-defined source functions, it is usually slower than `build_tree(...)` because it repeatedly callbacks from Fortran into Julia.

## Repository Layout

The source tree is grouped by responsibility:

- `src/core/`: types, utilities, basis/tensor/kernel primitives
- `src/tree/`: adaptive tree construction, tree data transforms, interaction lists
- `src/solver/`: SOG tables, proxy/passes, plane-wave pipeline, local tables, derivatives, top-level solver
- `src/fortran/`: vendored-library paths, hotpaths, wrappers, debug snapshots
- `benchmark/`: performance and parity drivers
- `test/`: package regression tests
- `deps/boxdmk_fortran/`: vendored Fortran reference sources

Generated build outputs under `deps/usr/` and `deps/boxdmk_fortran/build*` are local artifacts and should not be committed.

## Development Workflow

Useful commands:

Run a focused test:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_solver.jl")'
```

Run wrapper and hybrid parity checks:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_fortran_wrapper.jl"); include("test/test_hybrid_parity.jl")'
```

Run the reference hybrid benchmark:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project benchmark/hybrid_parity.jl
```

## Limitations

- The vendored Fortran solve library is a required runtime dependency for the public solver.
- The strongest parity/debug tooling is still focused on the Laplace 3D reference case.
- The native Julia solver internals remain in the repo for debugging and development and are still used for kernels outside the validated Laplace-backed path.

## License

MIT License. See [LICENSE](LICENSE) for details.
