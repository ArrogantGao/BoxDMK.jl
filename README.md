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

There are three practical paths in this repo.

1. Pure Julia: `build_tree(...)` + `bdmk(...)`
2. Fortran-backed solve on a Julia tree: `build_tree(...)` + `bdmk_fortran(tree, fvals, ...)`
3. Full Fortran tree + solve: `build_tree_fortran(...)` + `bdmk_fortran(...)`

The validated reference configuration today is:

- Laplace
- 3D
- `LegendreBasis()`
- `norder = 16`
- `eps = 1e-6`
- potentials only

For that case, the repo includes a working hybrid path based on:

- Julia tree construction
- Fortran solve stages

The public `bdmk(...)` entrypoint currently uses that hybrid solve path for the validated reference configuration only. Outside that slice, `bdmk(...)` continues to use the native Julia solver path.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/ArrogantGao/BoxDMK.jl")
```

Requires Julia `1.10+`.

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

## Building The Vendored Fortran Libraries

The repo vendors the Fortran reference sources under `deps/boxdmk_fortran/`.

Build the local shared libraries with:

```bash
julia --project deps/build_fortran_ref.jl
```

This produces repo-local libraries under `deps/usr/lib/` and is the supported way to enable:

- wrapper tests,
- hybrid parity/debug tooling,
- the validated hybrid reference solve path.

## Fortran And Hybrid APIs

Build the tree with Fortran:

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

Solve with Fortran using either the wrapper object or a Julia tree:

```julia
result1 = bdmk_fortran(ftree, LaplaceKernel(); eps = 1e-6)
result2 = bdmk_fortran(tree, fvals, LaplaceKernel(); eps = 1e-6)
```

The second form is the main hybrid entrypoint for parity/debug work.

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

- The strongest parity/debug tooling is focused on the Laplace 3D reference case.
- The native Julia solver path is not yet at full Fortran parity across all configurations.
- The public hybrid dispatch in `bdmk(...)` is intentionally narrow and currently applies only to the validated reference slice.

## License

MIT License. See [LICENSE](LICENSE) for details.
