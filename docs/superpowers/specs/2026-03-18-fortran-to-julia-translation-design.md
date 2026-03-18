# BoxDMK.jl — Fortran-to-Julia Translation Design Spec

## Overview

Translate the Fortran BoxDMK library (~37K lines) into an idiomatic Julia package. BoxDMK computes volume potentials and their derivatives on adaptive hierarchical box trees using a kernel-independent method: Sum-of-Gaussians (SOG) decomposition + Box Fast Gauss Transform (FGT) + multi-level FMM-like hierarchy.

### Scope

- **3D only** (Laplace, Yukawa, sqrt-Laplace kernels). 2D support deferred.
- **Idiomatic Julia** with type-parameterized kernel dispatch, not a line-by-line port.
- **Julia LinearAlgebra** stdlib for all BLAS/LAPACK operations.
- **SOG tables** loaded from data files (JLD2), not hardcoded.
- **Threading** via `Threads.@threads` from the start.
- **Independent tests** against analytical solutions.

---

## Type Hierarchy

```julia
# Kernels
abstract type AbstractKernel end
struct LaplaceKernel <: AbstractKernel end
struct YukawaKernel{T<:Real} <: AbstractKernel
    beta::T
end
struct SqrtLaplaceKernel <: AbstractKernel end

# Polynomial bases
abstract type AbstractBasis end
struct LegendreBasis <: AbstractBasis end
struct ChebyshevBasis <: AbstractBasis end

# Adaptive tree
struct BoxTree{T<:Real, B<:AbstractBasis}
    ndim::Int
    nlevels::Int
    centers::Matrix{T}           # (ndim, nboxes)
    boxsize::Vector{T}           # one per level
    parent::Vector{Int}
    children::Matrix{Int}        # (2^ndim, nboxes); 0 = leaf
    colleagues::Vector{Vector{Int}}
    level::Vector{Int}
    basis::B
    norder::Int
    fvals::Array{T,3}            # (nd, npbox, nboxes)
end

# SOG data
struct SOGNodes{T<:Real}
    weights::Vector{T}
    deltas::Vector{T}
end

# Local interaction precomputed data
struct LocalTables{T<:Real}
    tables::Array{T}
    indices::Vector{Vector{Int}}
end

# Solver output
struct SolverResult{T<:Real}
    pot::Array{T,3}                              # (nd, npbox, nboxes)
    grad::Union{Nothing, Array{T,4}}             # (nd, 3, npbox, nboxes)
    hess::Union{Nothing, Array{T,4}}             # (nd, 6, npbox, nboxes)
    target_pot::Union{Nothing, Matrix{T}}        # (nd, ntarg)
    target_grad::Union{Nothing, Array{T,3}}      # (nd, 3, ntarg)
    target_hess::Union{Nothing, Array{T,3}}      # (nd, 6, ntarg)
end
```

---

## Public API

```julia
# Build adaptive tree by sampling user function
tree = build_tree(f, kernel, basis;
    ndim=3, norder=6, eps=1e-6, boxlen=1.0,
    nd=1, dpars=nothing, eta=1.0)

# Evaluate volume potential
result = bdmk(tree, kernel;
    eps=1e-6, grad=false, hess=false,
    targets=nothing)

# Access
result.pot
result.grad         # nothing if grad=false
result.hess         # nothing if hess=false
result.target_pot   # nothing if targets=nothing
```

---

## File Layout

```
src/
├── BoxDMK.jl          # Module definition, includes, exports
├── types.jl           # All type definitions above
├── kernels.jl         # Kernel definitions + dispatch helpers
├── basis.jl           # Legendre/Chebyshev: nodes, weights, derivative matrices
├── tree.jl            # Adaptive tree construction + refinement
├── tree_data.jl       # Tree data transforms (interpolation, anterpolation)
├── tensor.jl          # Tensor product operations (1D→3D)
├── sog.jl             # SOG table loading + lookup
├── boxfgt.jl          # Box Fast Gauss Transform
├── local.jl           # Near-field local interactions
├── planewave.jl       # Plane wave expansion routines
├── solver.jl          # Main bdmk() orchestration
├── derivatives.jl     # Gradient + Hessian computation
└── utils.jl           # Miscellaneous utilities
data/
└── sog/               # SOG node tables as JLD2 files
test/
└── runtests.jl        # Analytical solution tests
```

---

## Algorithm Pipeline (inside `bdmk()`)

### 1. Setup Phase
- Load SOG nodes for `(kernel, eps)` via `load_sog_nodes(kernel, 3, eps)`
- Build transformation matrices from basis: `p2c_transform(basis, norder, 3)`, `c2p_transform(basis, norder, 3)`
- Precompute local interaction tables: `build_local_tables(kernel, basis, norder, 3)`
- Determine cutoff level `npwlevel` separating far/near field

### 2. Upward Pass
- Traverse tree bottom-up
- At each non-leaf box: interpolate children's `fvals` to parent grid via `p2c_transform`
- Uses tensor product structure for efficiency

### 3. Plane Wave Pass (Far-Field)
- For each SOG component `(weight, delta)`:
  - At each level above cutoff: run `boxfgt!` to accumulate far-field contributions
  - Box FGT handles plane wave expansions and translations
- Accumulate all SOG contributions into potential

### 4. Downward Pass
- Traverse tree top-down
- At each non-leaf box: anterpolate parent potential to children via `c2p_transform`
- Accumulate with existing children contributions

### 5. Local Pass (Near-Field)
- For each leaf box and its colleagues (near neighbors):
  - Apply precomputed local interaction tables
  - Direct tensor product evaluation
- Handles same-level and different-level neighbors via sparse indexing

### 6. Derivative Computation (Optional)
- Apply 1D derivative matrices along each dimension via tensor product
- Gradient: 3 components from first derivative matrix
- Hessian: 6 components (xx, yy, zz, xy, xz, yz) from second derivative matrix
- Scale by inverse box size

### 7. Target Evaluation (Optional)
- For arbitrary target points: locate containing box, interpolate from box grid

---

## Key Internal Interfaces

### Basis Functions
```julia
nodes_and_weights(::LegendreBasis, norder) → (Vector, Vector)
nodes_and_weights(::ChebyshevBasis, norder) → (Vector, Vector)
derivative_matrix(basis, norder) → Matrix
second_derivative_matrix(basis, norder) → Matrix
interpolation_matrix(basis, from_nodes, to_nodes) → Matrix
```

### Tensor Product Operations
```julia
# Apply 1D matrices along each dimension of a 3D tensor
tensor_product_apply!(out, matrices::NTuple{3,Matrix}, vals)
# Build parent-to-child / child-to-parent transforms
p2c_transform(basis, norder, ndim) → Array
c2p_transform(basis, norder, ndim) → Array
```

### SOG Tables
```julia
load_sog_nodes(kernel, ndim, eps) → SOGNodes
# Files in data/sog/: laplace_3d.jld2, yukawa_3d.jld2, sqrtlaplace_3d.jld2
# Each contains Dict: eps_level => (weights, deltas)
```

### Box FGT
```julia
boxfgt!(pot, tree, fvals, delta, weight, level)
```

### Local Interactions
```julia
build_local_tables(kernel, basis, norder, ndim) → LocalTables
apply_local!(pot, tree, fvals, tables)
```

---

## Threading Strategy

- `Threads.@threads` at the **box loop level** in:
  - Upward pass (independent per-parent operations)
  - Downward pass (independent per-child operations)
  - Local pass (independent per-box operations)
  - Box FGT (independent per-box at each level)
- No threading inside per-box tensor product operations (small matrices)
- Thread-local accumulators for downward pass to avoid write races
- Level-by-level synchronization (barrier between levels)

---

## Dependencies

```toml
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
```

---

## Testing Strategy

Independent tests against analytical solutions:
- **Laplace 3D**: Known Green's function `1/(4πr)`, test with polynomial source densities where exact integrals are computable
- **Yukawa 3D**: Known Green's function `exp(-βr)/(4πr)`, test against quadrature-computed reference
- **Sqrt-Laplace 3D**: Test against reference computations
- **Tree construction**: Verify refinement criterion, level-restricted property, colleague relationships
- **Basis functions**: Verify orthogonality, interpolation accuracy, derivative correctness
- **Tensor products**: Verify against explicit Kronecker product computation
- **Convergence tests**: Verify error decreases with `norder` and `eps`

---

## Fortran → Julia Translation Notes

| Fortran Pattern | Julia Equivalent |
|---|---|
| `ikernel` flag + if/else | Multiple dispatch on `AbstractKernel` subtypes |
| `ipoly` flag | Multiple dispatch on `AbstractBasis` subtypes |
| Packed `itree(ltree)` + `iptr(8)` | `BoxTree` struct with named fields |
| Subroutine with many output arrays | Return `SolverResult` struct |
| `fun` callback with `dpars/zpars/ipars` | Julia closure or callable |
| OpenMP `!$omp parallel do` | `Threads.@threads for` |
| BLAS `dgemm` calls | `mul!` from LinearAlgebra |
| Hardcoded SOG tables in source | JLD2 data files |
| `ifpgh` flag for output level | Keyword arguments `grad=false, hess=false` |
