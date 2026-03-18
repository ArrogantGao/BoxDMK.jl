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

# Adaptive tree (geometry only — no function data)
struct BoxTree{T<:Real, B<:AbstractBasis}
    ndim::Int
    nlevels::Int
    centers::Matrix{T}           # (ndim, nboxes)
    boxsize::Vector{T}           # one per level (0:nlevels)
    parent::Vector{Int}
    children::Matrix{Int}        # (2^ndim, nboxes); 0 = no child
    colleagues::Vector{Vector{Int}}
    level::Vector{Int}
    basis::B
    norder::Int
end

# Interaction lists (distinct from colleagues)
struct InteractionLists
    list1::Vector{Vector{Int}}    # direct (local/near-field) interaction list
    listpw::Vector{Vector{Int}}   # plane wave (far-field) interaction list
end

# SOG data
struct SOGNodes{T<:Real}
    weights::Vector{T}
    deltas::Vector{T}
    r0::T                         # cutoff radius for SOG approximation accuracy
end

# Proxy charge/potential subsystem
struct ProxyData{T<:Real}
    porder::Int                    # proxy polynomial order (depends on eps)
    ncbox::Int                     # porder^ndim
    den2pc_mat::Matrix{T}         # (porder, norder) density-to-proxy-charge
    poteval_mat::Matrix{T}        # (norder, porder) proxy-potential-to-potential
    p2c_transmat::Array{T,4}      # (porder, porder, ndim, 2^ndim)
    c2p_transmat::Array{T,4}      # (porder, porder, ndim, 2^ndim)
end

# Local interaction precomputed data (per-level, per-delta structure)
struct LocalTables{T<:Real}
    tab::Array{T,5}               # (norder, norder, nloctab2, ndeltas, 0:nlevels)
    tabx::Array{T,5}              # gradient tables (same shape)
    tabxx::Array{T,5}             # hessian tables (same shape)
    ind::Array{Int,5}             # (2, norder+1, nloctab2, ndeltas, 0:nlevels) sparse ranges
end

# Plane wave expansion workspace
struct PlaneWaveData{T<:Real}
    rmlexp::Vector{ComplexF64}     # multipole/local expansion storage
    iaddr::Matrix{Int}             # (2, nboxes) pointers into rmlexp
    npw::Vector{Int}               # PW term count per level
    pw_nodes::Vector{Vector{T}}    # PW quadrature nodes per level
    pw_weights::Vector{Vector{T}}  # PW quadrature weights per level
    wpwshift::Vector{Matrix{ComplexF64}}  # PW shift matrices per level
    tab_coefs2pw::Vector{Matrix{ComplexF64}}  # coeffs-to-PW per level
    tab_pw2pot::Vector{Matrix{ComplexF64}}    # PW-to-potential per level
    ifpwexp::Vector{Bool}          # which boxes need PW processing
end

# Solver output
struct SolverResult{T<:Real}
    pot::Array{T,3}                              # (nd, npbox, nboxes)
    grad::Union{Nothing, Array{T,4}}             # (nd, ndim, npbox, nboxes)
    hess::Union{Nothing, Array{T,4}}             # (nd, nhess, npbox, nboxes)
    target_pot::Union{Nothing, Matrix{T}}        # (nd, ntarg)
    target_grad::Union{Nothing, Array{T,3}}      # (nd, ndim, ntarg)
    target_hess::Union{Nothing, Array{T,3}}      # (nd, nhess, ntarg)
end
```

---

## Public API

```julia
# Step 1: Build adaptive tree by sampling user function (fvals returned separately)
tree, fvals = build_tree(f, kernel, basis;
    ndim=3, norder=6, eps=1e-6, boxlen=1.0,
    nd=1, dpars=nothing, eta=1.0)

# Step 2: Evaluate volume potential
result = bdmk(tree, fvals, kernel;
    eps=1e-6, grad=false, hess=false,
    targets=nothing)

# Access
result.pot
result.grad         # nothing if grad=false
result.hess         # nothing if hess=false
result.target_pot   # nothing if targets=nothing
```

Note: `fvals` is separate from `tree` so the same tree can be reused with different densities.

---

## File Layout

```
src/
├── BoxDMK.jl          # Module definition, includes, exports
├── types.jl           # All type definitions above
├── kernels.jl         # Kernel definitions + Taylor correction coefficients
├── basis.jl           # Legendre/Chebyshev: nodes, weights, derivative matrices
├── tree.jl            # Adaptive tree construction + refinement
├── tree_data.jl       # Tree data transforms (Laplacian, biLaplacian, asymptotic)
├── tensor.jl          # Tensor product operations (1D→3D)
├── sog.jl             # SOG table loading + lookup
├── proxy.jl           # Proxy charge/potential setup + porder selection
├── boxfgt.jl          # Box Fast Gauss Transform (multi-delta batched)
├── local.jl           # Near-field local interactions + table construction
├── planewave.jl       # PW expansion: form multipole, shift, convert, evaluate
├── interaction_lists.jl  # Build list1 (direct) and listpw (far-field)
├── solver.jl          # Main bdmk() orchestration (9-step pipeline)
├── derivatives.jl     # Gradient + Hessian computation
└── utils.jl           # Miscellaneous utilities
data/
└── sog/               # SOG node tables as JLD2 files
test/
└── runtests.jl        # Analytical solution tests
```

---

## Algorithm Pipeline (inside `bdmk()`)

The pipeline has 9 steps, matching the Fortran's `bdmk4.f` structure:

### Step 1: Setup / Precomputation
- Load SOG nodes: `load_sog_nodes(kernel, 3, eps)` → `SOGNodes` (weights, deltas, r0)
- Select proxy order `porder` based on `eps` (lookup table, see `bdmk4.f:263-283`)
- Build `ProxyData`: density-to-proxy and proxy-to-potential transformation matrices
- Build proxy-level parent-to-child (`p2c_transmat`) and child-to-parent (`c2p_transmat`) transforms (always Legendre basis for proxy, regardless of user's basis)
- Determine cutoff level `npwlevel` per SOG delta
- Classify deltas into: normal (FGT-handled), fat (root-box-only), and asymptotic (small delta)
- Build interaction lists: `list1` (direct/near-field) and `listpw` (far-field)
- Precompute local interaction tables `LocalTables` (5D, per-level per-delta with sparse indices)
- Set up `PlaneWaveData`: PW term counts, nodes, weights, shift matrices, conversion tables (all complex-valued)

### Step 2: Local Taylor Expansion Corrections
- Compute Taylor coefficients `c(0), c(1), c(2)` from SOG residual (kernel-dependent, dispatched on `AbstractKernel`)
- Compute Laplacian `flvals` and biLaplacian `fl2vals` of density on each leaf box using derivative matrices
- If grad/hess requested: also compute `gvals`, `hvals` (gradient/hessian of density) and their Laplacians `glvals`, `hlvals`
- Apply correction: `pot += c(0)*fvals + c(1)*flvals + c(2)*fl2vals`
- This corrects for the part of the kernel not captured by the SOG decomposition at small `r`

### Step 3: Upward Pass (Density → Proxy Charges)
- Convert density `fvals` to proxy charges via `den2pc_mat` on leaf boxes
- Traverse tree bottom-up: anterpolate children's proxy charges to parent via `c2p_transmat`
- Note: uses child-to-parent (anterpolation), NOT parent-to-child

### Step 4: Form Multipole Plane Wave Expansions
- For boxes that need PW processing (`ifpwexp == true`):
  - Convert proxy charges to multipole PW expansions via `tab_coefs2pw`
  - Store in complex workspace `rmlexp` at locations given by `iaddr`

### Step 5: Multipole-to-Local PW Translation
- For each box pair in `listpw`:
  - Shift multipole PW expansion to local PW expansion via `wpwshift`
  - Multiply by kernel Fourier transform (`dmk_multiply_kernelFT`)

### Step 6: Evaluate Local PW → Proxy Potential + Downward Pass
- Convert local PW expansions to proxy potentials via `tab_pw2pot`
- Traverse tree top-down: interpolate parent's proxy potential to children via `p2c_transmat`
- Note: uses parent-to-child (interpolation) in downward direction

### Step 6b: Fat Gaussian Handling
- For SOG components with `npwlevel < 0` (cutoff below root):
  - Process on root box alone with component-specific PW expansions
  - Separate code path from the main multi-level FGT

### Step 7: Direct (Local Table) Interactions
- For each box and its `list1` neighbors:
  - Apply precomputed `LocalTables` via tensor product evaluation
  - Uses sparse indexing (`ind`) for efficiency when Gaussians are sharply peaked
  - Handles same-level and different-level neighbor boxes

### Step 8: Asymptotic Expansion (Small Deltas)
- For SOG components with very small variance (delta → 0):
  - Use asymptotic formulas: `sqrt(π*δ)` coefficients applied to `fvals`, `flvals`, `fl2vals`
  - Tracked by `idelta` array (count of asymptotic deltas per level)
  - Avoids building expensive local tables for these components

### Step 9: Proxy Potential → Final Output
- Convert proxy potentials back to user grid via `poteval_mat`
- If grad requested: apply derivative matrices, scale by inverse box size
- If hess requested: apply second derivative matrices
- If targets provided: locate containing box, interpolate to target points

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

### Kernel-Specific Taylor Coefficients
```julia
# Dispatch on kernel type — returns (c0, c1, c2) for Taylor correction
taylor_coefficients(::LaplaceKernel, sog::SOGNodes) → NTuple{3,Float64}
taylor_coefficients(::YukawaKernel, sog::SOGNodes) → NTuple{3,Float64}
taylor_coefficients(::SqrtLaplaceKernel, sog::SOGNodes) → NTuple{3,Float64}
```

### Tensor Product Operations
```julia
tensor_product_apply!(out, matrices::NTuple{3,Matrix}, vals)
p2c_transform(basis, norder, ndim) → Array   # parent-to-child interpolation
c2p_transform(basis, norder, ndim) → Array   # child-to-parent anterpolation
```

### Proxy System
```julia
select_porder(eps) → Int                    # lookup table for proxy order
build_proxy_data(basis, norder, porder, ndim) → ProxyData
density_to_proxy!(charge, fvals, proxy::ProxyData)
proxy_to_potential!(pot, proxy_pot, proxy::ProxyData)
```

### SOG Tables
```julia
load_sog_nodes(kernel, ndim, eps) → SOGNodes
# Files in data/sog/: laplace_3d.jld2, yukawa_3d.jld2, sqrtlaplace_3d.jld2
# Each contains Dict: eps_level => (weights, deltas, r0)
```

### Box FGT (Multi-Delta Batched)
```julia
# Accepts arrays of deltas/weights sharing the same cutoff level
boxfgt!(pot, tree, proxy_charges, deltas::Vector, weights::Vector,
        level, pw_data::PlaneWaveData, lists::InteractionLists)
```

### Plane Wave Sub-Operations
```julia
proxycharge_to_pw!(rmlexp, proxy_charges, tab_coefs2pw, iaddr)  # Step 4
shift_pw!(rmlexp_local, rmlexp_mp, wpwshift, listpw)            # Step 5
multiply_kernel_ft!(rmlexp, kernel, pw_nodes)                     # Step 5
pw_to_proxypot!(proxy_pot, rmlexp, tab_pw2pot, iaddr)            # Step 6
```

### Local Interactions
```julia
build_local_tables(kernel, basis, norder, ndim, deltas, boxsizes, nlevels) → LocalTables
apply_local!(pot, tree, fvals, tables::LocalTables, lists::InteractionLists)
```

### Interaction Lists
```julia
build_interaction_lists(tree::BoxTree) → InteractionLists
```

---

## Threading Strategy

- `Threads.@threads` at the **box loop level** in:
  - Upward pass (independent per-parent operations)
  - Downward pass (independent per-child operations)
  - Local pass (independent per-box operations)
  - Box FGT (independent per-box at each level)
  - Taylor corrections (independent per-leaf operations)
- No threading inside per-box tensor product operations (small matrices)
- Thread-local accumulators for downward pass to avoid write races
- Level-by-level synchronization (barrier between levels)

---

## Complex Arithmetic

Plane wave routines use complex arithmetic natively:
- `rmlexp::Vector{ComplexF64}` — PW expansion workspace
- `wpwshift::Matrix{ComplexF64}` — shift matrices
- `tab_coefs2pw`, `tab_pw2pot` — conversion tables
- Julia's native `ComplexF64` replaces Fortran's packed real/imaginary layout

---

## Dependencies

```toml
[deps]
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
JLD2 = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
```

---

## Multiple RHS Support

The `nd` parameter (number of right-hand sides) threads through all data arrays as the leading dimension:
- `fvals`: `(nd, npbox, nboxes)`
- `proxy_charges`: `(ncbox, nd, nboxes)` — note Fortran ordering preserved for BLAS efficiency
- `pot/grad/hess`: `(nd, ...)` leading dimension

All per-box operations handle `nd` RHS simultaneously via matrix operations.

---

## Testing Strategy

Independent tests against analytical solutions:
- **Laplace 3D**: Known Green's function `1/(4πr)`, test with polynomial source densities where exact integrals are computable
- **Yukawa 3D**: Known Green's function `exp(-βr)/(4πr)`, test against quadrature-computed reference
- **Sqrt-Laplace 3D**: Test against reference computations
- **Tree construction**: Verify refinement criterion, level-restricted property, colleague relationships
- **Basis functions**: Verify orthogonality, interpolation accuracy, derivative correctness
- **Tensor products**: Verify against explicit Kronecker product computation
- **Proxy system**: Verify density→proxy→potential roundtrip accuracy
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
| Packed `real*8` for complex PW data | Native `ComplexF64` |
| `porder` lookup table in source | `select_porder(eps)` function |
| `list1`/`listpw` integer arrays | `InteractionLists` struct with `Vector{Vector{Int}}` |
| `proxycharge`/`proxypotential` arrays | `ProxyData` struct + separate working arrays |
