# BoxDMK.jl Fortran-to-Julia Translation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Translate the Fortran BoxDMK volume potential solver into an idiomatic Julia package with type-parameterized kernel dispatch, 3D support for Laplace/Yukawa/SqrtLaplace kernels.

**Architecture:** Type hierarchy with `AbstractKernel` and `AbstractBasis` for dispatch. `BoxTree` struct for geometry, separate `fvals` for density. 9-step solver pipeline: precomputation → Taylor corrections → upward pass (proxy charges) → multipole PW → M2L translation → local PW eval + downward → direct local → asymptotic → output conversion.

**Tech Stack:** Julia 1.10+, LinearAlgebra (stdlib), JLD2 (SOG data files)

**Spec:** `docs/superpowers/specs/2026-03-18-fortran-to-julia-translation-design.md`

**Fortran source:** `/mnt/home/xgao1/codes/boxdmk/`

---

## File Structure

```
src/
├── BoxDMK.jl              # Module: includes, exports, using statements
├── types.jl               # All struct definitions (BoxTree, SOGNodes, ProxyData, etc.)
├── kernels.jl             # Kernel types + taylor_coefficients() dispatch
├── basis.jl               # LegendreBasis/ChebyshevBasis: nodes, weights, derivative matrices
├── tensor.jl              # Tensor product apply, p2c/c2p transform construction
├── tree.jl                # build_tree(): adaptive refinement, colleague lists
├── tree_data.jl           # Laplacian, biLaplacian, asymptotic expansion on tree
├── sog.jl                 # load_sog_nodes(), SOG data file management
├── proxy.jl               # select_porder(), build_proxy_data(), den↔proxy transforms
├── interaction_lists.jl   # build_interaction_lists(): list1, listpw
├── local_tables.jl        # build_local_tables(): 5D sparse tables
├── local.jl               # apply_local!(): direct near-field via sparse tensor products
├── planewave.jl           # PW nodes, shift matrices, conversion tables, kernel FT
├── boxfgt.jl              # boxfgt!(): multi-delta batched Box FGT
├── derivatives.jl         # Gradient + Hessian from polynomial coefficients
├── solver.jl              # bdmk(): 9-step orchestration
└── utils.jl               # Misc helpers
data/
└── sog/                   # JLD2 files with SOG nodes per kernel/precision
test/
├── runtests.jl            # Test runner
├── test_basis.jl          # Legendre/Chebyshev accuracy tests
├── test_tensor.jl         # Tensor product correctness
├── test_tree.jl           # Tree construction + refinement tests
├── test_proxy.jl          # Proxy roundtrip tests
├── test_kernels.jl        # Taylor correction coefficient tests
├── test_tree_data.jl      # Laplacian/biLaplacian/asymptotic tests
├── test_local.jl          # Local table + application tests
├── test_planewave.jl      # PW expansion tests
├── test_solver.jl         # End-to-end solver tests (all 3 kernels)
├── test_sog.jl            # SOG loading tests
└── test_cross_validation.jl  # Cross-validation against Fortran
```

---

## Phase 1: Foundation Types and Polynomial Bases

### Task 1: Types and Module Skeleton

**Files:**
- Create: `src/types.jl`
- Create: `src/utils.jl`
- Modify: `src/BoxDMK.jl`
- Modify: `Project.toml`

**Fortran reference:** Type definitions from spec; `bdmk4.f:263-283` for porder table

- [ ] **Step 1: Create `src/types.jl` with all struct definitions**

```julia
# All types from the design spec:
# AbstractKernel hierarchy (LaplaceKernel, YukawaKernel, SqrtLaplaceKernel)
# AbstractBasis hierarchy (LegendreBasis, ChebyshevBasis)
# BoxTree{T,B} — geometry only, no fvals
# InteractionLists — list1, listpw
# SOGNodes{T} — weights, deltas, r0
# ProxyData{T} — porder, ncbox, transformation matrices
# LocalTables{T} — 5D tab/tabx/tabxx/ind arrays
# PlaneWaveData{T} — complex workspace, PW nodes/weights/shifts/conversions
# SolverResult{T} — pot, grad, hess, target_*
```

- [ ] **Step 2: Create `src/utils.jl` with helper functions**

```julia
# nboxes(tree) — number of boxes
# nleaves(tree) — count leaf boxes
# isleaf(tree, ibox) — check if box is a leaf (all children == 0)
# leaves(tree) — iterator over leaf box indices
# npbox(norder, ndim) — norder^ndim
# nhess(ndim) — ndim*(ndim+1)÷2
```

- [ ] **Step 3: Update `src/BoxDMK.jl` module skeleton**

```julia
module BoxDMK
using LinearAlgebra
using JLD2

include("types.jl")
include("utils.jl")
# ... (other includes added in later tasks)

export LaplaceKernel, YukawaKernel, SqrtLaplaceKernel
export LegendreBasis, ChebyshevBasis
export build_tree, bdmk
end
```

- [ ] **Step 4: Add JLD2 to Project.toml**

```bash
cd /mnt/home/xgao1/codes/BoxDMK.jl && julia -e 'using Pkg; Pkg.add("JLD2")'
```

- [ ] **Step 5: Verify module loads**

```bash
cd /mnt/home/xgao1/codes/BoxDMK.jl && julia -e 'using BoxDMK; println("OK")'
```

- [ ] **Step 6: Commit**

```bash
git add src/types.jl src/utils.jl src/BoxDMK.jl Project.toml Manifest.toml
git commit -m "feat: add type hierarchy and module skeleton"
```

---

### Task 2: Legendre Basis Implementation

**Files:**
- Create: `src/basis.jl`
- Create: `test/test_basis.jl`
- Modify: `src/BoxDMK.jl` (add include)

**Fortran reference:** `src/common/specialfunctions/legeexps.f` — `legeexps`, `legeexps2`, `legepols`

- [ ] **Step 1: Write tests for Legendre basis**

```julia
# test/test_basis.jl
using BoxDMK, Test, LinearAlgebra

@testset "LegendreBasis" begin
    basis = LegendreBasis()
    for n in [4, 8, 16]
        x, w = BoxDMK.nodes_and_weights(basis, n)
        # Test nodes in [-1,1]
        @test all(-1 .< x .< 1)
        # Test weights sum to 2
        @test sum(w) ≈ 2.0 atol=1e-14
        # Test Gauss quadrature exactness for polynomial of degree 2n-1
        @test sum(w .* x.^(2n-1)) ≈ 0.0 atol=1e-12  # odd polynomial
        @test sum(w .* x.^(2n-2)) ≈ 2/(2n-1) atol=1e-12  # even polynomial

        # Test derivative matrix: d/dx(x^k) = k*x^(k-1)
        D = BoxDMK.derivative_matrix(basis, n)
        vals_x2 = x.^2
        dvals = D * vals_x2
        @test dvals ≈ 2 .* x atol=1e-12

        # Test second derivative matrix
        D2 = BoxDMK.second_derivative_matrix(basis, n)
        d2vals = D2 * vals_x2
        @test d2vals ≈ fill(2.0, n) atol=1e-12

        # Test forward/inverse transform roundtrip
        U = BoxDMK.forward_transform(basis, n)  # values → coefficients
        V = BoxDMK.inverse_transform(basis, n)  # coefficients → values
        @test U * V ≈ I atol=1e-13
    end
end
```

- [ ] **Step 2: Run tests — verify they fail**

```bash
cd /mnt/home/xgao1/codes/BoxDMK.jl && julia --project -e 'using Pkg; Pkg.test()'
```

- [ ] **Step 3: Implement `src/basis.jl` for LegendreBasis**

Port from `legeexps.f`. Key functions:
- `nodes_and_weights(::LegendreBasis, n)` — Gauss-Legendre nodes and weights via eigenvalue method or Newton iteration (port `legeexps`)
- `derivative_matrix(::LegendreBasis, n)` — first derivative at nodes (port from `legeexps2`, the `vp` output)
- `second_derivative_matrix(::LegendreBasis, n)` — second derivative at nodes (port `vpp` output)
- `forward_transform(::LegendreBasis, n)` — values→coefficients matrix `u`
- `inverse_transform(::LegendreBasis, n)` — coefficients→values matrix `v`
- `interpolation_matrix(::LegendreBasis, from_nodes, to_nodes)` — via Legendre polynomial evaluation at target nodes

- [ ] **Step 4: Add include to BoxDMK.jl, run tests — verify they pass**

- [ ] **Step 5: Commit**

```bash
git add src/basis.jl test/test_basis.jl src/BoxDMK.jl
git commit -m "feat: implement Legendre basis (nodes, weights, derivatives)"
```

---

### Task 3: Chebyshev Basis Implementation

**Files:**
- Modify: `src/basis.jl`
- Modify: `test/test_basis.jl`

**Fortran reference:** `src/common/specialfunctions/chebexps.f` — `chebexps`, `chebexps2`

- [ ] **Step 1: Add Chebyshev tests to `test/test_basis.jl`**

```julia
@testset "ChebyshevBasis" begin
    basis = ChebyshevBasis()
    for n in [4, 8, 16]
        x, w = BoxDMK.nodes_and_weights(basis, n)
        # Chebyshev nodes: cos((2i-1)π/(2n))
        @test all(-1 .< x .< 1)
        # Weights sum to 2
        @test sum(w) ≈ 2.0 atol=1e-14
        # Derivative and transform tests (same as Legendre)
        D = BoxDMK.derivative_matrix(basis, n)
        vals_x2 = x.^2
        @test D * vals_x2 ≈ 2 .* x atol=1e-12
    end
end
```

- [ ] **Step 2: Implement Chebyshev methods in `src/basis.jl`**

Port from `chebexps.f`:
- Nodes: `cos((2i-1)π/(2n))` sorted ascending
- Weights: Clenshaw-Curtis or via DCT
- Transforms: `chebexps` u/v matrices
- Derivatives: `chebexps2` vp/vpp matrices

- [ ] **Step 3: Run tests — verify they pass**

- [ ] **Step 4: Commit**

```bash
git add src/basis.jl test/test_basis.jl
git commit -m "feat: implement Chebyshev basis"
```

---

### Task 4: Tensor Product Operations

**Files:**
- Create: `src/tensor.jl`
- Create: `test/test_tensor.jl`
- Modify: `src/BoxDMK.jl`

**Fortran reference:** `src/common/tensor_prod_routs.f` — `tens_prod_trans`, `ortho_evalg_nd`, `ortho_eval_laplacian_nd`

- [ ] **Step 1: Write tensor product tests**

```julia
# test/test_tensor.jl
@testset "Tensor Products 3D" begin
    n = 4; ndim = 3; nd = 1
    # Test: applying identity in each dim preserves values
    I_mat = Matrix{Float64}(I, n, n)
    vals = rand(nd, n^3)
    out = similar(vals)
    BoxDMK.tensor_product_apply!(out, (I_mat, I_mat, I_mat), vals, n, ndim, nd)
    @test out ≈ vals

    # Test: tensor product of 1D matrices = Kronecker product
    A = rand(n, n); B = rand(n, n); C = rand(n, n)
    K = kron(C, kron(B, A))  # Kronecker product
    BoxDMK.tensor_product_apply!(out, (A, B, C), vals, n, ndim, nd)
    @test out ≈ vals * K' atol=1e-12  # verify against explicit Kronecker
end
```

- [ ] **Step 2: Implement `src/tensor.jl`**

Port from `tens_prod_trans`. Key: apply 1D matrix sequentially along each dimension using reshape + `mul!`:
```julia
function tensor_product_apply!(out, mats::NTuple{N,Matrix}, vals, n, ndim, nd) where N
    # For 3D: Apply mat[1] along dim 1, mat[2] along dim 2, mat[3] along dim 3
    # Use workspace to avoid allocation
    # Reshape vals as (nd, n, n, n), apply each matrix in sequence
end
```

Also implement:
- `p2c_transform(basis, norder, ndim)` — parent-to-child interpolation matrices (one set per child octant). Port from `dmk_get_coefs_translation_matrices` in `dmk_routs.f`. The 1D interpolation maps [-1,1] → [-1,0] or [0,1] for each child.
- `c2p_transform(basis, norder, ndim)` — child-to-parent anterpolation (transpose of p2c weighted by quadrature)

- [ ] **Step 3: Run tests — verify pass**

- [ ] **Step 4: Commit**

```bash
git add src/tensor.jl test/test_tensor.jl src/BoxDMK.jl
git commit -m "feat: implement tensor product operations and p2c/c2p transforms"
```

---

## Phase 2: Tree Construction

### Task 5: Adaptive Tree Construction

**Files:**
- Create: `src/tree.jl`
- Create: `test/test_tree.jl`
- Modify: `src/BoxDMK.jl`

**Fortran reference:** `src/common/tree_vol_coeffs.f` — `vol_tree_mem`, `vol_tree_build`

- [ ] **Step 1: Write tree construction tests**

```julia
# test/test_tree.jl
@testset "Tree Construction" begin
    # Smooth function → few boxes
    f_smooth(x) = [sin(π * x[1]) * cos(π * x[2]) * exp(x[3])]
    tree, fvals = build_tree(f_smooth, LaplaceKernel(), LegendreBasis();
        ndim=3, norder=6, eps=1e-6, boxlen=1.0, nd=1)
    @test tree.nlevels >= 0
    @test size(tree.centers, 2) > 0
    @test all(tree.level .>= 0)

    # Sharp Gaussian → deep tree
    f_sharp(x) = [exp(-1000 * sum((x .- 0.5).^2))]
    tree2, fvals2 = build_tree(f_sharp, LaplaceKernel(), LegendreBasis();
        ndim=3, norder=6, eps=1e-6, boxlen=1.0, nd=1)
    @test tree2.nlevels > tree.nlevels  # should refine more

    # Level-restricted: no more than 1 level difference between neighbors
    for ibox in 1:size(tree2.centers, 2)
        for jbox in tree2.colleagues[ibox]
            @test abs(tree2.level[ibox] - tree2.level[jbox]) <= 1
        end
    end
end
```

- [ ] **Step 2: Implement `src/tree.jl`**

Port from `tree_vol_coeffs.f`. Algorithm:
1. Start with root box covering `[0, boxlen]^3`
2. Sample `f` on tensor grid (norder^3 points) per box
3. Check refinement criterion: compare interpolant error vs `eps`
   - Criterion: `‖ũ(f) - f‖_p * h^η < ε` (relative to global norm)
4. If error exceeds threshold: split into 8 children, recurse
5. Enforce level-restriction: if neighbors differ by >1 level, refine
6. Build colleague lists after refinement complete

Return `(BoxTree, fvals)` where `fvals` is `(nd, npbox, nboxes)`.

- [ ] **Step 3: Run tests — verify pass**

- [ ] **Step 4: Commit**

```bash
git add src/tree.jl test/test_tree.jl src/BoxDMK.jl
git commit -m "feat: implement adaptive tree construction with refinement"
```

---

### Task 6: Interaction List Construction

**Files:**
- Create: `src/interaction_lists.jl`
- Modify: `test/test_tree.jl`
- Modify: `src/BoxDMK.jl`

**Fortran reference:** `src/bdmk/boxfgt_md.f` — interaction list construction logic

- [ ] **Step 1: Add interaction list tests**

```julia
@testset "Interaction Lists" begin
    f(x) = [exp(-100 * sum((x .- 0.5).^2))]
    tree, _ = build_tree(f, LaplaceKernel(), LegendreBasis();
        ndim=3, norder=6, eps=1e-6, boxlen=1.0, nd=1)
    lists = BoxDMK.build_interaction_lists(tree)

    # list1 and listpw should cover all box pairs
    # list1 entries should be near-field (adjacent or same-level)
    # listpw entries should be well-separated
    nboxes = size(tree.centers, 2)
    @test length(lists.list1) == nboxes
    @test length(lists.listpw) == nboxes
end
```

- [ ] **Step 2: Implement `src/interaction_lists.jl`**

Port the list construction from `boxfgt_md.f`:
- `list1`: boxes in near-field (colleagues + their children for different-level interactions)
- `listpw`: boxes in plane-wave interaction range (well-separated, same level)

- [ ] **Step 3: Run tests, commit**

```bash
git add src/interaction_lists.jl test/test_tree.jl src/BoxDMK.jl
git commit -m "feat: implement interaction list construction (list1, listpw)"
```

---

## Phase 3: SOG Data and Proxy System

### Task 7: SOG Table Loading

**Files:**
- Create: `src/sog.jl`
- Create: `data/sog/` directory with JLD2 files
- Create: `test/test_sog.jl`
- Modify: `src/BoxDMK.jl`

**Fortran reference:** `src/bdmk/sogapproximation/get_sognodes.f`, `l3dsognodes.f`, `y3dsognodes.f`, `sl3dsognodes.f`

- [ ] **Step 1: Extract SOG tables from Fortran source into JLD2 files**

Write a one-time script that reads the hardcoded SOG node data from `l3dsognodes.f`, `y3dsognodes.f`, `sl3dsognodes.f` and saves as JLD2:
```julia
# Each file contains: Dict mapping eps_level => (weights::Vector, deltas::Vector, r0::Float64)
# Keys: "1e-2", "1e-3", ..., "1e-12"
```

- [ ] **Step 2: Write SOG loading tests**

```julia
@testset "SOG Loading" begin
    for kernel in [LaplaceKernel(), YukawaKernel(1.0), SqrtLaplaceKernel()]
        sog = BoxDMK.load_sog_nodes(kernel, 3, 1e-6)
        @test length(sog.weights) == length(sog.deltas)
        @test length(sog.weights) > 0
        @test all(sog.deltas .> 0)
        @test sog.r0 > 0
    end
end
```

- [ ] **Step 3: Implement `src/sog.jl`**

```julia
function load_sog_nodes(kernel::AbstractKernel, ndim::Int, eps::Float64) → SOGNodes
    # Determine file path based on kernel type
    # Load from JLD2
    # Select closest precision level
    # Return SOGNodes(weights, deltas, r0)
end
```

Port the scaling logic from `get_sognodes.f`:
- For `ikernel > 0` (Laplace, SqrtLaplace): scale `ts(i) *= bsize²`

- [ ] **Step 4: Run tests, commit**

```bash
git add src/sog.jl data/sog/ test/test_sog.jl src/BoxDMK.jl
git commit -m "feat: implement SOG table loading from JLD2 files"
```

---

### Task 8: Proxy System

**Files:**
- Create: `src/proxy.jl`
- Create: `test/test_proxy.jl`
- Modify: `src/BoxDMK.jl`

**Fortran reference:** `src/common/dmk_routs.f` — `dmk_get_coefs_translation_matrices`; `bdmk4.f:263-283` for porder table

- [ ] **Step 1: Write proxy system tests**

```julia
@testset "Proxy System" begin
    # Test porder selection
    @test BoxDMK.select_porder(1e-3) == 16
    @test BoxDMK.select_porder(1e-6) == 30
    @test BoxDMK.select_porder(1e-12) == 62

    # Test density → proxy → potential roundtrip
    basis = LegendreBasis()
    norder = 6; porder = 16; ndim = 3; nd = 1
    proxy = BoxDMK.build_proxy_data(basis, norder, porder, ndim)

    # Smooth polynomial on one box should roundtrip accurately
    npbox = norder^ndim
    fvals = rand(nd, npbox)
    charge = zeros(proxy.ncbox, nd)
    BoxDMK.density_to_proxy!(charge, fvals, proxy)
    pot_back = zeros(nd, npbox)
    BoxDMK.proxy_to_potential!(pot_back, charge, proxy)
    # Not exact roundtrip, but should preserve low-order content
    @test norm(pot_back - fvals) / norm(fvals) < 0.1  # coarse check
end
```

- [ ] **Step 2: Implement `src/proxy.jl`**

```julia
function select_porder(eps::Float64) → Int
    # Lookup table from bdmk4.f:263-283
    eps >= 0.8e-3  && return 16
    eps >= 0.8e-4  && return 22
    eps >= 0.8e-5  && return 26
    eps >= 0.8e-6  && return 30
    eps >= 0.8e-7  && return 36
    eps >= 0.8e-8  && return 42
    eps >= 0.8e-9  && return 46
    eps >= 0.8e-10 && return 50
    eps >= 0.8e-11 && return 56
    return 62
end

function build_proxy_data(basis, norder, porder, ndim) → ProxyData
    # Proxy always uses Legendre basis regardless of user's choice
    # den2pc_mat: interpolation from user nodes (norder) to proxy nodes (porder)
    # poteval_mat: interpolation from proxy nodes (porder) to user nodes (norder)
    # p2c_transmat / c2p_transmat: proxy-level parent↔child transforms
end

function density_to_proxy!(charge, fvals, proxy::ProxyData)
    # charge = den2pc_mat * fvals (tensor product in 3D)
end

function proxy_to_potential!(pot, proxy_pot, proxy::ProxyData)
    # pot = poteval_mat * proxy_pot (tensor product in 3D)
end
```

- [ ] **Step 3: Run tests, commit**

```bash
git add src/proxy.jl test/test_proxy.jl src/BoxDMK.jl
git commit -m "feat: implement proxy charge/potential system"
```

---

## Phase 4: Kernel-Specific Logic and Tree Data Transforms

### Task 9: Kernel Taylor Corrections

**Files:**
- Create: `src/kernels.jl`
- Create: `test/test_kernels.jl`
- Modify: `src/BoxDMK.jl`

**Fortran reference:** `bdmk4.f` Step 2 Taylor correction logic

**Dependencies:** Task 1 (types), Task 7 (SOG loading for SOGNodes)

- [ ] **Step 1: Write Taylor coefficient tests**

```julia
# test/test_kernels.jl
using BoxDMK, Test

@testset "Taylor Coefficients" begin
    # For each kernel, verify coefficients are finite and have expected signs
    for kernel in [LaplaceKernel(), YukawaKernel(1.0), SqrtLaplaceKernel()]
        sog = BoxDMK.load_sog_nodes(kernel, 3, 1e-6)
        c0, c1, c2 = BoxDMK.taylor_coefficients(kernel, sog)
        @test isfinite(c0)
        @test isfinite(c1)
        @test isfinite(c2)
        # c0 should be close to 0 (SOG approximation residual)
        @test abs(c0) < 1.0
    end

    # Taylor correction for gradient/hessian density terms
    for kernel in [LaplaceKernel(), YukawaKernel(1.0), SqrtLaplaceKernel()]
        sog = BoxDMK.load_sog_nodes(kernel, 3, 1e-6)
        gc0, gc1 = BoxDMK.taylor_coefficients_grad(kernel, sog)
        @test isfinite(gc0) && isfinite(gc1)
    end
end
```

- [ ] **Step 2: Implement `src/kernels.jl`**

```julia
# Taylor coefficients c(0), c(1), c(2) for potential correction:
# pot += c(0)*f + c(1)*lap(f) + c(2)*bilap(f)
# These depend on kernel type and SOG residual

function taylor_coefficients(::LaplaceKernel, sog::SOGNodes)
    # c(0) = 1 - sum(ws * sqrt(pi*deltas))  (residual of SOG for r→0)
    # c(1), c(2) from Taylor expansion of kernel - SOG
end

function taylor_coefficients(::YukawaKernel, sog::SOGNodes)
    # Similar but includes beta-dependent terms
end

function taylor_coefficients(::SqrtLaplaceKernel, sog::SOGNodes)
    # Kernel-specific coefficients
end

# Gradient/Hessian density Taylor corrections (spec Step 2):
# When grad/hess requested, also need corrections for gradient/hessian of density
function taylor_coefficients_grad(kernel::AbstractKernel, sog::SOGNodes)
    # Returns (gc0, gc1) for: grad_correction += gc0*gvals + gc1*glvals
end

function taylor_coefficients_hess(kernel::AbstractKernel, sog::SOGNodes)
    # Returns (hc0, hc1) for: hess_correction += hc0*hvals + hc1*hlvals
end
```

- [ ] **Step 3: Run tests, commit**

```bash
git add src/kernels.jl test/test_kernels.jl src/BoxDMK.jl
git commit -m "feat: implement kernel Taylor correction coefficients"
```

---

### Task 10: Tree Data Transforms (Laplacian, BiLaplacian, Asymptotic)

**Files:**
- Create: `src/tree_data.jl`
- Create: `test/test_tree_data.jl`
- Modify: `src/BoxDMK.jl`

**Fortran reference:** `src/common/tree_data_routs.f` — `treedata_eval_laplacian_nd`, `treedata_eval_pot_nd_asym`; `src/common/tensor_prod_routs.f` — `ortho_eval_laplacian_nd`

**Dependencies:** Task 2 (basis — derivative matrices), Task 4 (tensor products)

- [ ] **Step 1: Write Laplacian/biLaplacian tests**

```julia
# test/test_tree_data.jl
using BoxDMK, Test

@testset "Tree Data Transforms" begin
    # Test Laplacian of known polynomial: f(x,y,z) = x² + y² + z²
    # Laplacian = 6 (constant)
    f_quad(x) = [sum(x.^2)]
    tree, fvals = build_tree(f_quad, LaplaceKernel(), LegendreBasis();
        ndim=3, norder=6, eps=1e-6, boxlen=1.0, nd=1)
    flvals = similar(fvals)
    BoxDMK.compute_laplacian!(flvals, tree, fvals, LegendreBasis())
    # On each box, Laplacian of x²+y²+z² should be ≈ 6
    for ibox in BoxDMK.leaves(tree)
        @test all(abs.(flvals[1, :, ibox] .- 6.0) .< 1e-10)
    end

    # Test biLaplacian of quartic: f = x⁴ → Lap = 12x² → BiLap = 24
    # (simplified: use sum of quartics)

    # Test asymptotic expansion: for very small delta, result should
    # approximate weight * sqrt(π*δ)^3 * f to leading order
end
```

- [ ] **Step 2: Implement `src/tree_data.jl`**

```julia
function compute_laplacian!(flvals, tree, fvals, basis)
    # For each leaf box: compute Laplacian = Σ_d d²f/dx_d²
    # Uses second_derivative_matrix and tensor product
    # Scale by (2/boxsize)^2 for each dimension
end

function compute_bilaplacian!(fl2vals, tree, fvals, flvals, basis)
    # BiLaplacian = Laplacian of Laplacian
    # Apply Laplacian operator to flvals
end

function compute_gradient_density!(gvals, tree, fvals, basis)
    # Gradient of density (needed for Taylor correction of grad output)
end

function compute_hessian_density!(hvals, tree, fvals, basis)
    # Hessian of density (needed for Taylor correction of hess output)
end

function eval_asymptotic!(pot, tree, fvals, flvals, fl2vals, delta, weight)
    # Asymptotic formula for small delta:
    # pot += weight * sqrt(π*δ)^ndim * (fvals + δ/4 * flvals + δ²/32 * fl2vals)
end

function apply_asymptotic!(pot, tree, fvals, flvals, fl2vals, sog)
    # Loop over SOG components, identify asymptotic ones (small delta per level),
    # call eval_asymptotic! for each
    # Uses idelta array: count of asymptotic deltas per level
end
```

- [ ] **Step 3: Run tests, commit**

```bash
git add src/tree_data.jl test/test_tree_data.jl src/BoxDMK.jl
git commit -m "feat: implement tree data transforms (Laplacian, biLaplacian, asymptotic)"
```

---

## Phase 5: Local Interactions

### Task 11: Local Table Construction

**Files:**
- Create: `src/local_tables.jl`
- Create: `test/test_local.jl`
- Modify: `src/BoxDMK.jl`

**Fortran reference:** `src/bdmk/bdmk_local_tables.f` — `mk_loctab_all`, `mk_loctab`

- [ ] **Step 1: Write local table tests**

```julia
@testset "Local Tables" begin
    basis = LegendreBasis()
    norder = 6; ndim = 3; nlevels = 3
    deltas = [0.01, 0.001]
    boxsizes = [1.0, 0.5, 0.25, 0.125]
    tables = BoxDMK.build_local_tables(LaplaceKernel(), basis, norder, ndim, deltas, boxsizes, nlevels)

    # Tables should have correct dimensions
    @test ndims(tables.tab) == 5
    @test size(tables.tab, 1) == norder
    @test size(tables.tab, 2) == norder
end
```

- [ ] **Step 2: Implement `src/local_tables.jl`**

Port from `bdmk_local_tables.f`:
- 1D table construction via numerical quadrature (nquad=50)
- `tab_loc(m, j, k, id, ilev)` = integral of `P_m(y) * exp(-(ξ_j - y)²/delta)` over source box
- Sparse index computation: find first/last nonzero columns per row
- Gradient tables (`tabx`) and Hessian tables (`tabxx`) via derivative of kernel

- [ ] **Step 3: Run tests, commit**

```bash
git add src/local_tables.jl test/test_local.jl src/BoxDMK.jl
git commit -m "feat: implement local interaction table construction"
```

---

### Task 12: Local Interaction Application

**Files:**
- Create: `src/local.jl`
- Modify: `test/test_local.jl`
- Modify: `src/BoxDMK.jl`

**Fortran reference:** `src/bdmk/bdmk_local.f` — `bdmk_tens_prod_to_potloc`

- [ ] **Step 1: Add application tests**

```julia
@testset "Local Application" begin
    # Test: apply local tables to known density, compare with direct quadrature
    # Use a simple Gaussian source and verify against direct integration
end
```

- [ ] **Step 2: Implement `src/local.jl`**

Port from `bdmk_local.f`:
```julia
function apply_local!(pot, tree, fvals, tables::LocalTables, lists::InteractionLists)
    # For each box and its list1 neighbors:
    #   Determine table offset index from box positions
    #   Apply 3D tensor product using sparse tables
    #   Use sparse indexing: only multiply non-zero ranges
    Threads.@threads for ibox in leaves(tree)
        for jbox in lists.list1[ibox]
            # Compute offset, look up table, apply via tensor product
        end
    end
end
```

- [ ] **Step 3: Run tests, commit**

```bash
git add src/local.jl test/test_local.jl src/BoxDMK.jl
git commit -m "feat: implement local interaction application with sparse tensor products"
```

---

## Phase 6: Plane Wave Expansions

### Task 13: Plane Wave Setup and Conversion Tables

**Files:**
- Create: `src/planewave.jl`
- Create: `test/test_planewave.jl`
- Modify: `src/BoxDMK.jl`

**Fortran reference:** `src/common/dmk_routs.f` — `dmk_mk_coefs_pw_conversion_tables`; `src/bdmk/bdmk_pwrouts.f` — `mk_kernel_Fourier_transform`; `src/bdmk/bdmk_pwterms.f` — PW term count tables

- [ ] **Step 1: Write PW setup tests**

```julia
@testset "Plane Wave Setup" begin
    # Test PW node generation
    npw = BoxDMK.get_pw_term_count(1e-6, 0)  # level 0
    @test npw > 0

    # Test conversion table dimensions
    porder = 16
    tab_c2pw, tab_pw2c = BoxDMK.build_pw_conversion_tables(
        LegendreBasis(), porder, npw, 1.0)
    @test size(tab_c2pw) == (npw, porder)
    @test size(tab_pw2c) == (npw, porder)
end
```

- [ ] **Step 2: Implement `src/planewave.jl`**

Port from multiple Fortran files:
- `get_pw_term_count(eps, level)` — from `bdmk_pwterms.f` lookup tables
- `get_pw_nodes(eps, level)` — PW quadrature nodes and weights
- `build_pw_conversion_tables(basis, porder, npw, boxdim)` — `tab_coefs2pw` and `tab_pw2pot` (complex matrices)
- `build_pw_shift_matrices(npw, boxdim)` — shift matrices for M2L translation
- `kernel_fourier_transform!(wpwexp, kernel, deltas, weights, pw_nodes)` — from `mk_kernel_Fourier_transform`
- `setup_planewave_data(tree, proxy, kernel, sog, eps)` → `PlaneWaveData`

All PW arrays use `ComplexF64` natively.

- [ ] **Step 3: Run tests, commit**

```bash
git add src/planewave.jl test/test_planewave.jl src/BoxDMK.jl
git commit -m "feat: implement plane wave setup, conversion tables, kernel FT"
```

---

### Task 14: Box FGT (Multi-Delta Batched)

**Files:**
- Create: `src/boxfgt.jl`
- Modify: `src/BoxDMK.jl`

**Fortran reference:** `src/bdmk/boxfgt_md.f` — full Box FGT algorithm

- [ ] **Step 1: Implement `src/boxfgt.jl`**

Port from `boxfgt_md.f`. The Box FGT processes multiple deltas sharing the same cutoff level:

```julia
function boxfgt!(pot, tree, proxy_charges, deltas, weights, level,
                 pw_data::PlaneWaveData, lists::InteractionLists, proxy::ProxyData)
    # Step 4: proxy charges → multipole PW expansions
    proxycharge_to_pw!(pw_data, proxy_charges, level)

    # Step 5: M2L translation via shift + kernel FT multiply
    shift_and_translate_pw!(pw_data, lists, level)

    # Step 6: local PW → proxy potential
    pw_to_proxypot!(pot, pw_data, level)
end
```

Sub-operations:
```julia
function proxycharge_to_pw!(pw_data, charges, level)
    # rmlexp[iaddr[1,ibox]:...] = tab_coefs2pw * charges[:,:,ibox]
    Threads.@threads for ibox in boxes_at_level(level)
        # Complex matrix-vector multiply
    end
end

function shift_and_translate_pw!(pw_data, lists, level)
    # For each (ibox, jbox) in listpw at this level:
    #   rmlexp_local[ibox] += wpwshift * rmlexp_mp[jbox]
    Threads.@threads for ibox in boxes_at_level(level)
        for jbox in lists.listpw[ibox]
            # shift + accumulate
        end
    end
end

function pw_to_proxypot!(pot, pw_data, level)
    # proxy_pot[:,:,ibox] = tab_pw2pot * rmlexp_local[ibox]
end
```

- [ ] **Step 2: Commit**

```bash
git add src/boxfgt.jl src/BoxDMK.jl
git commit -m "feat: implement Box FGT with multi-delta batching"
```

---

## Phase 7: Upward/Downward Passes, Derivatives, and Solver Assembly

### Task 15: Upward and Downward Passes

**Files:**
- Create: `src/passes.jl`
- Create: `test/test_passes.jl`
- Modify: `src/BoxDMK.jl`

**Fortran reference:** `bdmk4.f` Steps 3 and 6 (proxy charge anterpolation / proxy potential interpolation)

**Dependencies:** Task 4 (tensor products), Task 8 (proxy system), Task 5 (tree)

- [ ] **Step 1: Write upward/downward pass tests**

```julia
# test/test_passes.jl
using BoxDMK, Test

@testset "Upward/Downward Passes" begin
    # Test: smooth polynomial proxy charges should anterpolate cleanly
    f(x) = [sin(π * x[1])]
    tree, fvals = build_tree(f, LaplaceKernel(), LegendreBasis();
        ndim=3, norder=6, eps=1e-6, boxlen=1.0, nd=1)
    proxy = BoxDMK.build_proxy_data(LegendreBasis(), 6, 16, 3)
    nboxes_val = size(tree.centers, 2)
    nd = 1; ncbox = proxy.ncbox

    proxy_charges = zeros(ncbox, nd, nboxes_val)
    BoxDMK.density_to_proxy!(proxy_charges, fvals, proxy)
    charges_before = copy(proxy_charges)
    BoxDMK.upward_pass!(proxy_charges, tree, proxy)

    # After upward pass, root should have accumulated content
    @test norm(proxy_charges[:, :, 1]) > 0

    # Downward pass should distribute back
    proxy_pot = copy(proxy_charges)
    BoxDMK.downward_pass!(proxy_pot, tree, proxy)
    # Leaf boxes should have content after downward pass
end
```

- [ ] **Step 2: Implement `src/passes.jl`**

```julia
function upward_pass!(proxy_charges, tree, proxy)
    # Bottom-up: for each level from nlevels-1 down to 0
    for ilev in tree.nlevels-1:-1:0
        Threads.@threads for ibox in boxes_at_level(tree, ilev)
            if !isleaf(tree, ibox)
                for ic in 1:8
                    ichild = tree.children[ic, ibox]
                    ichild == 0 && continue
                    # c2p_transmat tensor product: anterpolate child → parent
                end
            end
        end
    end
end

function downward_pass!(proxy_pot, tree, proxy)
    # Top-down: for each level from 1 to nlevels
    for ilev in 1:tree.nlevels
        Threads.@threads for ibox in boxes_at_level(tree, ilev)
            if !isleaf(tree, ibox)
                for ic in 1:8
                    ichild = tree.children[ic, ibox]
                    ichild == 0 && continue
                    # p2c_transmat tensor product: interpolate parent → child, accumulate
                end
            end
        end
    end
end
```

- [ ] **Step 3: Run tests, commit**

```bash
git add src/passes.jl test/test_passes.jl src/BoxDMK.jl
git commit -m "feat: implement upward/downward passes for proxy charges/potentials"
```

---

### Task 15b: Delta Classification and Fat Gaussian Handling

**Files:**
- Modify: `src/sog.jl` (add `group_deltas_by_level`)
- Modify: `src/boxfgt.jl` (add `handle_fat_gaussian!`)
- Modify: `src/tree_data.jl` (add `evaluate_at_targets`)
- Modify: `src/BoxDMK.jl`

**Fortran reference:** `bdmk4.f:957-1001` (fat Gaussians), `bdmk4.f:1544-1574` (npwlevel/cutoff)

**Dependencies:** Task 7 (SOG), Task 14 (Box FGT)

- [ ] **Step 1: Implement delta classification in `src/sog.jl`**

```julia
struct DeltaGroups
    normal::Vector{Tuple{Int, Vector{Float64}, Vector{Float64}}}  # (level, deltas, weights)
    fat::Vector{Tuple{Float64, Float64}}         # (delta, weight) with npwlevel < 0
    asymptotic::Vector{Tuple{Int, Float64, Float64}}  # (level, delta, weight) for small delta
end

function group_deltas_by_level(sog::SOGNodes, tree::BoxTree)
    # For each SOG component:
    #   dcutoff = sqrt(delta * ln(1/eps))
    #   Find npwlevel: lowest level where boxsize >= dcutoff
    #   If npwlevel < 0: fat Gaussian
    #   If delta very small (asymptotic criterion): asymptotic
    #   Otherwise: normal, group by npwlevel
    return DeltaGroups(normal, fat, asymptotic)
end
```

- [ ] **Step 2: Implement fat Gaussian handling in `src/boxfgt.jl`**

```julia
function handle_fat_gaussian!(proxy_pot, tree, proxy_charges, delta, weight, pw_data)
    # Process on root box (level 0) alone
    # Single-component PW expansion at root level
end
```

- [ ] **Step 3: Implement target evaluation in `src/tree_data.jl`**

```julia
function evaluate_at_targets(pot, tree, targets, basis)
    # For each target point:
    #   Find containing leaf box
    #   Interpolate potential from box grid to target location
    ntarg = size(targets, 2)
    target_pot = zeros(size(pot, 1), ntarg)
    for itarg in 1:ntarg
        ibox = find_containing_box(tree, targets[:, itarg])
        # Build interpolation weights, apply
    end
    return target_pot
end
```

- [ ] **Step 4: Commit**

```bash
git add src/sog.jl src/boxfgt.jl src/tree_data.jl src/BoxDMK.jl
git commit -m "feat: implement delta classification, fat Gaussian handling, target evaluation"
```

---

### Task 16: Gradient and Hessian Computation

**Files:**
- Create: `src/derivatives.jl`
- Modify: `src/BoxDMK.jl`

**Fortran reference:** `src/common/tensor_prod_routs.f` — `ortho_evalg_nd`, `ortho_evalgh_nd`

**Dependencies:** Task 2 (basis), Task 4 (tensor products)

- [ ] **Step 1: Implement `src/derivatives.jl`**

```julia
function compute_gradient!(grad, pot_coeffs, tree, basis)
    # grad(nd, 3, npbox, nboxes)
    # For each box: apply derivative matrix along each dimension
    # Scale by 2/boxsize[level[ibox]] per dimension
    Threads.@threads for ibox in 1:nboxes(tree)
        sc = 2.0 / tree.boxsize[tree.level[ibox] + 1]
        # grad[:, d, :, ibox] = tensor_product with D in dim d, I in others
    end
end

function compute_hessian!(hess, pot_coeffs, tree, basis)
    # hess(nd, 6, npbox, nboxes) for 3D: xx, yy, zz, xy, xz, yz
    # Apply second derivative matrices and mixed derivative products
end
```

- [ ] **Step 2: Commit**

```bash
git add src/derivatives.jl src/BoxDMK.jl
git commit -m "feat: implement gradient and Hessian computation"
```

---

### Task 17: Main Solver Assembly

**Files:**
- Create: `src/solver.jl`
- Create: `test/test_solver.jl`
- Modify: `src/BoxDMK.jl`

**Fortran reference:** `bdmk4.f` — full 9-step pipeline

- [ ] **Step 1: Write end-to-end solver tests**

```julia
@testset "Solver - Laplace 3D" begin
    # Test with known Gaussian source: f(x) = exp(-α|x-x₀|²)
    # Exact potential for Laplace kernel convolution with Gaussian is known
    α = 100.0; x0 = [0.5, 0.5, 0.5]
    f(x) = [exp(-α * sum((x .- x0).^2))]

    tree, fvals = build_tree(f, LaplaceKernel(), LegendreBasis();
        ndim=3, norder=8, eps=1e-6, boxlen=1.0, nd=1)

    result = bdmk(tree, fvals, LaplaceKernel(); eps=1e-6)

    # Verify potential is non-trivial
    @test maximum(abs.(result.pot)) > 0

    # Compare against direct quadrature at a few sample points
    # (exact solution for Laplace * Gaussian is erf-based)
end

@testset "Solver - Yukawa 3D" begin
    β = 1.0; α = 100.0; x0 = [0.5, 0.5, 0.5]
    f(x) = [exp(-α * sum((x .- x0).^2))]

    tree, fvals = build_tree(f, YukawaKernel(β), LegendreBasis();
        ndim=3, norder=8, eps=1e-6, boxlen=1.0, nd=1)

    result = bdmk(tree, fvals, YukawaKernel(β); eps=1e-6)
    @test maximum(abs.(result.pot)) > 0
end

@testset "Solver - SqrtLaplace 3D" begin
    α = 100.0; x0 = [0.5, 0.5, 0.5]
    f(x) = [exp(-α * sum((x .- x0).^2))]

    tree, fvals = build_tree(f, SqrtLaplaceKernel(), LegendreBasis();
        ndim=3, norder=8, eps=1e-6, boxlen=1.0, nd=1)

    result = bdmk(tree, fvals, SqrtLaplaceKernel(); eps=1e-6)
    @test maximum(abs.(result.pot)) > 0
end

@testset "Solver - Convergence" begin
    # Verify error decreases with eps
    f(x) = [exp(-100 * sum((x .- 0.5).^2))]
    errors = Float64[]
    for eps in [1e-3, 1e-6, 1e-9]
        tree, fvals = build_tree(f, LaplaceKernel(), LegendreBasis();
            ndim=3, norder=8, eps=eps, boxlen=1.0, nd=1)
        result = bdmk(tree, fvals, LaplaceKernel(); eps=eps)
        # Compare against high-accuracy reference
        push!(errors, compute_error(result, reference))
    end
    # Errors should decrease
    @test errors[2] < errors[1]
    @test errors[3] < errors[2]
end
```

- [ ] **Step 2: Implement `src/solver.jl` — the 9-step `bdmk()` function**

```julia
function bdmk(tree::BoxTree, fvals::Array, kernel::AbstractKernel;
              eps=1e-6, grad=false, hess=false, targets=nothing)

    ndim = tree.ndim
    norder = tree.norder
    basis = tree.basis
    nd = size(fvals, 1)
    nboxes_val = size(fvals, 3)
    npbox = norder^ndim

    # === Step 1: Precomputation ===
    sog = load_sog_nodes(kernel, ndim, eps)
    porder = select_porder(eps)
    proxy = build_proxy_data(basis, norder, porder, ndim)
    lists = build_interaction_lists(tree)

    # Classify deltas: normal, fat, asymptotic
    # Build local tables, PW data
    local_tabs = build_local_tables(kernel, basis, norder, ndim, sog.deltas, tree.boxsize, tree.nlevels)
    pw_data = setup_planewave_data(tree, proxy, kernel, sog, eps)

    # Allocate output
    pot = zeros(nd, npbox, nboxes_val)

    # === Step 2: Taylor corrections ===
    c0, c1, c2 = taylor_coefficients(kernel, sog)
    flvals = similar(fvals)
    compute_laplacian!(flvals, tree, fvals, basis)
    fl2vals = similar(fvals)
    compute_bilaplacian!(fl2vals, tree, fvals, flvals, basis)
    @. pot += c0 * fvals + c1 * flvals + c2 * fl2vals

    # Gradient/Hessian density Taylor corrections (if requested)
    if grad
        gvals = zeros(nd, ndim, npbox, nboxes_val)
        compute_gradient_density!(gvals, tree, fvals, basis)
        glvals = zeros(nd, ndim, npbox, nboxes_val)
        compute_gradient_density!(glvals, tree, flvals, basis)  # grad of Laplacian
        gc0, gc1 = taylor_coefficients_grad(kernel, sog)
        @. grad_correction = gc0 * gvals + gc1 * glvals
    end
    if hess
        hvals = zeros(nd, nhess(ndim), npbox, nboxes_val)
        compute_hessian_density!(hvals, tree, fvals, basis)
        hlvals = zeros(nd, nhess(ndim), npbox, nboxes_val)
        compute_hessian_density!(hlvals, tree, flvals, basis)
        hc0, hc1 = taylor_coefficients_hess(kernel, sog)
        @. hess_correction = hc0 * hvals + hc1 * hlvals
    end

    # === Step 3: Upward pass ===
    proxy_charges = zeros(proxy.ncbox, nd, nboxes_val)
    density_to_proxy!(proxy_charges, fvals, proxy)  # leaf boxes
    upward_pass!(proxy_charges, tree, proxy)         # anterpolate up

    # === Steps 4-6: Plane wave (far-field) ===
    proxy_pot = zeros(proxy.ncbox, nd, nboxes_val)
    # Classify and group deltas by cutoff level
    delta_groups = group_deltas_by_level(sog, tree)
    for (level_group, delta_group, weight_group) in delta_groups.normal
        boxfgt!(proxy_pot, tree, proxy_charges, delta_group, weight_group,
                level_group, pw_data, lists, proxy)
    end

    # === Step 6b: Fat Gaussians (npwlevel < 0) ===
    for (delta, weight) in delta_groups.fat
        handle_fat_gaussian!(proxy_pot, tree, proxy_charges, delta, weight, pw_data)
    end

    # === Downward pass ===
    downward_pass!(proxy_pot, tree, proxy)

    # === Step 7: Direct local interactions ===
    apply_local!(pot, tree, fvals, local_tabs, lists)

    # === Step 8: Asymptotic expansion (small deltas) ===
    apply_asymptotic!(pot, tree, fvals, flvals, fl2vals, sog, delta_groups.asymptotic)

    # === Step 9: Proxy potential → final output ===
    proxy_to_potential_add!(pot, proxy_pot, proxy)

    # === Optional: derivatives ===
    grad_out = nothing
    hess_out = nothing
    if grad
        grad_out = zeros(nd, ndim, npbox, nboxes_val)
        compute_gradient!(grad_out, pot, tree, basis)
    end
    if hess
        hess_out = zeros(nd, nhess(ndim), npbox, nboxes_val)
        compute_hessian!(hess_out, pot, tree, basis)
    end

    # === Optional: target evaluation ===
    target_pot = target_grad = target_hess = nothing
    if targets !== nothing
        target_pot = evaluate_at_targets(pot, tree, targets, basis)
    end

    return SolverResult(pot, grad_out, hess_out, target_pot, target_grad, target_hess)
end
```

- [ ] **Step 3: Run end-to-end tests**

```bash
cd /mnt/home/xgao1/codes/BoxDMK.jl && julia --project -e 'using Pkg; Pkg.test()'
```

- [ ] **Step 4: Commit**

```bash
git add src/solver.jl test/test_solver.jl src/BoxDMK.jl
git commit -m "feat: implement 9-step bdmk solver with end-to-end tests"
```

---

## Phase 8: Polish and Validation

### Task 18: Test Runner Assembly

**Files:**
- Modify: `test/runtests.jl`

- [ ] **Step 1: Update test runner to include all test files**

```julia
using BoxDMK
using Test

@testset "BoxDMK.jl" begin
    include("test_basis.jl")
    include("test_tensor.jl")
    include("test_tree.jl")
    include("test_sog.jl")
    include("test_proxy.jl")
    include("test_kernels.jl")
    include("test_tree_data.jl")
    include("test_local.jl")
    include("test_planewave.jl")
    include("test_passes.jl")
    include("test_solver.jl")
end
```

- [ ] **Step 2: Run full test suite**

```bash
cd /mnt/home/xgao1/codes/BoxDMK.jl && julia --project -e 'using Pkg; Pkg.test()'
```

- [ ] **Step 3: Commit**

```bash
git add test/runtests.jl
git commit -m "feat: assemble full test suite"
```

---

### Task 19: Cross-Validation Against Fortran

**Files:**
- Create: `test/test_cross_validation.jl`

- [ ] **Step 1: Write cross-validation tests**

Use the same test parameters as Fortran's `testbdmk.f`:
- Gaussian sources with `rsig = 4e-3, 1e-3, 1e-4, 1e-5`
- `norder = 16`, `eps = 1e-3, 1e-6, 1e-9`
- All three 3D kernels
- Verify relative errors match Fortran's reported accuracy levels

- [ ] **Step 2: Run and verify**

- [ ] **Step 3: Commit**

```bash
git add test/test_cross_validation.jl
git commit -m "test: add cross-validation tests matching Fortran test suite"
```

---

## Task Dependency Graph

```
Task 1 (types/skeleton)
├── Task 2 (Legendre) → Task 3 (Chebyshev)
├── Task 4 (tensor products) [depends on Task 2 for basis]
│   ├── Task 5 (tree) → Task 6 (interaction lists)
│   ├── Task 10 (tree data) [also depends on Task 2]
│   ├── Task 11 (local tables) → Task 12 (local application)
│   └── Task 16 (derivatives)
├── Task 7 (SOG loading)
│   └── Task 9 (kernel Taylor) [depends on Task 7]
├── Task 8 (proxy system) [depends on Task 4]
│   └── Task 15 (upward/downward passes) [depends on Task 5, Task 8]
├── Task 13 (plane wave setup) [depends on Task 8]
│   └── Task 14 (Box FGT)
│       └── Task 15b (delta classification, fat Gaussians) [depends on Task 7, Task 14]

All above → Task 17 (solver assembly) → Task 18 (test runner) → Task 19 (cross-validation)
```

**Parallelizable groups (after Task 1):**
- Tasks 2, 7 can run in parallel
- Task 3 after Task 2; Task 4 after Task 2
- Tasks 5, 8, 9, 10, 11 can run in parallel after their dependencies
- Tasks 6, 12, 13, 15, 15b, 16 depend on their respective predecessors
- Task 17 requires all previous tasks
