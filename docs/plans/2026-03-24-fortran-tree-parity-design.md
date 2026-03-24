# Fortran Tree Parity Design

**Date:** 2026-03-24

## Goal

Make `build_tree(...)` a pure Julia implementation that reproduces the vendored Fortran tree builder exactly at the tree-data level for every currently supported tree-build configuration:

- same box ordering
- same `level`
- same `parent`
- same `children`
- same `colleagues`
- same sampled `fvals`

This parity target applies to the public tree-build API, not just the validated Laplace reference benchmark.

## Current State

The current Julia tree builder in [src/tree/tree.jl](/mnt/home/xgao1/codes/BoxDMK.jl/src/tree/tree.jl) is a simplified implementation built around `_TreeBoxState`. It uses:

- a Julia-native adaptive refinement loop
- a simplified `_enforce_level_restriction!`
- a naive same-level colleague builder

The vendored Fortran tree builder in [deps/boxdmk_fortran/src/common/tree_vol_coeffs.f](/mnt/home/xgao1/codes/BoxDMK.jl/deps/boxdmk_fortran/src/common/tree_vol_coeffs.f) and [deps/boxdmk_fortran/src/common/tree_routs.f](/mnt/home/xgao1/codes/BoxDMK.jl/deps/boxdmk_fortran/src/common/tree_routs.f) is materially different:

- refinement state is stored in Fortran-shaped arrays (`laddr`, `ilevel`, `iparent`, `nchild`, `ichild`, `fvals`, `rintbs`, `rintl`)
- adaptive thresholds are updated incrementally through `update_rints`
- colleagues are built through `computecoll`
- level restriction is enforced by `vol_tree_fix_lr` with `flag`, `flag+`, and `flag++` passes
- boxes are reordered through `vol_tree_reorg`

At `tol=1e-3`, the current mismatch is large:

- Fortran final tree: `713` boxes, histogram `[1, 8, 64, 512, 112, 16]`
- Julia final tree: `241` boxes, histogram `[1, 8, 64, 96, 64, 8]`

The strongest confirmed divergence is the level-restriction closure. Julia currently produces a valid level-restricted tree in the minimal touching-leaf sense, but not the stronger closure produced by Fortran.

## Non-Goals

- Replacing the public tree build with a Fortran call
- Preserving the current `_TreeBoxState` implementation as the primary backend
- Optimizing beyond parity before parity is proven

## Constraints

- Runtime tree construction must remain pure Julia
- Public API shape must remain unchanged
- Existing solver code must continue to consume `BoxTree` and `fvals`
- Exact parity is required at the tree-data level, not only up to isomorphism

## Options Considered

### 1. Patch the current object-based Julia builder

This would keep `_TreeBoxState` and incrementally patch:

- adaptive threshold updates
- level restriction
- colleague ordering
- final box ordering

This is not recommended. The implementation model is already too different from Fortran, and exact parity would likely require repeated special-case fixes.

### 2. Add a new internal Julia backend that mirrors Fortran structurally

Implement the tree builder with Fortran-shaped arrays and port the relevant Fortran routines directly into Julia:

- adaptive refinement loop
- `update_rints`
- `computecoll`
- `vol_tree_fix_lr`
- `vol_updateflags`
- `vol_tree_refine_boxes_flag`
- `vol_tree_reorg`

Then convert the final arrays into `BoxTree`.

This is the recommended approach. It is the most reliable way to achieve exact parity while staying pure Julia.

### 3. Keep both builders and add a separate parity-only builder

This would reduce migration risk, but it would not satisfy the requirement that `build_tree(...)` itself reproduce the Fortran output.

## Recommended Architecture

Replace the current authoritative tree-construction pipeline with a Fortran-shaped Julia backend.

### Internal State

Introduce an internal mutable state structure that mirrors the arrays used by Fortran:

- `centers::Matrix{Float64}`
- `boxsize::Vector{Float64}`
- `fvals::Array{Float64,3}`
- `laddr::Matrix{Int}`
- `ilevel::Vector{Int}`
- `iparent::Vector{Int}`
- `nchild::Vector{Int}`
- `ichild::Matrix{Int}`
- `nnbors::Vector{Int}`
- `nbors::Matrix{Int}`
- `rintbs::Vector{Float64}`
- `rintl::Vector{Float64}`
- `iflag::Vector{Int}`
- capacity bookkeeping for `nboxes`, `nlevels`, and dynamic growth

### Control Flow

The Julia backend should follow the same high-level order as Fortran:

1. Build tensor grid and weights in Fortran-compatible ordering
2. Sample root box and initialize `rint`, `rintbs`, `rintl`
3. Run adaptive refinement loop
4. Update `rint` incrementally using the Fortran `update_rints` logic
5. Compute colleagues with the Fortran `computecoll` algorithm
6. Apply full Fortran-style level restriction using `vol_tree_fix_lr`
7. Reorganize boxes with the Fortran `vol_tree_reorg` mapping
8. Recompute colleagues after reorganization
9. Materialize the final `BoxTree` and `fvals`

### Ordering Rules

Exact parity depends on more than geometry. The Julia backend must preserve Fortran conventions for:

- child numbering
- levelwise box numbering
- box append order during refinement
- tail-box append order during LR closure
- colleague discovery order
- reorganization order
- sampled grid-point order in `fvals`

The existing Fortran unpacking logic in [src/fortran/fortran_wrapper.jl](/mnt/home/xgao1/codes/BoxDMK.jl/src/fortran/fortran_wrapper.jl) is the canonical interpretation of those conventions and should be reused as a reference for the final Julia packing.

## Testing Strategy

The change should be driven by exact-parity regression tests against `build_tree_fortran(...)`.

### Required parity assertions

For each covered configuration:

- `tree.nlevels`
- `tree.centers`
- `tree.boxsize`
- `tree.parent`
- `tree.children`
- `tree.colleagues`
- `tree.level`
- `fvals`

All must match exactly, not approximately, except where floating-point sampling requires a tight numeric tolerance.

### Coverage

Tests should cover the currently supported tree-build surface:

- kernels: `LaplaceKernel`, `YukawaKernel`, `SqrtLaplaceKernel`
- bases: `LegendreBasis`, `ChebyshevBasis`
- dimensions: `1`, `2`, `3`
- representative tolerances including at least one loose case that exercises LR closure

The first failing regression should be the known `tol=1e-3` 3D Laplace case.

## Migration Plan

1. Add exact-parity tests against the Fortran wrapper
2. Introduce the Fortran-shaped Julia state and helper routines
3. Port adaptive refinement and `update_rints`
4. Port `computecoll`
5. Port `vol_tree_fix_lr` and reorganization
6. Switch `build_tree(...)` to the new backend
7. Remove or demote the old `_TreeBoxState` path once parity is established

## Risks

- The current tensor-grid ordering may differ subtly from Fortran and break exact `fvals` parity
- Fortran array semantics around tail reorganization can be easy to mistranslate
- Periodic branches exist in Fortran code even though the current public API does not expose them directly
- A partial port could make benchmark cases pass while still failing exact box-order parity elsewhere

## Recommendation

Proceed with a clean Fortran-structured Julia port inside `src/tree/`, driven by exact tree-data parity tests against `build_tree_fortran(...)`. The current object-based builder is not the right foundation for exact reproduction.
