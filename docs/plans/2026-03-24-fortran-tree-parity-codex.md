# Fortran Tree Parity: Codex Execution Plan

**Goal:** Rewrite `build_tree(...)` in `src/tree/tree.jl` so that the Julia tree builder produces **exactly** the same tree data as the vendored Fortran tree builder, for all supported configurations. The public API signature must remain unchanged.

**Oracle:** `build_tree_fortran(...)` from `src/fortran/fortran_wrapper.jl` is the ground truth. Every task ends with a parity test comparing Julia output to Fortran output.

**Working directory:** `/mnt/home/xgao1/codes/BoxDMK.jl`
**Run tests:** `julia --project --threads=1 -e 'using BoxDMK; include("test/test_tree.jl")'`
**Run full suite:** `julia --project --threads=auto -e 'using Pkg; Pkg.test()'`

---

## Critical Background

### Current Divergence

At `tol=1e-3` 3D Laplace:
- Fortran: 713 boxes, histogram `[1, 8, 64, 512, 112, 16]`
- Julia: 241 boxes, histogram `[1, 8, 64, 96, 64, 8]`

Root causes (ordered by impact):
1. **Level restriction algorithm** — Julia uses a simplified all-pairs touching check; Fortran uses a 4-phase flag/flag+/flag++ algorithm with reorganization
2. **Integral norm updates** — Julia recomputes global L2 norm from scratch each level; Fortran uses incremental `update_rints` (subtract parent, add children) and saves per-level norms in `rintl`
3. **Forced refinement via `zk`** — Fortran uses `zk = ComplexF64(30.0, 0.0)` and forces all-box refinement when `real(zk) * boxsize > 5`; Julia's `_kernel_requires_refinement` returns `false` for `LaplaceKernel`
4. **Colleague computation** — Julia does naive O(n²) same-level scan; Fortran traverses parent's colleagues' children
5. **Box ordering** — Julia doesn't maintain levelwise contiguity after LR; Fortran uses `vol_tree_reorg`

### Conventions That Already Match

- **Child numbering** — Both use Morton order: child `j` has sign `((j-1) >> (d-1)) & 1 == 0 ? -1 : +1` per dimension. Fortran's `get_child_box_sign` produces the same pattern as Julia's `_child_center`.
- **Grid ordering** — Fortran's `mesh3d` loops `z(outer) → y → x(inner)`. Julia's `Iterators.product(nodes, nodes, nodes)` iterates first arg fastest. Both produce x-fastest ordering — they match.
- **Coordinate shift** — Fortran centers root at origin, evaluates on `[-L/2, L/2]^d`. Julia centers root at origin too, but adds `coord_shift = boxlen/2` in `_sample_box` so user function sees `[0, L]^d`. The wrapper's Fortran callback adds the same shift. This is cosmetic — parity comparison should use the Fortran convention (centered at origin) and shift at the final `_pack_tree` step.
- **Error norm** — Julia uses `iptype=2` (L2). The `_modal_tail_mask` and `_modal_tail_error` already match Fortran's `tens_prod_get_rmask` + `fun_err` for iptype=2.
- **`wts2` and `quadrature_weights`** — Fortran's `polytens_exps_nd` with `itype=1` produces the same tensor-product weights as Julia's `_reference_weights`. Both are tensor products of 1D quadrature weights on `[-1,1]`. Since multiplication commutes, ordering differences are irrelevant for weight values.

### The `vol_tree_mem` / `vol_tree_build` Duality

`build_tree_fortran(...)` calls **two** Fortran routines sequentially:
1. `vol_tree_mem` — builds the tree and computes `nboxes`, `nlevels`, and `rintl(0:nlevels)` (the per-level integral norms). This is a dry run that uses incremental `update_rints` and saves `rintl[ilev]` after each level.
2. `vol_tree_build` — rebuilds the tree using the **saved** `rintl` values from step 1.

The critical difference: `vol_tree_build` uses `rintl(ilev)` at line 622, **NOT** a live `rint`. Since both calls evaluate the same function at the same points (deterministic), the trees are identical. **The Julia implementation should mirror `vol_tree_mem`**: compute `rintl` per level during adaptive refinement, then use `rintl[ilev]` when computing the refinement threshold `rsc`.

### `iperiod = 0` Always

The Fortran wrapper always passes `iperiod = 0` (free space, no periodic boundaries). All periodic distance handling branches in the Fortran source (`if (iperiod.eq.1) ...`) are dead code for our use cases. **Omit periodic branches in the Julia port** to reduce complexity and bug surface.

### Key Fortran Data Structures to Mirror

```
laddr(2, 0:nlmax)     — level address pointers: [first_box, last_box] per level
ilevel(nboxes)         — level of each box
iparent(nboxes)        — parent (Fortran uses -1 for root; Julia uses 0)
nchild(nboxes)         — number of children (0 for leaf, mc for non-leaf)
ichild(mc, nboxes)     — child indices (Fortran uses -1 for absent; Julia uses 0)
centers(ndim, nboxes)  — box centers (Fortran: centered at origin)
fvals(nd, npbox, nboxes) — function values
rintbs(nboxes)         — per-box integral norms (for iptype=2, stores SQUARED norms)
rintl(0:nlmax)         — per-level global integral norms (saved after each level's update_rints)
iflag(nboxes)          — flags for level restriction (0/1/2/3)
boxsize(0:nlevels)     — size per level
laddrtail(2, 0:nlmax)  — tail address pointers for LR (init: [0, -1] means empty range)
```

### Key Fortran Source Files

- `deps/boxdmk_fortran/src/common/tree_vol_coeffs.f` — `vol_tree_mem`, `vol_tree_build`, `vol_tree_find_box_refine`, `vol_tree_refine_boxes`, `update_rints`, `vol_tree_fix_lr`, `vol_tree_reorg`, **`vol_updateflags`** (NOT `updateflags` from tree_routs.f), `vol_tree_refine_boxes_flag`, `cumsum_nz`
- `deps/boxdmk_fortran/src/common/tree_routs.f` — `computecoll`, `tree_refine_boxes`
- `deps/boxdmk_fortran/src/common/pts_tree.f` — `get_child_box_sign`
- `deps/boxdmk_fortran/src/common/cumsum.f` — `cumsum`, `cumsum1`
- `deps/boxdmk_fortran/src/common/polytens.f` — `tens_prod_get_rmask`
- `deps/boxdmk_fortran/src/common/voltab3d.f` — `mesh3d`

**WARNING:** There are two different `updateflags` routines. Use `vol_updateflags` from `tree_vol_coeffs.f` (line 1733), NOT `updateflags` from `tree_routs.f` (line 247). The `vol_` variant includes `iperiod` handling (though we always use `iperiod=0`).

---

## Task 1: Add Exact Parity Test Scaffold

**Files to modify:** `test/test_tree.jl`

Add a new `@testset "Exact Fortran Tree Parity"` that:

1. Defines a test function (e.g., `f(x) = [exp(-40 * sum((x .- 0.5) .^ 2))]`)
2. Builds tree with both `build_tree(...)` and `build_tree_fortran(...)` using identical parameters: `ndim=3, norder=4, eps=1e-3, boxlen=1.0, nd=1, eta=1.0, kernel=LaplaceKernel(), basis=LegendreBasis()`
3. Compares **exactly**:
   - `tree.nlevels == ftree.tree.nlevels`
   - `tree.level == ftree.tree.level`
   - `tree.parent == ftree.tree.parent`
   - `tree.children == ftree.tree.children`
   - `tree.colleagues == ftree.tree.colleagues` (element-wise, including ordering)
   - `tree.centers ≈ ftree.tree.centers` (atol=1e-14)
   - `tree.boxsize ≈ ftree.tree.boxsize` (atol=1e-14)
   - `fvals ≈ ftree.fvals` (atol=1e-12)

Run the test — it should FAIL on the current code. Commit the failing test.

**Commit:** `test: add exact fortran tree parity regression`

---

## Task 2: Introduce Fortran-shaped Internal State

**Files to modify:** `src/tree/tree.jl`

Add a new internal mutable struct (not exported) that mirrors Fortran arrays:

```julia
mutable struct _FortranTreeState
    ndim::Int
    mc::Int                        # 2^ndim
    mnbors::Int                    # 3^ndim
    nboxes::Int                    # current box count
    nlevels::Int
    nbmax::Int                     # allocated capacity
    nlmax::Int                     # max levels (200)

    laddr::Matrix{Int}             # (2, nlmax+1) — level address pointers, 1-indexed
                                   #   laddr[1, ilev+1] = first box at level ilev
                                   #   laddr[2, ilev+1] = last box at level ilev
    ilevel::Vector{Int}            # (nbmax)
    iparent::Vector{Int}           # (nbmax) — use -1 for root (Fortran convention)
    nchild::Vector{Int}            # (nbmax) — 0 for leaf, mc for non-leaf
    ichild::Matrix{Int}            # (mc, nbmax) — use -1 for absent child
    centers::Matrix{Float64}       # (ndim, nbmax) — centered at origin
    boxsize::Vector{Float64}       # (nlmax+1) — boxsize[ilev+1] = size at level ilev
    fvals::Array{Float64,3}        # (nd, npbox, nbmax)
    rintbs::Vector{Float64}        # (nbmax) — per-box integral (SQUARED for iptype=2)
    rintl::Vector{Float64}         # (nlmax+1) — per-level integral norms, rintl[ilev+1] for level ilev
    iflag::Vector{Int}             # (nbmax) — for level restriction
    nnbors::Vector{Int}            # (nbmax) — colleague count
    nbors::Matrix{Int}             # (mnbors, nbmax) — colleague indices, use -1 for absent
end
```

**Level indexing convention:** Since Julia arrays are 1-indexed, access level `ilev` as index `ilev+1`. Define a helper:
```julia
_lev(ilev) = ilev + 1  # Fortran level 0 → Julia index 1
```

Use this consistently throughout to prevent off-by-one bugs.

Add helper functions:
- `_ftstate_init(ndim, nd, npbox, nbmax, nlmax)` — allocate and initialize root box
- `_ftstate_grow!(state, new_nbmax)` — resize arrays when capacity exceeded
- `_ftstate_to_boxtree(state, basis, norder, boxlen)` — materialize into `BoxTree` + `fvals`

The `_ftstate_to_boxtree` function must:
- Shift centers by `+boxlen/2` (Fortran origin → Julia `[0, L]^d`)
- Convert `iparent=-1` to `0` for root
- Convert `ichild=-1` to `0` for absent children
- Trim arrays to `nboxes` (remove allocated-but-unused tail)
- Build `boxsize` vector as `[boxlen / 2^i for i in 0:nlevels]`
- Copy colleague data from `nnbors`/`nbors` into `Vector{Vector{Int}}`

Do NOT switch `build_tree(...)` to use this yet. Just add the types and helpers.

Run existing tests — they should still pass (no behavior change).

**Commit:** `refactor: add fortran-shaped julia tree state`

---

## Task 3: Port Adaptive Refinement with `update_rints`

**Files to modify:** `src/tree/tree.jl`

### 3a: Port `vol_tree_find_box_refine`

Write `_ftstate_find_box_refine!(state, ...)` that mirrors the Fortran routine:

```
For each box in [ifirstbox, ilastbox]:
  1. Transform fvals to modal coefficients via tensor product with umat
  2. Compute error via fun_err (L2 version: sqrt(sum(coeff^2 * rmask)) * rscale)
  3. Divide by rsum
  4. If error > eps * rsc: mark for refinement

CRITICAL — forced refinement check (tree_vol_coeffs.f line 722):
  if 30.0 * boxsize > 5:    # zk = ComplexF64(30.0, 0.0) hardcoded in fortran_wrapper.jl:654
      mark ALL boxes at this level for refinement
      skip the error-based check entirely
```

**Why `zk = 30.0`:** The Fortran wrapper at `src/fortran/fortran_wrapper.jl:654` hardcodes `zk = ComplexF64(30.0, 0.0)` for ALL kernel types during tree building. The Fortran `vol_tree_find_box_refine` at line 722 checks `real(zk) * boxsize > 5`. For `boxlen=1.0`:
- Level 0 (boxsize=1.0): `30*1.0 = 30 > 5` → forced
- Level 1 (boxsize=0.5): `30*0.5 = 15 > 5` → forced
- Level 2 (boxsize=0.25): `30*0.25 = 7.5 > 5` → forced
- Level 3 (boxsize=0.125): `30*0.125 = 3.75 < 5` → error-based

This means the first 3 levels are ALWAYS refined regardless of function values. The current Julia `_kernel_requires_refinement` returns `false` for Laplace, which is a major source of divergence.

The existing `_modal_tail_error` function can be reused for the error-based check — it already implements the L2 version of `fun_err` correctly.

For the refinement threshold `rsc` (iptype=2):
```
rsc = sqrt(1.0 / boxsize[0]^ndim) * rintl[ilev]
```

**IMPORTANT:** Use `rintl[ilev]` (the saved per-level norm from the previous level's `update_rints`), NOT a live recomputed norm. This matches `vol_tree_build` at line 622.

### 3b: Port `vol_tree_refine_boxes`

Write `_ftstate_refine_boxes!(state, irefinebox, ...)` that:

1. Computes cumulative sum of `irefinebox` (prefix scan: `isum[i] = sum(irefinebox[1:i])`)
2. For each offset `i` in `1:nbloc` where `irefinebox[i] == 1`:
   - `ibox = ifirstbox + i - 1`
   - `nbl = nbctr + (isum[i] - 1) * mc`
   - Creates `mc` children at positions `nbl+1` to `nbl+mc`
   - For child `j` in `1:mc`:
     - `jbox = nbl + j`
     - `centers[k, jbox] = centers[k, ibox] + isgn[k, j] * bsh` where `bsh = bs / 2`
     - Evaluates function: `xyz[k] = centers[k, jbox] + grid[k, l] * bs` then `f(xyz + boxlen/2)`
     - `iparent[jbox] = ibox`
     - `nchild[jbox] = 0`
     - `ichild[:, jbox] .= -1`
     - `ichild[j, ibox] = jbox`
     - `ilevel[jbox] = nlctr` (where `nlctr = ilev + 1`, the child level)
   - `nchild[ibox] = mc`
3. Updates `state.nboxes = nbctr + isum[nbloc] * mc`

**Reuse `_sample_box`:** The existing `_sample_box(f, center, boxsize, grid, nd, coord_shift)` can be called with `coord_shift = boxlen/2` since:
- `_sample_box` computes `center[d] + (boxsize/2) * grid[d] + shift`
- Fortran computes `centers[k,jbox] + grid[k,l] * bs` where grid is on `[-1/2, 1/2]`
- Julia's grid is on `[-1, 1]`, so `(boxsize/2) * grid = bs * (grid/2)` which matches Fortran's `grid_half * bs`
- Adding `coord_shift = boxlen/2` maps from Fortran convention to user convention

### 3c: Port `update_rints` (iptype=2 version)

Write `_ftstate_update_rints!(state, ifirstbox, nbloc, wts, rsc)`.

**CRITICAL: Two-pass structure.** The Fortran code at lines 1047-1078 uses two SEPARATE loops:

```
# Pass 1: Subtract ALL parent contributions first
rintsq = state.rintl_current^2
for i in 1:nbloc:
    ibox = ifirstbox + i - 1
    if nchild[ibox] > 0:
        rintsq -= rintbs[ibox]
rintsq = max(rintsq, 0.0)

# Pass 2: Add ALL children contributions
for i in 1:nbloc:
    ibox = ifirstbox + i - 1
    if nchild[ibox] > 0:
        for j in 1:mc:
            jbox = ichild[j, ibox]
            rintbs[jbox] = 0.0
            for l in 1:npbox:
                for idim in 1:nd:
                    rintbs[jbox] += fvals[idim, l, jbox]^2 * wts[l] * rsc
            rintsq += rintbs[jbox]

state.rint = sqrt(rintsq)
```

Do NOT interleave subtract and add in a single loop — that changes the floating-point result.

The `wts` here are the tensor-product quadrature weights on `[-1,1]^d` (same as `wts2` in Fortran), and `rsc = boxsize[ilev+1]^ndim / mc`.

### 3d: Port root box initialization

The root box initialization in Fortran computes `rintbs(1)` during function evaluation:
```
rintbs[1] = 0.0
for i in 1:npbox:
    for idim in 1:nd:
        rintbs[1] += fvals[idim, i, 1]^2 * wts2[i] * rsc
rint = sqrt(rintbs[1])
rintl[0] = rint
```

**WARNING — `rsc` for root uses `boxlen**2`, NOT `boxlen**ndim`:**
```
rsc = boxlen^2 / mc     # Fortran: tree_vol_coeffs.f line 244
```
This is `boxlen^2` regardless of `ndim`. The Fortran comment says "extra factor of 4 since wts2 are on [-1,1]^2 as opposed to [-1/2,1/2]^2". This appears to be a 2D-era comment that wasn't updated for general dimensions. **Replicate this exactly for parity — do NOT "fix" it to `boxlen^ndim`.**

For subsequent levels, `update_rints` uses `rsc = boxsize[ilev+1]^ndim / mc` (line 358), which IS dimension-dependent. This inconsistency exists in the Fortran and must be preserved.

### 3e: Assemble the adaptive refinement loop

Write `_ftstate_adaptive_refine!(state, f, ...)` that:

```
for ilev in 0:nlmax-1:
    # Compute refinement threshold using saved per-level norm
    rsc = sqrt(1.0 / boxsize[0]^ndim) * rintl[ilev]

    call _ftstate_find_box_refine!(...)
    if no refinement needed: break

    # Check capacity and grow if needed
    nbadd = count(irefinebox .== 1) * mc
    if state.nboxes + nbadd > state.nbmax:
        _ftstate_grow!(state, state.nboxes + nbadd)

    boxsize[ilev+1] = boxsize[ilev] / 2
    laddr[1, ilev+1] = state.nboxes + 1

    call _ftstate_refine_boxes!(...)

    # Update integral norms
    rsc_update = boxsize[ilev+1]^ndim / mc     # NOTE: uses ^ndim, unlike root's ^2
    call _ftstate_update_rints!(...)
    rintl[ilev+1] = state.rint                  # Save per-level norm

    laddr[2, ilev+1] = state.nboxes

state.nlevels = ilev
```

Run parity test — adaptive box counts should now match Fortran (before LR). The full parity test will still fail because LR hasn't been ported yet. Print/compare the box count and level histogram to verify progress.

**Commit:** `feat: port adaptive tree refinement with incremental rints`

---

## Task 4: Port `computecoll`

**Files to modify:** `src/tree/tree.jl`

Write `_ftstate_computecoll!(state)` mirroring `computecoll` from `tree_routs.f` (line 105):

```
# Initialize
for i in 1:state.nboxes:
    nnbors[i] = 0
    for j in 1:mnbors:
        nbors[j, i] = -1

# Root is its own colleague
nnbors[1] = 1
nbors[1, 1] = 1

for ilev in 1:nlevels:
    for ibox in laddr[1, _lev(ilev)] : laddr[2, _lev(ilev)]:
        dad = iparent[ibox]
        for i in 1:nnbors[dad]:
            jbox = nbors[i, dad]
            for j in 1:mc:
                kbox = ichild[j, jbox]
                if kbox > 0:
                    # Check if kbox is a neighbor of ibox
                    ifnbor = true
                    for k in 1:ndim:
                        dis = abs(centers[k, kbox] - centers[k, ibox])
                        if dis > 1.05 * boxsize[_lev(ilev)]:
                            ifnbor = false; break
                    if ifnbor:
                        nnbors[ibox] += 1
                        nbors[nnbors[ibox], ibox] = kbox
```

**Note:** Colleague computation is called unconditionally in `vol_tree_build` (line 656), not guarded by `nlevels >= 2`. Only level restriction has the `nlevels >= 2` guard.

This algorithm naturally produces colleagues in the same order as Fortran because it traverses parent's colleagues in stored order, then children in octant order.

Run parity test — colleague data should now match for the adaptive tree (before LR).

**Commit:** `feat: port fortran colleague construction`

---

## Task 5: Port Level Restriction (`vol_tree_fix_lr`)

**Files to modify:** `src/tree/tree.jl`

This is the most complex task. The algorithm has 4 phases. Port `vol_tree_fix_lr` from `tree_vol_coeffs.f` (line 1234).

**Guard:** Only run if `nlevels >= 2` (matching `vol_tree_build` line 660).

### 5a: Phase 1 — Flag initial violations (iflag=1)

```
# Initialize flags
for i in 1:state.nboxes:
    iflag[i] = 0

for ilev in nlevels:-1:2:
    distest = 1.05 * (boxsize[_lev(ilev-1)] + boxsize[_lev(ilev-2)]) / 2
    for ibox in laddr[1, _lev(ilev)] : laddr[2, _lev(ilev)]:
        idad = iparent[ibox]
        igranddad = iparent[idad]
        for i in 1:nnbors[igranddad]:
            jbox = nbors[i, igranddad]
            if nchild[jbox] == 0 && iflag[jbox] == 0:
                ict = 0
                for k in 1:ndim:
                    dis = centers[k, jbox] - centers[k, idad]
                    if abs(dis) <= distest: ict += 1
                if ict == ndim:
                    iflag[jbox] = 1
```

### 5b: Phase 2 — Flag+ boxes (iflag=2)

```
for ilev in nlevels:-1:1:
    distest = 1.05 * (boxsize[_lev(ilev)] + boxsize[_lev(ilev-1)]) / 2
    for ibox in laddr[1, _lev(ilev)] : laddr[2, _lev(ilev)]:
        if iflag[ibox] == 1 || iflag[ibox] == 2:
            idad = iparent[ibox]
            for i in 1:nnbors[idad]:
                jbox = nbors[i, idad]
                if nchild[jbox] == 0 && iflag[jbox] == 0:
                    ict = 0
                    for k in 1:ndim:
                        dis = centers[k, jbox] - centers[k, ibox]
                        if abs(dis) <= distest: ict += 1
                    if ict == ndim:
                        iflag[jbox] = 2
```

### 5c: Phase 3 — Subdivide flagged boxes + reorganize

Port `vol_tree_refine_boxes_flag` from `tree_vol_coeffs.f` line 1857.

**Key differences from `vol_tree_refine_boxes`:**
1. Uses `cumsum_nz` instead of `cumsum`: `isum[i] = cumulative count of (iflag[ifirstbox + i - 1] > 0)` for i=1..nbloc
2. Level assignment: `ilevel[jbox] = nlctr + 1` (not `nlctr` — the flag variant adds 1)
3. **CRITICAL — iflag propagation to children (line 1920-1921):**
   - If parent has `iflag == 1`: children get `iflag = 3` (flag++)
   - If parent has `iflag == 2`: children get `iflag = 0`
4. Indexing: `nbl = nbctr + (isum[ibox - ifirstbox + 1] - 1) * mc` where `ibox` ranges from `ifirstbox` to `ilastbox`

```
# Initialize laddrtail — empty ranges mean [0, -1]
for ilev in 0:nlevels:
    laddrtail[1, _lev(ilev)] = 0
    laddrtail[2, _lev(ilev)] = -1

for ilev in 1:nlevels-2:
    laddrtail[1, _lev(ilev+1)] = state.nboxes + 1

    nbloc = laddr[2, _lev(ilev)] - laddr[1, _lev(ilev)] + 1
    _ftstate_refine_boxes_flag!(state, iflag, laddr[1, _lev(ilev)], nbloc,
        boxsize[_lev(ilev+1)], ilev, f, ...)
    # _refine_boxes_flag sets:
    #   ilevel[jbox] = nlctr + 1  (= ilev + 1)
    #   iflag[jbox] = 3 if iflag[ibox] == 1
    #   iflag[jbox] = 0 if iflag[ibox] == 2

    laddrtail[2, _lev(ilev+1)] = state.nboxes

# Reorganize tree
_ftstate_reorg!(state, laddrtail)

# Recompute colleagues
_ftstate_computecoll!(state)
```

### 5d: Port `vol_tree_reorg`

This reorganization merges `laddr` and `laddrtail` ranges into contiguous levelwise ordering. Source: `tree_vol_coeffs.f` line 1555.

```
1. Copy all current data to temporary arrays (tilevel, tiparent, tnchild, tichild, tcenters, tfvals, tiflag)
2. Save old laddr as tladdr

3. For levels 0-1: identity mapping
   for ilev in 0:1:
       for ibox in laddr[1, _lev(ilev)] : laddr[2, _lev(ilev)]:
           iboxtocurbox[ibox] = ibox

4. Compute new levelwise layout:
   curbox = laddr[1, _lev(2)]   # start of level 2 (unchanged)
   for ilev in 2:nlevels:
       laddr[1, _lev(ilev)] = curbox
       # First: copy boxes from original laddr range
       for ibox in tladdr[1, _lev(ilev)] : tladdr[2, _lev(ilev)]:
           copy ilevel, nchild, centers, fvals, iflag from temp[ibox] to curbox
           iboxtocurbox[ibox] = curbox
           curbox += 1
       # Then: copy boxes from laddrtail range
       for ibox in laddrtail[1, _lev(ilev)] : laddrtail[2, _lev(ilev)]:
           copy ilevel, nchild, centers, fvals, iflag from temp[ibox] to curbox
           iboxtocurbox[ibox] = curbox
           curbox += 1
       laddr[2, _lev(ilev)] = curbox - 1

5. Remap parent/child indices via iboxtocurbox:
   for ibox in 1:nboxes:
       newbox = iboxtocurbox[ibox]
       if tiparent[ibox] == -1:
           iparent[newbox] = -1
       else:
           iparent[newbox] = iboxtocurbox[tiparent[ibox]]
       for j in 1:mc:
           if tichild[j, ibox] == -1:
               ichild[j, newbox] = -1
           else:
               ichild[j, newbox] = iboxtocurbox[tichild[j, ibox]]
```

**Note on empty ranges:** When `laddrtail[2, _lev(ilev)] < laddrtail[1, _lev(ilev)]` (i.e., `[0, -1]`), the inner loop body executes zero times. Julia's `for ibox in 0:-1` does nothing, which is correct.

### 5e: Phase 4 — Process flag++ boxes

```
# Reset flags: keep only flag++ (iflag=3), clear flag=1/2 to 0
for ibox in 1:state.nboxes:
    if iflag[ibox] != 3: iflag[ibox] = 0

# Re-init laddrtail
for ilev in 0:nlevels:
    laddrtail[1, _lev(ilev)] = 0
    laddrtail[2, _lev(ilev)] = -1

for ilev in 2:nlevels-2:
    # Step 1: vol_updateflags — check if flag++ boxes need subdivision
    _ftstate_vol_updateflags!(state, ilev, laddr)
    _ftstate_vol_updateflags!(state, ilev, laddrtail)

    # Step 2: Subdivide
    laddrtail[1, _lev(ilev+1)] = state.nboxes + 1

    nbloc = laddr[2, _lev(ilev)] - laddr[1, _lev(ilev)] + 1
    _ftstate_refine_boxes_flag!(state, iflag, laddr[1, _lev(ilev)], nbloc,
        boxsize[_lev(ilev+1)], ilev, f, ...)

    nbloc_tail = laddrtail[2, _lev(ilev)] - laddrtail[1, _lev(ilev)] + 1
    _ftstate_refine_boxes_flag!(state, iflag, laddrtail[1, _lev(ilev)], nbloc_tail,
        boxsize[_lev(ilev+1)], ilev, f, ...)

    laddrtail[2, _lev(ilev+1)] = state.nboxes

    # Step 3: Compute colleagues for newly created boxes ONLY (incremental)
    for ibox in laddrtail[1, _lev(ilev+1)] : laddrtail[2, _lev(ilev+1)]:
        nnbors[ibox] = 0
        idad = iparent[ibox]
        for i in 1:nnbors[idad]:
            jbox = nbors[i, idad]
            for j in 1:mc:
                kbox = ichild[j, jbox]
                if kbox > 0:
                    ifnbor = true
                    for k in 1:ndim:
                        dis = abs(centers[k, kbox] - centers[k, ibox])
                        if dis > 1.05 * boxsize[_lev(ilev+1)]:
                            ifnbor = false; break
                    if ifnbor:
                        nnbors[ibox] += 1
                        nbors[nnbors[ibox], ibox] = kbox

# Final reorganize
_ftstate_reorg!(state, laddrtail)

# Final colleague recompute
_ftstate_computecoll!(state)
```

### 5f: Port `vol_updateflags`

Source: `tree_vol_coeffs.f` line 1733 — **`vol_updateflags`**, NOT `updateflags` from `tree_routs.f`.

```
distest = 1.05 * (boxsize[_lev(curlev)] + boxsize[_lev(curlev+1)]) / 2

for ibox in laddr[1, _lev(curlev)] : laddr[2, _lev(curlev)]:
    if iflag[ibox] == 3:
        iflag[ibox] = 0
        for i in 1:nnbors[ibox]:
            jbox = nbors[i, ibox]
            for j in 1:mc:
                kbox = ichild[j, jbox]
                if kbox > 0 && nchild[kbox] > 0:
                    ict = 0
                    for k in 1:ndim:
                        dis = centers[k, kbox] - centers[k, ibox]
                        if abs(dis) <= distest: ict += 1
                    if ict == ndim:
                        iflag[ibox] = 1
                        @goto done_checking
        @label done_checking
```

After porting all phases, ensure `_ftstate_grow!` is called when capacity is exceeded. Before LR, grow to at least `2 * mc * nboxes`.

Run parity test — the tree structure should now match exactly.

**Commit:** `feat: port fortran level-restricted tree closure and reorganization`

---

## Task 6: Wire Up `build_tree(...)` to Use Parity Backend

**Files to modify:** `src/tree/tree.jl`

Replace the body of `build_tree(...)` to:

1. Set up quadrature (nodes, weights, umat) — same as current code
2. Compute `wts2 = _reference_weights(Float64.(weights), ndim)` — tensor-product weights on `[-1,1]^d` for `update_rints`
3. Compute grid on `[-1/2, 1/2]^d`: divide reference_grid by 2 (Fortran convention, `xq(i) = xq(i)/2`)
4. Initialize `_FortranTreeState` with root box at center = origin
5. Evaluate root function: `xyz[k] = grid[k, i] * boxlen`, then call `f(xyz .+ boxlen/2)` (shift to user convention)
6. Compute root `rintbs` and `rint`:
   ```julia
   rsc = boxlen^2 / mc   # NOTE: boxlen^2, NOT boxlen^ndim — matches Fortran exactly
   rintbs[1] = sum(fvals[idim, i, 1]^2 * wts2[i] * rsc for idim, i)
   rint = sqrt(rintbs[1])
   rintl[0] = rint
   ```
7. Run `_ftstate_adaptive_refine!(state, ...)`
8. Compute colleagues via `_ftstate_computecoll!(state)` (unconditionally)
9. If `nlevels >= 2`:
   - Grow capacity if needed (`2 * mc * nboxes`)
   - Run `_ftstate_fix_lr!(state, ...)`
10. Materialize via `_ftstate_to_boxtree(state, basis, norder, boxlen)`

**Function evaluation convention:** During tree building, all coordinates are in Fortran convention (origin-centered, `[-L/2, L/2]^d`). The user's Julia function expects `[0, L]^d`. Add `boxlen/2` before calling:
```julia
user_point = fortran_point .+ boxlen/2
result = f(user_point)
```

The old `_TreeBoxState` type and associated functions can be left as dead code for now (removed in Task 8).

Run the parity test — it should now PASS for the `eps=1e-3` case.

**Commit:** `feat: make julia tree builder mirror fortran exactly`

---

## Task 7: Expand Parity Coverage and Run Full Suite

**Files to modify:** `test/test_tree.jl`

### 7a: Add more parity test cases

Expand the parity testset to cover:
- `YukawaKernel(1.0)` with `LegendreBasis`, ndim=3
- `SqrtLaplaceKernel()` with `LegendreBasis`, ndim=3
- `LaplaceKernel()` with `ChebyshevBasis`, ndim=3
- `LaplaceKernel()` with `LegendreBasis`, ndim=1 and ndim=2

Use the same test function and the same comparison logic.

### 7b: Run all test files

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_tree.jl")'
julia --project --threads=1 -e 'using BoxDMK; include("test/test_tree_data.jl")'
julia --project --threads=1 -e 'using BoxDMK; include("test/test_fortran_wrapper.jl")'
julia --project --threads=1 -e 'using BoxDMK; include("test/test_solver.jl")'
```

### 7c: Run benchmark

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project benchmark/hybrid_parity.jl
```

Fix any regressions discovered during verification.

**Commit:** `test: expand fortran tree parity coverage`

---

## Task 8: Clean Up Dead Code

**Files to modify:** `src/tree/tree.jl`

Remove the old tree-building infrastructure that is no longer used:
- `_TreeBoxState` struct
- `_enforce_level_restriction!`
- `_build_colleagues`
- `_split_box!`
- `_touches`
- `_global_l2_scale`
- `_box_l2_scale`
- `_pack_tree`

Keep these helper functions that are still used or could be used by the new code:
- `_reference_grid`
- `_reference_weights`
- `_sample_box` (if still used for function evaluation during refinement)
- `_child_center` (if still used)
- `_maxabs` (if still used)
- `_kernel_requires_refinement` (may need updating for `zk` check)
- `_modal_tail_mask`
- `_modal_tail_error`
- `_check_basis_order`

Before removing anything, grep for usages across the entire `src/` and `test/` directories to make sure nothing else depends on them.

Run the full test suite to confirm nothing breaks.

**Commit:** `refactor: remove old tree-building infrastructure`

---

## Implementation Notes

### The `wts2` vs `quadrature_weights` distinction

Fortran uses `wts2` from `polytens_exps_nd(ndim, ipoly, itype=1, ...)` which returns weights on `[-1,1]^d`. The current Julia code computes `quadrature_weights = _reference_weights(weights, ndim)` which are the tensor product of 1D weights. These are equivalent since `polytens_exps_nd` with `itype=1` returns the same tensor-product weights. Since multiplication commutes, ordering differences are irrelevant for weight values.

### Array indexing

Fortran uses 1-based indexing for boxes, -1 for "no parent/child". Julia's `BoxTree` uses 0 for "no parent/child". The internal `_FortranTreeState` should use Fortran convention (-1) during construction, then convert in `_ftstate_to_boxtree`.

For `laddr` and `boxsize`, Fortran uses 0-based level indexing. Define a helper `_lev(ilev) = ilev + 1` and use it consistently throughout. This prevents the #1 source of off-by-one bugs.

### Thread safety

The Fortran code uses OpenMP parallel loops in several places. For the initial Julia port, single-threaded is fine — the goal is correctness/parity, not performance. Do NOT use `@threads` until parity is established.

### Capacity management

Follow the Fortran pattern: start with `nbmax = 10_000`, grow by copying when needed. The growth check in Fortran is:
```
nbtot = nbctr + nbadd
if nbtot > nbmax: reallocate
```
And before LR:
```
nbtot = 2 * mc * nboxes
if nbtot > nbmax: reallocate
```

### `cumsum` vs `cumsum_nz`

- `cumsum(n, a, b)` — standard prefix sum: `b[i] = sum(a[1:i])`
- `cumsum_nz(n, a, b)` — prefix sum of positive indicators: `b[i] = count(a[j] > 0 for j in 1:i)`

Both are used:
- `cumsum` in `vol_tree_refine_boxes` (adaptive refinement)
- `cumsum_nz` in `vol_tree_refine_boxes_flag` (level restriction refinement)

### Naming disambiguation

Two pairs of confusingly similar routines exist in the Fortran:
1. **`vol_updateflags`** (tree_vol_coeffs.f:1733) vs **`updateflags`** (tree_routs.f:247) — use the `vol_` variant
2. **`vol_tree_refine_boxes_flag`** (tree_vol_coeffs.f:1857) vs **`tree_refine_boxes_flag`** (tree_routs.f) — use the `vol_` variant

The `vol_` variants include function evaluation (`fvals`) and iflag propagation; the non-`vol_` variants are geometry-only.
