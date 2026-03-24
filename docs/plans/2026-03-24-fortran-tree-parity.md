# Fortran Tree Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make `build_tree(...)` a pure Julia implementation that reproduces the vendored Fortran tree builder exactly at the tree-data level for every currently supported tree-build configuration.

**Architecture:** Replace the current object-based tree-construction backend with a Fortran-shaped Julia implementation that mirrors the vendored adaptive refinement, colleague construction, level-restriction closure, and reorganization passes. Keep the public `build_tree(...)` API unchanged and materialize the final result as `BoxTree` plus `fvals`.

**Tech Stack:** Julia, `Test`, existing Fortran wrapper as parity oracle, vendored Fortran source as behavioral reference.

---

### Task 1: Add exact tree-data parity tests

**Files:**
- Modify: `test/test_tree.jl`
- Reference: `test/test_fortran_wrapper.jl`
- Reference: `src/fortran/fortran_wrapper.jl`

**Step 1: Write the failing test**

Add an exact-parity helper and a test matrix that compares `build_tree(...)` to `build_tree_fortran(...)` for:

- the known loose-tolerance 3D Laplace case
- at least one small case each for `YukawaKernel`, `SqrtLaplaceKernel`, and `ChebyshevBasis`

The assertions should cover:

- `nlevels`
- `centers`
- `boxsize`
- `parent`
- `children`
- `colleagues`
- `level`
- `fvals`

**Step 2: Run test to verify it fails**

Run: `julia --project --threads=1 -e 'using BoxDMK; include("test/test_tree.jl")'`

Expected: the new exact-parity test fails on the current Julia builder.

**Step 3: Keep the failing assertion minimal**

Start with the `tol=1e-3` 3D Laplace case if the broader matrix produces too much noise before the first implementation pass.

**Step 4: Commit**

```bash
git add test/test_tree.jl
git commit -m "test: add fortran tree parity regression"
```

### Task 2: Introduce a Fortran-shaped Julia tree state

**Files:**
- Modify: `src/tree/tree.jl`

**Step 1: Write the failing test**

If needed, add a small internal-state sanity test in `test/test_tree.jl` that checks resulting tree ordering stays levelwise and parent/child indices are valid after materialization.

**Step 2: Run test to verify it fails**

Run the targeted tree test and confirm the failure is due to missing internal parity support.

**Step 3: Write minimal implementation**

Add an internal state container and helpers for:

- array allocation and growth
- root initialization
- child allocation
- level-address bookkeeping
- final materialization to `BoxTree`

Do not switch `build_tree(...)` yet.

**Step 4: Run tests**

Run: `julia --project --threads=1 -e 'using BoxDMK; include("test/test_tree.jl")'`

Expected: existing tests still pass except the parity regression.

**Step 5: Commit**

```bash
git add src/tree/tree.jl test/test_tree.jl
git commit -m "refactor: add fortran-shaped julia tree state"
```

### Task 3: Port adaptive refinement and `update_rints`

**Files:**
- Modify: `src/tree/tree.jl`
- Reference: `deps/boxdmk_fortran/src/common/tree_vol_coeffs.f`

**Step 1: Write the failing test**

Extend the parity test with explicit checks on box count and level histogram for the loose-tolerance case if not already present.

**Step 2: Run test to verify it fails**

Run the targeted tree parity test and confirm adaptive counts still diverge.

**Step 3: Write minimal implementation**

Port the Fortran logic for:

- `vol_tree_find_box_refine`
- `vol_tree_refine_boxes`
- `update_rints`

Use Fortran-style levelwise append order and incremental `rint` updates.

**Step 4: Run tests**

Run the tree parity test again and record whether the pre-LR structure now matches better.

**Step 5: Commit**

```bash
git add src/tree/tree.jl test/test_tree.jl
git commit -m "feat: port adaptive tree refinement from fortran"
```

### Task 4: Port `computecoll`

**Files:**
- Modify: `src/tree/tree.jl`
- Reference: `deps/boxdmk_fortran/src/common/tree_routs.f`

**Step 1: Write the failing test**

Ensure the parity test compares colleague ordering exactly, not just set equality.

**Step 2: Run test to verify it fails**

Run the targeted tree parity test and confirm colleague ordering still differs.

**Step 3: Write minimal implementation**

Replace the current same-level all-pairs colleague builder with a Julia port of `computecoll`.

**Step 4: Run tests**

Run the targeted tree test and confirm colleague arrays now match where adaptive structures already align.

**Step 5: Commit**

```bash
git add src/tree/tree.jl test/test_tree.jl
git commit -m "feat: port fortran colleague construction"
```

### Task 5: Port Fortran level-restriction closure and reorganization

**Files:**
- Modify: `src/tree/tree.jl`
- Reference: `deps/boxdmk_fortran/src/common/tree_vol_coeffs.f`

**Step 1: Write the failing test**

Keep the exact-parity regression centered on the `tol=1e-3` case and confirm it still fails after the adaptive and colleague ports.

**Step 2: Run test to verify it fails**

Run the targeted tree parity test and observe remaining differences in counts, ordering, or colleagues.

**Step 3: Write minimal implementation**

Port:

- `vol_tree_fix_lr`
- `vol_updateflags`
- `vol_tree_refine_boxes_flag`
- `vol_tree_reorg`

Use the same flag transitions and tail-box reorganization order as Fortran.

**Step 4: Run tests**

Run the targeted tree parity test and confirm the known `tol=1e-3` case passes exactly.

**Step 5: Commit**

```bash
git add src/tree/tree.jl test/test_tree.jl
git commit -m "feat: port fortran level-restricted tree closure"
```

### Task 6: Switch `build_tree(...)` to the parity backend for all supported configs

**Files:**
- Modify: `src/tree/tree.jl`
- Modify: `README.md` if behavior notes change

**Step 1: Write the failing test**

Expand the parity test matrix to include the representative supported configurations:

- kernels: Laplace, Yukawa, Sqrt-Laplace
- bases: Legendre, Chebyshev
- dimensions: 1, 2, 3 where supported by the wrapper

**Step 2: Run test to verify it fails**

Run the full tree test file and observe any remaining unsupported combinations.

**Step 3: Write minimal implementation**

Make the new backend the only path behind `build_tree(...)` and remove dependence on `_TreeBoxState` for public tree construction.

**Step 4: Run tests**

Run: `julia --project --threads=1 -e 'using BoxDMK; include("test/test_tree.jl"); include("test/test_fortran_wrapper.jl")'`

Expected: exact parity passes across the representative coverage matrix.

**Step 5: Commit**

```bash
git add src/tree/tree.jl test/test_tree.jl test/test_fortran_wrapper.jl README.md
git commit -m "feat: make julia tree builder mirror fortran exactly"
```

### Task 7: Run broader verification

**Files:**
- No code changes required unless failures reveal missing parity cases

**Step 1: Run focused tests**

Run:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_tree.jl")'
julia --project --threads=1 -e 'using BoxDMK; include("test/test_tree_data.jl")'
julia --project --threads=1 -e 'using BoxDMK; include("test/test_fortran_wrapper.jl")'
```

**Step 2: Run solver smoke coverage**

Run:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_solver.jl")'
```

**Step 3: Run the hybrid parity benchmark sanity check**

Run:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project benchmark/hybrid_parity.jl
```

**Step 4: Investigate and fix any regressions**

Only patch behavior if verification exposes a parity bug or an API break caused by the new backend.

**Step 5: Commit**

```bash
git add -A
git commit -m "test: verify fortran-parity tree backend"
```
