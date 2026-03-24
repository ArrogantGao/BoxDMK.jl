# BoxDMK Fortran Hotpaths Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the remaining Julia tensor/local hot loops with the preselected Fortran routines while preserving solver behavior.

**Architecture:** Add explicit Julia wrappers for the required Fortran entry points, then route the local interactions and tree passes through those wrappers using the existing array layouts. Reuse the already-established Fortran PW conversion path inside fat-Gaussian handling so the single-box fallback matches `boxfgt!`.

**Tech Stack:** Julia, `ccall`, BoxDMK local tables / proxy passes / plane-wave tables, libboxdmk Fortran routines

---

### Task 1: Add failing tests for new wrappers and hotpaths

**Files:**
- Modify: `test/test_local.jl`
- Modify: `test/test_passes.jl`

**Step 1: Write the failing tests**

Add a test that compares `BoxDMK._f_tens_prod_to_potloc!` against the current local sparse reference implementation for one 3D box interaction.

Add a test that compares `BoxDMK._f_tens_prod_trans!` against `BoxDMK.tensor_product_apply!` for one child transform.

**Step 2: Run tests to verify they fail**

Run: `julia --project --threads=1 -e 'using BoxDMK; include("test/test_local.jl"); include("test/test_passes.jl")'`

Expected: failure because the new wrapper functions are not defined yet.

### Task 2: Implement the wrappers and route the hotpaths

**Files:**
- Modify: `src/fortran_hotpaths.jl`
- Modify: `src/local.jl`
- Modify: `src/passes.jl`
- Modify: `src/boxfgt.jl`

**Step 1: Add `ccall` wrappers**

Implement `_f_tens_prod_to_potloc!` and `_f_tens_prod_trans!` with the exact libboxdmk ABI.

**Step 2: Replace local interaction inner loop**

Pre-convert `tables.ind` to `Array{Cint}` once at the start of `apply_local!`, then call `_f_tens_prod_to_potloc!` for each `(ibox, jbox, idelta)` triple.

**Step 3: Replace upward/downward tensor products**

Build `umat_nd(::Array{Float64,3})` per child and call `_f_tens_prod_trans!` once per density for both upward and downward passes.

**Step 4: Route fat-Gaussian PW conversions through Fortran**

Use the same `_FORTRAN_HOTPATHS_AVAILABLE[] && tree.ndim == 3` switch already used by `boxfgt!` when converting proxy charges to/from PW space.

### Task 3: Verify green

**Files:**
- Test: `test/test_solver.jl`
- Test: `test/test_local.jl`
- Test: `test/test_passes.jl`

**Step 1: Run targeted tests**

Run: `julia --project --threads=1 -e 'using BoxDMK; include("test/test_solver.jl")'`

Run: `julia --project --threads=1 -e 'using BoxDMK; include("test/test_local.jl")'`

Run: `julia --project --threads=1 -e 'using BoxDMK; include("test/test_passes.jl")'`

Run: `julia --project --threads=4 -e 'using BoxDMK; include("test/test_solver.jl")'`

Expected: all pass with the new Fortran-backed paths enabled when `libboxdmk.so` is present.
