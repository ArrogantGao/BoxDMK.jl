# Hybrid Default Solver Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make validated public Laplace solves default to Julia tree construction plus Fortran solve, and fail package initialization when the vendored Fortran solve library is missing.

**Architecture:** Keep tree construction in Julia, route validated public Laplace solves through the existing Fortran wrapper, and restore Julia tree ordering on returned box-based arrays. Keep the native Julia solver for kernels outside that safe slice. Add a small initialization helper so the package validates the Fortran solve dependency up front with a clear error.

**Tech Stack:** Julia, vendored Fortran shared libraries, Julia `ccall`, package tests, Markdown docs.

---

### Task 1: Add a failing public-dispatch regression

**Files:**
- Modify: `test/test_hybrid_parity.jl`
- Test: `test/test_hybrid_parity.jl`

**Step 1: Write the failing test**

Add a test that:

- builds a small Julia tree outside the current narrow reference slice,
- calls `bdmk(tree, fvals, kernel; eps=..., targets=...)`,
- compares its outputs to `bdmk_fortran(tree, fvals, kernel; eps=..., targets=...)`,
- checks `pot`, `target_pot`, and, for an additional case, `grad` and `hess`.

**Step 2: Run test to verify it fails**

Run:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_hybrid_parity.jl")'
```

Expected: FAIL because public `bdmk(...)` still uses the native Julia solver outside the old reference gate.

**Step 3: Write minimal implementation**

Generalize the public hybrid wrapper and dispatch gate in `src/solver/solver.jl`.

**Step 4: Run test to verify it passes**

Run:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_hybrid_parity.jl")'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add test/test_hybrid_parity.jl src/solver/solver.jl
git commit -m "feat: default public solver to hybrid fortran path"
```

### Task 2: Add a failing init-check regression

**Files:**
- Modify: `test/test_module_surface.jl`
- Modify: `src/BoxDMK.jl`
- Modify: `src/fortran/fortran_paths.jl`
- Test: `test/test_module_surface.jl`

**Step 1: Write the failing test**

Add a test for a helper like `_require_fortran_solve_library!()` that:

- succeeds when the solve library exists,
- throws an error with the build command when passed a missing path or a mocked missing-path resolver.

**Step 2: Run test to verify it fails**

Run:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_module_surface.jl")'
```

Expected: FAIL because the helper does not exist yet.

**Step 3: Write minimal implementation**

Add the helper and call it from `__init__`.

**Step 4: Run test to verify it passes**

Run:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_module_surface.jl")'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/BoxDMK.jl src/fortran/fortran_paths.jl test/test_module_surface.jl
git commit -m "feat: require fortran solve library at package init"
```

### Task 3: Restore Julia box order for all public hybrid outputs

**Files:**
- Modify: `src/solver/solver.jl`
- Test: `test/test_hybrid_parity.jl`

**Step 1: Write the failing test**

Extend the public hybrid test to assert that:

- `pot` matches the reorder-corrected Fortran reference,
- `grad` and `hess` also match reorder-corrected references in a small case.

**Step 2: Run test to verify it fails**

Run:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_hybrid_parity.jl")'
```

Expected: FAIL if gradients and Hessians are not reordered or are dropped.

**Step 3: Write minimal implementation**

Generalize the reorder helper in `src/solver/solver.jl` for 3D, 4D, and 5D box-indexed outputs and return the reordered solver result from the public hybrid wrapper.

**Step 4: Run test to verify it passes**

Run:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_hybrid_parity.jl")'
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/solver/solver.jl test/test_hybrid_parity.jl
git commit -m "fix: preserve julia box order in hybrid solver outputs"
```

### Task 4: Rewrite the README around the default hybrid workflow

**Files:**
- Modify: `README.md`

**Step 1: Write the failing test**

There is no automated README test in this repo. Use a manual checklist instead:

- default workflow shown as `build_tree(...)` + `bdmk(...)`,
- Fortran build step appears before normal usage,
- direct `bdmk_fortran(...)` described as advanced/debug API,
- limitations section updated to match the new default.

**Step 2: Run test to verify it fails**

Inspect the current README and confirm the default workflow is not described that way.

**Step 3: Write minimal implementation**

Update the status, installation, quick-start, and Fortran sections in `README.md`.

**Step 4: Run test to verify it passes**

Re-read the updated README and verify the checklist above.

**Step 5: Commit**

```bash
git add README.md
git commit -m "docs: make hybrid workflow the default in readme"
```

### Task 5: Verify end-to-end behavior

**Files:**
- No code changes expected

**Step 1: Run focused tests**

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_module_surface.jl"); include("test/test_hybrid_parity.jl")'
julia --project --threads=1 -e 'using BoxDMK; include("test/test_solver.jl"); include("test/test_fortran_wrapper.jl")'
```

Expected: PASS.

**Step 2: Run reference benchmark**

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project benchmark/hybrid_parity.jl
```

Expected: the benchmark still reports low hybrid target error on the validated Laplace reference case.

**Step 3: Clean generated artifacts if needed**

```bash
git clean -fdX -- deps/usr deps/boxdmk_fortran/build deps/boxdmk_fortran/build_callback deps/boxdmk_fortran/build_hot benchmark/results.txt
```

Expected: only ignored generated artifacts are removed.

**Step 4: Commit**

```bash
git add docs/plans/2026-03-24-hybrid-default-solver-design.md docs/plans/2026-03-24-hybrid-default-solver.md
git commit -m "docs: add hybrid default solver design and plan"
```
