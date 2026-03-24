# BoxDMK Repo Cleanup and Refactor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Clean up generated artifacts, reorganize the internal source tree, and rewrite the top-level README without changing package behavior.

**Architecture:** Treat this as a behavior-preserving maintenance refactor. First tighten repo hygiene, then move implementation files into clearer internal groups while preserving include order, exports, and wrapper semantics, and finally rewrite the repo documentation to match the current hybrid Julia/Fortran workflow.

**Tech Stack:** Julia package layout, vendored Fortran build outputs, CMake-generated artifacts, Markdown documentation, existing BoxDMK test suite

---

### Task 1: Expand Ignore Rules For Generated Artifacts

**Files:**
- Modify: `.gitignore`

**Step 1: Inspect the current generated artifacts that should be ignored**

Check:

- `deps/boxdmk_fortran/build/`
- `deps/boxdmk_fortran/build_callback/`
- `deps/boxdmk_fortran/build_hot/`
- `deps/usr/lib/`
- benchmark result dumps and local temporary outputs

**Step 2: Update `.gitignore` with repo-specific generated outputs**

Add ignore rules for:

- Fortran/C/CMake outputs: `*.so`, `*.o`, `*.mod`, `*.a`, `*.dll`, `*.dylib`, `a.out`
- vendored Fortran build dirs: `/deps/boxdmk_fortran/build*/`
- local install dirs: `/deps/usr/`
- CMake metadata: `CMakeCache.txt`, `CMakeFiles/`, `cmake_install.cmake`, `Makefile`, `CTestTestfile.cmake`
- generated benchmark outputs such as `/benchmark/results*.txt`

Do not ignore:

- `deps/boxdmk_fortran/src/`
- `deps/boxdmk_fortran/README.md`
- tracked docs/plans content

**Step 3: Verify the ignore rules match the repo state**

Run:

```bash
git status --short
```

Expected:

- source files still appear when modified
- generated Fortran/CMake outputs stop showing up as untracked changes after cleanup

### Task 2: Add A Module Surface Regression Guard

**Files:**
- Modify: `test/runtests.jl`
- Create or Modify: `test/test_module_surface.jl`

**Step 1: Write a small regression test for the package surface**

Cover:

- `using BoxDMK` loads successfully
- `build_tree`, `bdmk`, `build_tree_fortran`, `bdmk_fortran` are defined
- key exported types such as `LaplaceKernel`, `LegendreBasis`, `SolverResult` remain available

**Step 2: Run the new test to verify the current baseline**

Run:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_module_surface.jl")'
```

Expected:

- pass on the current baseline before moving files

### Task 3: Reorganize Core Files Into `src/core/`

**Files:**
- Move: `src/types.jl` -> `src/core/types.jl`
- Move: `src/utils.jl` -> `src/core/utils.jl`
- Move: `src/basis.jl` -> `src/core/basis.jl`
- Move: `src/tensor.jl` -> `src/core/tensor.jl`
- Move: `src/kernels.jl` -> `src/core/kernels.jl`
- Modify: `src/BoxDMK.jl`

**Step 1: Move the files without changing their contents**

Keep file contents unchanged except for path-sensitive comments if any.

**Step 2: Update `src/BoxDMK.jl` include paths**

Preserve the same effective load order.

**Step 3: Run the module-surface and basic tests**

Run:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_module_surface.jl")'
julia --project --threads=1 -e 'using BoxDMK; include("test/test_types.jl"); include("test/test_basis.jl"); include("test/test_tensor.jl"); include("test/test_kernels.jl")'
```

Expected:

- all pass

### Task 4: Reorganize Tree Files Into `src/tree/`

**Files:**
- Move: `src/tree.jl` -> `src/tree/tree.jl`
- Move: `src/tree_data.jl` -> `src/tree/tree_data.jl`
- Move: `src/interaction_lists.jl` -> `src/tree/interaction_lists.jl`
- Modify: `src/BoxDMK.jl`

**Step 1: Move tree-related files as a batch**

Do not alter behavior or APIs.

**Step 2: Update include paths in `src/BoxDMK.jl`**

Keep tree-related includes after core includes and before solver components that depend on them.

**Step 3: Run tree-focused tests**

Run:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_tree.jl"); include("test/test_tree_data.jl")'
```

Expected:

- all pass

### Task 5: Reorganize Solver Files Into `src/solver/`

**Files:**
- Move: `src/sog.jl` -> `src/solver/sog.jl`
- Move: `src/proxy.jl` -> `src/solver/proxy.jl`
- Move: `src/passes.jl` -> `src/solver/passes.jl`
- Move: `src/planewave.jl` -> `src/solver/planewave.jl`
- Move: `src/boxfgt.jl` -> `src/solver/boxfgt.jl`
- Move: `src/local_tables.jl` -> `src/solver/local_tables.jl`
- Move: `src/local.jl` -> `src/solver/local.jl`
- Move: `src/derivatives.jl` -> `src/solver/derivatives.jl`
- Move: `src/solver.jl` -> `src/solver/solver.jl`
- Modify: `src/BoxDMK.jl`

**Step 1: Move the solver implementation files**

Keep the current dependency order:

- SOG/proxy/passes
- planewave/boxfgt/local
- derivatives
- top-level solver

**Step 2: Update include paths while preserving load order**

Be careful that helper functions used across files are still defined before use.

**Step 3: Run solver component tests**

Run:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_sog.jl"); include("test/test_proxy.jl"); include("test/test_planewave.jl"); include("test/test_passes.jl"); include("test/test_boxfgt.jl"); include("test/test_local.jl"); include("test/test_derivatives.jl"); include("test/test_solver.jl")'
```

Expected:

- all pass

### Task 6: Reorganize Fortran Integration Files Into `src/fortran/`

**Files:**
- Move: `src/fortran_paths.jl` -> `src/fortran/fortran_paths.jl`
- Move: `src/fortran_hotpaths.jl` -> `src/fortran/fortran_hotpaths.jl`
- Move: `src/fortran_wrapper.jl` -> `src/fortran/fortran_wrapper.jl`
- Move: `src/fortran_debug_wrapper.jl` -> `src/fortran/fortran_debug_wrapper.jl`
- Modify: `src/BoxDMK.jl`

**Step 1: Move the Fortran support files**

Keep current wrapper semantics and exported names unchanged.

**Step 2: Update include paths in `src/BoxDMK.jl`**

Keep Fortran path resolution loaded before hotpaths and wrappers, and keep the debug wrapper last.

**Step 3: Run wrapper and hybrid tests**

Run:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_fortran_wrapper.jl"); include("test/test_hybrid_parity.jl")'
```

Expected:

- all pass

### Task 7: Simplify `src/BoxDMK.jl`

**Files:**
- Modify: `src/BoxDMK.jl`

**Step 1: Replace the flat include list with grouped sections**

Use clearly labeled sections for:

- core
- tree
- solver
- Fortran

Avoid changing exports or public names.

**Step 2: Remove stale conditional includes if no longer needed**

Only remove `isfile(...)` guards when the moved files are now required package files.

**Step 3: Run the module-surface test again**

Run:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_module_surface.jl")'
```

Expected:

- pass

### Task 8: Rewrite `README.md`

**Files:**
- Modify: `README.md`

**Step 1: Rewrite the opening description**

Describe the package as:

- a Julia implementation with a current hybrid Julia/Fortran workflow,
- a repo that vendors the Fortran reference for parity/debugging,
- a package with both user-facing and developer-facing paths.

**Step 2: Add practical setup and usage sections**

Cover:

- package usage
- building vendored Fortran libraries with `julia --project deps/build_fortran_ref.jl`
- when `bdmk` uses native Julia vs the validated hybrid dispatch
- key benchmarks/tests

**Step 3: Add contributor-oriented repo layout**

Document:

- `src/core/`
- `src/tree/`
- `src/solver/`
- `src/fortran/`
- `benchmark/`
- `deps/boxdmk_fortran/`

**Step 4: Review for accuracy against the current codebase**

Check that the README matches actual commands, paths, and current limitations.

### Task 9: Run The Verification Sweep

**Files:**
- Test only

**Step 1: Run targeted regression tests**

Run:

```bash
julia --project --threads=1 -e 'using BoxDMK; include("test/test_module_surface.jl"); include("test/test_tree.jl"); include("test/test_tree_data.jl"); include("test/test_passes.jl"); include("test/test_boxfgt.jl"); include("test/test_local.jl"); include("test/test_solver.jl"); include("test/test_fortran_wrapper.jl"); include("test/test_hybrid_parity.jl")'
```

Expected:

- all pass

**Step 2: Run the reference benchmark/debug driver**

Run:

```bash
JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project benchmark/hybrid_parity.jl
```

Expected:

- benchmark completes
- the hybrid candidate still reports correct parity for the validated reference case

**Step 3: Review git status**

Run:

```bash
git status --short
```

Expected:

- only intentional source/doc changes remain
- generated build artifacts are no longer part of the intended diff
