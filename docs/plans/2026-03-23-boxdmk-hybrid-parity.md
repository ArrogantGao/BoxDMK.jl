# BoxDMK Hybrid Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a repo-local hybrid Julia-Fortran benchmark/debug pipeline for the Laplace 3D reference case that measures stage-by-stage error against vendored Fortran and chooses Julia vs Fortran per step.

**Architecture:** Vendor the minimal Fortran source closure into `deps/boxdmk_fortran/`, compile a repo-local debug shared library, and expose new debug entrypoints for the reference tree/solver stages. Add a Julia driver that runs the reference problem on the same tree and `fvals`, compares each stage against Fortran, and can swap stages between Julia and Fortran to converge on a correct and fast split.

**Tech Stack:** Julia, `ccall`, vendored Fortran sources, C ABI shims via `bind(C)`, BoxDMK solver/tree code, benchmark harnesses

---

### Task 1: Add failing tests for the hybrid parity harness

**Files:**
- Create: `test/test_hybrid_parity.jl`
- Modify: `test/runtests.jl`

**Step 1: Write the failing test for the repo-local Fortran library locator**

Add a test that expects a new helper to resolve the vendored Fortran shared library path.

**Step 2: Write the failing test for the reference-case driver**

Add a test that expects a callable Julia entrypoint for the hybrid parity benchmark to return:

- a tree comparison summary,
- a per-step timing table,
- a per-step error table.

**Step 3: Run the tests to verify they fail**

Run: `julia --project --threads=1 -e 'using Pkg; Pkg.test(; test_args=["test_hybrid_parity"])'`

Expected: failure because the new helpers and driver do not exist yet.

### Task 2: Vendor the minimal Fortran source closure

**Files:**
- Create: `deps/boxdmk_fortran/README.md`
- Create: `deps/boxdmk_fortran/src/...`

**Step 1: Create the vendor directory skeleton**

Create the repo-local layout under `deps/boxdmk_fortran/`.

**Step 2: Copy the required Fortran sources**

Copy only the needed source closure from `/mnt/home/xgao1/codes/boxdmk/src` into the vendor directory.

**Step 3: Record provenance**

Document the upstream commit / source path and the reference-case scope in `deps/boxdmk_fortran/README.md`.

**Step 4: Verify the vendor tree is self-contained enough for build scripting**

Run: `find deps/boxdmk_fortran -type f | sort`

Expected: the required source set is present in-repo.

### Task 3: Add repo-local build helpers for the vendored Fortran library

**Files:**
- Create: `deps/build_fortran_ref.jl`
- Create: `deps/fortran_paths.jl`
- Modify: `src/fortran_wrapper.jl`
- Modify: `src/fortran_hotpaths.jl`

**Step 1: Implement path resolution**

Add helpers that prefer the vendored shared library path over the external `/mnt/home/xgao1/codes/boxdmk/build/libboxdmk.so`.

**Step 2: Implement the repo-local build script**

Write a Julia build helper that compiles the vendored Fortran sources into a shared library in a deterministic location under `deps/`.

**Step 3: Route existing wrappers through the new resolver**

Replace hard-coded library paths in the Julia wrappers with the resolver.

**Step 4: Run the build helper**

Run: `julia --project deps/build_fortran_ref.jl`

Expected: a repo-local shared library is produced and can be found by the resolver.

### Task 4: Add failing tests for Fortran debug entrypoints

**Files:**
- Modify: `test/test_hybrid_parity.jl`
- Modify: `deps/boxdmk_fortran/src/bdmk/bdmk_c_api.f90`

**Step 1: Define the expected debug payloads**

Add tests that expect debug entrypoints for:

- tree build summary,
- final solver output,
- intermediate stage arrays for the reference case.

**Step 2: Run the tests to verify they fail**

Run: `julia --project --threads=1 -e 'using Pkg; Pkg.test(; test_args=["test_hybrid_parity"])'`

Expected: failure because the debug entrypoints are not implemented yet.

### Task 5: Implement the vendored Fortran debug ABI

**Files:**
- Modify: `deps/boxdmk_fortran/src/bdmk/bdmk_c_api.f90`
- Modify: `deps/boxdmk_fortran/src/bdmk/bdmk4.f`
- Modify: vendored helper sources only if required by the debug export

**Step 1: Add debug-only C entrypoints**

Expose stable debug entrypoints for the reference tree/solver path.

**Step 2: Capture per-step intermediate arrays**

Store the stage arrays needed for parity checks:

- post-step-2 potential,
- post-step-3 proxy charges,
- post-step-4 PW or equivalent exposed state,
- post-step-6 proxy potential,
- post-step-7 accumulated potential,
- post-step-8 accumulated potential,
- final potential.

**Step 3: Rebuild the vendored shared library**

Run: `julia --project deps/build_fortran_ref.jl`

Expected: the shared library rebuilds successfully with the debug symbols.

### Task 6: Add Julia wrappers for the debug ABI

**Files:**
- Create: `src/fortran_debug_wrapper.jl`
- Modify: `src/BoxDMK.jl`

**Step 1: Wrap the new debug entrypoints**

Add Julia `ccall` wrappers for the reference-case debug API.

**Step 2: Normalize outputs into Julia-native arrays**

Convert the raw Fortran buffers into shapes that match the Julia pipeline outputs.

**Step 3: Export only the benchmark/debug surface needed now**

Keep the public package API unchanged except for explicit debug helpers.

### Task 7: Add the Julia hybrid parity driver

**Files:**
- Create: `benchmark/hybrid_parity.jl`

**Step 1: Build the reference-case problem generator**

Reuse the exact benchmark RHS and reference targets already established in `benchmark/julia_vs_fortran.jl`.

**Step 2: Run both trees and compare tree outputs**

Add a driver mode for:

- Julia tree build,
- vendored Fortran tree build,
- sampled `fvals` comparison.

**Step 3: Run the Julia and Fortran stage pipelines on the same tree**

Use the debug wrappers to get stage arrays from Fortran and compare them against `timed_julia_pipeline`.

**Step 4: Emit a machine-readable report**

Return:

- tree metrics,
- per-step errors,
- per-step timings,
- stage norms.

### Task 8: Add per-stage switching between Julia and Fortran

**Files:**
- Modify: `benchmark/hybrid_parity.jl`
- Modify: `src/fortran_debug_wrapper.jl`

**Step 1: Add a stage-selection configuration**

Represent the 9 solver stages as a switchable configuration with Julia or Fortran chosen per step.

**Step 2: Implement the first mixed runs**

Start with likely Fortran candidates:

- step 4 `charge->PW`
- step 6 `PW->proxy+down`
- step 7 `local`
- step 9 `proxy->pot`

**Step 3: Measure and record the mixed-path output**

For each mixed run, record:

- final error vs Fortran,
- per-step errors where available,
- wall time.

### Task 9: Lock the first correct split

**Files:**
- Modify: `benchmark/hybrid_parity.jl`
- Modify: `test/test_hybrid_parity.jl`

**Step 1: Find the minimal split with correct final potential**

Iterate on the stage-selection table until the mixed path matches Fortran on the reference case within tight tolerance.

**Step 2: Add a regression test for the chosen split**

Assert:

- tree parity if the chosen path uses Julia tree build,
- final potential parity,
- deterministic stage-selection configuration.

**Step 3: Run targeted tests**

Run: `julia --project --threads=1 -e 'using Pkg; Pkg.test(; test_args=["test_hybrid_parity"])'`

Expected: the hybrid parity tests pass.

### Task 10: Verify existing wrappers and benchmark output still work

**Files:**
- Test: `test/test_fortran_wrapper.jl`
- Test: `test/test_local.jl`
- Test: `test/test_passes.jl`
- Test: `test/test_boxfgt.jl`
- Test: `benchmark/julia_vs_fortran.jl`

**Step 1: Run the wrapper and hotpath tests**

Run: `julia --project --threads=1 -e 'using Pkg; Pkg.test(; test_args=["test_fortran_wrapper","test_local","test_passes","test_boxfgt"])'`

Expected: existing wrapper-level tests still pass with the vendored library path.

**Step 2: Run the reference benchmark**

Run: `JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project benchmark/hybrid_parity.jl`

Expected: it prints a per-step error table and timing table for the chosen hybrid split.

**Step 3: Run the current comparison benchmark**

Run: `JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 julia --project benchmark/julia_vs_fortran.jl`

Expected: the report still runs and now uses the repo-local library resolver.
