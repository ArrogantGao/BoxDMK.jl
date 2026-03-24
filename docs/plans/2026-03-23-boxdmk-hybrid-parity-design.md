# BoxDMK Hybrid Julia-Fortran Parity Design

Date: 2026-03-23

## Goal

For the reference case

- Laplace
- 3D
- `norder=16`
- `eps=1e-6`
- potentials only

build a benchmark/debug pipeline inside this repo that:

1. produces the same tree and sampled density values as the Fortran reference,
2. measures true stage-by-stage error against Fortran intermediate arrays,
3. times each stage,
4. determines which stages should stay in Julia and which should call Fortran.

This work is scoped to the benchmark/debug path first. It does not change the public package API yet.

## Problem Summary

The current repo already wraps the external Fortran solver and several low-level hotpaths, but the end-to-end Julia solver is still incorrect on the reference case even when evaluated on the exact Fortran tree and `fvals`.

Current evidence:

- Tree parity is now much closer than earlier reports and the benchmark path currently matches Fortran tree counts.
- The final potential is still wrong by essentially `O(1)` relative error.
- The local contribution remains on the correct scale, while the proxy / plane-wave path blows up by roughly eight orders of magnitude.
- The current Fortran C API exposes per-step timings and final outputs, but not the intermediate stage arrays needed for a true stage-by-stage error trace.

Therefore the next step is not another blind optimization pass. The next step is to make the reference implementation observable enough to compare each solver stage directly.

## Chosen Approach

Vendor the minimal Fortran reference sources into this Julia repo, add a repo-local debug-capable shared library, and build a Julia-side hybrid driver that can swap Julia vs Fortran per stage while comparing all intermediate arrays against the vendored Fortran reference.

This is preferred over a full Fortran fallback because a fallback hides the stage boundary decision. It is also preferred over continuing to use the existing external `libboxdmk.so` unchanged because that API cannot provide the required stage arrays.

## Non-Goals

- General parity for Yukawa, square-root Laplace, gradients, Hessians, or arbitrary `norder`.
- Replacing the public `build_tree` / `bdmk` path immediately.
- Rewriting the vendored Fortran solver into idiomatic Julia before the stage boundary is known.

## Repo Changes

### 1. Vendor a bounded Fortran subtree

Add a new repo-local directory:

- `deps/boxdmk_fortran/`

This directory will contain only the Fortran sources needed to build the reference tree and solver path:

- `src/bdmk/bdmk4.f`
- `src/bdmk/bdmk_c_api.f90`
- `src/bdmk/bdmk_local.f`
- `src/bdmk/bdmk_local_tables.f`
- `src/bdmk/bdmk_pwrouts.f`
- `src/bdmk/bdmk_pwterms.f`
- `src/bdmk/sogapproximation/*`
- `src/common/dmk_routs.f`
- `src/common/polytens.f`
- `src/common/tensor_prod_routs.f`
- `src/common/tree_data_routs.f`
- `src/common/tree_routs.f`
- `src/common/tree_vol_coeffs.f`
- supporting special-function / BLAS wrapper sources required by the build

The vendored copy must be sufficient to compile without depending on `/mnt/home/xgao1/codes/boxdmk`.

### 2. Add a repo-local build path

Add Julia build helpers under `deps/` to compile:

- the standard shared library used by current wrappers,
- a debug shared library that exports the intermediate arrays for the reference case.

Expected new files:

- `deps/build_fortran_ref.jl`
- `deps/fortran_paths.jl`

The build should prefer local vendored sources and only fall back to an existing external build if explicitly configured.

### 3. Add debug entrypoints to the vendored Fortran C API

Extend the vendored `bdmk_c_api.f90` with new `bind(C)` entrypoints that expose:

- tree outputs,
- final outputs,
- per-step timings,
- intermediate arrays after each solver stage for the reference case.

The key requirement is not a generic perfect ABI for every internal buffer. The key requirement is a stable enough debug ABI to compare the reference-case stages.

The minimum useful intermediate outputs are:

- Taylor-corrected potential after step 2
- proxy charges after step 3
- plane-wave multipole buffer after step 4
- local plane-wave buffer after step 5
- proxy potential after step 6
- accumulated potential after step 7
- accumulated potential after step 8
- final potential after step 9

If exposing steps 4 and 5 directly is too invasive, the fallback is to expose the post-step-4 and post-step-6 proxy-adjacent state first and then tighten observability later.

### 4. Build a Julia hybrid benchmark/debug driver

Add a dedicated Julia driver that:

- constructs the same reference problem,
- builds either a Julia tree or a Fortran tree,
- samples the same `fvals`,
- runs the vendored Fortran debug solver,
- runs the Julia stage pipeline,
- compares stage arrays,
- can replace individual Julia stages with Fortran calls.

Expected new files:

- `benchmark/hybrid_parity.jl`
- `test/test_hybrid_parity.jl`

This driver is the decision engine for the language split.

## Stage Boundary Strategy

The hybrid driver will evaluate the 9 stages in order.

### Tree construction

Keep Julia tree build only if it matches the vendored Fortran tree topology and sampled `fvals` for the reference case. Otherwise use the vendored Fortran tree build in the benchmark/debug path.

### Step 1: Precomputation

Compare:

- SOG nodes,
- PW levels,
- interaction lists,
- local tables.

This step is likely to remain Julia if the arrays match, because it has no callback issue and is not the dominant runtime relative to steps 4, 6, and 7.

### Step 2: Taylor correction

Compare the step-2 accumulated potential directly. This step is a correctness gate before any plane-wave work.

### Step 3: Upward / proxy charges

Current low-level pass kernels already have Fortran wrappers, so this step can be flipped independently and should be used to determine whether the mismatch starts before PW conversion.

### Steps 4-6: PW path

These are the most suspicious correctness region and one of the main performance bottlenecks.

Initial expectation:

- Julia may remain acceptable for setup and bookkeeping,
- Fortran is a strong candidate for `charge->PW`, `M2L`, and `PW->proxy`.

The stage comparison will decide this with direct array evidence.

### Step 7: Local interactions

This is already a strong Fortran candidate for runtime reasons. The driver will still verify that the Julia and Fortran local accumulation match exactly on the reference tree.

### Step 8: Asymptotic

This step is smaller in total cost but still needs correctness verification against vendored Fortran.

### Step 9: Proxy to potential

This step already has a Fortran hotpath wrapper and should be benchmarked as a final isolated choice.

## Decision Rule

For each stage:

- If Julia and Fortran match within tight tolerance and Julia time is acceptable, keep Julia.
- If Julia is correct but much slower, use Fortran in the hybrid path.
- If Julia is not correct, use Fortran immediately and treat the Julia implementation as a separate debugging task.

The end result should be a per-stage table:

- stage name,
- Julia error vs Fortran,
- Julia time,
- Fortran time,
- chosen implementation,
- rationale.

## Testing Plan

### Benchmark/debug regression

Add a regression test for the reference case that checks:

- tree parity,
- sampled `fvals` parity,
- final potential parity for the chosen hybrid split.

### Stage checks

Add targeted tests for each exported Fortran debug array:

- shape,
- deterministic values,
- tolerance against Julia stage output when Julia is expected to match.

### Build checks

Add a lightweight test that confirms the vendored shared library can be located and loaded from the repo-local build path.

## Risks

### ABI fragility

Fortran internal arrays are not currently designed as a stable debug ABI. The fix is to add an explicitly debug-only C layer and keep the surface small.

### Over-vendoring

Copying the whole external repo would add too much noise. The solution is to vendor only the source closure needed by the reference case.

### Confusing correctness with performance

The hybrid driver must compare stage arrays before using time to decide the language split. Correctness gates come first.

## Recommended Execution Order

1. Vendor the minimal Fortran source closure and compile a repo-local shared library.
2. Add debug C entrypoints for the reference solver stages.
3. Build the Julia hybrid parity driver.
4. Produce the first true stage-by-stage error table.
5. Flip stages to Fortran one by one until the final output matches.
6. Benchmark the chosen split and document the result.
