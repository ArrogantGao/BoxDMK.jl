# BoxDMK Repo Cleanup and Refactor Design

Date: 2026-03-24

## Goal

Clean up the repository layout and developer workflow without changing solver behavior.

This work covers three areas:

1. add proper ignore rules for generated artifacts,
2. reorganize `src/` into clearer internal groupings,
3. rewrite the top-level `README.md` so it reflects the actual current workflow.

## Scope

This is a maintenance refactor, not an algorithmic change.

The public API should remain unchanged:

- `build_tree`
- `bdmk`
- `build_tree_fortran`
- `bdmk_fortran`
- existing exported kernels, bases, and helper APIs

The internal include/load order may change, but runtime behavior and test results should not.

## Current Problems

### Generated artifacts are mixed into the repo tree

The repo currently contains generated outputs under:

- `deps/boxdmk_fortran/build/`
- `deps/boxdmk_fortran/build_callback/`
- `deps/boxdmk_fortran/build_hot/`
- `deps/usr/lib/`

These are build products, not source. The current `.gitignore` only covers Julia coverage files and docs output, so the Fortran/CMake outputs are not being treated as disposable.

### `src/` is flat and hard to scan

All implementation files currently live directly under `src/`, including:

- public entrypoints,
- core types/utilities,
- tree construction,
- solver stages,
- Fortran wrappers/debug interfaces.

This makes it harder to understand boundaries and load order, especially now that the hybrid Julia/Fortran path is a first-class part of the repo.

### The current README no longer matches the practical workflow

The top-level README still reads like a mostly pure-Julia package overview. The repo now also includes:

- vendored Fortran sources,
- local build helpers,
- hybrid parity/debug benchmarks,
- a practical hybrid path for the validated Laplace reference case.

The README should explain those paths directly.

## Proposed Structure

### 1. Repository hygiene

Expand `.gitignore` to ignore generated artifacts such as:

- shared libraries and object files,
- Fortran module files,
- CMake output directories and cache files,
- local benchmark result dumps,
- local build/install directories under `deps/usr/`.

The vendored Fortran source tree stays in the repo, but its generated build directories are treated as disposable.

### 2. Internal source layout

Reorganize `src/` into grouped subdirectories while keeping behavior unchanged.

Proposed layout:

- `src/core/`
  - `types.jl`
  - `utils.jl`
  - `basis.jl`
  - `tensor.jl`
  - `kernels.jl`
- `src/tree/`
  - `tree.jl`
  - `tree_data.jl`
  - `interaction_lists.jl`
- `src/solver/`
  - `sog.jl`
  - `proxy.jl`
  - `passes.jl`
  - `planewave.jl`
  - `boxfgt.jl`
  - `local_tables.jl`
  - `local.jl`
  - `derivatives.jl`
  - `solver.jl`
- `src/fortran/`
  - `fortran_paths.jl`
  - `fortran_hotpaths.jl`
  - `fortran_wrapper.jl`
  - `fortran_debug_wrapper.jl`

`src/BoxDMK.jl` should then become a thin manifest that loads these groups in a fixed order.

The move is intentionally organizational only:

- no namespace changes,
- no export changes,
- no semantic changes,
- no function renames unless required for load-order cleanup.

### 3. Documentation cleanup

Rewrite `README.md` with sections that match the actual repo:

- what BoxDMK is,
- what currently works well,
- Julia path vs Fortran path vs hybrid path,
- install and development setup,
- building vendored Fortran libraries,
- running tests and benchmarks,
- repository layout,
- current status/limitations.

The README should help both users and contributors understand the current state without reading benchmark scripts first.

## Behavior Preservation Strategy

The reorganization should preserve behavior by keeping:

- the same include order,
- the same exported names,
- the same test suite expectations,
- the same Fortran path resolution semantics.

This means the refactor should be incremental:

1. add or tighten tests around public/module surface where useful,
2. move files in small groups,
3. rerun targeted tests after each group,
4. finish with a broader verification pass.

## Testing Strategy

Use existing tests as the primary regression guard:

- module loading and API smoke tests,
- solver tests,
- Fortran wrapper tests,
- hybrid parity tests,
- selected low-level solver component tests.

The most important verification set after the refactor is:

- `test/test_solver.jl`
- `test/test_fortran_wrapper.jl`
- `test/test_hybrid_parity.jl`
- `test/test_boxfgt.jl`
- `test/test_passes.jl`
- `test/test_local.jl`

If include-order fragility shows up during moves, add a small module-surface regression test rather than changing behavior silently.

## Non-Goals

- Fixing remaining native Julia tree/solver parity gaps.
- Changing the hybrid dispatch policy.
- Changing benchmark math or tolerances.
- Removing the vendored Fortran source tree.
- Redesigning the public API.

## Risks

### Include-order breakage

Because the package uses many `include(...)` files with cross-file helpers, a path reorganization can accidentally change load order.

Mitigation:

- preserve current order exactly while moving files,
- verify after each small batch,
- keep `BoxDMK.jl` as the single authoritative load manifest.

### Accidental behavior changes during “cleanup”

Large refactors often slip in opportunistic edits. That should be avoided here.

Mitigation:

- separate structural moves from logic edits,
- keep diffs narrow,
- use existing tests as the acceptance criteria.

### Over-ignoring files

An aggressive `.gitignore` can hide files that should remain tracked.

Mitigation:

- target generated artifacts specifically,
- keep source/vendor files under `deps/boxdmk_fortran/src/` and docs tracked.

## Recommended Execution Order

1. Update `.gitignore` and remove generated-artifact expectations from the repo layout.
2. Add a small regression guard for module/API surface if needed.
3. Reorganize `src/` into grouped subdirectories while preserving load order.
4. Rewrite `README.md` to match the cleaned repo.
5. Run targeted tests, then a broader verification pass.
