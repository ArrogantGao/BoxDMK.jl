# Vendored BoxDMK Fortran Reference

This directory is reserved for the repo-local copy of the Fortran reference sources
used by the hybrid parity benchmark/debug path.

Current status:

- source vendoring scaffold is present under `src/`, `benchmark/`, and `test/`
- the Julia wrappers still fall back to the external `libboxdmk.so` when the
  vendored build does not exist
- `deps/build_fortran_ref.jl` is the repo-local build entrypoint

The intended build output location is:

- `deps/usr/lib/libboxdmk.so`
- `deps/usr/lib/libboxdmk_debug.so`
