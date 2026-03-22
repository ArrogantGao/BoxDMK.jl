# BoxDMK.jl Optimization Plan

## Current State
- Julia eps=1e-6: **127s** (was >30s timeout before Phase 1 fixes)
- Fortran eps=1e-6: **0.525s**
- Gap: **~240x** slower

## Phase 2: Fortran-Style Tensor Product (expected ~5-10x speedup)

### Problem
`_tensor_product_apply_rect!` in `src/proxy.jl` and `_tensor_product_apply_dim!` in `src/tensor.jl` loop over suffix slices doing many small gemm calls. The Fortran does 3 large zgemm calls total by reshaping arrays between dimensions.

### Changes

**File: `src/proxy.jl` — `_tensor_product_apply_rect!`**

Replace the current per-suffix-slice loop with the Fortran approach. For each dimension, reshape the data so the transform dimension is either the leading or trailing dimension of a 2D matrix, then do ONE gemm.

For `ndim=3`, `nd=1`, transforming from `from_order=n` to `to_order=m`:

**Dim 1 (z-transform):** Source is `(n, n, n)` in memory = `(n², n)` as 2D.
```julia
# src_2d = reshape(src, n², n)
# dest_2d = src_2d * mat'  →  (n², m)
# This is ONE gemm: (n², n) @ (n, m)
mul!(dest_2d, src_2d, transpose(mat))
```

**Dim 2 (y-transform):** After dim 1, data is `(n, m, n)` = need to transform middle dim.
Fortran transposes to `(n, m, n)` → `(n, m*n)` then transposes blocks to get `(n, m, n)` ready.
Actually, the Fortran approach for dim 2:
1. Transpose `ff(n, n, npw2)` → `fft(n, npw2, n)` (explicit loop)
2. Then `zgemm('n','n', npw, npw2*n, n, ...)` — one gemm treating `fft` as `(n, npw2*n)`.

In Julia, we can do:
```julia
# After dim 1: data is (n, n, m) in memory where m = to_order for z
# We want to apply mat along the middle (y) dimension
# Reshape as (n, n*m) and note data[:, j + (k-1)*n] = data[i, j, k]
# Permute dims 1,2: tmp[j, k, i] = data[i, j, k] for all i
# Then tmp as (n, m*n_x) matrix, apply mat: result = mat * tmp → (m_y, m*n_x)
permutedims!(tmp, reshape(data, n_x, n, m_z), (2, 3, 1))
tmp_2d = reshape(tmp, n, m_z * n_x)
result_2d = mat * tmp_2d  # (m_y, m_z * n_x)
```

**Dim 3 (x-transform):** After dims 1,2, data needs similar treatment.

**Simplified approach:** Instead of exactly matching Fortran's manual reshuffling, use a simpler pattern:
- For each dimension `d`, reshape the nd-dimensional tensor so that the `d`-th axis is the *last* axis of a 2D matrix, then apply `mat'` from the right:
  ```
  reshape(tensor, product_of_other_dims, from_order) @ mat' → (product_of_other_dims, to_order)
  ```
- This requires permuting/transposing between dimensions. Use `permutedims` for the rearrangement.

**Actually, the simplest correct approach:**

For a 3D tensor product `out[i,j,k] = sum_{a,b,c} mat[i,a]*mat[j,b]*mat[k,c]*src[a,b,c]`, we can apply dimension by dimension. For each dim, treat the data as a 2D matrix where the target dimension is one axis and everything else is the other axis, then do one gemm.

The current code already does this but with a per-slice loop when `prefix > 1`. The fix: **transpose the array so the target dimension becomes the leading dimension**, then reshape as 2D and do one gemm.

```julia
function _tensor_product_apply_rect!(out, mat, vals, from_order::Int, to_order::Int, ndim::Int, nd::Int)
    # ... existing checks ...

    src = copy(vals)  # or use a pre-allocated buffer

    for dim in 1:ndim
        # Current tensor shape: to_order^(dim-1) * from_order^(ndim-dim+1) total elements per density
        # We want to apply mat along the dim-th original axis

        prefix = to_order^(dim - 1)
        from_remaining = from_order^(ndim - dim + 1)
        suffix = from_order^(ndim - dim)

        ncols_out = to_order^dim * from_order^(ndim - dim)
        dest = dim == ndim ? out : Matrix{eltype(out)}(undef, nd, ncols_out)

        for iv in 1:nd
            if prefix == 1
                # First dim: src is (from_order, suffix) in memory
                src_2d = reshape(@view(src[iv, :]), from_order, suffix)
                dest_2d = reshape(@view(dest[iv, :]), to_order, suffix)
                mul!(dest_2d, mat, src_2d)
            elseif suffix == 1
                # Last dim: src is (prefix, from_order) in memory
                src_2d = reshape(@view(src[iv, :]), prefix, from_order)
                dest_2d = reshape(@view(dest[iv, :]), prefix, to_order)
                mul!(dest_2d, src_2d, transpose(mat))
            else
                # Middle dim: need to reshape
                # src layout: (prefix, from_order, suffix)
                # Reshape to (prefix * suffix, from_order) by permuting dims
                src_3d = reshape(@view(src[iv, :]), prefix, from_order, suffix)
                # Permute to (prefix, suffix, from_order) then reshape to (prefix*suffix, from_order)
                tmp = permutedims(src_3d, (1, 3, 2))
                tmp_2d = reshape(tmp, prefix * suffix, from_order)
                result_2d = tmp_2d * transpose(mat)  # (prefix*suffix, to_order)
                # Permute back: (prefix, suffix, to_order) → (prefix, to_order, suffix)
                result_3d = reshape(result_2d, prefix, suffix, to_order)
                dest_3d = reshape(@view(dest[iv, :]), prefix, to_order, suffix)
                permutedims!(dest_3d, result_3d, (1, 3, 2))
            end
        end
        src = dest
    end
    return out
end
```

This replaces all per-slice loops with single gemm calls plus permutedims. For the typical case (nd=1, ndim=3, n=30, m=44):
- Dim 1: `(44, 30) @ (30, 900)` — one gemm
- Dim 2: permute `(44, 30, 30)` → `(44, 30, 30)`, gemm `(1320, 30) @ (30, 44)`, permute back
- Dim 3: `(1936, 30) @ (30, 44)` — one gemm

**Also apply the same pattern to `_tensor_product_apply_dim!` in `src/tensor.jl`** for the square case.

### Pre-allocate workspace

Add workspace buffers to avoid allocation in inner loops:
- In `_proxycharge_to_pw!` and `_pw_to_proxy!`: currently allocate `Matrix{ComplexF64}` per call. Pre-allocate once and pass through.
- In `boxfgt!`: pre-allocate the ComplexF64 work matrices for charge→PW and PW→pot conversions outside the box loop.

### Test
```bash
julia --project -e 'using BoxDMK; include("test/test_solver.jl")'
julia --project -e 'using BoxDMK; include("test/test_proxy.jl")'
julia --project -e 'using BoxDMK; include("test/test_tensor.jl")'
julia --project -e 'using BoxDMK; include("test/test_passes.jl")'
```

---

## Phase 3: PW Half-Complex Symmetry (expected ~2x speedup)

### Problem
Julia uses full `nexp = npw^3` for PW expansions. Fortran uses `nexp_half = ((npw+1)/2) * npw^(ndim-1)`, exploiting the fact that for real source data, the PW expansion has conjugate symmetry along the last axis. This halves all PW storage and operations.

### Background from Fortran
The Fortran convention for 3D is: `pwexp(npw, npw, (npw+1)/2)`.
The last axis only stores half the frequencies. The other half is recovered via conjugation.

### Changes

**File: `src/planewave.jl`**

1. Add a function `pw_expansion_size_half(npw, ndim)`:
   ```julia
   pw_expansion_size_half(npw, ndim) = ((npw + 1) ÷ 2) * npw^(ndim - 1)
   ```

2. Modify `kernel_fourier_transform` to produce a half-sized output:
   Only iterate over `1:((npw+1)÷2)` for the first (outermost) dimension instead of `1:npw`. The kernel FT is real and symmetric, so `kernel_ft[i,j,k] == kernel_ft[npw+1-i,j,k]`.

3. Modify `build_pw_1d_phase_table` — no change needed (it's 1D).

4. Modify `build_pw_conversion_tables` to produce half-sized `tab_coefs2pw` and `tab_pw2pot`:
   `tab_coefs2pw` maps from `porder` coefficients to `nexp_half` PW entries.
   This is a tensor product where the last axis uses only `(npw+1)/2` terms.

5. Modify `compute_shift_vector!` to produce half-sized output.

**File: `src/boxfgt.jl`**

6. Update `_proxycharge_to_pw!` to produce half-sized PW expansion:
   The 3D tensor product transforms `(porder, porder, porder)` → `(npw, npw, (npw+1)/2)`.
   Only the last dimension uses half the frequencies.

7. Update `_pw_to_proxy!` to read half-sized PW expansion and reconstruct full using conjugation.

8. Update the M2L shift multiply to work on half-sized expansions.

9. Update `_pw_expansion_view` to use `nexp_half` instead of `nexp`.

**File: `src/types.jl`**

10. No structural change needed — just the sizes change.

### Test
```bash
julia --project -e 'using BoxDMK; include("test/test_planewave.jl")'
julia --project -e 'using BoxDMK; include("test/test_boxfgt.jl")'
julia --project -e 'using BoxDMK; include("test/test_solver.jl")'
```

---

## Phase 4: Threading (expected ~Nx for N threads)

### Problem
The Fortran uses `$OMP PARALLEL DO` for:
- Step 4: charge → PW conversion (per box at each level)
- Step 5: M2L shift + kernel FT multiply (per box at each level)
- Step 6: PW → potential conversion (per box at each level)

The Julia runs all of these single-threaded.

### Changes

**File: `src/boxfgt.jl`**

1. Thread the charge→PW loop:
   ```julia
   Threads.@threads for idx in eachindex(level_boxes)
       ibox = level_boxes[idx]
       # _proxycharge_to_pw!(...)
       # loc .= mp
   end
   ```

2. Thread the M2L + kernel_ft loop:
   ```julia
   Threads.@threads for idx in eachindex(level_boxes)
       ibox = level_boxes[idx]
       # shift_vec and loc are per-thread
       for jbox in lists.listpw[ibox]
           # shift + accumulate
       end
       # multiply kernel_ft
       # _pw_to_proxy!(...)
   end
   ```
   Note: each thread needs its own `shift_vec` buffer. Pre-allocate per thread.

**File: `src/passes.jl`**

3. The upward/downward passes already use `Threads.@threads` — verify they work correctly.

### Test
```bash
julia --project --threads=auto -e 'using BoxDMK; include("test/test_solver.jl")'
```

---

## Phase 5: Reduce Allocations (expected ~1.5-2x from GC reduction)

### Problem
212M allocations and 150 GC cycles for a single solve. Major offenders:
- `_proxycharge_to_pw!` / `_pw_to_proxy!` allocate temp matrices per call
- `_tensor_product_apply_rect!` allocates intermediate arrays per dimension
- `upward_pass!` / `downward_pass!` allocate `src_box` and `work` per iteration

### Changes

**File: `src/boxfgt.jl`**

1. Pre-allocate work arrays for charge↔PW conversions outside the box loop:
   ```julia
   src_work = Matrix{ComplexF64}(undef, nd, porder^ndim)
   pw_work = Matrix{ComplexF64}(undef, nd, nexp)
   # Pass these into _proxycharge_to_pw! and _pw_to_proxy!
   ```

2. Pre-allocate intermediate tensor product buffers.

**File: `src/proxy.jl`**

3. Add workspace-accepting variants of `_tensor_product_apply_rect!` that take pre-allocated buffers instead of allocating internally.

**File: `src/passes.jl`**

4. Pre-allocate `src_box` and `work` outside the thread loop (one per thread).

### Test
```bash
julia --project -e 'using BoxDMK; include("test/test_solver.jl")'
```

---

## Phase 6: Unified Multi-Level boxfgt (expected ~1.3x from reduced overhead)

### Problem
Currently `bdmk()` calls `setup_planewave_data` + `boxfgt!` in a loop per delta group (7 calls for eps=1e-6). Each call allocates and zeros a new `rmlexp` array. The Fortran allocates ONE `rmlexp` for all levels and processes all levels in a single pass.

### Changes

**File: `src/solver.jl`**

1. Allocate ONE `PlaneWaveData` for all needed levels (now feasible with half-complex nexp after Phase 3).

2. Process all delta groups in a structured multi-level pass:
   - Step 4: For each level with deltas, convert charges → PW (all levels)
   - Step 5: For each level, do M2L + kernel_ft multiply
   - Step 6: For each level, convert PW → potential

This eliminates redundant allocation/deallocation of rmlexp per delta group.

### Test
```bash
julia --project -e 'using BoxDMK; include("test/test_solver.jl")'
```

---

## Execution Order

| Phase | Expected Speedup | Cumulative | Risk |
|-------|-----------------|------------|------|
| 2: Tensor product gemm | 5-10x | 5-10x | Low — pure BLAS optimization |
| 3: PW half-complex | 2x | 10-20x | Medium — changes data layout everywhere |
| 4: Threading | 4-8x (4-8 threads) | 40-160x | Low — independent box processing |
| 5: Reduce allocations | 1.5-2x | 60-320x | Low — mechanical pre-allocation |
| 6: Unified multi-level | 1.3x | ~80-400x | Medium — architectural change |

Target: **Julia within 2-5x of Fortran** after all phases.

## Verification

After each phase:
1. Run `julia --project -e 'using BoxDMK; include("test/test_solver.jl")'` — must pass 24/24
2. Run the benchmark timing script to measure speedup
3. Run `julia --project -e 'using BoxDMK; include("test/test_cross_validation.jl")'` if time permits
