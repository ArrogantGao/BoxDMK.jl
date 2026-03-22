# BoxDMK.jl Optimization Plan — Phase 2: Fortran ccall for Hot Paths

## Current Bottlenecks (1 thread, eps=1e-6, norder=8, 393 boxes)

| Step | Time (ms) | % Total |
|------|-----------|---------|
| BoxFGT (charge↔PW + M2L) | 9,510 | 72% |
| Local/near-field | 1,479 | 11% |
| Upward pass | 663 | 5% |
| Downward pass | 680 | 5% |
| Density → proxy | 457 | 3% |
| Other | 484 | 4% |
| **Total** | **13,293** | |

## Strategy

Replace the 3 hottest Julia functions with direct ccall to Fortran compiled routines from `libboxdmk.so`. These routines use optimized BLAS zgemm with Fortran-native memory layout, avoiding Julia's overhead from transposes, permutedims, and GC.

## Step 1: Replace PW conversions with Fortran ccall

### 1a. `dmk_proxycharge2pw_3d_` — proxy charges → PW expansion

**Fortran signature:**
```fortran
subroutine dmk_proxycharge2pw_3d(nd, n, coefs, npw, tab_coefs2pw, pwexp)
  integer :: nd, n, npw
  real*8 :: coefs(n, n, n, nd)           ! input: proxy charges
  complex*16 :: tab_coefs2pw(npw, n)     ! input: 1D conversion table
  complex*16 :: pwexp(npw, npw, (npw+1)/2, nd)  ! output: half-complex PW expansion
```

**Julia ccall:**
```julia
ccall((:dmk_proxycharge2pw_3d_, LIBBOXDMK), Cvoid,
    (Ref{Cint}, Ref{Cint}, Ptr{Cdouble}, Ref{Cint}, Ptr{ComplexF64}, Ptr{ComplexF64}),
    nd, n, coefs, npw, tab_coefs2pw, pwexp)
```

**Integration:** Replace `_proxycharge_to_pw!` body with this ccall when ndim==3.

Key data layout difference: Fortran expects `coefs(n, n, n, nd)` — this is the proxy charge data stored as a real array in column-major, with the proxy grid varying fastest and nd last. The Julia proxy_charges are stored as `(ncbox, nd, nboxes)` where ncbox = porder^3. So `proxy_charges[:, id, ibox]` is already `(porder^3,)` which reshapes to `(porder, porder, porder)` in column-major. For nd=1, this is just a pointer to the start.

BUT: the Fortran coefs are `real*8` while the Julia function currently converts to ComplexF64 before applying the tensor product. The Fortran does the real→complex conversion internally. So we pass the real proxy charges directly.

### 1b. `dmk_pw2proxypot_3d_` — PW expansion → proxy potential

**Fortran signature:**
```fortran
subroutine dmk_pw2proxypot_3d(nd, n, npw, pwexp, tab_pw2coefs, coefs)
  integer :: nd, n, npw
  complex*16 :: pwexp(npw, npw, (npw+1)/2, nd)  ! input: half-complex PW expansion
  complex*16 :: tab_pw2coefs(npw, n)             ! input: 1D inverse conversion table
  real*8 :: coefs(n, n, n, nd)                   ! output: proxy potential (REAL part)
```

**Integration:** Replace `_pw_to_proxy!` body with this ccall when ndim==3.

NOTE: the Fortran `tab_pw2coefs` is `(npw, n)` while Julia's `tab_pw2pot` is `(porder, npw)` — these are TRANSPOSES of each other! Need to pass `transpose(tab_pw2pot)` or `conj(tab_coefs2pw)`.

Actually, checking the Fortran code: `tab_pw2pot(j,i)` where j is PW index and i is coeff index. So it's `(npw, porder)`. Julia's `tab_pw2pot` from `build_pw_conversion_tables` is `(porder, npw)`. So we need the transpose for the Fortran call.

### 1c. `dmk_shiftpw_` — PW shift (M2L translation)

**Fortran signature:**
```fortran
subroutine dmk_shiftpw(nd, nexp, pwexp1, pwexp2, wshift)
  integer :: nd, nexp
  complex*16 :: pwexp1(nexp, nd)  ! input: source multipole PW
  complex*16 :: pwexp2(nexp, nd)  ! output: target local PW (accumulated)
  complex*16 :: wshift(nexp)      ! input: diagonal shift operator
```

**Integration:** Replace `loc .+= src .* shift_vec` in boxfgt! with this ccall.

### 1d. `dmk_multiply_kernelft_` — kernel Fourier transform multiply

**Fortran signature:**
```fortran
subroutine dmk_multiply_kernelft(nd, nexp, pwexp, wpwexp)
  integer :: nd, nexp
  complex*16 :: pwexp(nexp, nd)   ! input/output: PW expansion
  real*8 :: wpwexp(nexp)          ! input: kernel FT weights
```

**Integration:** Replace `loc .*= kernel_ft` in boxfgt! with this ccall.

### Implementation for Step 1

Create `src/fortran_hotpaths.jl` with:

```julia
const LIBBOXDMK = "/mnt/home/xgao1/codes/boxdmk/build/libboxdmk.so"

function _fortran_proxycharge2pw_3d!(pwexp, coefs, tab_coefs2pw, nd, porder, npw)
    # coefs: (porder^3,) real array for one box, one density
    # pwexp: (nexp_half,) complex array
    # tab_coefs2pw: (npw, porder) complex matrix
    ccall((:dmk_proxycharge2pw_3d_, LIBBOXDMK), Cvoid,
        (Ref{Cint}, Ref{Cint}, Ptr{Cdouble}, Ref{Cint}, Ptr{ComplexF64}, Ptr{ComplexF64}),
        Cint(nd), Cint(porder), coefs, Cint(npw), tab_coefs2pw, pwexp)
end

function _fortran_pw2proxypot_3d!(coefs, pwexp, tab_pw2coefs, nd, porder, npw)
    ccall((:dmk_pw2proxypot_3d_, LIBBOXDMK), Cvoid,
        (Ref{Cint}, Ref{Cint}, Ref{Cint}, Ptr{ComplexF64}, Ptr{ComplexF64}, Ptr{Cdouble}),
        Cint(nd), Cint(porder), Cint(npw), pwexp, tab_pw2coefs, coefs)
end

function _fortran_shiftpw!(pwexp2, pwexp1, wshift, nd, nexp)
    ccall((:dmk_shiftpw_, LIBBOXDMK), Cvoid,
        (Ref{Cint}, Ref{Cint}, Ptr{ComplexF64}, Ptr{ComplexF64}, Ptr{ComplexF64}),
        Cint(nd), Cint(nexp), pwexp1, pwexp2, wshift)
end

function _fortran_multiply_kernelft!(pwexp, wpwexp, nd, nexp)
    ccall((:dmk_multiply_kernelft_, LIBBOXDMK), Cvoid,
        (Ref{Cint}, Ref{Cint}, Ptr{ComplexF64}, Ptr{Cdouble}),
        Cint(nd), Cint(nexp), pwexp, wpwexp)
end
```

Then modify `boxfgt!` in `src/boxfgt.jl` to call these instead of the Julia implementations.

**CRITICAL DATA LAYOUT ISSUE:**
- Fortran `dmk_proxycharge2pw_3d` expects `coefs(n,n,n,nd)` as real*8
- The Julia proxy_charges are `(ncbox, nd, nboxes)` where `proxy_charges[:, id, ibox]` is `(porder^3,)`
- For nd=1, `proxy_charges[:, 1, ibox]` is contiguous and can be passed directly as `Ptr{Cdouble}`
- The Fortran output `pwexp(npw, npw, npw2, nd)` is the half-complex expansion stored column-major
- The Julia `_pw_expansion_view` returns a view into rmlexp as `(nexp_half, nd)` — this should match `(npw*npw*npw2, nd)` in column-major

ALSO: the Fortran uses `tab_coefs2pw(npw, n)` while Julia stores `tab_coefs2pw` as `(npw, porder)` from `build_pw_conversion_tables`. These should match since `n = porder`.

## Step 2: Replace local/near-field with Fortran ccall

### `bdmk_tens_prod_to_potloc_`

**Fortran signature:**
```fortran
subroutine bdmk_tens_prod_to_potloc(ndim, nd, n, ws, fvals, pot,
    ntab, tab_loc, ind_loc, ixyz)
  integer :: ndim, nd, n, ntab
  real*8 :: ws(*)         ! SOG weights
  real*8 :: fvals(n**ndim, nd)  ! source values
  real*8 :: pot(n**ndim, nd)    ! output potential (accumulated)
  real*8 :: tab_loc(n, n, ntab) ! 1D local tables
  integer :: ind_loc(2, n+1, ntab) ! sparse patterns
  integer :: ixyz(ndim)   ! offset indices for each dimension
```

This is the function that applies the sparse tensor product for ONE source box → ONE target box, for ALL deltas at once.

**Integration:** Replace `_apply_local_sparse_3d!` loop over deltas with ONE call to this Fortran routine.

The key difference: the Fortran function processes all deltas (ntab = ndeltas) in one call, while the Julia loops over deltas individually. The Fortran also handles the sparse pattern internally.

Actually, looking more carefully at the Fortran interface, `ntab` is the number of 1D tables, and `ixyz` selects which table to use per dimension. The Julia's `_apply_local_sparse_3d!` is called per-delta, so we'd need to restructure the caller.

Better approach: call `bdmk_tens_prod_to_potloc_` for each (target, source) pair, passing all delta tables at once.

## Step 3: Replace density↔proxy conversions with Fortran ccall

### `bdmk_density2proxycharge_`

**Fortran signature:**
```fortran
subroutine bdmk_density2proxycharge(ndim, nd, nin, fin, nout, fout, umat0, sc)
  integer :: ndim, nd, nin, nout
  real*8 :: fin(nin**ndim, nd)    ! input density values
  real*8 :: fout(nout**ndim, nd)  ! output proxy charges
  real*8 :: umat0(nout, nin)      ! 1D interpolation matrix
  real*8 :: sc                    ! scaling factor
```

This handles the density → proxy charge conversion via 3D tensor product of a real interpolation matrix. Much faster than the Julia version which converts to complex.

### `bdmk_proxypot2pot_`

```fortran
subroutine bdmk_proxypot2pot(ndim, nd, nin, fin, nout, fout, umat0)
```
Same pattern for proxy potential → user grid potential.

## Step 4: Replace upward/downward pass tensor products

The upward/downward passes use `tensor_product_apply!` with square `porder × porder` matrices. The Fortran equivalent is also `bdmk_density2proxycharge_` with `nin = nout = porder` and `umat0 = c2p_transmat` or `p2c_transmat`.

## Execution Plan

### Phase A: PW conversions (highest impact, ~72% of time)

1. Create `src/fortran_hotpaths.jl` with ccall wrappers for:
   - `dmk_proxycharge2pw_3d_`
   - `dmk_pw2proxypot_3d_`
   - `dmk_shiftpw_`
   - `dmk_multiply_kernelft_`

2. Modify `boxfgt!` to use Fortran routines:
   - Replace `_proxycharge_to_pw!` call with `_fortran_proxycharge2pw_3d!`
   - Replace `loc .+= src .* shift_vec` with `_fortran_shiftpw!`
   - Replace `loc .*= kernel_ft` with `_fortran_multiply_kernelft!`
   - Replace `_pw_to_proxy!` call with `_fortran_pw2proxypot_3d!`

3. Key data layout changes needed in boxfgt!:
   - The Fortran shift matrix needs to be pre-computed in Fortran format `(nexp_half, ntranslations)` instead of computing shift vectors on-the-fly
   - OR keep the on-the-fly approach and use `_fortran_shiftpw!` with the computed shift_vec
   - The proxy charges need to be passed as real*8 pointers, not wrapped in complex transposes
   - The conversion table `tab_coefs2pw` layout must match Fortran's `(npw, porder)`

4. Test: `julia --project -e 'using BoxDMK; include("test/test_solver.jl")'`

### Phase B: Local/near-field (11% of time)

5. Add ccall wrapper for `bdmk_tens_prod_to_potloc_`
6. Modify `apply_local!` to call Fortran per (target, source) box pair
7. Test

### Phase C: Density↔proxy + passes (8% of time)

8. Add ccall wrappers for `bdmk_density2proxycharge_` and `bdmk_proxypot2pot_`
9. Modify `_density_to_proxy_leaves!`, `proxy_to_potential!`, `upward_pass!`, `downward_pass!`
10. Test

## Expected Results

| Step | Current (ms) | Expected (ms) | Speedup |
|------|-------------|---------------|---------|
| BoxFGT | 9,510 | ~1,000 | ~10x |
| Local | 1,479 | ~300 | ~5x |
| Passes + proxy | 1,800 | ~200 | ~9x |
| **Total** | **13,293** | **~2,000** | **~7x** |

Target: Julia solve in **~2s** for the 393-box case (vs Fortran's 0.525s for 1129 boxes).
