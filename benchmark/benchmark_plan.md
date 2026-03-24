# Benchmark Plan: BoxDMK.jl vs Fortran BoxDMK

## Benchmark 1: Accuracy — Self-convergence
Test that increasing precision (decreasing eps) improves accuracy.
- Fix source: 2 Gaussians (matching Fortran benchmark)
- Fix norder=8, boxlen=1.0
- Sweep eps = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
- Use eps=1e-10 as "reference" solution
- Report: relative L2 error vs eps for each kernel (Laplace, Yukawa, SqrtLaplace)

## Benchmark 2: Accuracy — Against Fortran (via ccall)
Call the Fortran shared library from Julia via ccall to get Fortran results on the same problem.
- Use libboxdmk.so C API
- Same 2-Gaussian source, norder=16, eps=1e-6, boxlen=1.18
- Compare Julia pot vs Fortran pot on the same tree grid
- Report: relative L2 difference

## Benchmark 3: Speed — Julia timing
- Same 2-Gaussian source
- norder = 8, 12, 16
- eps = 1e-3, 1e-6, 1e-9
- Report: tree build time, solve time, total time, nboxes, points/sec

## Benchmark 4: Speed — Julia vs Fortran
- Same problem as Fortran benchmark: 2 Gaussians, norder=16, eps=1e-6, boxlen=1.18
- Compare tree_build_s and solve_s
- Report: speedup ratio

## Benchmark 5: Gradient & Hessian accuracy
- Use polynomial source where exact potential/gradient/Hessian are known
- Verify gradient and Hessian converge with norder
