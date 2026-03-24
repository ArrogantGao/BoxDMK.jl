using BoxDMK
using LinearAlgebra
using Printf
using Random

println("=" ^ 72)
println("  BoxDMK.jl — Accuracy & Speed Benchmark")
println("=" ^ 72)
println()
@printf("Julia threads: %d\n", Threads.nthreads())
@printf("BLAS threads:  %d\n", BLAS.get_num_threads())
println()

# ── Source function (same as Fortran benchmark) ──────────────────────
const rsig = 4e-3
const ndim = 3
const rsign = (rsig * 1.0)^(ndim / 2.0)

function gaussian_2_rhs(x)
    c1 = [0.1, 0.02, 0.04]; s1 = rsig
    c2 = [0.03, -0.1, 0.05]; s2 = rsig / 2
    v1 = exp(-sum((x .- c1).^2) / s1^2) / (π * rsign)
    v2 = -0.5 * exp(-sum((x .- c2).^2) / s2^2) / (π * rsign)
    return [v1 + v2]
end

# ── Target points for accuracy comparison ────────────────────────────
Random.seed!(42)
targets = 0.1 .+ 0.8 .* rand(3, 50)

# ══════════════════════════════════════════════════════════════════════
# PART 1: Self-convergence (accuracy)
# ══════════════════════════════════════════════════════════════════════
println("─" ^ 72)
println("PART 1: Self-Convergence Accuracy (reference: eps=1e-7)")
println("─" ^ 72)

for (kernel, kname) in [
    (LaplaceKernel(), "Laplace"),
    (YukawaKernel(1.0), "Yukawa(β=1)"),
    (SqrtLaplaceKernel(), "SqrtLaplace"),
]
    println("\n  Kernel: $kname")
    @printf("  %10s  %8s  %6s  %12s  %12s\n", "eps", "nboxes", "nlev", "max|pot|", "rel_err_L2")
    @printf("  %10s  %8s  %6s  %12s  %12s\n", "-"^10, "-"^8, "-"^6, "-"^12, "-"^12)

    # Reference at eps=1e-7
    tree_ref, fvals_ref = build_tree(gaussian_2_rhs, kernel, LegendreBasis();
        ndim=3, norder=8, eps=1e-7, boxlen=1.0, nd=1)
    res_ref = bdmk(tree_ref, fvals_ref, kernel; eps=1e-7, targets=targets)
    pot_ref = res_ref.target_pot

    for eps_val in [1e-3, 1e-4, 1e-5, 1e-6]
        tree, fvals = build_tree(gaussian_2_rhs, kernel, LegendreBasis();
            ndim=3, norder=8, eps=eps_val, boxlen=1.0, nd=1)
        res = bdmk(tree, fvals, kernel; eps=eps_val, targets=targets)
        pot_test = res.target_pot

        nb = size(tree.centers, 2)
        nl = tree.nlevels
        maxpot = maximum(abs.(res.pot))
        rel_err = norm(pot_test .- pot_ref) / norm(pot_ref)

        @printf("  %10.0e  %8d  %6d  %12.4e  %12.4e\n", eps_val, nb, nl, maxpot, rel_err)
    end
end

# ══════════════════════════════════════════════════════════════════════
# PART 2: Speed benchmark
# ══════════════════════════════════════════════════════════════════════
println("\n" * "─" ^ 72)
println("PART 2: Speed Benchmark (LaplaceKernel)")
println("─" ^ 72)

@printf("\n  %8s  %8s  %8s  %6s  %10s  %10s  %10s  %12s\n",
    "norder", "eps", "nboxes", "nlev", "tree(s)", "solve(s)", "total(s)", "pts/sec")
@printf("  %8s  %8s  %8s  %6s  %10s  %10s  %10s  %12s\n",
    "-"^8, "-"^8, "-"^8, "-"^6, "-"^10, "-"^10, "-"^10, "-"^12)

for norder in [8, 12, 16]
    for eps_val in [1e-3, 1e-6]
        # Warm-up
        tree, fvals = build_tree(gaussian_2_rhs, LaplaceKernel(), LegendreBasis();
            ndim=3, norder=norder, eps=eps_val, boxlen=1.0, nd=1)
        bdmk(tree, fvals, LaplaceKernel(); eps=eps_val)

        # Timed tree build
        t_tree = @elapsed begin
            tree, fvals = build_tree(gaussian_2_rhs, LaplaceKernel(), LegendreBasis();
                ndim=3, norder=norder, eps=eps_val, boxlen=1.0, nd=1)
        end

        # Timed solve
        t_solve = @elapsed begin
            result = bdmk(tree, fvals, LaplaceKernel(); eps=eps_val)
        end

        nb = size(tree.centers, 2)
        nl = tree.nlevels
        total_pts = nb * norder^3
        pps = total_pts / t_solve

        @printf("  %8d  %8.0e  %8d  %6d  %10.3f  %10.3f  %10.3f  %12.2e\n",
            norder, eps_val, nb, nl, t_tree, t_solve, t_tree + t_solve, pps)
    end
end

# ══════════════════════════════════════════════════════════════════════
# PART 3: Fortran comparison
# ══════════════════════════════════════════════════════════════════════
println("\n" * "─" ^ 72)
println("PART 3: Julia vs Fortran (norder=16, eps=1e-6, Laplace)")
println("─" ^ 72)

# Run Fortran benchmark
fortran_bin = "/mnt/home/xgao1/codes/boxdmk/build/bench-fortran-same-problem"
fortran_output = ""
try
    fortran_output = read(`$fortran_bin`, String)
catch e
    println("  WARNING: could not run Fortran benchmark: $e")
end

f_tree = NaN; f_solve = NaN; f_nboxes = 0
if !isempty(fortran_output)
    m = match(r"tree_build_s=\s*([\d.]+)\s+solve_s=\s*([\d.]+).*nboxes=\s*(\d+)", fortran_output)
    if m !== nothing
        f_tree = parse(Float64, m[1])
        f_solve = parse(Float64, m[2])
        f_nboxes = parse(Int, m[3])
    end
end

# Julia equivalent
tree_j, fvals_j = build_tree(gaussian_2_rhs, LaplaceKernel(), LegendreBasis();
    ndim=3, norder=16, eps=1e-6, boxlen=1.18, nd=1)
bdmk(tree_j, fvals_j, LaplaceKernel(); eps=1e-6)  # warm-up

t_tree_j = @elapsed begin
    tree_j, fvals_j = build_tree(gaussian_2_rhs, LaplaceKernel(), LegendreBasis();
        ndim=3, norder=16, eps=1e-6, boxlen=1.18, nd=1)
end
t_solve_j = @elapsed begin
    res_j = bdmk(tree_j, fvals_j, LaplaceKernel(); eps=1e-6)
end
j_nboxes = size(tree_j.centers, 2)

@printf("\n  %15s  %12s  %12s  %12s\n", "Metric", "Fortran", "Julia", "Ratio(F/J)")
@printf("  %15s  %12s  %12s  %12s\n", "-"^15, "-"^12, "-"^12, "-"^12)
@printf("  %15s  %12.3f  %12.3f  %12.2f\n", "Tree build (s)", f_tree, t_tree_j, f_tree / t_tree_j)
@printf("  %15s  %12.3f  %12.3f  %12.2f\n", "Solve (s)", f_solve, t_solve_j, f_solve / t_solve_j)
@printf("  %15s  %12.3f  %12.3f  %12.2f\n", "Total (s)", f_tree + f_solve, t_tree_j + t_solve_j, (f_tree + f_solve) / (t_tree_j + t_solve_j))
@printf("  %15s  %12d  %12d  %12s\n", "nboxes", f_nboxes, j_nboxes, "-")

# ══════════════════════════════════════════════════════════════════════
# PART 4: Gradient & Hessian
# ══════════════════════════════════════════════════════════════════════
println("\n" * "─" ^ 72)
println("PART 4: Gradient & Hessian (norder=8, eps=1e-6, Laplace)")
println("─" ^ 72)

f_simple(x) = [exp(-100 * sum((x .- 0.5).^2))]
tree_g, fvals_g = build_tree(f_simple, LaplaceKernel(), LegendreBasis();
    ndim=3, norder=8, eps=1e-6, boxlen=1.0, nd=1)

# Pot only
bdmk(tree_g, fvals_g, LaplaceKernel(); eps=1e-6)  # warm-up
t_pot = @elapsed res_pot = bdmk(tree_g, fvals_g, LaplaceKernel(); eps=1e-6)

# Pot + grad
bdmk(tree_g, fvals_g, LaplaceKernel(); eps=1e-6, grad=true)
t_grad = @elapsed res_grad = bdmk(tree_g, fvals_g, LaplaceKernel(); eps=1e-6, grad=true)

# Pot + grad + hess
bdmk(tree_g, fvals_g, LaplaceKernel(); eps=1e-6, grad=true, hess=true)
t_full = @elapsed res_full = bdmk(tree_g, fvals_g, LaplaceKernel(); eps=1e-6, grad=true, hess=true)

@printf("\n  %20s  %12s  %12s  %12s\n", "Output", "Time (s)", "max|val|", "overhead")
@printf("  %20s  %12s  %12s  %12s\n", "-"^20, "-"^12, "-"^12, "-"^12)
@printf("  %20s  %12.3f  %12.4e  %12s\n", "Potential only", t_pot, maximum(abs.(res_pot.pot)), "-")
@printf("  %20s  %12.3f  %12.4e  %12.1f%%\n", "Pot + Gradient", t_grad, maximum(abs.(res_grad.grad)), 100 * (t_grad - t_pot) / t_pot)
@printf("  %20s  %12.3f  %12.4e  %12.1f%%\n", "Pot + Grad + Hess", t_full, maximum(abs.(res_full.hess)), 100 * (t_full - t_pot) / t_pot)

println("\n" * "=" ^ 72)
println("  BENCHMARK COMPLETE")
println("=" ^ 72)
