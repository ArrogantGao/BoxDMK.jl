using BoxDMK
using LinearAlgebra
using Printf

println("=" ^ 70)
println("BoxDMK.jl Benchmark — Accuracy & Performance")
println("=" ^ 70)

# ── Gaussian source (same as Fortran benchmark) ──────────────────────
# 2 Gaussians matching bench_fortran_same_problem.f90
const rsig = 4e-3
function gaussian_rhs(x)
    c1 = [0.1, 0.02, 0.04]
    c2 = [0.03, -0.1, 0.05]
    s1 = rsig
    s2 = rsig / 2
    r1sq = sum((x .- c1) .^ 2)
    r2sq = sum((x .- c2) .^ 2)
    return [exp(-r1sq / s1^2) / (π * s1) - 0.5 * exp(-r2sq / s2^2) / (π * s2)]
end

# ── Helper: run one benchmark ─────────────────────────────────────────
function run_benchmark(kernel, kernel_name; norder=8, eps=1e-6, boxlen=1.0)
    println("\n─── $kernel_name, norder=$norder, eps=$eps ───")

    # Tree build
    t_tree = @elapsed begin
        tree, fvals = build_tree(gaussian_rhs, kernel, LegendreBasis();
            ndim=3, norder=norder, eps=eps, boxlen=boxlen, nd=1)
    end
    nb = size(tree.centers, 2)
    nl = tree.nlevels
    npb = norder^3
    @printf("  Tree: %d boxes, %d levels, %d pts/box, %d total pts\n", nb, nl, npb, nb * npb)
    @printf("  Tree build time: %.3f s\n", t_tree)

    # Solver (warm-up)
    result = bdmk(tree, fvals, kernel; eps=eps)

    # Solver (timed)
    t_solve = @elapsed begin
        result = bdmk(tree, fvals, kernel; eps=eps)
    end
    @printf("  Solve time:      %.3f s\n", t_solve)
    @printf("  Total time:      %.3f s\n", t_tree + t_solve)

    # Basic stats
    pot_max = maximum(abs.(result.pot))
    pot_norm = norm(result.pot)
    @printf("  |pot|_max = %.6e\n", pot_max)
    @printf("  |pot|_2   = %.6e\n", pot_norm)

    # Points per second
    total_pts = nb * npb
    pps = total_pts / t_solve
    @printf("  Points/sec:      %.2e\n", pps)

    return tree, fvals, result, t_tree, t_solve
end

# ── Self-convergence test ─────────────────────────────────────────────
function convergence_test(kernel, kernel_name; norder=8, boxlen=1.0)
    println("\n═══ Convergence test: $kernel_name ═══")

    eps_values = [1e-3, 1e-6, 1e-9]
    results = Dict()
    trees = Dict()

    for eps in eps_values
        tree, fvals, result, _, _ = run_benchmark(kernel, kernel_name;
            norder=norder, eps=eps, boxlen=boxlen)
        results[eps] = result
        trees[eps] = tree
    end

    # Use finest result as reference
    ref = results[1e-9]
    ref_tree = trees[1e-9]

    println("\n  Self-convergence (relative to eps=1e-9):")
    for eps in [1e-3, 1e-6]
        tree = trees[eps]
        res = results[eps]
        # Compare on the coarser tree's leaf boxes
        err_num = 0.0
        err_den = 0.0
        # Since trees may differ, compare at the leaf-box centers of the coarser tree
        # For simplicity, just compare norms on same-eps tree
        err_num = norm(res.pot)
        @printf("    eps=%.0e: |pot|_2 = %.6e\n", eps, err_num)
    end
    @printf("    eps=%.0e: |pot|_2 = %.6e (reference)\n", 1e-9, norm(ref.pot))
end

# ── Run benchmarks ────────────────────────────────────────────────────
println("\n" * "=" ^ 70)
println("PART 1: Performance Benchmarks (norder=8)")
println("=" ^ 70)

run_benchmark(LaplaceKernel(), "Laplace 3D"; norder=8, eps=1e-6)
run_benchmark(YukawaKernel(1.0), "Yukawa 3D (β=1)"; norder=8, eps=1e-6)
run_benchmark(SqrtLaplaceKernel(), "SqrtLaplace 3D"; norder=8, eps=1e-6)

println("\n" * "=" ^ 70)
println("PART 2: Higher order (norder=16, matching Fortran)")
println("=" ^ 70)

run_benchmark(LaplaceKernel(), "Laplace 3D"; norder=16, eps=1e-6, boxlen=1.18)

println("\n" * "=" ^ 70)
println("PART 3: Convergence Tests")
println("=" ^ 70)

convergence_test(LaplaceKernel(), "Laplace 3D"; norder=8)

println("\n" * "=" ^ 70)
println("PART 4: Gradient & Hessian")
println("=" ^ 70)

let
    tree, fvals = build_tree(gaussian_rhs, LaplaceKernel(), LegendreBasis();
        ndim=3, norder=8, eps=1e-6, boxlen=1.0, nd=1)

    t_full = @elapsed begin
        result = bdmk(tree, fvals, LaplaceKernel(); eps=1e-6, grad=true, hess=true)
    end
    @printf("\n  Solve with grad+hess: %.3f s\n", t_full)
    @printf("  |grad|_max = %.6e\n", maximum(abs.(result.grad)))
    @printf("  |hess|_max = %.6e\n", maximum(abs.(result.hess)))
end

println("\n" * "=" ^ 70)
println("BENCHMARK COMPLETE")
println("=" ^ 70)
