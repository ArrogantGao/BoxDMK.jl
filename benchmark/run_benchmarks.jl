using BoxDMK
using LinearAlgebra
using Printf
using Random

const TWO_GAUSSIAN_C1 = [0.1, 0.02, 0.04]
const TWO_GAUSSIAN_C2 = [0.03, -0.1, 0.05]
const TWO_GAUSSIAN_S1 = 4e-3
const TWO_GAUSSIAN_S2 = 2e-3
const SELF_CONVERGENCE_EPS = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
const SELF_CONVERGENCE_REF_EPS = 1e-9
const SPEED_NORDERS = [8, 12, 16]
const SPEED_EPS = [1e-3, 1e-6]
const FORTRAN_BENCHMARK = "/mnt/home/xgao1/codes/boxdmk/build/bench-fortran-same-problem"

function two_gaussian_source(x)
    r1sq = sum((x .- TWO_GAUSSIAN_C1) .^ 2)
    r2sq = sum((x .- TWO_GAUSSIAN_C2) .^ 2)
    return [
        exp(-r1sq / TWO_GAUSSIAN_S1^2) / (π * TWO_GAUSSIAN_S1) -
        0.5 * exp(-r2sq / TWO_GAUSSIAN_S2^2) / (π * TWO_GAUSSIAN_S2),
    ]
end

gaussian_bump_source(x) = [exp(-100.0 * sum((x .- 0.5) .^ 2))]

kernel_label(::LaplaceKernel) = "LaplaceKernel"
kernel_label(kernel::YukawaKernel) = @sprintf("YukawaKernel(%.1f)", Float64(kernel.beta))
kernel_label(::SqrtLaplaceKernel) = "SqrtLaplaceKernel"

function relative_l2_error(approx, reference)
    approx_vec = vec(Float64.(approx))
    reference_vec = vec(Float64.(reference))
    denom = norm(reference_vec)
    denom == 0.0 && return norm(approx_vec)
    return norm(approx_vec - reference_vec) / denom
end

function generate_target_points(ntargets::Integer; seed::Integer = 20260319)
    rng = MersenneTwister(seed)
    return 0.1 .+ 0.8 .* rand(rng, 3, Int(ntargets))
end

function parse_fortran_benchmark_output(text::AbstractString)
    line = nothing
    for candidate in eachline(IOBuffer(text))
        occursin("BENCH_FORTRAN", candidate) || continue
        line = candidate
    end

    line === nothing && error("Could not find BENCH_FORTRAN summary line in Fortran benchmark output")

    float_pattern = "([+-]?(?:\\d+(?:\\.\\d*)?|\\.\\d+)(?:[Ee][+-]?\\d+)?)"
    int_pattern = "([+-]?\\d+)"

    function capture_float(name::AbstractString)
        matchobj = match(Regex("$(name)\\s*=\\s*$(float_pattern)"), line)
        matchobj === nothing && error("Could not parse $(name) from Fortran benchmark output")
        return parse(Float64, matchobj.captures[1])
    end

    function capture_int(name::AbstractString)
        matchobj = match(Regex("$(name)\\s*=\\s*$(int_pattern)"), line)
        matchobj === nothing && error("Could not parse $(name) from Fortran benchmark output")
        return parse(Int, matchobj.captures[1])
    end

    return (
        tree_build_s = capture_float("tree_build_s"),
        solve_s = capture_float("solve_s"),
        total_s = capture_float("total_s"),
        nboxes = capture_int("nboxes"),
        nlevels = capture_int("nlevels"),
    )
end

function build_problem(source, kernel; norder::Integer, eps::Real, boxlen::Real, ndim::Integer = 3, nd::Integer = 1)
    return build_tree(
        source,
        kernel,
        LegendreBasis();
        ndim = Int(ndim),
        norder = Int(norder),
        eps = Float64(eps),
        boxlen = Float64(boxlen),
        nd = Int(nd),
    )
end

function solve_problem(tree, fvals, kernel; eps::Real, grad::Bool = false, hess::Bool = false, targets = nothing)
    return bdmk(
        tree,
        fvals,
        kernel;
        eps = Float64(eps),
        grad = grad,
        hess = hess,
        targets = targets,
    )
end

function run_twice_timed(f)
    f()
    GC.gc()
    value = nothing
    elapsed = @elapsed value = f()
    return value, elapsed
end

function total_box_points(tree, fvals)
    return size(fvals, 2) * size(fvals, 3)
end

function warmup_solver()
    kernel = LaplaceKernel()
    tree, fvals = build_problem(two_gaussian_source, kernel; norder = 8, eps = 1e-3, boxlen = 1.0)
    solve_problem(tree, fvals, kernel; eps = 1e-3, targets = generate_target_points(1; seed = 1))
    return nothing
end

function print_section(title::AbstractString)
    println()
    println(title)
    println(repeat("=", length(title)))
end

function print_self_convergence_table(kernel, targets)
    label = kernel_label(kernel)
    print_section("Part 1: Self-Convergence Accuracy Test - $label")

    ref_tree, ref_fvals = build_problem(two_gaussian_source, kernel; norder = 8, eps = SELF_CONVERGENCE_REF_EPS, boxlen = 1.0)
    ref_result = solve_problem(ref_tree, ref_fvals, kernel; eps = SELF_CONVERGENCE_REF_EPS, targets = targets)
    ref_pot = vec(ref_result.target_pot)

    @printf("Reference eps: %.0e, nboxes: %d, nlevels: %d\n", SELF_CONVERGENCE_REF_EPS, size(ref_tree.centers, 2), ref_tree.nlevels)
    @printf("%10s | %10s | %16s\n", "eps", "nboxes", "relative_error")
    println(repeat("-", 43))

    for eps in SELF_CONVERGENCE_EPS
        tree, fvals = build_problem(two_gaussian_source, kernel; norder = 8, eps = eps, boxlen = 1.0)
        result = solve_problem(tree, fvals, kernel; eps = eps, targets = targets)
        err = relative_l2_error(vec(result.target_pot), ref_pot)
        @printf("%10.0e | %10d | %16.8e\n", eps, size(tree.centers, 2), err)
    end
end

function print_speed_benchmark_table()
    kernel = LaplaceKernel()
    print_section("Part 2: Speed Benchmark - LaplaceKernel")
    @printf("%6s | %10s | %8s | %8s | %11s | %11s | %11s | %13s\n",
        "norder", "eps", "nboxes", "nlevels", "tree_time", "solve_time", "total_time", "pts/sec")
    println(repeat("-", 97))

    for norder in SPEED_NORDERS
        for eps in SPEED_EPS
            (tree, fvals), tree_time = run_twice_timed(() -> build_problem(two_gaussian_source, kernel; norder = norder, eps = eps, boxlen = 1.0))
            _, solve_time = run_twice_timed(() -> solve_problem(tree, fvals, kernel; eps = eps))
            total_time = tree_time + solve_time
            pps = total_box_points(tree, fvals) / solve_time

            @printf("%6d | %10.0e | %8d | %8d | %11.6f | %11.6f | %11.6f | %13.3f\n",
                norder, eps, size(tree.centers, 2), tree.nlevels, tree_time, solve_time, total_time, pps)
        end
    end
end

function print_fortran_comparison_table()
    print_section("Part 3: Fortran Comparison")

    fortran_text = read(`$(FORTRAN_BENCHMARK)`, String)
    fortran_metrics = parse_fortran_benchmark_output(fortran_text)

    kernel = LaplaceKernel()
    (tree, fvals), tree_time = run_twice_timed(() -> build_problem(two_gaussian_source, kernel; norder = 16, eps = 1e-6, boxlen = 1.18))
    _, solve_time = run_twice_timed(() -> solve_problem(tree, fvals, kernel; eps = 1e-6))
    julia_total = tree_time + solve_time

    rows = [
        ("tree_build_s", fortran_metrics.tree_build_s, tree_time),
        ("solve_s", fortran_metrics.solve_s, solve_time),
        ("total_s", fortran_metrics.total_s, julia_total),
        ("nboxes", Float64(fortran_metrics.nboxes), Float64(size(tree.centers, 2))),
    ]

    @printf("%14s | %12s | %12s | %12s\n", "metric", "Fortran", "Julia", "ratio")
    println(repeat("-", 60))

    for (metric, fortran_value, julia_value) in rows
        ratio = julia_value / fortran_value
        if metric == "nboxes"
            @printf("%14s | %12d | %12d | %12.6f\n", metric, Int(round(fortran_value)), Int(round(julia_value)), ratio)
        else
            @printf("%14s | %12.6f | %12.6f | %12.6f\n", metric, fortran_value, julia_value, ratio)
        end
    end
end

function print_derivative_benchmark_table()
    kernel = LaplaceKernel()
    print_section("Part 4: Gradient/Hessian Check")

    tree, fvals = build_problem(gaussian_bump_source, kernel; norder = 8, eps = 1e-6, boxlen = 1.0)
    _, solve_time_plain = run_twice_timed(() -> solve_problem(tree, fvals, kernel; eps = 1e-6))
    result_with_derivs, solve_time_derivs = run_twice_timed(() -> solve_problem(tree, fvals, kernel; eps = 1e-6, grad = true, hess = true))

    max_grad = maximum(abs.(result_with_derivs.grad))
    max_hess = maximum(abs.(result_with_derivs.hess))

    @printf("%22s | %16s\n", "metric", "value")
    println(repeat("-", 42))
    @printf("%22s | %16d\n", "nboxes", size(tree.centers, 2))
    @printf("%22s | %16d\n", "nlevels", tree.nlevels)
    @printf("%22s | %16.6f\n", "solve_no_derivs_s", solve_time_plain)
    @printf("%22s | %16.6f\n", "solve_with_derivs_s", solve_time_derivs)
    @printf("%22s | %16.8e\n", "max|grad|", max_grad)
    @printf("%22s | %16.8e\n", "max|hess|", max_hess)
end

function selected_parts(args::Vector{String})
    isempty(args) && return ["part1", "part2", "part3", "part4"]
    valid = Set(["part1", "part2", "part3", "part4"])
    lowered = lowercase.(args)
    all(in(valid), lowered) || error("Unknown benchmark part in ARGS. Expected any of: part1, part2, part3, part4")
    return lowered
end

function main()
    BLAS.set_num_threads(1)
    warmup_solver()

    println("BoxDMK.jl Comprehensive Benchmark")
    println("=================================")
    @printf("Julia threads: %d\n", Threads.nthreads())
    @printf("BLAS threads: %d\n", BLAS.get_num_threads())

    parts = Set(selected_parts(collect(ARGS)))
    targets = generate_target_points(50)
    if "part1" in parts
        for kernel in (LaplaceKernel(), YukawaKernel(1.0), SqrtLaplaceKernel())
            print_self_convergence_table(kernel, targets)
        end
    end

    "part2" in parts && print_speed_benchmark_table()
    "part3" in parts && print_fortran_comparison_table()
    "part4" in parts && print_derivative_benchmark_table()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
