using BoxDMK
using Test

function _gaussian_source(alpha::Real, center::AbstractVector{<:Real})
    return x -> [exp(-Float64(alpha) * sum((x .- center) .^ 2))]
end

function _run_solver_smoke(kernel::BoxDMK.AbstractKernel; eps::Float64 = 1e-3)
    f = _gaussian_source(0.1, fill(0.5, 3))
    tree, fvals = build_tree(
        f,
        kernel,
        LegendreBasis();
        ndim = 3,
        norder = 6,
        eps = eps,
        boxlen = 1.0,
        nd = 1,
    )

    result = bdmk(tree, fvals, kernel; eps = eps)

    @test result isa BoxDMK.SolverResult
    @test size(result.pot) == size(fvals)
    @test result.grad === nothing
    @test result.hess === nothing
    @test result.target_pot === nothing
    @test result.target_grad === nothing
    @test result.target_hess === nothing
    @test maximum(abs.(result.pot)) > 0
end

@testset "Solver" begin
    @testset "Laplace" begin
        _run_solver_smoke(LaplaceKernel())
    end

    @testset "Yukawa" begin
        _run_solver_smoke(YukawaKernel(1.0))
    end

    @testset "SqrtLaplace" begin
        _run_solver_smoke(SqrtLaplaceKernel())
    end
end
