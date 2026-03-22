using BoxDMK
using Test

function _gaussian_source(alpha::Real, center::AbstractVector{<:Real})
    return x -> [exp(-Float64(alpha) * sum((x .- center) .^ 2))]
end

function _shared_pw_setup_tree()
    centers = reshape([0.5, 0.25, 0.75, 0.125, 0.375, 0.625, 0.875], 1, 7)
    children = zeros(Int, 2, 7)
    children[:, 1] .= (2, 3)
    children[:, 2] .= (4, 5)
    children[:, 3] .= (6, 7)

    return BoxDMK.BoxTree(
        1,
        2,
        centers,
        [1.0, 0.5, 0.25],
        [0, 1, 1, 2, 2, 3, 3],
        children,
        [[1], [2, 3], [2, 3], [4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7], [4, 5, 6, 7]],
        [0, 1, 1, 2, 2, 2, 2],
        LegendreBasis(),
        2,
    )
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

    @testset "Shared Normal PW Setup" begin
        tree = _shared_pw_setup_tree()
        proxy = BoxDMK.build_proxy_data(LegendreBasis(), tree.norder, 1, tree.ndim)
        delta_groups = BoxDMK.DeltaGroups(
            [(1, [0.01], [1.0]), (2, [0.001], [0.5])],
            Tuple{Float64, Float64}[],
            Tuple{Float64, Float64}[],
        )

        pw_data = BoxDMK._setup_normal_pw_data(tree, proxy, 1e-6; nd = 1, delta_groups = delta_groups)

        @test pw_data isa BoxDMK.PlaneWaveData
        @test pw_data.ifpwexp == [false, true, true, true, true, true, true]
        @test isempty(pw_data.tab_coefs2pw[1])
        @test !isempty(pw_data.tab_coefs2pw[2])
        @test !isempty(pw_data.tab_coefs2pw[3])
        @test pw_data.iaddr[:, 1] == [0, 0]
        @test all(pw_data.iaddr[1, ibox] > 0 for ibox in 2:7)
    end
end
